"""FlexScribe: local web app that transcribes therapy sessions (Whisper), writes clinical summaries
(Gemma via llama.cpp) and tracks psychological-flexibility processes (hexaflex) across sessions.

Folder layout (one folder per patient, one subfolder per session):
    <patients_dir>/<Paziente>/<YYYY-MM-DD_titolo>/
        audio.<ext>        original upload
        audio16k.wav       converted for whisper
        transcript.txt     editable text
        transcript.json    whisper segments with timestamps
        meta.json          date, title, model, status
        sintesi_<skill>.md clinical summaries written by the local LLM (llama.cpp + Gemma)

Summary "skills" are the Markdown prompt files in ./skills (add or edit them freely).
"""
import atexit
import glob
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("TRASCRIZIONI_CONFIG", APP_DIR / "config.json"))
DEFAULT_CONFIG = {
    # optional whisper.cpp build (whisper-cli.exe + ggml-*.bin models), only used for the slow CPU fallback
    "whisper_dir": str(APP_DIR / "whisper.cpp"),
    "patients_dir": str(Path.home() / "Documents" / "Pazienti"),
    "language": "it",
    "default_model": "ov:whisper-large-v3-turbo-int8-ov:GPU",
    "threads": max(1, (os.cpu_count() or 4) - 1),
    "host": "127.0.0.1",
    "port": 8765,
    # local LLM for clinical summaries (llama.cpp "llama serve")
    "llm_exe": shutil.which("llama") or str(Path.home() / "AppData/Local/Microsoft/WindowsApps/llama.exe"),
    "llm_model": next((p for p in glob.glob(str(Path.home() / ".cache/huggingface/hub/models--google--gemma-4*/snapshots/*/*.gguf"))
                       if "mmproj" not in p), ""),
    "llm_port": 8766,
    "llm_ctx": 32768,
    "llm_threads": max(1, (os.cpu_count() or 4) - 1),
    "llm_gpu_layers": 999,  # all layers on the GPU; 0 = CPU only
    "llm_device": "Vulkan0",  # Intel GPU via Vulkan ("none" = CPU only; see `llama serve --list-devices`)
    "llm_temperature": 0.3,
    "llm_idle_minutes": 15,  # keep Gemma (and the transcript it already read) loaded between edits
    # speculative decoding that copies text already in the prompt: faster rewrites, identical output ("" = off)
    "llm_spec": "ngram-simple",
    "default_skill": "02-seduta-psicoterapia",
    "auto_process_card": True,  # fill the hexaflex card right after each new summary (transcript still cached)
    # vocabulary hint for Whisper: helps it spell ACT terms correctly (and nudges punctuation)
    "whisper_prompt": "Seduta di psicoterapia cognitivo-comportamentale di terza generazione: ACT (Acceptance and Commitment Therapy), mindfulness, CFT, DBT. Accettazione, defusione, valori, azione impegnata, autocompassione, rimuginio, evitamento esperienziale.",
}
AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".aac", ".wma", ".opus", ".mp4", ".webm", ".amr", ".3gp"}
SKILLS_DIR = APP_DIR / "skills"
MODELS_DIR = APP_DIR / "models"


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig")) if CONFIG_PATH.exists() else {}
    cfg.update(saved)
    if set(cfg) - set(saved):  # write back new keys so they are visible and editable
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


CFG = load_config()
ROOT = Path(CFG["patients_dir"])
ROOT.mkdir(parents=True, exist_ok=True)
WHISPER_DIR = Path(CFG["whisper_dir"])
WHISPER_CLI = WHISPER_DIR / "whisper-cli.exe"


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    local = APP_DIR / "ffmpeg.exe"
    if not local.exists():
        # copy the binary bundled with imageio-ffmpeg next to the app: its site-packages path
        # can exceed the Windows 260-char limit for launching executables
        import imageio_ffmpeg
        hits = glob.glob(str(Path(imageio_ffmpeg.__file__).parent / "binaries" / "ffmpeg*.exe"))
        shutil.copy2(hits[0] if hits else imageio_ffmpeg.get_ffmpeg_exe(), local)
    return str(local)


FFMPEG = find_ffmpeg()

# ---------------------------------------------------------------- helpers

def safe_name(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name).strip().strip(".")
    if not name:
        raise HTTPException(400, "Nome non valido")
    return name[:100]


def inside_root(*parts: str) -> Path:
    p = ROOT.joinpath(*parts).resolve()
    if ROOT.resolve() not in p.parents and p != ROOT.resolve():
        raise HTTPException(400, "Percorso non valido")
    return p


def patient_dir(patient: str) -> Path:
    p = inside_root(patient)
    if not p.is_dir():
        raise HTTPException(404, "Paziente non trovato")
    return p


def session_dir(patient: str, session: str) -> Path:
    s = inside_root(patient, session)
    if not s.is_dir():
        raise HTTPException(404, "Sessione non trovata")
    return s


def read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def write_json(p: Path, data):
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def find_audio(sdir: Path):
    """The original audio: prefer audio.*, else any audio file dropped in manually."""
    files = [f for f in sdir.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTS and f.name != "audio16k.wav"]
    files.sort(key=lambda f: (not f.stem.lower().startswith("audio"), f.name))
    return files[0] if files else None


def session_info(sdir: Path):
    meta = read_json(sdir / "meta.json", {}) or {}
    m = re.match(r"(\d{4}-\d{2}-\d{2})_?(.*)", sdir.name)
    has_text = (sdir / "transcript.txt").exists()
    job = JOBS.get(job_key(sdir.parent.name, sdir.name))
    if job and job["status"] in ("queued", "converting", "loading", "transcribing"):
        status = job["status"]
    elif has_text:
        status = "done"
    elif job and job["status"] == "error":
        status = "error"
    else:
        status = "pending" if find_audio(sdir) else "empty"
    preview = ""
    if has_text:
        preview = (sdir / "transcript.txt").read_text(encoding="utf-8", errors="replace")[:220].strip()
    return {
        "id": sdir.name,
        "date": meta.get("date") or (m.group(1) if m else ""),
        "title": meta.get("title") or ((m.group(2) if m else sdir.name).replace("_", " ")),
        "model": meta.get("model", ""),
        "status": status,
        "has_audio": find_audio(sdir) is not None,
        "preview": preview,
        "has_summary": any(sdir.glob("sintesi_*.md")),
        "has_processi": (sdir / "processi.json").exists(),
        "progress": job["progress"] if job else None,
        "error": job.get("error") if job else None,
        "summary_job": summary_job_info(sdir.parent.name, sdir.name),
    }


def summary_job_info(patient, session):
    j = JOBS.get(summary_key(patient, session))
    if not j or (j["status"] == "done" and time.time() - j["started"] > 60):
        return None
    return {k: j.get(k) for k in ("status", "progress", "skill", "mode", "partial", "error", "details")}


DATE_IN_NAME = re.compile(r"(20\d{2})[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])")


def adopt_loose_audio(pdir: Path):
    """Audio dropped straight into a patient folder gets its own dated session folder.

    The date comes from the file name when it contains one (e.g. 'REC_20261018_1430.m4a',
    'Registrazione 2026-10-18.mp3'), otherwise from the file's modification time.
    """
    for f in list(pdir.iterdir()):
        if not f.is_file() or f.suffix.lower() not in AUDIO_EXTS:
            continue
        st = f.stat()
        if time.time() - st.st_mtime < 15:
            continue  # probably still being copied; pick it up on a later refresh
        m = DATE_IN_NAME.search(f.stem)
        d = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else time.strftime("%Y-%m-%d", time.localtime(st.st_mtime))
        title = re.sub(r"[_\s]+", " ", DATE_IN_NAME.sub("", f.stem)).strip(" -_.")
        title = re.sub(r"\b([01]\d|2[0-3])([0-5]\d)([0-5]\d)?\b", r"ore \1:\2", title) or "sessione"  # REC_..._143000
        try:
            base = f"{d}_{safe_name(title)[:60].replace(' ', '_')}"
        except HTTPException:
            base = f"{d}_sessione"
        name, n = base, 2
        while (pdir / name).exists():
            name, n = f"{base}_{n}", n + 1
        sdir = pdir / name
        try:
            sdir.mkdir()
            shutil.move(str(f), sdir / f"audio{f.suffix.lower()}")
        except OSError:  # file locked (still copying / open in a player): retry on the next refresh
            if sdir.exists() and not any(sdir.iterdir()):
                sdir.rmdir()
            continue
        write_json(sdir / "meta.json", {"date": d, "title": title, "original_filename": f.name, "added": "cartella"})


def list_sessions(pdir: Path):
    adopt_loose_audio(pdir)
    sessions = [session_info(s) for s in pdir.iterdir() if s.is_dir()]
    sessions.sort(key=lambda s: (s["date"], s["id"]), reverse=True)
    return sessions


# ---------------------------------------------------------------- job queue

JOBS: dict = {}
JOB_QUEUE: "queue.Queue[str]" = queue.Queue()
PROGRESS_RE = re.compile(r"progress\s*=\s*(\d+)%")
ACTIVE = ("queued", "converting", "transcribing", "loading", "reading", "writing")


def job_key(patient, session):
    return f"{patient}/{session}"


def summary_key(patient, session):
    return f"{patient}/{session}#sintesi"


def enqueue(patient: str, session: str, model: str, auto_skill: str = ""):
    key = job_key(patient, session)
    if key in JOBS and JOBS[key]["status"] in ACTIVE:
        return
    JOBS[key] = {"kind": "transcribe", "patient": patient, "session": session, "status": "queued", "progress": 0,
                 "model": model, "auto_skill": auto_skill, "started": time.time()}
    JOB_QUEUE.put(key)


def enqueue_summary(patient: str, session: str, skill: str, instructions: str = "", mode: str = "generate"):
    """mode: generate (new summary) | revise (Gemma applies the clinician's request) | grammar (proofreading only)"""
    key = summary_key(patient, session)
    if key in JOBS and JOBS[key]["status"] in ACTIVE:
        raise HTTPException(409, "C'è già una sintesi in corso per questa sessione")
    JOBS[key] = {"kind": "summary", "patient": patient, "session": session, "status": "queued", "progress": 0,
                 "skill": skill, "instructions": instructions, "mode": mode, "partial": "", "started": time.time()}
    JOB_QUEUE.put(key)


# ---------------------------------------------------------------- skills (summary templates)

def load_context():
    """skills/_contesto.md: clinical context (ACT orientation) shared by every template."""
    f = SKILLS_DIR / "_contesto.md"
    return f.read_text(encoding="utf-8-sig").strip() if f.exists() else ""


def load_skills():
    skills = []
    for f in sorted(SKILLS_DIR.glob("*.md")):
        if f.name.startswith("_"):
            continue
        raw = f.read_text(encoding="utf-8-sig")
        meta, body = {}, raw
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", raw, re.S)
        if m:
            for line in m.group(1).splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip().lower()] = v.strip()
            body = m.group(2)
        skills.append({"id": f.stem, "name": meta.get("nome", f.stem), "description": meta.get("descrizione", ""),
                       "prompt": body.strip()})
    return skills


def get_skill(skill_id: str):
    for s in load_skills():
        if s["id"] == skill_id:
            return s
    raise HTTPException(404, "Modello di sintesi non trovato")


def list_summaries(sdir: Path):
    names = {s["id"]: s["name"] for s in load_skills()}
    out = []
    for f in sorted(sdir.glob("sintesi_*.md"), key=lambda f: f.stat().st_mtime, reverse=True):
        sid = f.stem[len("sintesi_"):]
        out.append({"skill": sid, "name": names.get(sid, sid), "text": f.read_text(encoding="utf-8-sig", errors="replace"),
                    "updated": time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime)),
                    "versions": len(list_versions(sdir, sid))})
    return out


def list_versions(sdir: Path, skill_id: str):
    """Earlier versions of a summary, oldest first, kept in <session>/versioni/ for undo."""
    return sorted((sdir / "versioni").glob(f"sintesi_{skill_id}_*.md"))


def save_version(sdir: Path, skill_id: str):
    cur = sdir / f"sintesi_{skill_id}.md"
    if cur.exists():
        (sdir / "versioni").mkdir(exist_ok=True)
        shutil.copy2(cur, sdir / "versioni" / f"sintesi_{skill_id}_{time.strftime('%Y%m%d-%H%M%S')}.md")


# ---------------------------------------------------------------- local LLM (llama.cpp server)

class LLM:
    """Starts `llama serve` on demand and stops it when idle, so Gemma's RAM is free while whisper runs."""

    def __init__(self):
        self.proc = None
        self.last_used = 0.0
        self.busy = False
        self.lock = threading.Lock()
        self.base = f"http://127.0.0.1:{CFG['llm_port']}"
        self.log = APP_DIR / "llama.log"

    def healthy(self):
        try:
            with urllib.request.urlopen(self.base + "/health", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    def ensure(self):
        with self.lock:
            if self.proc and self.proc.poll() is None and self.healthy():
                return
            self._stop()
            exe, model = CFG["llm_exe"], CFG["llm_model"]
            if not exe or not Path(exe).exists():
                raise RuntimeError(f"llama.cpp non trovato: imposta 'llm_exe' in config.json (attuale: {exe!r})")
            if not model or not Path(model).exists():
                raise RuntimeError(f"Modello Gemma non trovato: imposta 'llm_model' in config.json (attuale: {model!r})")
            cmd = [exe, "serve", "-m", model, "--host", "127.0.0.1", "--port", str(CFG["llm_port"]),
                   "-c", str(CFG["llm_ctx"]), "-np", "1", "-t", str(CFG["llm_threads"]),
                   "-ngl", str(CFG["llm_gpu_layers"]), "--device", CFG["llm_device"],
                   *(["--spec-type", CFG["llm_spec"]] if CFG["llm_spec"] else []),
                   "--reasoning", "off", "--offline"]
            logf = open(self.log, "w", encoding="utf-8", errors="replace")
            self.proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                         creationflags=subprocess.CREATE_NO_WINDOW)
            deadline = time.time() + 300
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError("Il server LLM si è chiuso all'avvio:\n" + self._log_tail())
                if self.healthy():
                    return
                time.sleep(1)
            self._stop()
            raise RuntimeError("Il server LLM non ha risposto entro 5 minuti:\n" + self._log_tail())

    def _log_tail(self):
        try:
            return self.log.read_text(encoding="utf-8", errors="replace")[-800:]
        except Exception:
            return ""

    def _stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def stop(self):
        with self.lock:
            self._stop()

    def running(self):
        return bool(self.proc and self.proc.poll() is None)

    def stream_chat(self, messages, on_prompt_progress, on_text, response_format=None, max_tokens=3000):
        payload = {"messages": messages, "stream": True, "temperature": CFG["llm_temperature"],
                   "max_tokens": max_tokens, "return_progress": True}
        if response_format:
            payload["response_format"] = response_format  # JSON schema -> grammar-constrained output
        body = json.dumps(payload).encode()
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        text, timings = "", {}
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                timings = chunk.get("timings") or timings
                pp = chunk.get("prompt_progress")
                if pp and pp.get("total"):
                    on_prompt_progress(int(100 * pp.get("processed", 0) / pp["total"]))
                for ch in chunk.get("choices", []):
                    piece = (ch.get("delta") or {}).get("content") or ""
                    if piece:
                        text += piece
                        on_text(text)
        return text, timings


LLM_SERVER = LLM()
atexit.register(LLM_SERVER.stop)


def kill_orphan_llm():
    """A previous run closed abruptly (window closed) can leave its llama server holding ~5 GB of RAM."""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='llama.exe'\" | "
          f"Where-Object {{ $_.CommandLine -match '--port {CFG['llm_port']}( |$)' }} | "
          "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True,
                   creationflags=subprocess.CREATE_NO_WINDOW)


kill_orphan_llm()  # synchronously, before the job worker can start a new server on the same port


def llm_idle_watcher():
    while True:
        time.sleep(30)
        idle = time.time() - LLM_SERVER.last_used > CFG["llm_idle_minutes"] * 60
        if LLM_SERVER.running() and not LLM_SERVER.busy and idle:
            LLM_SERVER.stop()


def prompt_prefix(patient: str, meta: dict, skill: dict, transcript: str):
    """System prompt + transcript block shared by every Gemma job on a session (summary, edits, process card).
    Keeping it byte-identical lets llama.cpp reuse the transcript it already read."""
    system = (load_context() + "\n\n" + skill["prompt"]).strip()
    head = (f"Paziente: {patient}\nData della sessione: {meta.get('date', '')}\n"
            f"Titolo: {meta.get('title', '')}\n\nTRASCRIZIONE:\n\"\"\"\n{transcript}\n\"\"\"")
    return system, head


def run_summary(job: dict):
    sdir = ROOT / job["patient"] / job["session"]
    txt = sdir / "transcript.txt"
    if not txt.exists():
        raise RuntimeError("La sessione non ha ancora una trascrizione")
    transcript = txt.read_text(encoding="utf-8-sig", errors="replace").strip()
    est_tokens = len(transcript) / 3.2
    if est_tokens > CFG["llm_ctx"] - 4000:
        raise RuntimeError(f"Trascrizione troppo lunga per il contesto del modello (~{int(est_tokens)} token): "
                           f"aumenta 'llm_ctx' in config.json")
    skill = get_skill(job["skill"])
    meta = read_json(sdir / "meta.json", {}) or {}
    mode = job.get("mode", "generate")
    request = job["instructions"].strip()
    current_file = sdir / f"sintesi_{skill['id']}.md"
    current = current_file.read_text(encoding="utf-8-sig", errors="replace").strip() if current_file.exists() else ""
    if mode != "generate" and not current:
        raise RuntimeError("Non c'è ancora una sintesi da modificare")
    # The system prompt and the transcript come first and are identical for generate and revise, so
    # llama.cpp can reuse the transcript it already read (prompt cache) and only reads the new tail.
    system, head = prompt_prefix(job["patient"], meta, skill, transcript)
    sectioned = mode == "revise" and sum(1 for h, _ in split_sections(current) if h) >= 2

    if mode == "grammar":
        # only a list of corrections is generated (a few dozen tokens), then applied here
        system = GRAMMAR_PROMPT
        user = f"TESTO:\n\"\"\"\n{current}\n\"\"\""
    elif mode == "revise":
        how = (REVISE_SECTIONS_PROMPT if sectioned else
               "Restituisci il testo completo della sintesi modificata, senza commenti.")
        user = (head + f"\n\nSINTESI ATTUALE:\n\"\"\"\n{current}\n\"\"\"\n\n"
                "STAI MODIFICANDO LA SINTESI ATTUALE. Applica solo la richiesta del clinico qui sotto, lascia invariato "
                "tutto il resto e mantieni lo stesso formato Markdown. Le regole restano valide: usa solo informazioni "
                f"presenti nella trascrizione. {how}\n\nRICHIESTA DI MODIFICA DEL CLINICO: {request}")
    else:
        user = head + ("\n\nIstruzioni aggiuntive del clinico: " + request if request else "")

    job["run_started"] = time.time()
    LLM_SERVER.busy = True
    try:
        job["status"] = "loading"
        LLM_SERVER.ensure()
        job["status"] = "reading"

        def on_pp(p):
            job["progress"] = p

        def on_text(t):
            job["status"] = "writing"
            job["partial"] = t

        text, timings = LLM_SERVER.stream_chat([{"role": "system", "content": system},
                                                {"role": "user", "content": user}], on_pp, on_text)
    finally:
        LLM_SERVER.busy = False
        LLM_SERVER.last_used = time.time()
    text = re.sub(r'^\s*"""\s*|\s*"""\s*$', "", text.strip())  # drop quotes the model may echo back
    if not text.strip():
        raise RuntimeError("Il modello non ha prodotto testo")

    details = {}
    if mode == "grammar":
        new, applied, skipped = apply_corrections(current, text)
        details = {"corrections": applied, "skipped": skipped}
    elif sectioned:
        new, changed = merge_sections(current, text)
        if not changed:
            raise RuntimeError("Gemma non ha restituito sezioni riconoscibili: prova a riformulare la richiesta.\n\n"
                               + text[:500])
        details = {"sections": changed}
    else:
        new = text
    job["details"] = details

    if new.strip() != current.strip():
        save_version(sdir, skill["id"])
        current_file.write_text(new.strip() + "\n", encoding="utf-8")
    meta = read_json(sdir / "meta.json", {}) or {}
    info = meta.setdefault("summaries", {}).setdefault(skill["id"], {})
    info.update({"llm": Path(CFG["llm_model"]).stem, "duration_s": round(time.time() - job["run_started"])})
    stamp = time.strftime("%Y-%m-%d %H:%M")
    if mode == "generate":
        info.update({"generated_at": stamp, "instructions": request, "history": []})
    info.setdefault("history", []).append({
        "at": stamp, "action": {"generate": "generata", "revise": "modificata con Gemma",
                                "grammar": "controllo grammaticale"}[mode],
        "request": request, "seconds": round(time.time() - job["run_started"]),
        "cached_tokens": timings.get("cache_n"), "read_tokens": timings.get("prompt_n"),
        "written_tokens": timings.get("predicted_n"), **details})
    write_json(sdir / "meta.json", meta)
    job["status"] = "done"
    job["progress"] = 100
    if mode == "generate" and CFG["auto_process_card"] and not (sdir / "processi.json").exists():
        enqueue_process_card(job["patient"], job["session"], skill["id"])  # transcript still cached: fast


# ---------------------------------------------------------------- process cards (hexaflex per session)

PROCESSES = [  # key, label; score 1 = rigid pole, 5 = flexible pole, 0 = no evidence in the session
    ("accettazione", "Accettazione"), ("defusione", "Defusione"), ("presente", "Momento presente"),
    ("se_contesto", "Sé come contesto"), ("valori", "Valori"), ("azione", "Azione impegnata"),
    ("autocompassione", "Autocompassione"),
]
HEXAFLEX = [k for k, _ in PROCESSES[:6]]

PROCESS_PROMPT = """COMPITO DIVERSO: invece della sintesi, compila una SCHEDA DEI PROCESSI di questa seduta in JSON.

Per ciascun processo assegna un punteggio sulla base di ciò che il paziente porta IN QUESTA SEDUTA:
1 = forte rigidità (polo inflessibile), 2 = rigidità prevalente, 3 = misto, 4 = flessibilità prevalente, 5 = forte flessibilità.
0 = nessuna evidenza nella trascrizione (usalo quando il processo non emerge: non indovinare).
- accettazione: disponibilità a sentire emozioni e sensazioni difficili (1 = evitamento esperienziale marcato)
- defusione: capacità di vedere i pensieri come pensieri (1 = fusione, pensieri presi alla lettera)
- presente: contatto flessibile con il qui e ora (1 = rimuginio, ruminazione, pilota automatico)
- se_contesto: prospettiva del sé osservante (1 = identificazione rigida con storie su di sé)
- valori: chiarezza e contatto con ciò che conta (1 = valori assenti o confusi)
- azione: comportamenti coerenti con i valori (1 = inazione, evitamento, impulsività)
- autocompassione: atteggiamento verso di sé nelle difficoltà (1 = autocritica dura)
Per ogni processo scrivi in "evidenza" un esempio telegrafico dalla trascrizione (massimo 12 parole), o "" se il punteggio è 0.
Compila anche, in modo telegrafico: temi (max 4, 1-3 parole ciascuno), valori_emersi (1-4 parole ciascuno),
azioni_concordate per la prossima volta (max 8 parole ciascuna), compiti_precedenti (esito dei compiti della seduta
prima: svolti, parziali, non svolti, non discussi; nota max 10 parole) e rischio (max 10 parole).
Rispondi solo con il JSON."""

PROCESS_SCHEMA = {
    "type": "object",
    "properties": {
        "processi": {"type": "object", "properties": {
            k: {"type": "object", "properties": {"punteggio": {"type": "integer", "enum": [0, 1, 2, 3, 4, 5]},
                                                 "evidenza": {"type": "string"}},
                "required": ["punteggio", "evidenza"]} for k, _ in PROCESSES},
            "required": [k for k, _ in PROCESSES]},
        "temi": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "valori_emersi": {"type": "array", "items": {"type": "string"}},
        "azioni_concordate": {"type": "array", "items": {"type": "string"}},
        "compiti_precedenti": {"type": "object", "properties": {
            "esito": {"type": "string", "enum": ["svolti", "parziali", "non svolti", "non discussi"]},
            "nota": {"type": "string"}}, "required": ["esito", "nota"]},
        "rischio": {"type": "string"},
    },
    "required": ["processi", "temi", "valori_emersi", "azioni_concordate", "compiti_precedenti", "rischio"],
}


def process_key(patient, session):
    return f"{patient}/{session}#processi"


def enqueue_process_card(patient: str, session: str, skill_id: str = ""):
    key = process_key(patient, session)
    if key in JOBS and JOBS[key]["status"] in ACTIVE:
        return
    JOBS[key] = {"kind": "processi", "patient": patient, "session": session, "status": "queued", "progress": 0,
                 "skill": skill_id or CFG["default_skill"], "started": time.time()}
    JOB_QUEUE.put(key)


def run_process_card(job: dict):
    sdir = ROOT / job["patient"] / job["session"]
    txt = sdir / "transcript.txt"
    if not txt.exists():
        raise RuntimeError("La sessione non ha ancora una trascrizione")
    transcript = txt.read_text(encoding="utf-8-sig", errors="replace").strip()
    meta = read_json(sdir / "meta.json", {}) or {}
    try:
        skill = get_skill(job["skill"])
    except HTTPException:
        skill = load_skills()[0]
    system, head = prompt_prefix(job["patient"], meta, skill, transcript)
    job["run_started"] = time.time()
    LLM_SERVER.busy = True
    try:
        job["status"] = "loading"
        LLM_SERVER.ensure()
        job["status"] = "reading"
        text, timings = LLM_SERVER.stream_chat(
            [{"role": "system", "content": system}, {"role": "user", "content": head + "\n\n" + PROCESS_PROMPT}],
            lambda p: job.update(progress=p), lambda t: job.update(status="writing", partial=t),
            response_format={"type": "json_schema", "json_schema": {"name": "scheda", "schema": PROCESS_SCHEMA}},
            max_tokens=1200)
    finally:
        LLM_SERVER.busy = False
        LLM_SERVER.last_used = time.time()
    try:
        card = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except ValueError:
        raise RuntimeError("Gemma non ha restituito una scheda valida:\n" + text[:400])
    old = read_json(sdir / "processi.json", {}) or {}
    write_json(sdir / "processi.json", {
        "generated_at": time.strftime("%Y-%m-%d %H:%M"), "llm": Path(CFG["llm_model"]).stem,
        "seconds": round(time.time() - job["run_started"]), "cached_tokens": timings.get("cache_n"),
        "written_tokens": timings.get("predicted_n"),
        "proposed": card, "confirmed": old.get("confirmed", {})})  # clinician's confirmations survive a re-run
    job["status"] = "done"
    job["progress"] = 100


def effective_card(sdir: Path):
    """Scores to plot: the clinician's value when confirmed, otherwise Gemma's proposal; 0 -> None (no evidence)."""
    data = read_json(sdir / "processi.json")
    if not data:
        return None
    prop, conf = data.get("proposed", {}), data.get("confirmed", {})
    procs = {}
    for k, label in PROCESSES:
        p = (prop.get("processi") or {}).get(k) or {}
        proposed = p.get("punteggio") if isinstance(p.get("punteggio"), int) else 0
        value = conf[k] if k in conf else proposed
        procs[k] = {"label": label, "score": value or None, "proposed": proposed or None,
                    "confirmed": k in conf, "evidence": p.get("evidenza", "")}
    return {"processi": procs, "temi": prop.get("temi", []), "valori": prop.get("valori_emersi", []),
            "azioni": prop.get("azioni_concordate", []), "compiti": prop.get("compiti_precedenti", {}),
            "rischio": prop.get("rischio", ""), "generated_at": data.get("generated_at"),
            "all_confirmed": all(k in conf for k, _ in PROCESSES)}


GRAMMAR_PROMPT = (
    "Sei un correttore di bozze per testi clinici in italiano. Trova SOLO errori di ortografia, grammatica, "
    "punteggiatura, concordanza e refusi (parole storpiate o inesistenti). Non correggere lo stile, non riformulare, "
    "non cambiare contenuto né terminologia clinica (inclusi termini inglesi come willingness, workability, "
    "hexaflex, mindfulness).\n\n"
    "Rispondi SOLO con l'elenco delle correzioni, una per riga, nel formato:\n"
    "testo errato => testo corretto\n\n"
    "Il testo errato deve essere copiato ESATTAMENTE dal testo: solo la parola sbagliata con 1-4 parole vicine, "
    "MAI una frase intera. Non toccare il testo tra virgolette: sono citazioni testuali del paziente e vanno "
    "lasciate come sono. Se una frase è corretta anche se migliorabile, non segnalarla.\n"
    "Esempio:\nha dormito pocco => ha dormito poco\n"
    "Se non ci sono errori rispondi solo: NESSUNA CORREZIONE")

REVISE_SECTIONS_PROMPT = (
    "Restituisci SOLO le sezioni che modifichi, ciascuna completa e preceduta dal suo titolo '## ...' scritto "
    "esattamente come nella sintesi attuale. Non restituire le sezioni che restano invariate. Per aggiungere una "
    "sezione nuova usa un titolo nuovo. Per eliminare una sezione scrivi il suo titolo seguito da una riga con "
    "[ELIMINATA]. Nessun commento.")


def split_sections(text: str):
    """[(heading_line or '', body)] split on Markdown '## ' headings; the first item is any preamble."""
    parts, head, buf = [], "", []
    for line in text.splitlines():
        if re.match(r"^#{1,3}\s+\S", line):
            if head or "".join(buf).strip():
                parts.append((head, "\n".join(buf).strip()))
            head, buf = line.strip(), []
        else:
            buf.append(line)
    if head or "".join(buf).strip():
        parts.append((head, "\n".join(buf).strip()))
    return parts


def _key(heading: str):
    return re.sub(r"[^a-zà-ù0-9]+", " ", heading.lstrip("#").lower()).strip()


def _find_section(sections, heading: str):
    """The existing section Gemma meant, tolerating small title differences ('Interventi' vs
    'Interventi del terapeuta:', 'S - Soggettivo' vs 'S – Soggettivo')."""
    import difflib
    k = _key(heading)
    if not k:
        return None
    exact = next((c for c in sections if _key(c[0]) == k), None)
    if exact:
        return exact
    contained = [c for c in sections if k in _key(c[0]) or _key(c[0]) in k]
    if len(contained) == 1:
        return contained[0]
    best = max(sections, key=lambda c: difflib.SequenceMatcher(None, _key(c[0]), k).ratio(), default=None)
    if best and difflib.SequenceMatcher(None, _key(best[0]), k).ratio() >= 0.75:
        return best
    return None


def merge_sections(current: str, revised: str):
    """Replace the sections Gemma returned; returns (new_text, [changed headings])."""
    cur, preamble = [], ""
    for head, body in split_sections(current):
        if head:
            cur.append([head, body])
        else:
            preamble = body
    changed = []
    for head, body in split_sections(revised):
        if not head:
            continue
        match = _find_section(cur, head)
        if body.strip().upper().strip("[]") == "ELIMINATA":
            if match:
                cur.remove(match)
                changed.append(match[0].lstrip("# ") + " (eliminata)")
            continue
        if match:
            if match[1].strip() != body.strip():
                match[1] = body
                changed.append(match[0].lstrip("# "))
        else:
            cur.append([head if head.startswith("## ") else "## " + head.lstrip("# "), body])
            changed.append(head.lstrip("# ") + " (nuova)")
    parts = ([preamble] if preamble else []) + [f"{h}\n{b}".rstrip() for h, b in cur]
    return "\n\n".join(parts), changed


DOUBLED = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)


def _inside_quote(text: str, wrong: str, right: str):
    """True when the first changed character, located in the text, lies inside a quotation ("..." or “...”) on its
    line: patient quotes stay verbatim."""
    start = text.find(wrong)
    if start < 0:
        return False
    i = next((k for k, (a, b) in enumerate(zip(wrong, right)) if a != b), min(len(wrong), len(right)))
    pos = start + i
    before = text[text.rfind("\n", 0, pos) + 1:pos]
    return before.count('"') % 2 == 1 or before.count("“") > before.count("”")


def apply_corrections(text: str, answer: str):
    """Apply 'wrong => right' lines; returns (new_text, applied, skipped)."""
    applied, skipped = [], []
    if "NESSUNA CORREZIONE" in answer.upper():
        return text, applied, skipped
    for line in answer.splitlines():
        if "=>" not in line:
            continue
        wrong, right = (s.strip().strip("`*\"“”'-•").strip() for s in line.split("=>", 1))
        wrong = re.sub(r"^\d+[.)]\s*", "", wrong)
        if not wrong or wrong == right:
            continue
        if _inside_quote(text, wrong, right):
            skipped.append({"da": wrong, "a": right, "motivo": "citazione del paziente"})
            continue
        if len(wrong.split()) > 12:
            skipped.append({"da": wrong, "a": right, "motivo": "riformulazione, non un errore"})
            continue
        if wrong in text:
            candidate = text.replace(wrong, right)
            if len(DOUBLED.findall(candidate)) > len(DOUBLED.findall(text)):
                # e.g. Gemma copied "riuscita" instead of "è riuscita" and added "è" again -> "è è riuscita"
                skipped.append({"da": wrong, "a": right, "motivo": "creerebbe una parola ripetuta"})
                continue
            text = candidate
            applied.append({"da": wrong, "a": right})
        else:
            skipped.append({"da": wrong, "a": right})
    return text, applied, skipped


def run_job(job: dict):
    patient, session, model = job["patient"], job["session"], job["model"]
    job["run_started"] = time.time()
    sdir = ROOT / patient / session
    src = find_audio(sdir)
    if not src:
        raise RuntimeError("Nessun file audio nella sessione")

    job["status"] = "converting"
    wav = sdir / "audio16k.wav"
    # run inside the session folder with relative file names to keep paths short
    r = subprocess.run([FFMPEG, "-y", "-i", src.name, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav.name],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(sdir),
                       creationflags=subprocess.CREATE_NO_WINDOW)
    if r.returncode != 0:
        raise RuntimeError("Conversione audio fallita: " + r.stderr[-400:])

    job["status"] = "transcribing"
    if model.startswith("ov:"):
        transcribe_openvino(job, sdir, wav, model)
    else:
        transcribe_whispercpp(job, sdir, wav, model)

    meta = read_json(sdir / "meta.json", {}) or {}
    meta.update({"model": model, "transcribed_at": time.strftime("%Y-%m-%d %H:%M"),
                 "duration_s": round(time.time() - job["run_started"])})
    write_json(sdir / "meta.json", meta)
    wav.unlink(missing_ok=True)
    job["status"] = "done"
    job["progress"] = 100
    if job.get("auto_skill"):
        try:
            enqueue_summary(patient, session, job["auto_skill"])
        except HTTPException:
            pass  # a summary for this session is already running


def transcribe_whispercpp(job: dict, sdir: Path, wav: Path, model: str):
    cmd = [str(WHISPER_CLI), "-m", str(WHISPER_DIR / model), "-l", CFG["language"], "-t", str(CFG["threads"]),
           "-pp", "-otxt", "-oj", "-of", "transcript", "-f", wav.name] + (["--prompt", CFG["whisper_prompt"]] if CFG["whisper_prompt"] else [])
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, cwd=str(sdir),
                            text=True, encoding="utf-8", errors="replace",
                            creationflags=subprocess.CREATE_NO_WINDOW)
    tail = []
    for line in proc.stderr:
        m = PROGRESS_RE.search(line)
        if m:
            job["progress"] = int(m.group(1))
        tail = (tail + [line])[-15:]
    proc.wait()
    if proc.returncode != 0 or not (sdir / "transcript.txt").exists():
        raise RuntimeError("Whisper ha restituito un errore: " + "".join(tail)[-600:])


# ---------------------------------------------------------------- OpenVINO engine (Intel GPU / NPU)

OV_PIPES: dict = {}  # (model_dir, device) -> loaded WhisperPipeline, kept between jobs
OV_BLOCK_S = 120     # audio is fed in ~2-minute blocks, cut at quiet points, for progress reporting


def list_ov_models():
    return sorted(p.name for p in MODELS_DIR.glob("*-ov") if (p / "openvino_encoder_model.xml").exists())


def ov_pipeline(model_dir: Path, device: str):
    key = (str(model_dir), device)
    if key not in OV_PIPES:
        import openvino_genai as ov_genai
        OV_PIPES.clear()  # keep one model in memory at a time
        # CACHE_DIR stores the compiled model: only the first load per device is slow (NPU ≈ 5 min)
        OV_PIPES[key] = ov_genai.WhisperPipeline(str(model_dir), device, CACHE_DIR=str(model_dir / f"cache_{device}"))
    return OV_PIPES[key]


def quiet_cuts(audio, sr=16000, block_s=OV_BLOCK_S, search_s=10):
    """Block boundaries near every block_s seconds, moved to the quietest 50 ms frame within ±search_s."""
    import numpy as np
    frame = sr // 20
    cuts, pos, n = [0], block_s * sr, len(audio)
    while pos < n - search_s * sr:
        lo, hi = pos - search_s * sr, pos + search_s * sr
        seg = audio[lo:hi][: (hi - lo) // frame * frame].reshape(-1, frame)
        cuts.append(lo + int(np.argmin((seg ** 2).mean(axis=1))) * frame + frame // 2)
        pos = cuts[-1] + block_s * sr
    cuts.append(n)
    return cuts


def transcribe_openvino(job: dict, sdir: Path, wav: Path, model: str):
    import wave
    import numpy as np
    _, name, device = model.split(":")
    model_dir = MODELS_DIR / name
    if not model_dir.exists():
        raise RuntimeError(f"Modello OpenVINO non trovato: {model_dir}")
    if (model_dir, device) not in [(Path(k[0]), k[1]) for k in OV_PIPES]:
        job["status"] = "loading"  # first load on the NPU compiles the model for a few minutes
    pipe = ov_pipeline(model_dir, device)
    job["status"] = "transcribing"
    with wave.open(str(wav)) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    cuts = quiet_cuts(audio)
    segments, texts = [], []
    for i in range(len(cuts) - 1):
        a, b = cuts[i], cuts[i + 1]
        res = pipe.generate(audio[a:b].tolist(), language=f"<|{CFG['language']}|>", task="transcribe",
                            return_timestamps=True, **({"initial_prompt": CFG["whisper_prompt"]} if CFG["whisper_prompt"] else {}))
        off = a / 16000
        for ch in res.chunks or []:
            t = ch.text.strip()
            if t:
                segments.append({"offsets": {"from": int((off + ch.start_ts) * 1000),
                                             "to": int((off + max(ch.end_ts, ch.start_ts)) * 1000)}, "text": " " + t})
        texts.append(res.texts[0].strip())
        job["progress"] = int(100 * b / len(audio))
    # same files the whisper.cpp engine writes, so the UI works unchanged
    lines = [s["text"].strip() for s in segments] or [t for t in texts if t]
    (sdir / "transcript.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(sdir / "transcript.json", {"transcription": segments})


def worker():
    while True:
        key = JOB_QUEUE.get()
        job = JOBS[key]
        try:
            if job["kind"] in ("summary", "processi"):
                OV_PIPES.clear()  # free the Whisper model before Gemma loads
                (run_summary if job["kind"] == "summary" else run_process_card)(job)
            else:
                LLM_SERVER.stop()  # free Gemma's RAM before whisper starts
                run_job(job)
        except Exception as e:  # noqa: BLE001
            job.update(status="error", error=str(e))
        finally:
            JOB_QUEUE.task_done()


threading.Thread(target=llm_idle_watcher, daemon=True).start()

threading.Thread(target=worker, daemon=True).start()

# ---------------------------------------------------------------- API

app = FastAPI(title="FlexScribe")


class NewPatient(BaseModel):
    name: str


class TranscriptEdit(BaseModel):
    text: str


class SummaryRequest(BaseModel):
    skill: str
    instructions: str = ""
    mode: str = "generate"


@app.get("/api/config")
def get_config():
    models = []
    for name in list_ov_models():
        short = name.replace("whisper-", "").replace("-ov", "")
        models.append({"id": f"ov:{name}:GPU", "label": f"{short} · scheda grafica (veloce)"})
    models += [{"id": p.name, "label": p.stem.replace("ggml-", "") + " · CPU (lento)"}
               for p in sorted(WHISPER_DIR.glob("ggml-*.bin"))]
    return {"patients_dir": str(ROOT), "models": models, "default_model": CFG["default_model"], "language": CFG["language"],
            "llm_model": Path(CFG["llm_model"]).stem if CFG["llm_model"] else "", "default_skill": CFG["default_skill"]}


@app.get("/api/skills")
def get_skills():
    return [{k: s[k] for k in ("id", "name", "description")} for s in load_skills()]


@app.get("/api/patients")
def get_patients():
    out = []
    for p in sorted(ROOT.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_dir():
            continue
        sessions = list_sessions(p)
        out.append({
            "name": p.name,
            "sessions": len(sessions),
            "last_date": sessions[0]["date"] if sessions else "",
            "busy": sum(1 for s in sessions if s["status"] in ("queued", "converting", "loading", "transcribing")),
            "pending": sum(1 for s in sessions if s["status"] in ("pending", "error")),
        })
    return out


@app.post("/api/patients")
def create_patient(body: NewPatient):
    name = safe_name(body.name)
    p = inside_root(name)
    if p.exists():
        raise HTTPException(409, "Esiste già un paziente con questo nome")
    p.mkdir(parents=True)
    return {"name": name}


@app.get("/api/patients/{patient}")
def get_patient(patient: str):
    pdir = patient_dir(patient)
    return {"name": pdir.name, "sessions": list_sessions(pdir)}


@app.post("/api/patients/{patient}/sessions")
async def create_session(patient: str, audio: UploadFile = File(...), session_date: str = Form(""),
                         title: str = Form(""), model: str = Form(""), auto_skill: str = Form("")):
    pdir = patient_dir(patient)
    d = session_date or date.today().isoformat()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
        raise HTTPException(400, "Data non valida")
    base = f"{d}_{safe_name(title).replace(' ', '_')}" if title.strip() else f"{d}_sessione"
    name, n = base, 2
    while (pdir / name).exists():
        name, n = f"{base}_{n}", n + 1
    sdir = pdir / name
    sdir.mkdir()
    ext = Path(audio.filename or "").suffix.lower() or ".bin"
    with open(sdir / f"audio{ext}", "wb") as f:
        while chunk := await audio.read(1024 * 1024):
            f.write(chunk)
    write_json(sdir / "meta.json", {"date": d, "title": title.strip() or "Sessione", "original_filename": audio.filename})
    if auto_skill:
        get_skill(auto_skill)
    enqueue(pdir.name, name, model or CFG["default_model"], auto_skill)
    return session_info(sdir)


@app.post("/api/patients/{patient}/sessions/{session}/transcribe")
def retranscribe(patient: str, session: str, model: str = ""):
    sdir = session_dir(patient, session)
    if not find_audio(sdir):
        raise HTTPException(400, "Nessun file audio in questa sessione")
    enqueue(sdir.parent.name, sdir.name, model or CFG["default_model"])
    return session_info(sdir)


@app.get("/api/patients/{patient}/sessions/{session}")
def get_session(patient: str, session: str):
    sdir = session_dir(patient, session)
    info = session_info(sdir)
    txt = sdir / "transcript.txt"
    info["text"] = txt.read_text(encoding="utf-8-sig", errors="replace") if txt.exists() else ""
    data = read_json(sdir / "transcript.json", {}) or {}
    info["segments"] = [
        {"from": s["offsets"]["from"] / 1000, "to": s["offsets"]["to"] / 1000, "text": s["text"].strip()}
        for s in data.get("transcription", [])
    ]
    info["meta"] = read_json(sdir / "meta.json", {}) or {}
    info["summaries"] = list_summaries(sdir)
    info["processi"] = effective_card(sdir)
    info["processi_job"] = process_job_info(sdir.parent.name, sdir.name)
    return info


def process_job_info(patient, session):
    j = JOBS.get(process_key(patient, session))
    if not j or (j["status"] == "done" and time.time() - j["started"] > 60):
        return None
    return {k: j.get(k) for k in ("status", "progress", "error")}


class ScoreUpdate(BaseModel):
    score: int | None = None  # 0-5 confirms that value (0 = no evidence); None = back to Gemma's proposal


@app.post("/api/patients/{patient}/sessions/{session}/processi")
def create_process_card(patient: str, session: str):
    sdir = session_dir(patient, session)
    if not (sdir / "transcript.txt").exists():
        raise HTTPException(400, "Prima serve la trascrizione")
    enqueue_process_card(sdir.parent.name, sdir.name)
    return process_job_info(sdir.parent.name, sdir.name)


@app.get("/api/patients/{patient}/sessions/{session}/processi-job")
def get_process_job(patient: str, session: str):
    sdir = session_dir(patient, session)
    return process_job_info(sdir.parent.name, sdir.name)


@app.put("/api/patients/{patient}/sessions/{session}/processi/{process}")
def set_process_score(patient: str, session: str, process: str, body: ScoreUpdate):
    sdir = session_dir(patient, session)
    data = read_json(sdir / "processi.json")
    if not data:
        raise HTTPException(404, "Scheda dei processi non trovata")
    if process not in dict(PROCESSES):
        raise HTTPException(400, "Processo sconosciuto")
    if body.score is None:
        data.setdefault("confirmed", {}).pop(process, None)
    elif 0 <= body.score <= 5:
        data.setdefault("confirmed", {})[process] = body.score
    else:
        raise HTTPException(400, "Punteggio tra 0 e 5")
    write_json(sdir / "processi.json", data)
    return effective_card(sdir)


@app.post("/api/patients/{patient}/sessions/{session}/processi/conferma")
def confirm_all_scores(patient: str, session: str):
    sdir = session_dir(patient, session)
    card = effective_card(sdir)
    if not card:
        raise HTTPException(404, "Scheda dei processi non trovata")
    data = read_json(sdir / "processi.json")
    conf = data.setdefault("confirmed", {})
    for k, p in card["processi"].items():
        conf.setdefault(k, p["proposed"] or 0)
    write_json(sdir / "processi.json", data)
    return effective_card(sdir)


@app.get("/api/patients/{patient}/andamento")
def get_trend(patient: str):
    """All the session cards of a patient, oldest first: the charts are computed in the browser."""
    pdir = patient_dir(patient)
    sessions = sorted(list_sessions(pdir), key=lambda s: (s["date"] or "9999", s["id"]))
    out = []
    for s in sessions:
        sdir = pdir / s["id"]
        out.append({"id": s["id"], "date": s["date"], "title": s["title"],
                    "has_transcript": (sdir / "transcript.txt").exists(), "card": effective_card(sdir),
                    "job": process_job_info(pdir.name, s["id"])})
    return {"patient": pdir.name, "processes": [{"key": k, "label": l} for k, l in PROCESSES],
            "hexaflex": HEXAFLEX, "sessions": out}


@app.post("/api/patients/{patient}/andamento/analizza")
def analyze_missing(patient: str):
    """Queue a process card for every transcribed session that doesn't have one yet."""
    pdir = patient_dir(patient)
    n = 0
    for s in list_sessions(pdir):
        sdir = pdir / s["id"]
        if (sdir / "transcript.txt").exists() and not (sdir / "processi.json").exists():
            enqueue_process_card(pdir.name, s["id"])
            n += 1
    return {"queued": n}


@app.post("/api/patients/{patient}/sessions/{session}/summary")
def create_summary(patient: str, session: str, body: SummaryRequest):
    sdir = session_dir(patient, session)
    if not (sdir / "transcript.txt").exists():
        raise HTTPException(400, "Prima serve la trascrizione")
    get_skill(body.skill)
    if body.mode not in ("generate", "revise", "grammar"):
        raise HTTPException(400, "Modalità non valida")
    if body.mode == "revise" and not body.instructions.strip():
        raise HTTPException(400, "Scrivi cosa vuoi modificare")
    enqueue_summary(sdir.parent.name, sdir.name, body.skill, body.instructions, body.mode)
    return summary_job_info(sdir.parent.name, sdir.name)


@app.post("/api/patients/{patient}/sessions/{session}/summary/{skill}/restore")
def restore_summary(patient: str, session: str, skill: str):
    """Undo: put back the most recent earlier version (the current text is discarded)."""
    sdir = session_dir(patient, session)
    versions = list_versions(sdir, safe_name(skill))
    if not versions:
        raise HTTPException(404, "Nessuna versione precedente")
    shutil.move(str(versions[-1]), sdir / f"sintesi_{safe_name(skill)}.md")
    return {"ok": True, "versions_left": len(versions) - 1}


@app.get("/api/patients/{patient}/sessions/{session}/summary-job")
def get_summary_job(patient: str, session: str):
    sdir = session_dir(patient, session)
    return summary_job_info(sdir.parent.name, sdir.name)


@app.put("/api/patients/{patient}/sessions/{session}/summary/{skill}")
def save_summary(patient: str, session: str, skill: str, body: TranscriptEdit):
    sdir = session_dir(patient, session)
    f = sdir / f"sintesi_{safe_name(skill)}.md"
    if not f.exists():
        raise HTTPException(404, "Sintesi non trovata")
    save_version(sdir, safe_name(skill))
    f.write_text(body.text, encoding="utf-8")
    return {"ok": True}


@app.put("/api/patients/{patient}/sessions/{session}/transcript")
def save_transcript(patient: str, session: str, body: TranscriptEdit):
    sdir = session_dir(patient, session)
    txt = sdir / "transcript.txt"
    if txt.exists() and not (sdir / "transcript.original.txt").exists():
        shutil.copy2(txt, sdir / "transcript.original.txt")  # keep whisper's output once
    txt.write_text(body.text, encoding="utf-8")
    return {"ok": True}


@app.get("/api/patients/{patient}/sessions/{session}/audio")
def get_audio(patient: str, session: str):
    a = find_audio(session_dir(patient, session))
    if not a:
        raise HTTPException(404, "Audio non trovato")
    return FileResponse(a)


@app.post("/api/patients/{patient}/sessions/{session}/open")
def open_folder(patient: str, session: str):
    os.startfile(session_dir(patient, session))
    return {"ok": True}


@app.get("/api/jobs")
def get_jobs():
    return [{k: v for k, v in j.items() if k != "partial"} for j in JOBS.values()
            if j["status"] != "done" or time.time() - j["started"] < 3600]


app.mount("/", StaticFiles(directory=APP_DIR / "static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    url = f"http://{CFG['host']}:{CFG['port']}"
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=CFG["host"], port=CFG["port"])
