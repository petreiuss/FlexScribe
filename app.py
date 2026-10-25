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
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------- privacy: nothing leaves this computer
# Set before the libraries are imported. OpenVINO sends usage statistics to Google Analytics unless opted out
# (https://docs.openvino.ai/2025/about-openvino/additional-resources/telemetry.html): see ensure_openvino_opt_out().
os.environ.update({
    "OTEL_SDK_DISABLED": "true",         # FastAPI / OpenTelemetry: no tracing or exporters
    "HF_HUB_DISABLE_TELEMETRY": "1",     # Hugging Face: no usage pings
    "HF_HUB_OFFLINE": "1",               # the app itself never downloads (tools/setup_llm.py does, at install)
    "DO_NOT_TRACK": "1",                 # generic opt-out honoured by several libraries
})


def ensure_openvino_opt_out():
    """Make sure OpenVINO's telemetry consent is 'declined' (file %LOCALAPPDATA%\\Intel Corporation\\openvino_telemetry = 0)."""
    consent = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Intel Corporation" / "openvino_telemetry"
    try:
        if consent.read_text().strip() == "0":
            return
    except OSError:
        pass
    tool = Path(sys.executable).parent / "opt_in_out.exe"  # official command shipped with openvino-telemetry
    if tool.exists():
        subprocess.run([str(tool), "--opt_out"], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if not consent.exists() or consent.read_text().strip() != "0":
        consent.parent.mkdir(parents=True, exist_ok=True)
        consent.write_text("0")


ensure_openvino_opt_out()

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
    # llama.cpp: the copy downloaded by tools/setup_llm.py, else one on the PATH
    "llm_exe": next((str(p) for p in (APP_DIR / "llama.cpp").rglob("llama.exe")), None)
               or shutil.which("llama") or str(Path.home() / "AppData/Local/Microsoft/WindowsApps/llama.exe"),
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
    elif find_audio(sdir):
        status = "pending"
    else:
        status = "imported" if any(sdir.glob("sintesi_*.md")) else "empty"
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


DATE_PATTERNS = [  # (regex, order of the groups) - ISO first, then Italian day-month-year
    (re.compile(r"(?<!\d)(20\d{2})[-_./ ](\d{1,2})[-_./ ](\d{1,2})(?!\d)"), "ymd"),   # 2026-09-12
    (re.compile(r"(?<!\d)(\d{1,2})[-_./ ](\d{1,2})[-_./ ](20\d{2})(?!\d)"), "dmy"),   # 12-09-2026, 12/09/2026
    (re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)"), "ymd"),                   # 20260912
    (re.compile(r"(?<!\d)(\d{2})(\d{2})(20\d{2})(?!\d)"), "dmy"),                   # 12092026
    (re.compile(r"(?<!\d)(\d{1,2})[-_./](\d{1,2})[-_./](\d{2})(?!\d)"), "dmy2"),    # 12-09-26, 12.09.26
]
MONTHS = {m: i for i, m in enumerate(["gen", "feb", "mar", "apr", "mag", "giu", "lug", "ago", "set", "ott", "nov", "dic"], 1)}
MONTH_NAME_DATE = re.compile(r"(?<!\d)(\d{1,2})[-_. ]*(gen\w*|feb\w*|mar\w*|apr\w*|mag\w*|giu\w*|lug\w*|ago\w*|set\w*|ott\w*|"
                             r"nov\w*|dic\w*)[-_. ]*(20\d{2})(?!\d)", re.IGNORECASE)  # 12 settembre 2026
IMPORTED_SKILL = "importata"  # summaries brought in from another app (PDF)


def date_in_name(stem: str):
    """(ISO date, matched text) from a file name such as 'Rossi 12-09-2026', '2026-09-12_seduta', 'REC_20260912'."""
    for rx, order in DATE_PATTERNS + [(MONTH_NAME_DATE, "dmy")]:
        for m in rx.finditer(stem):
            g = m.groups()
            y, mo, d = (g[0], g[1], g[2]) if order == "ymd" else (g[2], g[1], g[0])
            if order == "dmy2":
                y = "20" + y
            mo = MONTHS.get(mo[:3].lower(), mo)
            try:
                return date(int(y), int(mo), int(d)).isoformat(), m.group(0)
            except ValueError:
                continue
    return None, ""


def session_title_from(stem: str, matched: str, patient: str):
    """Readable title from the rest of the file name, without the date and the patient's name."""
    t = stem.replace(matched, " ") if matched else stem
    for word in patient.split():
        t = re.sub(rf"\b{re.escape(word)}\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"[_\s]+", " ", t).strip(" -_.")
    return re.sub(r"\b([01]\d|2[0-3])([0-5]\d)([0-5]\d)?\b", r"ore \1:\2", t)  # REC_..._143000


def new_session_dir(pdir: Path, d: str, title: str):
    try:
        base = f"{d}_{safe_name(title)[:60].replace(' ', '_')}" if title else f"{d}_seduta"
    except HTTPException:
        base = f"{d}_seduta"
    name, n = base, 2
    while (pdir / name).exists():
        name, n = f"{base}_{n}", n + 1
    return pdir / name


BULLET = re.compile(r"^\s*[•●▪◦‣∙·\-–*]\s+")


def pdf_text(path: Path):
    """Text of a PDF as light Markdown: paragraphs re-joined, short title-like lines as '## ' headings,
    bullets as '- ', typographic ligatures (ﬀ, ﬁ) expanded."""
    import unicodedata
    from pypdf import PdfReader
    raw = "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    raw = unicodedata.normalize("NFKC", raw).replace(" ", " ")
    raw = re.sub(r"(\w)-\n(\w)", r"\1\2", raw)  # re-join words hyphenated at the end of a line
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in raw.splitlines()]
    out = []
    for ln in lines:
        if not ln:
            out.append("")
            continue
        is_bullet = bool(BULLET.match(ln))
        heading = (len(ln) <= 60 and not is_bullet and ln[0].isupper() and not re.search(r"[.,;:!?)]$", ln)
                   and ":" not in ln and len(ln.split()) <= 7)
        prev = out[-1] if out else ""
        # a line that continues the previous sentence (PDF line wrap): join it
        if (prev and not is_bullet and not heading and not prev.startswith("## ")
                and not re.search(r"[.:;!?]$", prev) and (ln[0].islower() or ln[0] in "(\"'«")):
            out[-1] = prev + " " + ln
        elif heading and out and not (prev and not re.search(r"[.:;!?\"»)]$", prev) and ln[0].islower()):
            out += ["", "## " + ln] if prev else ["## " + ln]
        elif is_bullet:
            out.append("- " + BULLET.sub("", ln))
        else:
            out.append(ln)
    text = "\n".join(out)
    text = re.sub(r"\n(## )", r"\n\n\1", text)
    text = re.sub(r"(## [^\n]+)\n(?!\n)", r"\1\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def import_summary_pdf(pdir: Path, src: Path, original_name: str, move: bool):
    """Create a session from a summary PDF made with another app. Returns the session folder name."""
    stem = Path(original_name).stem
    text = pdf_text(src)
    if len(text) < 20:
        raise ValueError("il PDF non contiene testo (forse è una scansione)")
    # session date: from the file name; else the first date written at the top of the PDF; else the file's date
    d, matched = date_in_name(stem)
    date_source = "nome del file"
    if not d:
        d, _ = date_in_name(text[:800])
        date_source = "testo del PDF"
    if not d:
        d = time.strftime("%Y-%m-%d", time.localtime(src.stat().st_mtime))
        date_source = "data del file (da verificare)"
    title = session_title_from(stem, matched, pdir.name) or "Seduta"
    sdir = new_session_dir(pdir, d, title)
    sdir.mkdir()
    try:
        (shutil.move if move else shutil.copy2)(str(src), sdir / "sintesi_originale.pdf")
    except OSError:
        shutil.rmtree(sdir, ignore_errors=True)
        raise
    (sdir / f"sintesi_{IMPORTED_SKILL}.md").write_text(text + "\n", encoding="utf-8")
    write_json(sdir / "meta.json", {"date": d, "date_source": date_source, "title": title,
                                    "original_filename": original_name, "source": "sintesi importata",
                                    "imported_at": time.strftime("%Y-%m-%d %H:%M")})
    if CFG["auto_process_card"]:
        enqueue_process_card(pdir.name, sdir.name)
    return sdir.name


def adopt_loose_files(pdir: Path):
    """Files dropped straight into a patient folder get their own dated session folder:
    audio becomes a session to transcribe, a PDF becomes a session with an imported summary.
    The date comes from the file name when it contains one, otherwise from the file's modification time.
    """
    for f in list(pdir.iterdir()):
        ext = f.suffix.lower()
        if not f.is_file() or (ext not in AUDIO_EXTS and ext != ".pdf"):
            continue
        st = f.stat()
        if time.time() - st.st_mtime < 15:
            continue  # probably still being copied; pick it up on a later refresh
        if ext == ".pdf":
            try:
                import_summary_pdf(pdir, f, f.name, move=True)
            except Exception as e:  # noqa: BLE001 - leave the file where it is, report it once in the log
                print(f"[import] {f.name}: {e}")
                f.rename(f.with_suffix(".pdf.non_importato")) if "testo" in str(e) else None
            continue
        d, matched = date_in_name(f.stem)
        d = d or time.strftime("%Y-%m-%d", time.localtime(st.st_mtime))
        title = session_title_from(f.stem, matched, pdir.name) or "sessione"
        sdir = new_session_dir(pdir, d, title)
        try:
            sdir.mkdir()
            shutil.move(str(f), sdir / f"audio{ext}")
        except OSError:  # file locked (still copying / open in a player): retry on the next refresh
            if sdir.exists() and not any(sdir.iterdir()):
                sdir.rmdir()
            continue
        write_json(sdir / "meta.json", {"date": d, "title": title, "original_filename": f.name, "added": "cartella"})


def list_sessions(pdir: Path):
    adopt_loose_files(pdir)
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
    if skill_id == IMPORTED_SKILL:
        return {"id": IMPORTED_SKILL, "name": "Sintesi importata", "description": "",
                "prompt": "Il testo è una sintesi clinica di seduta scritta in precedenza dal clinico con un'altra "
                          "applicazione (importata da PDF). Mantieni la sua struttura e il suo stile."}
    raise HTTPException(404, "Modello di sintesi non trovato")


def list_summaries(sdir: Path):
    names = {s["id"]: s["name"] for s in load_skills()} | {IMPORTED_SKILL: "Sintesi importata"}
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
            # keep previous runs in the log (a crash is diagnosed after the restart); rotate at 5 MB
            if self.log.exists() and self.log.stat().st_size > 5_000_000:
                self.log.replace(self.log.with_suffix(".old.log"))
            logf = open(self.log, "a", encoding="utf-8", errors="replace")
            logf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} avvio: {' '.join(cmd[1:])}\n")
            logf.flush()
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
        """One chat request; if the llama server dies mid-request it is restarted and the request retried once."""
        try:
            return self._stream_chat(messages, on_prompt_progress, on_text, response_format, max_tokens)
        except (ConnectionError, urllib.error.URLError, OSError) as e:
            code = self.proc.poll() if self.proc else None
            with open(self.log, "a", encoding="utf-8", errors="replace") as f:
                f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} richiesta interrotta ({e}); "
                        f"codice di uscita del server: {code}. Riavvio e riprovo una volta.\n")
            self.stop()
            self.ensure()
            on_text("")
            return self._stream_chat(messages, on_prompt_progress, on_text, response_format, max_tokens)

    def _stream_chat(self, messages, on_prompt_progress, on_text, response_format=None, max_tokens=3000):
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
    head = f"Paziente: {patient}\nData della sessione: {meta.get('date', '')}\nTitolo: {meta.get('title', '')}"
    if transcript:
        head += f"\n\nTRASCRIZIONE:\n\"\"\"\n{transcript}\n\"\"\""
    return system, head


def run_summary(job: dict):
    sdir = ROOT / job["patient"] / job["session"]
    txt = sdir / "transcript.txt"
    mode = job.get("mode", "generate")
    if not txt.exists() and mode == "generate":
        raise RuntimeError("La sessione non ha ancora una trascrizione")
    # imported summaries have no transcript: edits and grammar then work on the summary alone
    transcript = txt.read_text(encoding="utf-8-sig", errors="replace").strip() if txt.exists() else ""
    est_tokens = len(transcript) / 3.2
    if est_tokens > CFG["llm_ctx"] - 4000:
        raise RuntimeError(f"Trascrizione troppo lunga per il contesto del modello (~{int(est_tokens)} token): "
                           f"aumenta 'llm_ctx' in config.json")
    skill = get_skill(job["skill"])
    meta = read_json(sdir / "meta.json", {}) or {}
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
                "tutto il resto e mantieni lo stesso formato. Le regole restano valide: usa solo informazioni "
                f"presenti nella {'trascrizione' if transcript else 'sintesi attuale'}. {how}\n\n"
                f"RICHIESTA DI MODIFICA DEL CLINICO: {request}")
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

# Process models; each patient uses one or both (paziente.json "modelli"). Score 1 = rigid, 5 = flexible, 0 = no evidence.
MODELS = {
    "hexaflex": {"label": "Hexaflex (ACT)", "radar": ["accettazione", "defusione", "presente", "se_contesto", "valori", "azione"],
                 "processes": [
        ("accettazione", "Accettazione", "disponibilità a sentire emozioni e sensazioni difficili (1 = evitamento esperienziale marcato)"),
        ("defusione", "Defusione", "capacità di vedere i pensieri come pensieri (1 = fusione, pensieri presi alla lettera)"),
        ("presente", "Momento presente", "contatto flessibile con il qui e ora (1 = rimuginio, ruminazione, pilota automatico)"),
        ("se_contesto", "Sé come contesto", "prospettiva del sé osservante (1 = identificazione rigida con storie su di sé)"),
        ("valori", "Valori", "chiarezza e contatto con ciò che conta (1 = valori assenti o confusi)"),
        ("azione", "Azione impegnata", "comportamenti coerenti con i valori (1 = inazione, evitamento, impulsività)"),
        ("autocompassione", "Autocompassione", "atteggiamento verso di sé nelle difficoltà (1 = autocritica dura)")]},
    # DNA-v (Hayes & Ciarrochi): ACT for young people
    "dnav": {"label": "DNA-v", "radar": ["d_discoverer", "d_noticer", "d_advisor", "d_valori", "d_self", "d_social"],
             "processes": [
        ("d_discoverer", "Esploratore", "(Discoverer) esplorare, provare comportamenti nuovi e imparare da come vanno "
                                       "(1 = evita il nuovo o ripete strategie che non funzionano)"),
        ("d_noticer", "Osservatore", "(Noticer) notare sensazioni del corpo, emozioni e pensieri nel momento presente "
                                 "(1 = disconnesso dal corpo e dalle emozioni, oppure sopraffatto)"),
        ("d_advisor", "Consigliere", "(Advisor) rapporto con la voce interna di regole, giudizi e previsioni "
                                 "(1 = segue rigidamente un consigliere critico anche quando non aiuta; 5 = lo usa quando è utile e lo lascia andare quando no)"),
        ("d_valori", "Valori / vitalità", "(values) contatto con ciò che conta e dà vitalità (1 = assente, scelte guidate solo dall'evitamento)"),
        ("d_self", "Visione di sé", "(self view) visione di sé flessibile (1 = etichette rigide su di sé, es. 'sono stupido')"),
        ("d_social", "Visione sociale", "(social view) connessione con gli altri, comprensione della loro prospettiva, appartenenza (1 = isolamento o ostilità)")]},
}
DEFAULT_MODELS = ["hexaflex"]
PROCESS_INFO = {k: (label, desc) for m in MODELS.values() for k, label, desc in m["processes"]}
PROCESSES = [(k, v[0]) for k, v in PROCESS_INFO.items()]  # every known process, for validation
HEXAFLEX = MODELS["hexaflex"]["radar"]


def patient_models(pdir: Path):
    models = (read_json(pdir / "paziente.json", {}) or {}).get("modelli") or DEFAULT_MODELS
    return [m for m in models if m in MODELS] or DEFAULT_MODELS


def model_processes(models):
    return [(k, label, desc) for m in models for k, label, desc in MODELS[m]["processes"]]

def load_vocabulary():
    """skills/_vocabolario.md -> {'temi': [...], 'valori': [...], 'interventi': [...]} (terms shared by all sessions)."""
    f, vocab, section = SKILLS_DIR / "_vocabolario.md", {}, None
    if f.exists():
        for line in f.read_text(encoding="utf-8-sig").splitlines():
            if line.startswith("## "):
                section = line[3:].strip().lower()
                vocab[section] = []
            elif section and line.lstrip().startswith("- "):
                vocab[section].append(line.lstrip()[2:].strip())
    return vocab


def vocabulary_prompt():
    v = load_vocabulary()
    if not v:
        return ""
    lines = [f"- {k}: {', '.join(terms)}" for k, terms in v.items() if terms]
    return ("\n\nVOCABOLARIO: per temi, valori_emersi e interventi usa questi termini, copiati esattamente, quando "
            "corrispondono; solo se nessuno corrisponde scrivi un termine nuovo breve (1-3 parole, minuscolo).\n"
            + "\n".join(lines))

PROCESS_PROMPT_HEAD = """COMPITO DIVERSO: invece della sintesi, compila una SCHEDA DEI PROCESSI di questa seduta in JSON.

Per ciascun processo assegna un punteggio sulla base di ciò che il paziente porta IN QUESTA SEDUTA:
1 = forte rigidità (polo inflessibile), 2 = rigidità prevalente, 3 = misto, 4 = flessibilità prevalente, 5 = forte flessibilità.
0 = nessuna evidenza nella trascrizione (usalo quando il processo non emerge: non indovinare).
"""
PROCESS_PROMPT_TAIL = """Per ogni processo scrivi in "evidenza" un esempio telegrafico dalla trascrizione (massimo 12 parole), o "" se il punteggio è 0.
Compila anche, in modo telegrafico: temi (max 4, 1-3 parole ciascuno), valori_emersi (1-4 parole ciascuno),
azioni_concordate per la prossima volta (max 8 parole ciascuna), compiti_precedenti (esito dei compiti della seduta
prima: svolti, parziali, non svolti, non discussi; nota max 10 parole), rischio (max 10 parole) e interventi (max 6:
tecniche, esercizi e metafore usati dal terapeuta in questa seduta, ognuno col suo nome specifico, per esempio
"passeggeri sull'autobus" e non "metafora").
Per compiti_precedenti guarda soprattutto l'inizio della seduta: se il terapeuta chiede com'è andato un compito o un esercizio della volta prima e il paziente risponde, l'esito NON è "non discussi" (usa svolti, parziali o non svolti).
Se in una lista non c'è nulla da mettere lasciala vuota ([]): non scrivere frasi come "non affrontati".
Rispondi solo con il JSON."""


def process_prompt(models):
    names = " e ".join(MODELS[m]["label"] for m in models)
    lines = [f"- {k}: {desc}" for k, _, desc in model_processes(models)]
    return PROCESS_PROMPT_HEAD + f"Modello: {names}.\n" + "\n".join(lines) + "\n" + PROCESS_PROMPT_TAIL


def process_schema(models):
    keys = [k for k, _, _ in model_processes(models)]
    return {
        "type": "object",
        "properties": {
            "processi": {"type": "object", "properties": {
                k: {"type": "object", "properties": {"punteggio": {"type": "integer", "enum": [0, 1, 2, 3, 4, 5]},
                                                     "evidenza": {"type": "string"}},
                    "required": ["punteggio", "evidenza"]} for k in keys},
                "required": keys},
            "temi": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
            "valori_emersi": {"type": "array", "items": {"type": "string"}},
            "azioni_concordate": {"type": "array", "items": {"type": "string"}},
            "compiti_precedenti": {"type": "object", "properties": {
                "esito": {"type": "string", "enum": ["svolti", "parziali", "non svolti", "non discussi"]},
                "nota": {"type": "string"}}, "required": ["esito", "nota"]},
            "rischio": {"type": "string"},
            "interventi": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        },
        "required": ["processi", "temi", "valori_emersi", "azioni_concordate", "compiti_precedenti", "rischio",
                     "interventi"],
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
    models = patient_models(sdir.parent)
    txt = sdir / "transcript.txt"
    meta = read_json(sdir / "meta.json", {}) or {}
    try:
        skill = get_skill(job["skill"])
    except HTTPException:
        skill = load_skills()[0]
    if txt.exists():
        source = "trascrizione"
        system, head = prompt_prefix(job["patient"], meta, skill,
                                     txt.read_text(encoding="utf-8-sig", errors="replace").strip())
    else:
        # no recording (e.g. a summary imported from another app): score from the summary text instead
        summary = session_summary_text(sdir)
        if not summary:
            raise RuntimeError("Serve una trascrizione o una sintesi per compilare la scheda")
        source = "sintesi"
        system, head = prompt_prefix(job["patient"], meta, skill, "")
        head += (f"\n\nNON C'È LA TRASCRIZIONE: hai solo la SINTESI DELLA SEDUTA scritta dal clinico. Basati su di essa "
                 f"(le evidenze saranno parafrasi, non citazioni) e usa 0 per i processi che non vi compaiono.\n\n"
                 f"SINTESI:\n\"\"\"\n{summary}\n\"\"\"")
    job["run_started"] = time.time()
    LLM_SERVER.busy = True
    try:
        job["status"] = "loading"
        LLM_SERVER.ensure()
        job["status"] = "reading"
        text, timings = LLM_SERVER.stream_chat(
            [{"role": "system", "content": system}, {"role": "user", "content": head + "\n\n" + process_prompt(models) + vocabulary_prompt()}],
            lambda p: job.update(progress=p), lambda t: job.update(status="writing", partial=t),
            response_format={"type": "json_schema", "json_schema": {"name": "scheda", "schema": process_schema(models)}},
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
        "written_tokens": timings.get("predicted_n"), "source": source,
        "proposed": card, "confirmed": old.get("confirmed", {})})  # clinician's confirmations survive a re-run
    job["status"] = "done"
    job["progress"] = 100


def session_summary_text(sdir: Path):
    """Text of the session's summary to analyze when there is no transcript: the imported one first."""
    files = sorted(sdir.glob("sintesi_*.md"), key=lambda f: (f.stem != f"sintesi_{IMPORTED_SKILL}", -f.stat().st_mtime))
    return files[0].read_text(encoding="utf-8-sig", errors="replace").strip() if files else ""


def analyzable(sdir: Path):
    return (sdir / "transcript.txt").exists() or any(sdir.glob("sintesi_*.md"))


def effective_card(sdir: Path):
    """Scores to plot: the clinician's value when confirmed, otherwise Gemma's proposal; 0 -> None (no evidence)."""
    data = read_json(sdir / "processi.json")
    if not data:
        return None
    prop, conf = data.get("proposed", {}), data.get("confirmed", {})
    models = patient_models(sdir.parent)
    procs, keys = {}, [k for k, _, _ in model_processes(models)]
    for k, label, _ in model_processes(models):
        p = (prop.get("processi") or {}).get(k) or {}
        proposed = p.get("punteggio") if isinstance(p.get("punteggio"), int) else 0
        value = conf[k] if k in conf else proposed
        procs[k] = {"label": label, "score": value or None, "proposed": proposed or None, "confirmed": k in conf,
                    "evidence": p.get("evidenza", ""), "model": next(m for m in models if k in dict((x[0], 1) for x in MODELS[m]["processes"]))}
    return {"processi": procs, "temi": prop.get("temi", []), "valori": prop.get("valori_emersi", []),
            "azioni": prop.get("azioni_concordate", []), "compiti": prop.get("compiti_precedenti", {}),
            "interventi": prop.get("interventi"),  # None on cards made before the graph existed
            "rischio": prop.get("rischio", ""), "generated_at": data.get("generated_at"),
            "all_confirmed": all(k in conf for k in keys), "source": data.get("source", "trascrizione"),
            "models": models,
            # card made before a model was added to this patient (e.g. DNA-v): "Aggiorna schede" completes it
            "incomplete": any(k not in (prop.get("processi") or {}) for k in keys)}


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
    labels: list[str] = []
    models: list[str] = []
    allow_suffix: bool = False  # on a name collision create 'NAME (2)' instead of refusing


class LabelsUpdate(BaseModel):
    labels: list[str]


class ModelsUpdate(BaseModel):
    models: list[str]  # ["hexaflex"], ["dnav"] or both


def patient_labels(pdir: Path):
    """Labels of a patient (e.g. the practice location), stored in <patient>/paziente.json."""
    return (read_json(pdir / "paziente.json", {}) or {}).get("etichette", [])


def save_patient_labels(pdir: Path, labels):
    # reuse the spelling already used by other patients ("bergamo" -> "Bergamo") so one location = one filter
    known = {lab.lower(): lab for p in ROOT.iterdir() if p.is_dir() and p != pdir for lab in patient_labels(p)}
    clean, seen = [], set()
    for lab in labels:
        lab = re.sub(r"\s+", " ", str(lab)).strip()[:40]
        lab = known.get(lab.lower(), lab)
        if lab and lab.lower() not in seen:
            seen.add(lab.lower())
            clean.append(lab)
    data = read_json(pdir / "paziente.json", {}) or {}
    data["etichette"] = clean
    write_json(pdir / "paziente.json", data)
    return clean


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
        if not p.is_dir() or p.name.startswith(("_", ".")):  # skip the trash (_Cestino) and hidden folders
            continue
        sessions = list_sessions(p)
        out.append({
            "name": p.name,
            "sessions": len(sessions),
            "last_date": sessions[0]["date"] if sessions else "",
            "busy": sum(1 for s in sessions if s["status"] in ("queued", "converting", "loading", "transcribing")),
            "pending": sum(1 for s in sessions if s["status"] in ("pending", "error")),
            "labels": patient_labels(p),
            "models": patient_models(p),
        })
    return out


@app.post("/api/patients")
def create_patient(body: NewPatient):
    """Patients are usually named by initials: on a collision the new one gets a number, 'M.R (2)', 'M.R (3)'…"""
    base = safe_name(body.name)
    if base.startswith(("_", ".")):
        raise HTTPException(400, "Nome non valido")
    existing = {p.name.lower() for p in ROOT.iterdir() if p.is_dir()}
    name, n = base, 2
    while name.lower() in existing:
        if not body.allow_suffix:
            raise HTTPException(409, "Esiste già un paziente con questo nome")
        name, n = f"{base} ({n})", n + 1
    p = inside_root(name)
    p.mkdir(parents=True)
    if body.labels:
        save_patient_labels(p, body.labels)
    if body.models:
        save_patient_models(p, body.models)
    return {"name": name}


def save_patient_models(pdir: Path, models):
    models = [m for m in dict.fromkeys(models) if m in MODELS]
    if not models:
        raise HTTPException(400, "Scegli almeno un modello")
    data = read_json(pdir / "paziente.json", {}) or {}
    data["modelli"] = models
    write_json(pdir / "paziente.json", data)
    return models


@app.put("/api/patients/{patient}/modelli")
def set_patient_models(patient: str, body: ModelsUpdate):
    return {"models": save_patient_models(patient_dir(patient), body.models)}


@app.put("/api/patients/{patient}/etichette")
def set_patient_labels(patient: str, body: LabelsUpdate):
    return {"labels": save_patient_labels(patient_dir(patient), body.labels)}


@app.get("/api/patients/{patient}")
def get_patient(patient: str):
    pdir = patient_dir(patient)
    return {"name": pdir.name, "sessions": list_sessions(pdir), "labels": patient_labels(pdir),
            "models": patient_models(pdir)}


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
    if not analyzable(sdir):
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
                    "analyzable": analyzable(sdir), "card": effective_card(sdir),
                    "job": process_job_info(pdir.name, s["id"])})
    models = patient_models(pdir)
    return {"patient": pdir.name,
            "processes": [{"key": k, "label": l, "model": m} for m in models for k, l, _ in MODELS[m]["processes"]],
            "models": [{"id": m, "label": MODELS[m]["label"], "radar": MODELS[m]["radar"]} for m in models],
            "hexaflex": HEXAFLEX, "sessions": out}


@app.post("/api/patients/{patient}/andamento/analizza")
def analyze_missing(patient: str):
    """Queue a process card for every transcribed session that doesn't have one yet."""
    pdir = patient_dir(patient)
    n = 0
    for s in list_sessions(pdir):
        sdir = pdir / s["id"]
        card = effective_card(sdir)
        if analyzable(sdir) and (card is None or card["incomplete"]):  # missing, or lacking a model added later
            enqueue_process_card(pdir.name, s["id"])
            n += 1
    return {"queued": n}


# ---------------------------------------------------------------- graph of sessions and concepts

GRAPH_KINDS = {"tema": ("temi", "temi"), "valore": ("valori", "valori"), "intervento": ("interventi", "interventi")}
# placeholders the model sometimes writes instead of an empty list, and generic words that are not a concept
PLACEHOLDER = re.compile(r"^(non |nessun|n/?a$|nulla|niente|-$)|^(metafor[ae]|esercizi[o]?|tecnic[ahe]+|interventi?)$")
ALIAS_FILE = lambda: ROOT / "_alias.json"  # merges done in the graph ("autocritica severa" -> "autocritica")


def load_aliases():
    return read_json(ALIAS_FILE(), {}) or {}


def concept_key(kind: str, label: str, vocab: dict, aliases: dict):
    """Normalized node key + display label: vocabulary spelling, then the clinician's merges."""
    s = re.sub(r"\s+", " ", str(label)).strip().strip(".;:,\"'").lower()
    if not s or PLACEHOLDER.search(s):
        return None, None
    display = vocab.get(s, s)
    key = f"{kind}:{display.lower()}"
    for _ in range(5):  # follow merge chains a -> b -> c
        if key not in aliases:
            break
        key = aliases[key]
    return key, vocab.get(key.split(":", 1)[1], key.split(":", 1)[1])


@app.get("/api/patients/{patient}/grafo")
def get_graph(patient: str):
    """Nodes: sessions + the themes, values and interventions of their process cards. Built from the cards, no AI."""
    pdir = patient_dir(patient)
    vocab_all, aliases = load_vocabulary(), load_aliases()
    sessions = sorted(list_sessions(pdir), key=lambda s: (s["date"] or "9999", s["id"]))
    nodes, edges, concepts, prev = [], [], {}, None
    outdated = missing = 0
    for s in sessions:
        sdir = pdir / s["id"]
        card = effective_card(sdir)
        if not card:
            missing += analyzable(sdir)
            continue
        if card.get("interventi") is None or card["incomplete"]:
            outdated += 1
        sid = "s:" + s["id"]
        nodes.append({"id": sid, "type": "seduta", "session": s["id"], "date": s["date"], "title": s["title"],
                      "compiti": (card.get("compiti") or {}).get("esito", ""), "azioni": card.get("azioni", []),
                      "source": card.get("source", "trascrizione")})
        if prev:
            edges.append({"source": prev, "target": sid, "type": "next",
                          "esito": (card.get("compiti") or {}).get("esito", "")})
        prev = sid
        for kind, (field, section) in GRAPH_KINDS.items():
            vocab = {t.lower(): t for t in vocab_all.get(section, [])}
            for label in card.get(field) or []:
                key, display = concept_key(kind, label, vocab, aliases)
                if not key:
                    continue
                c = concepts.setdefault(key, {"id": key, "type": kind, "label": display, "sessions": [],
                                              "in_vocab": display.lower() in vocab})
                if s["id"] not in c["sessions"]:
                    c["sessions"].append(s["id"])
                    edges.append({"source": sid, "target": key, "type": kind})
    for c in concepts.values():
        c["count"] = len(c["sessions"])
    return {"patient": pdir.name, "nodes": nodes + list(concepts.values()), "edges": edges,
            "outdated": outdated, "missing": missing,
            "aliases": [{"from": k, "to": v} for k, v in aliases.items()]}


@app.post("/api/patients/{patient}/grafo/aggiorna")
def refresh_graph_cards(patient: str):
    """Queue cards for sessions without one and re-run cards made before 'interventi' existed (confirmed scores kept)."""
    pdir = patient_dir(patient)
    n = 0
    for s in list_sessions(pdir):
        sdir = pdir / s["id"]
        card = effective_card(sdir)
        if analyzable(sdir) and (card is None or card.get("interventi") is None or card["incomplete"]):
            enqueue_process_card(pdir.name, s["id"])
            n += 1
    return {"queued": n}


class AliasRequest(BaseModel):
    source: str  # node key to merge, e.g. "tema:autocritica severa"
    target: str  # node key to keep, e.g. "tema:autocritica"


@app.post("/api/alias")
def add_alias(body: AliasRequest):
    if body.source.split(":")[0] != body.target.split(":")[0] or body.source == body.target:
        raise HTTPException(400, "Si possono unire solo concetti dello stesso tipo")
    aliases = load_aliases()
    aliases = {k: (body.target if v == body.source else v) for k, v in aliases.items()}  # re-point old merges
    aliases[body.source] = body.target
    aliases.pop(body.target, None)
    write_json(ALIAS_FILE(), aliases)
    return {"aliases": aliases}


@app.delete("/api/alias")
def remove_alias(source: str):
    aliases = load_aliases()
    aliases.pop(source, None)
    write_json(ALIAS_FILE(), aliases)
    return {"aliases": aliases}


@app.post("/api/patients/{patient}/import")
async def import_summaries(patient: str, files: list[UploadFile] = File(...)):
    """Import summary PDFs made with another app: one session per file, dated from the file name."""
    pdir = patient_dir(patient)
    created, errors = [], []
    for up in files:
        if not (up.filename or "").lower().endswith(".pdf"):
            errors.append({"file": up.filename, "error": "non è un PDF"})
            continue
        tmp = pdir / f".import_{uuid.uuid4().hex}.tmp"
        try:
            with open(tmp, "wb") as f:
                while chunk := await up.read(1024 * 1024):
                    f.write(chunk)
            created.append(import_summary_pdf(pdir, tmp, up.filename, move=True))
        except Exception as e:  # noqa: BLE001
            errors.append({"file": up.filename, "error": str(e)})
        finally:
            tmp.unlink(missing_ok=True)
    return {"created": created, "errors": errors}


class DateUpdate(BaseModel):
    date: str


class TitleUpdate(BaseModel):
    title: str


class MoveRequest(BaseModel):
    patient: str


class NameUpdate(BaseModel):
    name: str


TRASH = "_Cestino"  # deleted sessions and patients go here first (inside patients_dir), so they can be restored


def ensure_idle(prefix: str):
    """409 if a transcription/summary/card job is running on this patient or session (paths would change)."""
    if any(j["status"] in ACTIVE for k, j in JOBS.items() if k == prefix or k.startswith(prefix + "/")
           or k.startswith(prefix + "#")):
        raise HTTPException(409, "Attendi la fine dei lavori in corso")


def rename_session(sdir: Path, d: str, title: str):
    """Rename a session folder to '<date>_<title>' and update meta.json; returns the new folder."""
    meta = read_json(sdir / "meta.json", {}) or {}
    meta.update({"date": d, "title": title})
    write_json(sdir / "meta.json", meta)
    wanted = f"{d}_{safe_name(title).replace(' ', '_')}" if title else f"{d}_seduta"
    if sdir.name == wanted:
        return sdir
    target = new_session_dir(sdir.parent, d, title)
    sdir.rename(target)
    return target


def to_trash(path: Path, label: str):
    """Move a patient or session folder into _Cestino, with a note of where it came from."""
    tdir = ROOT / TRASH
    tdir.mkdir(exist_ok=True)
    target = tdir / f"{time.strftime('%Y%m%d-%H%M%S')}_{path.name}"
    shutil.move(str(path), target)
    write_json(target / "_cestino.json", {"origin": str(path.relative_to(ROOT)), "label": label,
                                          "deleted_at": time.strftime("%Y-%m-%d %H:%M")})
    return target


@app.put("/api/patients/{patient}/sessions/{session}/data")
def set_session_date(patient: str, session: str, body: DateUpdate):
    """Change a session's date: updates meta.json and renames the folder so it starts with the new date."""
    sdir = session_dir(patient, session)
    try:
        d = date.fromisoformat(body.date.strip()).isoformat()
    except ValueError:
        raise HTTPException(400, "Data non valida (formato AAAA-MM-GG)")
    ensure_idle(f"{sdir.parent.name}/{sdir.name}")
    title = session_info(sdir)["title"]
    target = rename_session(sdir, d, title)
    meta = read_json(target / "meta.json", {}) or {}
    meta["date_source"] = "modificata a mano"
    write_json(target / "meta.json", meta)
    return {"id": target.name, "date": d}


@app.put("/api/patients/{patient}/sessions/{session}/titolo")
def set_session_title(patient: str, session: str, body: TitleUpdate):
    sdir = session_dir(patient, session)
    title = body.title.strip()
    if not title:
        raise HTTPException(400, "Il titolo non può essere vuoto")
    safe_name(title)
    ensure_idle(f"{sdir.parent.name}/{sdir.name}")
    info = session_info(sdir)
    target = rename_session(sdir, info["date"] or date.today().isoformat(), title)
    return {"id": target.name}


@app.post("/api/patients/{patient}/sessions/{session}/sposta")
def move_session(patient: str, session: str, body: MoveRequest):
    """Move a session to another patient (e.g. imported or recorded under the wrong name)."""
    sdir = session_dir(patient, session)
    dest = patient_dir(body.patient)
    if dest == sdir.parent:
        raise HTTPException(400, "La seduta è già di questo paziente")
    ensure_idle(f"{sdir.parent.name}/{sdir.name}")
    info = session_info(sdir)
    target = new_session_dir(dest, info["date"] or date.today().isoformat(), info["title"])
    shutil.move(str(sdir), target)
    return {"patient": dest.name, "id": target.name}


@app.delete("/api/patients/{patient}/sessions/{session}")
def delete_session(patient: str, session: str):
    sdir = session_dir(patient, session)
    ensure_idle(f"{sdir.parent.name}/{sdir.name}")
    info = session_info(sdir)
    to_trash(sdir, f"{sdir.parent.name} · {info['date']} {info['title']}")
    return {"ok": True}


@app.put("/api/patients/{patient}")
def rename_patient(patient: str, body: NameUpdate):
    pdir = patient_dir(patient)
    name = safe_name(body.name)
    if name.startswith(("_", ".")):
        raise HTTPException(400, "Nome non valido")
    target = inside_root(name)
    if target.exists() and target.resolve() != pdir.resolve():
        raise HTTPException(409, "Esiste già un paziente con questo nome")
    ensure_idle(pdir.name)
    pdir.rename(target)
    return {"name": name}


@app.post("/api/patients/{patient}/open")
def open_patient_folder(patient: str):
    os.startfile(patient_dir(patient))
    return {"ok": True}


@app.delete("/api/patients/{patient}")
def delete_patient(patient: str):
    pdir = patient_dir(patient)
    if pdir.name.startswith(("_", ".")):
        raise HTTPException(400, "Non si può eliminare")
    ensure_idle(pdir.name)
    to_trash(pdir, f"Paziente {pdir.name}")
    return {"ok": True}


@app.get("/api/cestino")
def list_trash():
    tdir = ROOT / TRASH
    items = []
    for d in sorted(tdir.iterdir(), reverse=True) if tdir.exists() else []:
        if d.is_dir():
            info = read_json(d / "_cestino.json", {}) or {}
            items.append({"id": d.name, "label": info.get("label", d.name), "origin": info.get("origin", ""),
                          "deleted_at": info.get("deleted_at", "")})
    return items


@app.post("/api/cestino/{item}/ripristina")
def restore_trash(item: str):
    src = inside_root(TRASH, item)
    info = read_json(src / "_cestino.json", {}) or {}
    origin = Path(info.get("origin", ""))
    if not src.is_dir() or not origin.parts:
        raise HTTPException(404, "Elemento non trovato")
    (src / "_cestino.json").unlink(missing_ok=True)
    if len(origin.parts) == 1:  # a whole patient
        target, n = ROOT / origin.parts[0], 2
        while target.exists():
            target, n = ROOT / f"{origin.parts[0]} ({n})", n + 1
    else:  # a session: back into its patient (recreated if the patient was deleted meanwhile)
        pdir = ROOT / origin.parts[0]
        pdir.mkdir(exist_ok=True)
        m = re.match(r"(\d{4}-\d{2}-\d{2})_?(.*)", origin.parts[1])
        target = (pdir / origin.parts[1]) if not (pdir / origin.parts[1]).exists() else \
            new_session_dir(pdir, m.group(1) if m else date.today().isoformat(), (m.group(2) if m else "").replace("_", " "))
    shutil.move(str(src), target)
    return {"restored": str(target.relative_to(ROOT))}


@app.delete("/api/cestino/{item}")
def purge_trash(item: str):
    """Permanently delete one item from the trash."""
    src = inside_root(TRASH, item)
    if not src.is_dir():
        raise HTTPException(404, "Elemento non trovato")
    shutil.rmtree(src)
    return {"ok": True}


@app.get("/api/patients/{patient}/sessions/{session}/originale")
def get_original_pdf(patient: str, session: str):
    f = session_dir(patient, session) / "sintesi_originale.pdf"
    if not f.exists():
        raise HTTPException(404, "PDF originale non trovato")
    return FileResponse(f, media_type="application/pdf")


@app.post("/api/patients/{patient}/sessions/{session}/summary")
def create_summary(patient: str, session: str, body: SummaryRequest):
    sdir = session_dir(patient, session)
    if not (sdir / "transcript.txt").exists() and (body.mode == "generate" or
                                                   not (sdir / f"sintesi_{safe_name(body.skill)}.md").exists()):
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
