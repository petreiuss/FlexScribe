"""Download what the summaries need, only if it is missing: llama.cpp (Vulkan build for the Intel GPU) and the
Gemma model. Writes the paths into config.json. Run by avvia.bat; safe to run again (it skips what is there).

Usage: python tools/setup_llm.py [--force]
"""
import glob
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

os.environ.update({"HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1"})  # downloads only, no usage pings

APP_DIR = Path(__file__).resolve().parent.parent
CONFIG = APP_DIR / "config.json"
LLAMA_DIR = APP_DIR / "llama.cpp"
# pinned build: the app was tested with it (flags used: serve, --device, --spec-type, --reasoning, --offline)
LLAMA_BUILD = "b11879"
LLAMA_ASSET = f"llama-{LLAMA_BUILD}-bin-win-vulkan-x64.zip"
LLAMA_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/{LLAMA_ASSET}"
GEMMA_REPO = "google/gemma-4-E4B-it-qat-q4_0-gguf"   # Apache 2.0, no Hugging Face account needed
GEMMA_FILE = "gemma-4-E4B_q4_0-it.gguf"               # ~5 GB


def load_config():
    return json.loads(CONFIG.read_text(encoding="utf-8-sig")) if CONFIG.exists() else {}


def save_config(updates):
    cfg = load_config()
    cfg.update(updates)
    CONFIG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def find_llama(cfg):
    candidates = [cfg.get("llm_exe"), str(LLAMA_DIR / "llama.exe"), shutil.which("llama"),
                  str(Path.home() / "AppData/Local/Microsoft/WindowsApps/llama.exe")]
    return next((c for c in candidates if c and Path(c).is_file()), None)


def find_gemma(cfg):
    if cfg.get("llm_model") and Path(cfg["llm_model"]).is_file():
        return cfg["llm_model"]
    hits = [p for p in glob.glob(str(Path.home() / ".cache/huggingface/hub/models--google--gemma-4*/snapshots/*/*.gguf"))
            if "mmproj" not in p]
    return hits[0] if hits else None


def download(url, dest: Path):
    print(f"  {url}")
    with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
        total, done = int(r.headers.get("Content-Length") or 0), 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:5.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()


def install_llama():
    print(f"Scarico llama.cpp {LLAMA_BUILD} (build Vulkan per la scheda grafica, circa 40 MB)...")
    LLAMA_DIR.mkdir(exist_ok=True)
    tmp = LLAMA_DIR / LLAMA_ASSET
    download(LLAMA_URL, tmp)
    with zipfile.ZipFile(tmp) as z:
        z.extractall(LLAMA_DIR)
    tmp.unlink()
    exe = next(LLAMA_DIR.rglob("llama.exe"), None)
    if not exe:
        sys.exit("llama.exe non trovato nell'archivio scaricato")
    return str(exe)


def install_gemma():
    from huggingface_hub import hf_hub_download
    print(f"Scarico il modello Gemma ({GEMMA_FILE}, circa 5 GB): può richiedere diversi minuti...")
    return hf_hub_download(GEMMA_REPO, GEMMA_FILE)  # Hugging Face cache, shared with other tools


DIAR_DIR = APP_DIR / "models" / "diarization"
DIAR_BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
DIAR_SEGMENTATION = f"{DIAR_BASE}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"   # ~7 MB
DIAR_EMBEDDING = f"{DIAR_BASE}/speaker-recongition-models/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx"  # ~27 MB


def install_diarization():
    """Models that recognize who is speaking (therapist / patient), used by sherpa-onnx."""
    import tarfile
    seg = DIAR_DIR / "segmentation.onnx"
    emb = DIAR_DIR / "embedding.onnx"
    if seg.exists() and emb.exists():
        return
    print("Scarico i modelli per riconoscere chi parla (circa 35 MB)...")
    DIAR_DIR.mkdir(parents=True, exist_ok=True)
    if not seg.exists():
        tmp = DIAR_DIR / "segmentation.tar.bz2"
        download(DIAR_SEGMENTATION, tmp)
        with tarfile.open(tmp) as t:
            member = next(m for m in t.getmembers() if m.name.endswith("model.onnx"))
            with t.extractfile(member) as src, open(seg, "wb") as dst:
                dst.write(src.read())
        tmp.unlink()
    if not emb.exists():
        download(DIAR_EMBEDDING, emb)


def main():
    force = "--force" in sys.argv
    try:
        install_diarization()
    except Exception as e:  # noqa: BLE001 - the app works without it (no speaker labels)
        print(f"Attenzione: modelli per riconoscere chi parla non scaricati ({e})")
    cfg = load_config()
    exe = None if force else find_llama(cfg)
    model = None if force else find_gemma(cfg)
    if exe and model and cfg.get("llm_exe") == exe and cfg.get("llm_model") == model:
        return  # all set, nothing to do
    if not exe:
        exe = install_llama()
    if not model:
        model = install_gemma()
    save_config({"llm_exe": exe, "llm_model": model})
    print(f"Gemma pronto:\n  llama: {exe}\n  modello: {model}")


if __name__ == "__main__":
    main()
