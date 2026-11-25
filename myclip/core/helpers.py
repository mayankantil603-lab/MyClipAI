"""Optional AI helpers: registry, status and one-click installers.

Everything is optional. Each helper says whether it's ready; the job uses it only
when it is, and falls back to the normal path otherwise. Heavy models run in their
own process (see myclip/helpers_cli.py) so PyTorch never loads into the app."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .models import Cancelled
from .paths import config_dir

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
Progress = Callable[[float, str], None]

HINDI_REPO = "cutvideogaps/indicwhisper-hindi-ct2"
HINDI_BYTES = 780 * 1024 * 1024
YUNET_URLS = [
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
]
ESRGAN_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
              "realesrgan-ncnn-vulkan-20220424-windows.zip")
TORCH_INDEXES = ["https://download.pytorch.org/whl/cu128", "https://download.pytorch.org/whl/cu130"]
TORCH_CPU = "https://download.pytorch.org/whl/cpu"
VISION_DEFAULT = "gemma3:4b"


@dataclass
class Helper:
    id: str
    name: str
    what: str
    size: str


HELPERS = [
    Helper("voice", "Voice cleanup",
           "Removes music, gunshots and game sound before captions are made, so Whisper only hears the voice (Demucs).",
           "About 2.5 GB with an NVIDIA GPU (PyTorch with CUDA), about 300 MB without."),
    Helper("vision", "Vision",
           "Looks at frames of each possible clip to spot kills, wins, deaths and funny moments, and helps the titles.",
           "Uses a vision model in Ollama. Nothing to download if you already have gemma3 or qwen2.5vl, "
           "otherwise gemma3:4b (3.3 GB)."),
    Helper("faces", "Face tracking",
           "The vertical crop follows the streamer's face, and the Facecam layout finds your camera box.",
           "About 60 MB (OpenCV and a small face model)."),
    Helper("hindi", "Hindi speech model",
           "IndicWhisper Hindi by AI4Bharat. Use it for Hindi captions and compare it with Whisper on any clip.",
           "About 780 MB."),
    Helper("upscale", "Sharper thumbnails",
           "Picks the sharpest frame, ideally with a face, and upscales it with Real-ESRGAN.",
           "About 45 MB."),
]
BY_ID = {h.id: h for h in HELPERS}


def helpers_dir() -> Path:
    p = config_dir() / "helpers"
    p.mkdir(parents=True, exist_ok=True)
    return p


def app_root() -> Path:
    return Path(__file__).resolve().parents[2]


def python_exe() -> str:
    """python.exe next to pythonw.exe (pip needs a console-capable interpreter)."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and (exe.parent / "python.exe").exists():
        return str(exe.parent / "python.exe")
    return str(exe)


def _marker(hid: str) -> Path:
    return helpers_dir() / f"{hid}.ok"


def _has_module(name: str) -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


# ---------------------------------------------------------------- status

def hindi_dir() -> Path:
    return helpers_dir() / "hindi"


def yunet_path() -> Path:
    return helpers_dir() / "faces" / "yunet.onnx"


def esrgan_exe() -> Optional[Path]:
    root = helpers_dir() / "upscale"
    if not root.is_dir():
        return None
    for p in root.rglob("realesrgan-ncnn-vulkan*"):
        if p.is_file() and p.suffix.lower() in (".exe", ""):
            return p
    return None


def vision_model_saved() -> str:
    try:
        return json.loads((helpers_dir() / "vision.json").read_text(encoding="utf-8")).get("model", "")
    except Exception:
        return ""


def save_vision_model(model: str) -> None:
    (helpers_dir() / "vision.json").write_text(json.dumps({"model": model}), encoding="utf-8")


def installed(hid: str) -> bool:
    if hid == "voice":
        return _marker("voice").exists() and _has_module("torch") and _has_module("demucs")
    if hid == "faces":
        return _has_module("cv2")
    if hid == "hindi":
        return (hindi_dir() / "model.bin").exists()
    if hid == "upscale":
        return esrgan_exe() is not None
    if hid == "vision":
        return bool(vision_model_saved())
    return False


def status_text(hid: str) -> str:
    if hid == "vision":
        m = vision_model_saved()
        return f"Ready: {m}" if m else "No vision model found yet"
    if hid == "faces" and installed("faces"):
        return "Ready" if yunet_path().exists() else "Ready (basic detector; the better face model didn't download)"
    return "Ready" if installed(hid) else "Not installed"


# ---------------------------------------------------------------- install plumbing

def run_logged(cmd: list[str], cancel: threading.Event, on_line: Optional[Callable[[str], None]] = None,
               cwd: Optional[str] = None, timeout: float = 7200) -> int:
    """Run a command with no window, stream its output lines, kill on cancel."""
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            cwd=cwd, creationflags=NO_WINDOW)
    t0 = time.time()

    def pump() -> None:
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            print(f"[helper] {line}", flush=True)
            if on_line and line:
                on_line(line)

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    while proc.poll() is None:
        if cancel.wait(0.25) or time.time() - t0 > timeout:
            proc.kill()
            proc.wait()
            raise Cancelled()
    th.join(3)
    return proc.returncode


def pip(args: list[str], cancel: threading.Event, on_line: Optional[Callable[[str], None]] = None) -> bool:
    cmd = [python_exe(), "-m", "pip", "install", "--disable-pip-version-check", "--no-input", *args]
    return run_logged(cmd, cancel, on_line) == 0


def pip_uninstall(names: list[str], cancel: threading.Event) -> None:
    run_logged([python_exe(), "-m", "pip", "uninstall", "-y", *names], cancel)


def download(url: str, dest: Path, cancel: threading.Event, on_progress: Optional[Progress] = None,
             label: str = "Downloading") -> None:
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=(15, 120), allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done = 0
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                if cancel.is_set():
                    raise Cancelled()
                fh.write(chunk)
                done += len(chunk)
                if on_progress:
                    on_progress(done / total if total else 0.0,
                                f"{label} {done / 1048576:.0f} MB" + (f" of {total / 1048576:.0f} MB" if total else ""))
    os.replace(tmp, dest)


def has_nvidia() -> bool:
    return shutil.which("nvidia-smi") is not None


def _selftest(kind: str, cancel: threading.Event) -> tuple[bool, str]:
    out: list[str] = []
    code = run_logged([python_exe(), "-m", "myclip.helpers_cli", "selftest", kind], cancel, out.append,
                      cwd=str(app_root()), timeout=1800)
    last = next((l for l in reversed(out) if l.startswith("{")), "")
    try:
        d = json.loads(last)
        return bool(d.get("ok")), str(d.get("msg", ""))
    except Exception:
        return code == 0, (out[-1] if out else "")


# ---------------------------------------------------------------- installers

def install(hid: str, cancel: threading.Event, on_progress: Progress, ollama_url: str = "http://localhost:11434") -> str:
    """Install one helper. Returns a short result message, raises RuntimeError on failure."""
    if hid == "voice":
        return _install_voice(cancel, on_progress)
    if hid == "faces":
        return _install_faces(cancel, on_progress)
    if hid == "hindi":
        return _install_hindi(cancel, on_progress)
    if hid == "upscale":
        return _install_upscale(cancel, on_progress)
    if hid == "vision":
        return _install_vision(cancel, on_progress, ollama_url)
    raise RuntimeError("Unknown helper")


def _install_voice(cancel: threading.Event, on_progress: Progress) -> str:
    lines = lambda l: on_progress(-1, l[:120])
    have_torch = _has_module("torch")
    if not have_torch:
        idx = TORCH_INDEXES if has_nvidia() else [TORCH_CPU]
        ok = False
        for i, url in enumerate(idx):
            on_progress(0.05, "Installing PyTorch" + (" with CUDA (about 2.5 GB, this takes a while)" if has_nvidia()
                                                     else " (about 250 MB)"))
            if pip(["torch", "torchaudio", "--index-url", url], cancel, lines):
                ok = True
                break
        if not ok and has_nvidia():
            on_progress(0.05, "GPU build failed, installing the CPU build instead")
            ok = pip(["torch", "torchaudio", "--index-url", TORCH_CPU], cancel, lines)
        if not ok:
            raise RuntimeError("PyTorch couldn't be installed. Check your internet and try again.")
    on_progress(0.75, "Installing Demucs")
    # --no-deps so pip never swaps the CUDA build of PyTorch for a CPU one
    if not pip(["--no-deps", "demucs==4.0.1", "openunmix"], cancel, lines):
        raise RuntimeError("Demucs couldn't be installed.")
    if not pip(["dora-search", "julius", "einops", "pyyaml", "tqdm"], cancel, lines):
        raise RuntimeError("Demucs' helper packages couldn't be installed.")
    on_progress(0.9, "Testing (downloads the 80 MB voice model once)")
    ok, msg = _selftest("voice", cancel)
    if not ok:
        raise RuntimeError(f"Voice cleanup didn't pass its test: {msg[:200]}")
    _marker("voice").write_text(msg or "ok", encoding="utf-8")
    return msg or "Ready"


def _install_faces(cancel: threading.Event, on_progress: Progress) -> str:
    on_progress(0.1, "Installing OpenCV (about 40 MB)")
    if not _has_module("cv2"):
        if not pip(["opencv-python-headless>=4.8"], cancel, lambda l: on_progress(-1, l[:120])):
            raise RuntimeError("OpenCV couldn't be installed.")
    on_progress(0.7, "Downloading the face model")
    got = False
    for url in YUNET_URLS:
        try:
            download(url, yunet_path(), cancel)
            if yunet_path().stat().st_size > 100_000:
                got = True
                break
        except Cancelled:
            raise
        except Exception as exc:
            print(f"[helper] yunet download failed from {url}: {exc}", flush=True)
    on_progress(1.0, "Done")
    return "Ready" if got else "Ready (basic face detector; the better model couldn't be downloaded)"


def _dir_size(p: Path) -> int:
    try:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    except OSError:
        return 0


def _install_hindi(cancel: threading.Event, on_progress: Progress) -> str:
    from huggingface_hub import snapshot_download     # comes with faster-whisper
    dest = hindi_dir()
    dest.mkdir(parents=True, exist_ok=True)
    box: dict = {}

    def work() -> None:
        try:
            snapshot_download(HINDI_REPO, local_dir=str(dest))
            box["ok"] = True
        except Exception as exc:
            box["err"] = exc

    th = threading.Thread(target=work, daemon=True)
    th.start()
    while th.is_alive():
        if cancel.wait(1.0):
            raise Cancelled()
        sz = _dir_size(dest)
        on_progress(min(0.98, sz / HINDI_BYTES), f"Downloading the Hindi model {sz / 1048576:.0f} of about "
                                                 f"{HINDI_BYTES / 1048576:.0f} MB")
    if "err" in box:
        raise RuntimeError(f"Download failed: {box['err']}")
    if not (dest / "model.bin").exists():
        raise RuntimeError("The download finished but the model file is missing.")
    return "Ready"


def _install_upscale(cancel: threading.Event, on_progress: Progress) -> str:
    root = helpers_dir() / "upscale"
    z = helpers_dir() / "realesrgan.zip"
    download(ESRGAN_URL, z, cancel, lambda f, m: on_progress(f * 0.9, m), "Downloading Real-ESRGAN")
    on_progress(0.92, "Unpacking")
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(root)
    z.unlink(missing_ok=True)
    if esrgan_exe() is None:
        raise RuntimeError("Real-ESRGAN was downloaded but the program wasn't found in it.")
    return "Ready"


def _install_vision(cancel: threading.Event, on_progress: Progress, ollama_url: str) -> str:
    from . import vision
    on_progress(0.02, "Looking for a vision model in Ollama")
    found = vision.find_local_models(ollama_url)
    if found:
        best = vision.choose(found, "")
        save_vision_model(best)
        return f"Ready: {best}"
    import requests
    on_progress(0.05, f"Downloading {VISION_DEFAULT} in Ollama (3.3 GB)")
    try:
        with requests.post(ollama_url.rstrip("/") + "/api/pull", json={"model": VISION_DEFAULT, "stream": True},
                           stream=True, timeout=(10, 3600)) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if cancel.is_set():
                    raise Cancelled()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("error"):
                    raise RuntimeError(d["error"])
                tot, done = d.get("total") or 0, d.get("completed") or 0
                if tot:
                    on_progress(done / tot, f"Downloading {VISION_DEFAULT}: {done / 1e9:.1f} of {tot / 1e9:.1f} GB")
    except Cancelled:
        raise
    except Exception as exc:
        raise RuntimeError(f"Couldn't download {VISION_DEFAULT} with Ollama. Is Ollama running? ({exc})") from None
    save_vision_model(VISION_DEFAULT)
    return f"Ready: {VISION_DEFAULT}"


def remove(hid: str, cancel: threading.Event) -> None:
    if hid == "voice":
        _marker("voice").unlink(missing_ok=True)
        pip_uninstall(["demucs", "openunmix", "torchaudio", "torch"], cancel)
    elif hid == "faces":
        shutil.rmtree(helpers_dir() / "faces", ignore_errors=True)
        pip_uninstall(["opencv-python-headless"], cancel)
    elif hid == "hindi":
        shutil.rmtree(hindi_dir(), ignore_errors=True)
    elif hid == "upscale":
        shutil.rmtree(helpers_dir() / "upscale", ignore_errors=True)
    elif hid == "vision":
        (helpers_dir() / "vision.json").unlink(missing_ok=True)
