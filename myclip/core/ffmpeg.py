"""FFmpeg access. Uses the binary bundled with imageio-ffmpeg so the user
never installs anything; falls back to ffmpeg on PATH. Probing is done by
parsing `ffmpeg -i` so ffprobe is not needed."""
from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
import threading
from typing import Callable, Optional

from .models import Cancelled, MediaInfo

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@functools.lru_cache(maxsize=1)
def ffmpeg_path() -> str:
    env = os.environ.get("MYCLIP_FFMPEG")
    if env and os.path.isfile(env):
        return env
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise RuntimeError("FFmpeg is missing. Run: pip install imageio-ffmpeg")


def run(args: list[str], cancel: Optional[threading.Event] = None, cwd: Optional[str] = None,
        on_line: Optional[Callable[[str], None]] = None) -> str:
    """Run ffmpeg with args, killable via `cancel`. Returns stderr text."""
    proc = subprocess.Popen([ffmpeg_path(), "-hide_banner", *args], stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, cwd=cwd,
                            creationflags=NO_WINDOW)
    lines: list[str] = []

    def pump() -> None:
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            lines.append(line)
            if len(lines) > 4000:
                del lines[:2000]
            if on_line:
                on_line(line)

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    while proc.poll() is None:
        if cancel is not None and cancel.wait(0.15):
            proc.kill()
            proc.wait()
            raise Cancelled()
    t.join(2)
    text = "\n".join(lines)
    if proc.returncode != 0:
        tail = "\n".join(l for l in lines[-15:] if l.strip())
        raise RuntimeError(f"FFmpeg error:\n{tail}")
    return text


_DUR = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VID = re.compile(r"Stream #.*?Video:.*?(\d{2,5})x(\d{2,5})")
_FPS = re.compile(r"(\d+(?:\.\d+)?)\s*fps")
_ROT = re.compile(r"(?:rotate\s*:\s*|rotation of\s*)(-?\d+)")


def parse_probe(text: str) -> MediaInfo:
    m = _DUR.search(text)
    if not m:
        raise RuntimeError("Could not read this video (no duration found).")
    dur = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
    vline = next((l for l in text.splitlines() if "Video:" in l and "Stream #" in l), "")
    if not vline:
        raise RuntimeError("This file has no video stream.")
    v = _VID.search(vline)
    w, h = (int(v[1]), int(v[2])) if v else (1920, 1080)
    f = _FPS.search(vline)
    fps = float(f[1]) if f else 30.0
    r = _ROT.search(text)
    if r and abs(int(r[1])) % 180 == 90:
        w, h = h, w
    audio = any("Audio:" in l and "Stream #" in l for l in text.splitlines())
    return MediaInfo(dur, w, h, fps or 30.0, audio)


def probe(path: str) -> MediaInfo:
    p = subprocess.run([ffmpeg_path(), "-hide_banner", "-i", path], capture_output=True,
                       stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
    return parse_probe(p.stderr.decode("utf-8", "replace"))


@functools.lru_cache(maxsize=1)
def nvenc_ok() -> bool:
    """True only if NVENC actually encodes a frame (listing it isn't enough)."""
    try:
        p = subprocess.run([ffmpeg_path(), "-hide_banner", "-v", "error", "-f", "lavfi", "-i",
                            "color=c=black:s=256x256:d=0.1", "-c:v", "h264_nvenc", "-f", "null", "-"],
                           capture_output=True, stdin=subprocess.DEVNULL, timeout=20,
                           creationflags=NO_WINDOW)
        return p.returncode == 0
    except Exception:
        return False


def time_progress(line: str) -> Optional[float]:
    m = re.search(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", line)
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else None
