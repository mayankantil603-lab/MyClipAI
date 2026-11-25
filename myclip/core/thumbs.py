"""Thumbnails: the sharpest frame (Laplacian variance), preferring one with a face,
optionally upscaled with Real-ESRGAN; plus small hover-preview frames for the cards."""
from __future__ import annotations

import math
import os
import subprocess
import tempfile
import threading
from typing import Optional

import numpy as np

from . import ffmpeg, helpers
from .models import Cancelled


def sharpness(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def frame_score(gray: np.ndarray, has_face: bool = False) -> float:
    s = math.log1p(sharpness(gray))
    m = float(gray.mean())
    if m < 35:
        s -= (35 - m) / 12
    elif m > 225:
        s -= (m - 225) / 8
    s += 0.25 * min(1.0, float(gray.std()) / 60)        # some contrast, not a flat frame
    return s + (0.9 if has_face else 0.0)


def sample_times(duration: float, n: int = 12) -> list[float]:
    a, b = min(0.5, duration * 0.1), max(duration - 0.5, duration * 0.9)
    return [round(a + (b - a) * i / max(1, n - 1), 3) for i in range(n)]


def best_time(video: str, duration: float, W: int, H: int, cancel: threading.Event, use_faces: bool) -> float:
    w = 288
    h = max(2, int(round(w * H / max(W, 1) / 2)) * 2)
    best_t, best_s = min(duration * 0.35, 3.0), -1e9
    det = None
    if use_faces:
        try:
            from .faces import Detector
            det = Detector(w, h)
        except Exception:
            det = None
    for t in sample_times(duration):
        if cancel.is_set():
            raise Cancelled()
        fr = ffmpeg.raw_frames(video, t, 0.15, 10, w, h, "bgr24", cancel)
        if not len(fr):
            continue
        bgr = np.ascontiguousarray(fr[0])
        face = bool(det.detect(bgr)) if det is not None else False
        s = frame_score(bgr.mean(axis=2), face)
        if s > best_s:
            best_t, best_s = t, s
    return best_t


def _esrgan(src_png: str, dst_png: str, cancel: threading.Event) -> bool:
    exe = helpers.esrgan_exe()
    if exe is None:
        return False
    try:
        p = subprocess.Popen([str(exe), "-i", src_png, "-o", dst_png, "-n", "realesrgan-x4plus", "-s", "4", "-f", "png"],
                             cwd=str(exe.parent), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for _ in range(240):
            if p.poll() is not None:
                break
            if cancel.wait(0.25):
                p.kill()
                raise Cancelled()
        if p.poll() is None:
            p.kill()
            return False
        return p.returncode == 0 and os.path.isfile(dst_png)
    except Cancelled:
        raise
    except Exception:
        return False


def make(video: str, out_jpg: str, duration: float, W: int, H: int, cancel: threading.Event,
         smart: bool = True) -> dict:
    """Write the card thumbnail. Returns {"at": time, "upscaled": bool}."""
    at = min(duration * 0.35, 3.0)
    if smart:
        try:
            at = best_time(video, duration, W, H, cancel, helpers.installed("faces"))
        except Cancelled:
            raise
        except Exception as exc:
            print(f"[MyClip] sharp-frame pick skipped: {exc}", flush=True)
    up = False
    if smart and helpers.installed("upscale"):
        with tempfile.TemporaryDirectory(prefix="myclip_th_") as tmp:
            a, b = os.path.join(tmp, "a.png"), os.path.join(tmp, "b.png")
            ffmpeg.run(["-y", "-v", "error", "-ss", f"{at:.3f}", "-i", os.path.abspath(video), "-frames:v", "1",
                        "-vf", "scale=360:-2:flags=lanczos", a], cancel)
            if _esrgan(a, b, cancel):
                ffmpeg.run(["-y", "-v", "error", "-i", b, "-vf", "scale=720:-2:flags=lanczos", "-q:v", "3",
                            os.path.abspath(out_jpg)], cancel)
                up = True
    if not up:
        ffmpeg.run(["-y", "-v", "error", "-ss", f"{at:.3f}", "-i", os.path.abspath(video), "-frames:v", "1",
                    "-vf", "scale=720:-2:flags=lanczos" + (",unsharp=5:5:0.6" if smart else ""), "-q:v", "3",
                    os.path.abspath(out_jpg)], cancel)
    return {"at": at, "upscaled": up}


def hover_times(duration: float, n: int = 6) -> list[float]:
    """Evenly spread inside the clip, avoiding the very first and last moments."""
    if duration <= 0:
        return []
    return [round(duration * (i + 0.5) / n, 3) for i in range(n)]


def hover_frames(video: str, thumbs_dir: str, prefix: str, duration: float, cancel: threading.Event,
                 n: int = 6) -> list[str]:
    """Small frames for the card's hover preview. Returns file names relative to thumbs_dir's parent."""
    out = []
    for k, t in enumerate(hover_times(duration, n), 1):
        if cancel.is_set():
            raise Cancelled()
        name = f"{prefix}_h{k}.jpg"
        b = ffmpeg.grab_jpeg(video, t, 360, 5)
        if b:
            with open(os.path.join(thumbs_dir, name), "wb") as f:
                f.write(b)
            out.append(f"{os.path.basename(thumbs_dir)}/{name}")
    return out
