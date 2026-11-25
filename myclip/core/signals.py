"""Signals computed from the video itself: audio loudness and scene cuts."""
from __future__ import annotations

import math
import re
import subprocess
import threading
from typing import Callable, Optional

import numpy as np

from . import ffmpeg
from .models import Cancelled


def loudness(path: str, hop: float, cancel: threading.Event,
             on_progress: Optional[Callable[[float], None]] = None, duration: float = 0) -> np.ndarray:
    """RMS dBFS every `hop` seconds. Streamed; RAM use stays tiny on 5h VODs."""
    sr = 8000
    cmd = [ffmpeg.ffmpeg_path(), "-hide_banner", "-v", "error", "-i", path, "-vn", "-sn", "-dn",
           "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, creationflags=ffmpeg.NO_WINDOW)
    block = int(sr * hop) * 2
    vals: list[float] = []
    try:
        while True:
            buf = proc.stdout.read(block)
            if not buf:
                break
            a = np.frombuffer(buf[: len(buf) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0
            if a.size:
                vals.append(20 * math.log10(float(np.sqrt(np.mean(a * a))) + 1e-7))
            if len(vals) % 200 == 0:
                if cancel.is_set():
                    raise Cancelled()
                if on_progress and duration:
                    on_progress(min(len(vals) * hop / duration, 1.0))
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    return np.asarray(vals, dtype=np.float32)


_PTS = re.compile(r"pts_time:(\d+(?:\.\d+)?)")


def scene_cuts(path: str, cancel: threading.Event, threshold: float = 0.32,
               on_progress: Optional[Callable[[float], None]] = None, duration: float = 0) -> list[float]:
    """Hard cuts via ffmpeg scene score on a downscaled, 4 fps stream (fast)."""
    cuts: list[float] = []

    def line(l: str) -> None:
        m = _PTS.search(l)
        if m:
            cuts.append(float(m[1]))
        if on_progress and duration:
            t = ffmpeg.time_progress(l)
            if t is not None:
                on_progress(min(t / duration, 1.0))

    ffmpeg.run(["-nostats", "-stats_period", "2", "-stats", "-i", path, "-an", "-sn",
                "-vf", f"fps=4,scale=320:-2,select='gt(scene,{threshold})',showinfo",
                "-f", "null", "-"], cancel, on_line=line)
    return sorted(set(round(c, 2) for c in cuts))


def motion_center(path: str, start: float, end: float, cancel: threading.Event) -> Optional[float]:
    """Horizontal centre (0..1) of where the action is, for Smart framing.
    Samples small grayscale frames and weights columns by frame-difference
    energy plus a mild centre prior. Pure numpy; no OpenCV needed."""
    W, H = 160, 90
    dur = max(0.5, end - start)
    fps = min(4.0, 40.0 / dur) if dur > 10 else 4.0
    cmd = [ffmpeg.ffmpeg_path(), "-hide_banner", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}",
           "-i", path, "-an", "-vf", f"fps={fps:.3f},scale={W}:{H},format=gray", "-f", "rawvideo", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL, timeout=120,
                             creationflags=ffmpeg.NO_WINDOW).stdout
    except Exception:
        return None
    if cancel.is_set():
        raise Cancelled()
    n = len(out) // (W * H)
    if n < 3:
        return None
    frames = np.frombuffer(out[: n * W * H], dtype=np.uint8).reshape(n, H, W).astype(np.float32)
    diff = np.abs(np.diff(frames, axis=0)).mean(axis=(0, 1))       # per column motion
    edges = np.abs(np.diff(frames.mean(axis=0), axis=1)).mean(axis=0)
    edges = np.concatenate([edges, edges[-1:]])
    energy = diff / (diff.max() + 1e-6) + 0.35 * edges / (edges.max() + 1e-6)
    x = np.linspace(0, 1, W)
    energy = energy * (0.55 + 0.45 * np.exp(-((x - 0.5) ** 2) / 0.08))  # centre prior (HUDs at edges)
    if energy.sum() <= 0:
        return None
    c = float((energy * x).sum() / energy.sum())
    return min(max(c, 0.0), 1.0)
