"""Voice cleanup: isolates the voice of each chosen clip with Demucs (in a separate
process) so the caption pass hears speech, not gunshots and music."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from typing import Callable, Optional

from . import ffmpeg, helpers
from .models import Cancelled


def available() -> bool:
    return helpers.installed("voice")


def vocals_for(media: str, windows: list[tuple[float, float]], cancel: threading.Event,
               on_progress: Optional[Callable[[float, str], None]] = None, device: str = "auto"
               ) -> tuple[list[Optional[str]], str]:
    """windows: (start, duration) in video time. Returns (vocal wav per window or None, temp dir to delete)."""
    tmp = tempfile.mkdtemp(prefix="myclip_voice_")
    items = []
    for i, (a, d) in enumerate(windows):
        src = os.path.join(tmp, f"in_{i}.wav")
        ffmpeg.run(["-y", "-v", "error", "-ss", f"{max(0.0, a):.3f}", "-t", f"{max(0.5, d):.3f}", "-i",
                    os.path.abspath(media), "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", src], cancel)
        items.append({"in": src, "out": os.path.join(tmp, f"voice_{i}.wav")})
    jobs = os.path.join(tmp, "jobs.json")
    with open(jobs, "w", encoding="utf-8") as f:
        json.dump({"device": device, "items": items}, f)
    n = len(items)

    def line(l: str) -> None:
        if not l.startswith("{") or not on_progress:
            return
        try:
            d = json.loads(l)
        except Exception:
            return
        if "done" in d:
            on_progress(d["done"] / max(n, 1), f"Cleaning up the voice, clip {d['done']} of {n}")

    try:
        helpers.run_logged([helpers.python_exe(), "-m", "myclip.helpers_cli", "separate", jobs], cancel, line,
                           cwd=str(helpers.app_root()), timeout=600 + 120 * n)
    except Cancelled:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    out = [it["out"] if os.path.isfile(it["out"]) and os.path.getsize(it["out"]) > 1000 else None for it in items]
    return out, tmp
