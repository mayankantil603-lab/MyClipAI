"""Best-of compilation: the strongest part of the top clips, back to back, 60 s or less."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from . import ffmpeg
from .models import Cancelled
from .render import encoder_args


def plan(clips: list[dict], n: int = 5, total: float = 60.0, min_part: float = 4.0) -> list[dict]:
    """Choose clips (best score first) and a window inside each around its peak.
    Returns [{"file", "a", "b"}] in time order of the original video."""
    ranked = sorted([c for c in clips if c.get("file")], key=lambda c: -(c.get("score") or 0))[:max(1, n)]
    if not ranked:
        return []
    per = max(min_part, total / len(ranked))
    out = []
    for c in ranked:
        dur = max(0.5, float(c.get("end", 0)) - float(c.get("start", 0)))
        length = min(dur, per)
        peak = float((c.get("features") or {}).get("peak_pos", 0.68)) * dur
        a = min(max(0.0, peak - 0.6 * length), max(0.0, dur - length))
        out.append({"file": c["file"], "a": round(a, 2), "b": round(a + length, 2), "start": float(c.get("start", 0)),
                    "title": c.get("title", "")})
    used = sum(o["b"] - o["a"] for o in out)
    if used > total:                                 # trim evenly from each part's start
        k = total / used
        for o in out:
            keep = (o["b"] - o["a"]) * k
            o["a"] = round(o["b"] - keep, 2)
    out.sort(key=lambda o: o["start"])
    return out


def xfade_graph(n: int, durs: list[float], W: int, H: int, fade: float = 0.25, audio: bool = True) -> tuple[str, str, str]:
    """filter_complex chaining n inputs with quick crossfades. Returns (graph, video label, audio label)."""
    parts = []
    for i in range(n):
        parts.append(f"[{i}:v]scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,"
                     f"setsar=1,fps=30,format=yuv420p,settb=AVTB[v{i}]")
        if audio:
            parts.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo[a{i}]")
    if n == 1:
        return ";".join(parts), "[v0]", "[a0]" if audio else ""
    v, a = "[v0]", "[a0]"
    offset = 0.0
    for i in range(1, n):
        offset += durs[i - 1] - fade
        nv, na = f"[vx{i}]", f"[ax{i}]"
        parts.append(f"{v}[v{i}]xfade=transition=fade:duration={fade}:offset={offset:.3f}{nv}")
        if audio:
            parts.append(f"{a}[a{i}]acrossfade=d={fade}:c1=tri:c2=tri{na}")
        v, a = nv, na
    return ";".join(parts), v, a if audio else ""


def build(job_folder: str, clips: list[dict], cancel: threading.Event, n: int = 5, total: float = 60.0,
          on_progress: Optional[Callable[[float, str], None]] = None) -> dict:
    """Renders 'Best of.mp4' into the job folder and adds it to clips.json. Returns the new clip record."""
    folder = Path(job_folder)
    parts = plan(clips, n, total)
    if len(parts) < 2:
        raise RuntimeError("A compilation needs at least 2 clips.")
    first = ffmpeg.probe(str(folder / parts[0]["file"]))
    W, H = first.width, first.height
    audio = first.has_audio
    with tempfile.TemporaryDirectory(prefix="myclip_best_") as tmp:
        cuts = []
        for i, p in enumerate(parts):
            if cancel.is_set():
                raise Cancelled()
            if on_progress:
                on_progress(i / (len(parts) + 1), f"Cutting part {i + 1} of {len(parts)}")
            out = os.path.join(tmp, f"p{i}.mp4")
            ffmpeg.run(["-y", "-v", "error", "-ss", f"{p['a']:.3f}", "-t", f"{p['b'] - p['a']:.3f}", "-i",
                        str(folder / p["file"]), "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
                        *(["-c:a", "aac", "-b:a", "192k"] if audio else ["-an"]), out], cancel)
            cuts.append(out)
        durs = [ffmpeg.probe(c).duration for c in cuts]
        graph, vl, al = xfade_graph(len(cuts), durs, W, H, audio=audio)
        dest = folder / "Best of.mp4"
        k = 2
        while dest.exists():
            dest = folder / f"Best of ({k}).mp4"
            k += 1
        if on_progress:
            on_progress(len(parts) / (len(parts) + 1), "Joining the best moments")
        args = ["-y", "-v", "error"]
        for c in cuts:
            args += ["-i", c]
        args += ["-filter_complex", graph, "-map", vl]
        if audio:
            args += ["-map", al, "-c:a", "aac", "-b:a", "192k"]
        tail = ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)]
        try:
            ffmpeg.run(args + (encoder_args(True, H) if ffmpeg.nvenc_ok() else encoder_args(False, H)) + tail, cancel)
        except RuntimeError:
            ffmpeg.run(args + encoder_args(False, H) + tail, cancel)
    length = sum(durs) - 0.25 * (len(durs) - 1)
    rec = {"file": dest.name, "thumb": "", "start": 0.0, "end": round(length, 2), "score": None,
           "title": "Best of", "reason": f"The best {len(parts)} moments in one clip",
           "description": " · ".join(p["title"] for p in parts if p.get("title"))[:300], "hashtags": [],
           "source": "compilation", "parts_of": [p["file"] for p in parts]}
    try:
        thumbs = folder / ".thumbs"
        thumbs.mkdir(exist_ok=True)
        th = thumbs / (dest.stem + ".jpg")
        ffmpeg.run(["-y", "-v", "error", "-ss", "1.0", "-i", str(dest), "-frames:v", "1", "-vf", "scale=720:-2",
                    "-q:v", "3", str(th)], cancel)
        rec["thumb"] = f".thumbs/{th.name}"
    except Cancelled:
        raise
    except Exception:
        pass
    add_to_manifest(folder, rec)
    return rec


def add_to_manifest(folder: Path, rec: dict) -> None:
    m = Path(folder) / "clips.json"
    try:
        d = json.loads(m.read_text(encoding="utf-8"))
    except Exception:
        return
    d.setdefault("clips", []).append(rec)
    d["updated"] = time.strftime("%Y-%m-%d %H:%M")
    tmp = m.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, m)
