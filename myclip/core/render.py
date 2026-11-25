"""Rendering clips: framing (fit/fill/smart), captions, encode, thumbnail."""
from __future__ import annotations

import os
import tempfile
import threading
from typing import Callable, Optional

from . import ffmpeg
from .captions import build_ass
from .models import CAPTION_STYLES, Clip, JobConfig, MediaInfo, Transcript


def resolve_framing(cfg: JobConfig, info: MediaInfo, W: int, H: int) -> str:
    f = (cfg.framing or "Auto").lower()
    src_ar, dst_ar = info.width / max(info.height, 1), W / max(H, 1)
    if abs(src_ar - dst_ar) < 0.02:
        return "scale"
    if f == "auto":
        # wide gameplay into a tall frame: keep everything visible
        return "fit" if src_ar / dst_ar > 1.6 else "fill"
    return f


def video_filter(mode: str, W: int, H: int, center: float = 0.5) -> str:
    """filter_complex string; output label [v]."""
    if mode == "scale":
        return f"[0:v]scale={W}:{H}:flags=lanczos,setsar=1[v]"
    if mode in ("fill", "smart"):
        c = min(max(center, 0.0), 1.0)
        return (f"[0:v]scale='if(gt(a,{W}/{H}),-2,{W})':'if(gt(a,{W}/{H}),{H},-2)':flags=lanczos,"
                f"crop={W}:{H}:'max(0,min(iw-{W},(iw-{W})*{c:.4f}))':'(ih-{H})/2',setsar=1[v]")
    # fit with blurred background
    bw, bh = max(2, W // 8 * 2 // 2), max(2, H // 8 * 2 // 2)
    return (f"[0:v]split=2[a][b];"
            f"[a]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},"
            f"gblur=sigma=6,eq=brightness=-0.08:saturation=1.1,scale={W}:{H}:flags=bilinear[bg];"
            f"[b]scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v]")


def caption_position(mode: str, info: MediaInfo, W: int, H: int, default: float) -> float:
    """In fit mode, put captions just under the video band if there's room."""
    if mode != "fit":
        return default
    fg_h = W * info.height / max(info.width, 1)
    if fg_h >= H * 0.9:
        return default
    band_bottom = (H + fg_h) / 2
    gap = H - band_bottom
    return max(0.05, min(default, (gap * 0.55) / H))


def encoder_args(nvenc: bool, H: int) -> list[str]:
    if nvenc:
        return ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", "20",
                "-b:v", "0", "-maxrate", f"{max(8, H // 120)}M", "-bufsize", f"{max(16, H // 60)}M",
                "-profile:v", "high", "-spatial-aq", "1"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "19", "-profile:v", "high"]


def render(src: str, clip: Clip, out_path: str, cfg: JobConfig, info: MediaInfo, tr: Transcript,
           cancel: threading.Event, center: float = 0.5,
           on_progress: Optional[Callable[[float], None]] = None) -> bool:
    """Returns False if captions had to be dropped (e.g. no libass/fonts)."""
    try:
        _render(src, clip, out_path, cfg, info, tr, cancel, center, on_progress, captions=True)
        return True
    except RuntimeError:
        if cfg.captions == "Off" or tr.empty:
            raise
    cleanup_partial(out_path)
    _render(src, clip, out_path, cfg, info, tr, cancel, center, on_progress, captions=False)
    return False


def _render(src: str, clip: Clip, out_path: str, cfg: JobConfig, info: MediaInfo, tr: Transcript,
            cancel: threading.Event, center: float, on_progress, captions: bool) -> None:
    W, H = cfg.target_size(info)
    mode = resolve_framing(cfg, info, W, H)
    fc = video_filter(mode, W, H, center)
    label = "[v]"
    with tempfile.TemporaryDirectory(prefix="myclip_") as tmp:
        style = CAPTION_STYLES.get(cfg.captions)
        if captions and style and not tr.empty:
            words = tr.words_between(clip.start, clip.end)
            if words:
                pos = caption_position(mode, info, W, H, style.position)
                with open(os.path.join(tmp, "c.ass"), "w", encoding="utf-8") as f:
                    f.write(build_ass(words, clip.start, clip.end, W, H, style, pos))
                fc += ";[v]subtitles=c.ass[vc]"   # relative path: no Windows escaping issues
                label = "[vc]"
        fade_out = max(0.0, clip.duration - 0.25)
        def args(nv: bool) -> list[str]:
            a = ["-y", "-nostats", "-progress", "pipe:2", "-v", "error",
                 "-ss", f"{clip.start:.3f}", "-t", f"{clip.duration:.3f}", "-i", os.path.abspath(src),
                 "-filter_complex", fc, "-map", label]
            if info.has_audio:
                a += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                      "-af", f"afade=t=in:d=0.08,afade=t=out:st={fade_out:.2f}:d=0.25,"
                             "loudnorm=I=-14:TP=-1.5:LRA=11"]
            if info.fps > 61:
                a += ["-r", "60"]
            a += encoder_args(nv, H) + ["-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                        os.path.abspath(out_path)]
            return a

        def line(l: str) -> None:
            t = ffmpeg.time_progress(l)
            if t is not None and on_progress:
                on_progress(min(t / max(clip.duration, 0.1), 1.0))

        if ffmpeg.nvenc_ok():
            try:
                ffmpeg.run(args(True), cancel, cwd=tmp, on_line=line)
                return
            except RuntimeError:
                pass
        ffmpeg.run(args(False), cancel, cwd=tmp, on_line=line)


def thumbnail(video: str, out_jpg: str, at: float, cancel: threading.Event) -> None:
    ffmpeg.run(["-y", "-v", "error", "-ss", f"{max(0.0, at):.2f}", "-i", os.path.abspath(video),
                "-frames:v", "1", "-vf", "scale=360:-2", "-q:v", "4", os.path.abspath(out_jpg)], cancel)


def cleanup_partial(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
