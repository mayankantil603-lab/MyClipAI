"""Rendering clips: framing (fit / fill / follow face / facecam), captions, hook text,
bleeps, encode, thumbnail."""
from __future__ import annotations

import os
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import ffmpeg
from .captions import build_ass
from .models import CAPTION_STYLES, Clip, JobConfig, MediaInfo, Transcript


@dataclass
class RenderPlan:
    """Everything needed to render (or re-render) one clip. Stored in clips.json."""
    center: float = 0.5
    path: list = field(default_factory=list)      # [(t, c)] crop path that follows a face
    cam: list = field(default_factory=list)       # facecam box (x, y, w, h), 0..1 of the source
    hook: str = ""
    bleeps: list = field(default_factory=list)    # [(a, b)] in clip time
    bleep_mode: str = "off"

    def to_dict(self) -> dict:
        return {"center": self.center, "path": [list(p) for p in self.path], "cam": list(self.cam),
                "hook": self.hook, "bleeps": [list(b) for b in self.bleeps], "bleep_mode": self.bleep_mode}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "RenderPlan":
        d = d or {}
        return cls(float(d.get("center", 0.5)), [tuple(p) for p in d.get("path") or []], list(d.get("cam") or []),
                   str(d.get("hook", "")), [tuple(b) for b in d.get("bleeps") or []], str(d.get("bleep_mode", "off")))


def resolve_framing(cfg: JobConfig, info: MediaInfo, W: int, H: int) -> str:
    f = (cfg.framing or "Auto").lower()
    src_ar, dst_ar = info.width / max(info.height, 1), W / max(H, 1)
    if f == "facecam":
        return "facecam" if src_ar > dst_ar * 1.2 else ("fill" if abs(src_ar - dst_ar) >= 0.02 else "scale")
    if abs(src_ar - dst_ar) < 0.02:
        return "scale"
    if f == "auto":
        # wide gameplay into a tall frame: keep everything visible
        return "fit" if src_ar / dst_ar > 1.6 else "fill"
    return f


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def facecam_split(W: int, H: int, cam: list, info: MediaInfo) -> tuple[int, int]:
    """Heights of (facecam band, gameplay band). The band keeps the camera's own shape, within 25-45 %."""
    cw = max(0.05, float(cam[2])) * info.width
    chh = max(0.05, float(cam[3])) * info.height
    top = W * chh / max(cw, 1)
    top = _even(min(max(top, H * 0.25), H * 0.45))
    return top, H - top


def video_filter(mode: str, W: int, H: int, center: float = 0.5, path: Optional[list] = None,
                 cam: Optional[list] = None, info: Optional[MediaInfo] = None) -> str:
    """filter_complex string; output label [v]."""
    if mode == "scale":
        return f"[0:v]scale={W}:{H}:flags=lanczos,setsar=1[v]"
    fill_scale = f"scale='if(gt(a,{W}/{H}),-2,{W})':'if(gt(a,{W}/{H}),{H},-2)':flags=lanczos"
    if mode in ("fill", "smart"):
        if path and len(path) >= 2:
            from .faces import x_expr
            x = x_expr(path, W)
        else:
            c = min(max(center, 0.0), 1.0)
            x = f"max(0,min(iw-{W},(iw-{W})*{c:.4f}))"
        return f"[0:v]{fill_scale},crop={W}:{H}:'{x}':'(ih-{H})/2',setsar=1[v]"
    if mode == "facecam" and cam and info is not None:
        top, bot = facecam_split(W, H, cam, info)
        x, y, w, h = (max(0.0, min(1.0, float(v))) for v in cam[:4])
        c = min(max(center, 0.0), 1.0)
        gs = f"scale='if(gt(a,{W}/{bot}),-2,{W})':'if(gt(a,{W}/{bot}),{bot},-2)':flags=lanczos"
        return (f"[0:v]split=2[c][g];"
                f"[c]crop=iw*{w:.4f}:ih*{h:.4f}:iw*{x:.4f}:ih*{y:.4f},"
                f"scale={W}:{top}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{top}[top];"
                f"[g]{gs},crop={W}:{bot}:'max(0,min(iw-{W},(iw-{W})*{c:.4f}))':'(ih-{bot})/2'[bot];"
                f"[top][bot]vstack=inputs=2,setsar=1[v]")
    # fit with blurred background
    bw, bh = max(2, W // 8 * 2 // 2), max(2, H // 8 * 2 // 2)
    return (f"[0:v]split=2[a][b];"
            f"[a]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},"
            f"gblur=sigma=6,eq=brightness=-0.08:saturation=1.1,scale={W}:{H}:flags=bilinear[bg];"
            f"[b]scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v]")


def caption_position(mode: str, info: MediaInfo, W: int, H: int, default: float, cam: Optional[list] = None) -> float:
    """Distance of the captions from the bottom, as a fraction of the height."""
    if mode == "facecam" and cam:
        top, bot = facecam_split(W, H, cam, info)
        return round(max(0.1, (bot / H) - 0.10), 4)          # just under the seam, on the gameplay
    if mode != "fit":
        return default
    fg_h = W * info.height / max(info.width, 1)
    if fg_h >= H * 0.9:
        return default
    band_bottom = (H + fg_h) / 2
    gap = H - band_bottom
    return max(0.05, min(default, (gap * 0.55) / H))


def hook_top(mode: str, info: MediaInfo, W: int, H: int, cam: Optional[list] = None) -> int:
    """Pixels from the top for the hook text."""
    if mode == "facecam" and cam:
        top, _ = facecam_split(W, H, cam, info)
        return int(top + H * 0.03)
    if mode == "fit":
        fg_h = W * info.height / max(info.width, 1)
        band_top = (H - fg_h) / 2
        if band_top > H * 0.14:
            return int(max(H * 0.05, band_top * 0.35))
    return int(H * 0.09)


def encoder_args(nvenc: bool, H: int) -> list[str]:
    if nvenc:
        return ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", "20",
                "-b:v", "0", "-maxrate", f"{max(8, H // 120)}M", "-bufsize", f"{max(16, H // 60)}M",
                "-profile:v", "high", "-spatial-aq", "1"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "19", "-profile:v", "high"]


def render(src: str, clip: Clip, out_path: str, cfg: JobConfig, info: MediaInfo, tr: Transcript,
           cancel: threading.Event, center: float = 0.5,
           on_progress: Optional[Callable[[float], None]] = None, plan: Optional[RenderPlan] = None) -> bool:
    """Returns False if captions (and hook) had to be dropped (e.g. no libass/fonts)."""
    plan = plan or RenderPlan(center=center)
    try:
        _render(src, clip, out_path, cfg, info, tr, cancel, plan, on_progress, captions=True)
        return True
    except RuntimeError:
        if (cfg.captions == "Off" or tr.empty) and not plan.hook:
            raise
    cleanup_partial(out_path)
    _render(src, clip, out_path, cfg, info, tr, cancel, plan, on_progress, captions=False)
    return False


def _render(src: str, clip: Clip, out_path: str, cfg: JobConfig, info: MediaInfo, tr: Transcript,
            cancel: threading.Event, plan: RenderPlan, on_progress, captions: bool) -> None:
    W, H = cfg.target_size(info)
    mode = resolve_framing(cfg, info, W, H)
    if mode == "facecam" and not plan.cam:
        mode = "fit"                                 # no facecam found: keep everything visible
    fc = video_filter(mode, W, H, plan.center, plan.path, plan.cam, info)
    label = "[v]"
    with tempfile.TemporaryDirectory(prefix="myclip_") as tmp:
        style = CAPTION_STYLES.get(cfg.captions) or CAPTION_STYLES["Bold"]
        if captions:
            words = tr.words_between(clip.start, clip.end) if (cfg.captions != "Off" and not tr.empty) else []
            hook = plan.hook.strip()
            if words or hook:
                pos = caption_position(mode, info, W, H, style.position, plan.cam)
                hk = (hook, hook_top(mode, info, W, H, plan.cam), min(2.6, max(1.2, clip.duration * 0.25))) \
                    if hook else None
                ass = build_ass(words, clip.start, clip.end, W, H, style, pos, hk)
                with open(os.path.join(tmp, "c.ass"), "w", encoding="utf-8") as f:
                    f.write(ass)
                fc += ";[v]subtitles=c.ass[vc]"   # relative path: no Windows escaping issues
                label = "[vc]"
        fade_out = max(0.0, clip.duration - 0.25)
        post = f"afade=t=in:d=0.08,afade=t=out:st={fade_out:.2f}:d=0.25,loudnorm=I=-14:TP=-1.5:LRA=11"
        use_bleep = info.has_audio and plan.bleeps and plan.bleep_mode in ("beep", "mute")
        if use_bleep:
            from .bleep import audio_filter
            fc += ";" + audio_filter(plan.bleeps, plan.bleep_mode, clip.duration, post)

        def args(nv: bool) -> list[str]:
            a = ["-y", "-nostats", "-progress", "pipe:2", "-v", "error",
                 "-ss", f"{clip.start:.3f}", "-t", f"{clip.duration:.3f}", "-i", os.path.abspath(src),
                 "-filter_complex", fc, "-map", label]
            if info.has_audio:
                if use_bleep:
                    a += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
                else:
                    a += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-af", post]
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


def trim(src_clip: str, a: float, b: float, out_path: str, cancel: threading.Event) -> None:
    """Re-cut an already rendered clip (keeps its captions), accurate to the frame."""
    info = ffmpeg.probe(src_clip)
    a = max(0.0, min(a, info.duration - 0.2))
    b = max(a + 0.2, min(b, info.duration))
    d = b - a
    fade = max(0.0, d - 0.2)
    base = ["-y", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{d:.3f}", "-i", os.path.abspath(src_clip)]
    tail = (["-c:a", "aac", "-b:a", "192k", "-af", f"afade=t=in:d=0.05,afade=t=out:st={fade:.2f}:d=0.2"]
            if info.has_audio else ["-an"]) + ["-pix_fmt", "yuv420p", "-movflags", "+faststart", os.path.abspath(out_path)]
    if ffmpeg.nvenc_ok():
        try:
            ffmpeg.run(base + encoder_args(True, info.height) + tail, cancel)
            return
        except RuntimeError:
            pass
    ffmpeg.run(base + encoder_args(False, info.height) + tail, cancel)


def thumbnail(video: str, out_jpg: str, at: float, cancel: threading.Event) -> None:
    ffmpeg.run(["-y", "-v", "error", "-ss", f"{max(0.0, at):.2f}", "-i", os.path.abspath(video),
                "-frames:v", "1", "-vf", "scale=360:-2", "-q:v", "4", os.path.abspath(out_jpg)], cancel)


def cleanup_partial(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
