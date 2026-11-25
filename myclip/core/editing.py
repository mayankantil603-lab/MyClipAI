"""Editing a finished clip: fix caption words, change the hook or the facecam box and
re-render just that clip; trim a copy; compare Whisper with the Hindi model."""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import fields
from pathlib import Path
from typing import Callable, Optional

from . import ffmpeg, render
from .models import Cancelled, Clip, JobConfig, Segment, Transcript, Word

Progress = Optional[Callable[[float, str], None]]


def load_job(folder: str) -> dict:
    return json.loads((Path(folder) / "clips.json").read_text(encoding="utf-8"))


def save_job(folder: str, d: dict) -> None:
    m = Path(folder) / "clips.json"
    d["updated"] = time.strftime("%Y-%m-%d %H:%M")
    tmp = m.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, m)


def find_clip(d: dict, file_name: str) -> int:
    for i, c in enumerate(d.get("clips", [])):
        if c.get("file") == file_name:
            return i
    raise RuntimeError("This clip isn't in its job's clips.json any more.")


def media_of(d: dict) -> str:
    for p in (d.get("media"), d.get("source")):
        if p and os.path.isfile(p):
            return p
    return ""


def can_rerender(d: dict, c: dict) -> str:
    """'' if the clip can be re-rendered, otherwise the reason."""
    if not media_of(d):
        return "The original video isn't on this PC any more, so this clip can't be re-made."
    if c.get("source") == "compilation":
        return "A compilation is re-made from its clips: make it again from the results."
    if "words" not in c:
        return "This clip was made with an older version of MyClip, so its captions can't be edited."
    return ""


def config_of(d: dict) -> JobConfig:
    s = dict(d.get("settings") or {})
    known = {f.name for f in fields(JobConfig)}
    cfg = JobConfig(source=d.get("source", ""), **{k: v for k, v in s.items() if k in known and k != "source"})
    return cfg


def words_to_transcript(words: list, lang: str = "") -> Transcript:
    ws = [Word(float(a), float(b), str(t)) for a, b, t in words if str(t).strip()]
    if not ws:
        return Transcript([], lang)
    return Transcript([Segment(ws[0].start, ws[-1].end, " ".join(w.text for w in ws), ws)], lang)


def rerender(folder: str, file_name: str, cancel: threading.Event, words: Optional[list] = None,
             hook: Optional[str] = None, cam: Optional[list] = None, progress: Progress = None) -> dict:
    """Re-render one clip in place with new caption words / hook / facecam box. Returns its new record."""
    d = load_job(folder)
    i = find_clip(d, file_name)
    c = d["clips"][i]
    why = can_rerender(d, c)
    if why:
        raise RuntimeError(why)
    media = media_of(d)
    cfg = config_of(d)
    info = ffmpeg.probe(media)
    plan = render.RenderPlan.from_dict(c.get("plan"))
    if words is not None:
        c["words"] = [[round(float(a), 3), round(float(b), 3), str(t)] for a, b, t in words if str(t).strip()]
    if hook is not None:
        plan.hook = hook.strip()
        c["hook"] = plan.hook
    if cam is not None:
        plan.cam = [round(float(x), 4) for x in cam[:4]]
        if cfg.framing.lower() != "facecam":
            cfg.framing = "Facecam"
            d.setdefault("settings", {})["framing"] = "Facecam"
    tr = words_to_transcript(c.get("words", []), (d.get("settings") or {}).get("language", ""))
    if plan.bleep_mode in ("beep", "mute") and not tr.empty:
        from .bleep import Matcher, apply
        tr, plan.bleeps = apply(tr, float(c["start"]), float(c["end"]), Matcher())
        c["words"] = [[w.start, w.end, w.text] for w in tr.words()]
    clip = Clip(float(c["start"]), float(c["end"]))
    out = Path(folder) / file_name
    tmp = out.with_name(out.stem + ".new.mp4")
    try:
        render.render(media, clip, str(tmp), cfg, info, tr, cancel, plan.center,
                      (lambda f: progress(f * 0.9, "Rendering")) if progress else None, plan)
        for k in range(20):                           # the player may still hold the old file for a moment
            try:
                os.replace(tmp, out)
                break
            except PermissionError:
                if cancel.wait(0.25):
                    raise Cancelled()
        else:
            raise RuntimeError("The clip is open in another program, so it couldn't be replaced.")
    finally:
        render.cleanup_partial(str(tmp))
    c["plan"] = plan.to_dict()
    if progress:
        progress(0.93, "Thumbnail")
    from .engine import finish_thumbs
    clip.file = str(out)
    W, H = cfg.target_size(info)
    idx = i + 1
    try:
        idx = int(file_name.split(" ", 1)[0])
    except ValueError:
        pass
    finish_thumbs(clip, Path(folder), idx, W, H, cancel, cfg.sharp_thumbs)
    if clip.thumb:
        c["thumb"] = os.path.relpath(clip.thumb, folder).replace("\\", "/")
    c["hover"] = list(clip.hover)
    c["edited"] = time.strftime("%Y-%m-%d %H:%M")
    if cam is not None and d.get("source"):
        from .engine import save_facecam
        save_facecam(d["source"], plan.cam)
    save_job(folder, d)
    return c


def trim_copy(folder: str, file_name: str, a: float, b: float, cancel: threading.Event) -> dict:
    """'<name> (trimmed).mp4' from a..b (clip time), added to the job. Returns the new record."""
    d = load_job(folder)
    i = find_clip(d, file_name)
    c = d["clips"][i]
    src = Path(folder) / file_name
    stem = src.stem
    dest = src.with_name(f"{stem} (trimmed).mp4")
    k = 2
    while dest.exists():
        dest = src.with_name(f"{stem} (trimmed {k}).mp4")
        k += 1
    render.trim(str(src), a, b, str(dest), cancel)
    rec = dict(c)
    base = float(c.get("start", 0))
    rec.update({"file": dest.name, "start": round(base + a, 3), "end": round(base + b, 3), "trimmed_from": file_name,
                "hover": [], "uploads": []})
    rec["words"] = [w for w in c.get("words", []) if base + a <= float(w[0]) < base + b]
    rec["thumb"] = ""
    try:
        th = Path(folder) / ".thumbs" / (dest.stem + ".jpg")
        th.parent.mkdir(exist_ok=True)
        render.thumbnail(str(dest), str(th), min((b - a) * 0.35, 3.0), cancel)
        rec["thumb"] = f".thumbs/{th.name}"
    except Cancelled:
        raise
    except Exception:
        pass
    d["clips"].insert(i + 1, rec)
    save_job(folder, d)
    return rec


def compare_hindi(folder: str, file_name: str, cancel: threading.Event, progress: Progress = None) -> dict:
    """The clip heard by Whisper and by IndicWhisper Hindi, both shown in Hinglish."""
    from . import hindi, transcribe
    from .codeswitch import restore_english
    from .translit import roman_text
    d = load_job(folder)
    c = d["clips"][find_clip(d, file_name)]
    media = media_of(d)
    if not media:
        raise RuntimeError("The original video isn't on this PC any more.")
    cfg = config_of(d)
    a, b = float(c["start"]), float(c["end"])
    if progress:
        progress(0.05, "Reading the audio")
    audio, a0 = transcribe.clip_audio(media, a, b, cancel)
    if progress:
        progress(0.15, "Whisper is listening")
    w_segs, _, _ = transcribe.transcribe_clip(media, a, b, cfg.whisper_model, cfg.device, "hi", cancel, None, audio, a0)
    transcribe.release()
    if progress:
        progress(0.55, "Hindi model is listening")
    h_segs, dev = hindi.transcribe_audio(audio, a0, cancel)
    hindi.release()
    if progress:
        progress(0.8, "Keeping English words")
    try:
        h_segs, _ = restore_english(h_segs, transcribe.english_segments(audio, a0, cfg.whisper_model, cfg.device, cancel))
    except Cancelled:
        raise
    except Exception:
        pass
    transcribe.release()

    def text(segs) -> str:
        return roman_text(" ".join(" ".join(w.text for w in s.words) if s.words else s.text for s in segs)).strip()

    def words(segs) -> list:
        return [[round(w.start, 3), round(w.end, 3), w.text] for s in segs for w in s.words if a - 0.5 <= w.start < b]

    return {"whisper": text(w_segs), "hindi": text(h_segs), "whisper_words": words(w_segs),
            "hindi_words": words(h_segs)}


def roman_words(words: list) -> list:
    from .translit import roman_word
    return [[a, b, roman_word(t)] for a, b, t in words]


def delete_clip(folder: str, file_name: str) -> None:
    d = load_job(folder)
    i = find_clip(d, file_name)
    c = d["clips"].pop(i)
    for rel in [c.get("thumb", "")] + list(c.get("hover") or []):
        if rel:
            try:
                os.remove(Path(folder) / rel)
            except OSError:
                pass
    try:
        os.remove(Path(folder) / file_name)
    except OSError:
        pass
    save_job(folder, d)
