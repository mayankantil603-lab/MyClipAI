"""The job: source -> signals -> transcript -> moments -> AI -> render.

Runs on a worker thread. Reports through `emit(stage, fraction, message)`.
Stages have fixed weights so the overall bar moves smoothly."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

from . import ffmpeg, llm, moments, render, signals, sources, subtitles, transcribe
from .models import Cancelled, Clip, JobConfig, Signals, Transcript
from .paths import config_dir, shorts_dir, unique_dir
from .store import ConnectionStore
from .translit import has_devanagari, roman_text

STAGES = [  # name, weight
    ("Import", 10), ("Analyze", 11), ("Transcribe", 30), ("Find moments", 5),
    ("AI review", 10), ("Captions", 9), ("Render", 25),
]
_W = {n: w for n, w in STAGES}
_OFF: dict[str, int] = {}
_acc = 0
for _n, _w in STAGES:
    _OFF[_n] = _acc
    _acc += _w

Emit = Callable[[str, float, str], None]


def overall(stage: str, frac: float) -> int:
    return int(_OFF.get(stage, 0) + _W.get(stage, 0) * max(0.0, min(frac, 1.0)))


class JobResult:
    def __init__(self) -> None:
        self.folder = ""
        self.clips: list[Clip] = []
        self.used: list[str] = []
        self.title = ""
        self.source = ""
        self.warnings: list[str] = []


def run(cfg: JobConfig, store: Optional[ConnectionStore], emit: Emit, cancel: threading.Event,
        api_key: str = "") -> JobResult:
    res = JobResult()
    t0 = time.time()

    def check() -> None:
        if cancel.is_set():
            raise Cancelled()

    def warn(msg: str) -> None:
        res.warnings.append(msg)
        emit("", -1, msg)

    # ---------- import ----------
    src = cfg.source.strip().strip('"')
    sig = Signals()
    info_dict: dict = {}
    if sources.is_url(src):
        emit("Import", 0, "Reading link")
        info_dict = sources.fetch_info(src, cancel, want_comments=cfg.use_external)
        check()
        res.title = info_dict.get("title") or "Video"
        media = sources.download(src, info_dict, lambda f, m: emit("Import", f, m), cancel)
    else:
        if not os.path.isfile(src):
            raise RuntimeError("File not found. Check the path or drop the file again.")
        media = src
        res.title = Path(src).stem
    res.source = src
    check()
    info = ffmpeg.probe(media)
    if info.duration < 3:
        raise RuntimeError("This video is shorter than 3 seconds.")
    emit("Import", 1, f"{info.width}x{info.height}, {info.fps:.0f} fps, {_fmt(info.duration)}")

    # ---------- analyze ----------
    if info_dict:
        ext = sources.signals_from_info(info_dict, info.duration)
        ext.hop = sig.hop
        sig = ext
        if cfg.use_external:
            with tempfile.TemporaryDirectory(prefix="myclip_chat_") as tmp:
                emit("Analyze", 0.02, "Checking chat replay")
                chat = sources.fetch_chat_times(src, info_dict, cancel, tmp)
            if chat:
                sig.chat_times = chat
                sig.used.append(f"Chat replay ({len(chat)} messages)")
        else:
            sig.heatmap, sig.comment_times = [], []
            sig.used = [u for u in sig.used if u == "Chapters"]
    check()
    if info.has_audio:
        emit("Analyze", 0.05, "Measuring audio energy")
        sig.loudness = signals.loudness(media, sig.hop, cancel,
                                        lambda f: emit("Analyze", 0.05 + f * 0.45, "Measuring audio energy"),
                                        info.duration)
        sig.used.append("Audio energy")
    if cfg.use_scenes:
        emit("Analyze", 0.5, "Detecting scene changes")
        try:
            sig.scene_cuts = signals.scene_cuts(media, cancel,
                                                on_progress=lambda f: emit("Analyze", 0.5 + f * 0.5,
                                                                           "Detecting scene changes"),
                                                duration=info.duration)
            sig.used.append(f"Scene changes ({len(sig.scene_cuts)})")
        except Cancelled:
            raise
        except Exception as exc:
            warn(f"Scene detection skipped: {exc}")
    emit("Analyze", 1, "Signals ready")
    check()

    # ---------- transcribe ----------
    tr = Transcript()
    if not info.has_audio:
        emit("Transcribe", 1, "Skipped (no audio track)")
    elif not transcribe.available():
        warn("Whisper is not installed. Clips will be chosen from audio and video only.")
        emit("Transcribe", 1, "Skipped (Whisper not installed)")
    else:
        conn0 = store.get(cfg.connection_id) if (store and cfg.connection_id) else None
        if conn0 and conn0.provider == "Ollama":
            llm.unload_ollama(conn0.base_url)        # give Whisper the whole GPU
        tr, used = transcribe.transcribe(media, info.duration, cfg.whisper_model, cfg.device, cfg.language,
                                         config_dir() / "transcripts",
                                         lambda f, m: emit("Transcribe", f, m), cancel)
        res.used.append(used)
        from .gpu import gpu_failure
        if gpu_failure():
            warn(f"Transcribed on the CPU because the GPU failed. {gpu_failure()}")
        if store and cfg.connection_id:
            transcribe.release()                     # give the AI model the whole GPU
        emit("Transcribe", 1, f"{len(tr.segments)} lines · {transcribe.lang_name(tr.language)}")
    check()

    # ---------- moments ----------
    emit("Find moments", 0.1, "Scoring moments")
    cands = moments.candidates(info.duration, sig, tr, cfg.min_len, cfg.max_len, cfg.clip_count)
    if not cands:
        raise RuntimeError("Couldn't find any usable moments in this video.")
    emit("Find moments", 1, f"{len(cands)} candidates")

    # ---------- AI ----------
    conn = store.get(cfg.connection_id) if (store and cfg.connection_id) else None
    if conn and not tr.empty:
        key = api_key or (store.api_key(conn) if store else "")
        model = cfg.llm_model or conn.model
        if conn.needs_key and not key:
            warn(f"No API key for {conn.name}. Skipping AI review.")
        elif not model:
            warn(f"No model selected for {conn.name}. Skipping AI review.")
        else:
            try:
                emit("AI review", 0.05, f"{model} is reading the transcript")
                style = subtitles.title_style(cfg.caption_lang, tr)
                proposed = llm.propose(conn, key, model, tr, sig, cfg.clip_count, cfg.min_len, cfg.max_len,
                                       lambda m: emit("AI review", 0.3, m), cancel, style=style)
                cands = moments.merge_llm(cands, proposed, tr, sig, info.duration, cfg.min_len, cfg.max_len)
                shortlist = moments.select(cands, min(len(cands), cfg.clip_count + 4))
                emit("AI review", 0.7, "Final review of the shortlist")
                llm.review(conn, key, model, shortlist, sig, lambda m: emit("AI review", 0.8, m), cancel, style=style)
                cands = sorted(shortlist, key=lambda c: c.score, reverse=True) + \
                    [c for c in cands if c not in shortlist]
                res.used.append(f"{conn.name}: {model}")
            except Cancelled:
                raise
            except Exception as exc:
                warn(f"AI review failed, using signal scores only: {exc}")
    ai_done = bool(conn) and any(u.startswith(f"{conn.name}:") for u in res.used)
    emit("AI review", 1, "Done" if ai_done else ("Skipped (no AI selected)" if not conn else
                                                  "Skipped (no transcript)" if tr.empty else "Skipped"))
    check()

    final = moments.select(cands, cfg.clip_count)
    final.sort(key=lambda c: c.start)

    # ---------- captions ----------
    want_caps = cfg.captions != "Off" and not tr.empty and bool(final)
    cap_src: list[Transcript] = [tr] * len(final)
    if not want_caps:
        emit("Captions", 1, "Skipped (captions off)" if cfg.captions == "Off" else "Skipped (no speech found)")
    elif cfg.caption_quality == "best" and transcribe.available():
        if conn and conn.provider == "Ollama":
            llm.unload_ollama(conn.base_url)            # the whole GPU for the accurate model
        prompt = res.title[:200] if (tr.language or "").startswith("en") and res.title.isascii() else None
        desc, failed, last_err = "", 0, ""
        for i, c in enumerate(final):
            check()
            emit("Captions", 0.7 * i / len(final), f"Listening again to clip {i + 1} of {len(final)} for accurate captions")
            try:
                segs, desc, _ = transcribe.transcribe_clip(media, c.start, c.end, cfg.whisper_model, cfg.device,
                                                           tr.language or cfg.language, cancel, prompt)
                if segs:
                    cap_src[i] = Transcript(segs, tr.language)
            except Cancelled:
                raise
            except Exception as exc:
                failed, last_err = failed + 1, str(exc)
        transcribe.release()
        if desc:
            res.used.append(f"Captions: {desc}")
        if failed:
            warn(f"Accurate captions failed for {failed} clip{'s' if failed > 1 else ''}, so the first transcript "
                 f"was used ({last_err[:100]}).")

    cap_conn, cap_key, cap_model = None, "", ""
    if conn:
        cap_key = api_key or (store.api_key(conn) if store else "")
        cap_model = cfg.llm_model or conn.model
        if cap_model and (cap_key or not conn.needs_key):
            cap_conn = conn
    cap_trs = list(cap_src)
    if want_caps and cfg.caption_lang in ("roman", "english", "auto"):
        emit("Captions", 0.75, "Preparing subtitles")
        notes, useds = [], []
        for i, c in enumerate(final):
            check()
            try:
                out, note, used_sub = subtitles.prepare(cap_src[i], [c], cfg, media, cap_conn, cap_key, cap_model,
                                                        lambda msg, i=i: emit("Captions", 0.75 + 0.25 * i / len(final), msg),
                                                        cancel)
                cap_trs[i] = out[0]
                if note and note not in notes:
                    notes.append(note)
                if used_sub:
                    useds.append(used_sub)
            except Cancelled:
                raise
            except Exception as exc:
                msg = f"Couldn't change the subtitle language for some clips: {str(exc)[:100]}"
                if msg not in notes:
                    notes.append(msg)
        for n_ in notes:
            warn(n_)
        if useds:
            res.used.append(_summarise(useds))
    if want_caps:
        emit("Captions", 1, "Captions ready")
    for c, ctr in zip(final, cap_trs):
        if not c.title:
            probe_clip = Clip(c.start, c.end)
            probe_clip.text = ctr.text_between(c.start, c.end)
            if cfg.caption_lang in ("roman", "auto") and has_devanagari(probe_clip.text):
                probe_clip.text = roman_text(probe_clip.text)
            c.title = moments.fallback_title(probe_clip)

    # ---------- render ----------
    base = Path(cfg.output_dir) if cfg.output_dir else shorts_dir()
    base.mkdir(parents=True, exist_ok=True)
    folder = unique_dir(base, res.title)
    res.folder = str(folder)
    thumbs = folder / ".thumbs"
    thumbs.mkdir(exist_ok=True)
    W, H = cfg.target_size(info)
    n = len(final)
    for i, c in enumerate(final):
        check()
        center = 0.5
        mode = render.resolve_framing(cfg, info, W, H)
        if mode == "smart":
            emit("Render", i / n, f"Tracking action for clip {i + 1}")
            center = signals.motion_center(media, c.start, c.end, cancel) or 0.5
        name = f"{i + 1:02d} - {sources.safe_name(c.title, 50)}.mp4"
        out = folder / name
        msg = f"Clip {i + 1} of {n}"
        try:
            ok = render.render(media, c, str(out), cfg, info, cap_trs[i], cancel, center,
                               lambda f, i=i: emit("Render", (i + f) / n, msg))
            if not ok and "captions_dropped" not in res.warnings:
                warn("Captions couldn't be drawn on this system, so clips were made without them.")
                res.warnings.append("captions_dropped")
        except BaseException:
            render.cleanup_partial(str(out))
            raise
        c.file = str(out)
        try:
            th = thumbs / f"{i + 1:02d}.jpg"
            render.thumbnail(str(out), str(th), min(c.duration * 0.35, 3.0), cancel)
            c.thumb = str(th)
        except Cancelled:
            raise
        except Exception:
            pass
    if conn and conn.provider == "Ollama":
        llm.unload_ollama(conn.base_url)             # free VRAM for rendering / other apps
    res.clips = final
    res.used = sig.used + res.used
    write_manifest(res, cfg, info.duration, time.time() - t0)
    emit("Render", 1, f"Done in {_fmt(time.time() - t0)}")
    return res


def _summarise(useds: list[str]) -> str:
    import re as _re
    total = sum(int(x) for u in useds for x in _re.findall(r"with (\d+) translation", u))
    ai = any("(AI)" in u for u in useds)
    base = _re.sub(r" with \d+ translations?| \(AI\)", "", useds[0])
    return base + (f" with {total} translation{'s' if total != 1 else ''}" if total else "") + (" (AI)" if ai else "")


def write_manifest(res: JobResult, cfg: JobConfig, duration: float, took: float) -> None:
    data = {
        "app": "MyClip AI", "version": 1, "created": time.strftime("%Y-%m-%d %H:%M"),
        "title": res.title, "source": res.source, "duration": duration, "took": round(took, 1),
        "used": res.used, "warnings": [w for w in res.warnings if w != "captions_dropped"],
        "settings": {k: v for k, v in asdict(cfg).items() if k not in ("source",)},
        "clips": [{k: getattr(c, k) for k in ("start", "end", "score", "signal_score", "llm_score", "title",
                                               "hook", "reason", "description", "hashtags", "source")}
                  | {"file": os.path.basename(c.file), "thumb": os.path.relpath(c.thumb, res.folder) if c.thumb else ""}
                  for c in res.clips],
    }
    p = Path(res.folder) / "clips.json"
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_library(roots: Optional[list] = None) -> list[dict]:
    """All jobs from the default folder plus any custom output folders used."""
    folders = [shorts_dir()] + [Path(r) for r in (roots or []) if r]
    seen, items = set(), []
    for root in folders:
        try:
            manifests = list(Path(root).glob("*/clips.json"))
        except OSError:
            continue
        for m in manifests:
            key = os.path.normcase(str(m.resolve()))
            if key in seen:
                continue
            seen.add(key)
            try:
                d = json.loads(m.read_text(encoding="utf-8"))
                d["folder"] = str(m.parent)
                d["_mtime"] = m.stat().st_mtime
                items.append(d)
            except Exception:
                continue
    items.sort(key=lambda d: d["_mtime"], reverse=True)
    return items


def matches(job: dict, query: str) -> bool:
    q = query.strip().lower()
    if not q:
        return True
    hay = [job.get("title", ""), job.get("source", ""), job.get("created", "")]
    for c in job.get("clips", []):
        hay += [c.get("title", ""), c.get("description", ""), " ".join(c.get("hashtags") or [])]
    text = " ".join(hay).lower()
    return all(w in text for w in q.split())


def delete_job(folder: str) -> None:
    """Delete a job folder, but only if it really is a MyClip job folder."""
    p = Path(folder)
    if not (p / "clips.json").is_file():
        raise RuntimeError("Not a MyClip folder; nothing was deleted.")
    shutil.rmtree(p)


def _fmt(t: float) -> str:
    t = int(t)
    return f"{t // 3600}h {t % 3600 // 60:02d}m" if t >= 3600 else f"{t // 60}:{t % 60:02d}"
