"""Getting the video in: local files and public URLs (yt-dlp), plus the
external signals a URL can give us: YouTube "Most replayed" heatmap,
chapters, title/description/tags, live-chat replay and timestamped comments."""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Callable, Optional

from .models import Cancelled, Signals
from .paths import downloads_dir, safe_name

MAX_BYTES = 8 * 1024 ** 3
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv", ".ts", ".wmv"}
_URL = re.compile(r"^https?://", re.I)

Progress = Callable[[float, str], None]


def is_url(text: str) -> bool:
    return bool(_URL.match((text or "").strip()))


def is_video_file(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXT and os.path.isfile(path)


def _ydl_base(cancel: threading.Event) -> dict:
    from .ffmpeg import ffmpeg_path

    def hook(d: dict) -> None:
        if cancel.is_set():
            raise Cancelled()

    return {"quiet": True, "no_warnings": True, "noplaylist": True, "ffmpeg_location": ffmpeg_path(),
            "progress_hooks": [hook], "socket_timeout": 30, "retries": 5, "fragment_retries": 5}


def fetch_info(url: str, cancel: threading.Event, want_comments: bool = False) -> dict:
    import yt_dlp
    opts = _ydl_base(cancel) | {"skip_download": True}
    if want_comments:
        opts |= {"getcomments": True,
                 "extractor_args": {"youtube": {"max_comments": ["400", "all", "0", "0"],
                                                "comment_sort": ["top"]}}}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if info.get("_type") == "playlist":
        entries = [e for e in info.get("entries") or [] if e]
        if not entries:
            raise RuntimeError("That link is a playlist with no videos.")
        info = entries[0]
    if info.get("is_live"):
        raise RuntimeError("This stream is live right now. Use the Live page to clip it as it happens.")
    return info


def download(url: str, info: dict, on_progress: Progress, cancel: threading.Event) -> str:
    import yt_dlp
    size = info.get("filesize") or info.get("filesize_approx") or 0
    if size and size > MAX_BYTES:
        raise RuntimeError(f"Video is {size / 1024**3:.1f} GB; the limit is 8 GB.")
    out_dir = downloads_dir()
    stem = safe_name(info.get("title") or "video", 80) + f" [{info.get('id', 'x')}]"
    existing = next((p for p in out_dir.glob(glob_escape(stem) + ".*") if p.suffix.lower() in VIDEO_EXT), None)
    if existing and existing.stat().st_size > 0:
        on_progress(1.0, "Already downloaded.")
        return str(existing)

    def hook(d: dict) -> None:
        if cancel.is_set():
            raise Cancelled()
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            if total > MAX_BYTES:
                raise RuntimeError("Video is larger than the 8 GB limit.")
            speed = d.get("speed") or 0
            msg = f"Downloading {done / 1024**2:.0f} MB" + (f" of {total / 1024**2:.0f} MB" if total else "")
            if speed:
                msg += f" at {speed / 1024**2:.1f} MB/s"
            on_progress(done / total if total else 0.0, msg)

    opts = _ydl_base(cancel) | {
        "outtmpl": str(out_dir / (stem.replace("%", "%%") + ".%(ext)s")),
        "format": "bv*[height<=2160]+ba/b[height<=2160]/bv*+ba/b",
        "merge_output_format": "mp4",
        "max_filesize": MAX_BYTES,
        "progress_hooks": [hook],
        "windowsfilenames": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.process_ie_result(dict(info), download=True)
    except Cancelled:
        raise
    except Exception as exc:
        if cancel.is_set() or "Cancelled" in repr(exc):
            raise Cancelled()
        raise RuntimeError(f"Download failed: {_clean_err(exc)}") from None
    found = sorted((p for p in out_dir.glob(glob_escape(stem) + ".*") if p.suffix.lower() in VIDEO_EXT),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not found:
        raise RuntimeError("Download finished but the file was not found.")
    return str(found[0])


def glob_escape(s: str) -> str:
    return re.sub(r"([\[\]*?])", r"[\1]", s)


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\[[0-9;]{1,8}m")


def _clean_err(exc) -> str:
    return friendly_error(str(exc))


def friendly_error(msg: str) -> str:
    """yt-dlp errors without colour codes, and the common ones explained."""
    m = _ANSI.sub("", msg or "").replace("ERROR:", "").strip()
    m = re.sub(r"^\[[\w:-]+\]\s*(?:[\w-]+:\s*)?", "", m)     # "[youtube] abc123: " / "[generic] "
    low = m.lower()
    if "requested format is not available" in low or "no video formats found" in low:
        return ("The site didn't offer a stream MyClip can record. The downloader may be out of date: "
                "close MyClip and start run.bat again to update it.")
    if "sign in to confirm" in low or "not a bot" in low:
        return "YouTube is asking for a sign-in check from this network. Try again later or on another network."
    if "private video" in low or "members-only" in low or "join this channel" in low:
        return "This video is private or members-only."
    if "is offline" in low or "not currently live" in low or "will begin in" in low or "premieres in" in low:
        return "This stream isn't live right now."
    if "unsupported url" in low:
        return "This site isn't supported."
    if "http error 404" in low:
        return "Not found. Check the link."
    return m[:300]


# ---------------- external signals ----------------

_TS = re.compile(r"(?<![\d:])(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?![\d:])")


def parse_timestamps(text: str, duration: float) -> list[float]:
    out = []
    for m in _TS.finditer(text or ""):
        h = int(m[1] or 0)
        mm, ss = int(m[2]), int(m[3])
        if ss >= 60 or (m[1] and mm >= 60):
            continue
        t = h * 3600 + mm * 60 + ss
        if 0 <= t <= duration:
            out.append(float(t))
    return out


def signals_from_info(info: dict, duration: float) -> Signals:
    s = Signals()
    s.title = info.get("title") or ""
    s.description = (info.get("description") or "")[:3000]
    s.tags = list(info.get("tags") or [])[:30]
    hm = info.get("heatmap") or []
    if hm:
        s.heatmap = [(float(h.get("start_time", 0)), float(h.get("end_time", 0)), float(h.get("value", 0)))
                     for h in hm if h.get("end_time") is not None]
        if s.heatmap:
            s.used.append("YouTube most replayed")
    ch = info.get("chapters") or []
    if ch:
        s.chapters = [(float(c.get("start_time", 0)), float(c.get("end_time", 0)), c.get("title", ""))
                      for c in ch]
        s.used.append("Chapters")
    comments = info.get("comments") or []
    times: list[float] = []
    for c in comments:
        likes = int(c.get("like_count") or 0)
        for t in parse_timestamps(c.get("text", ""), duration):
            times += [t] * (1 + min(likes // 25, 5))   # liked comments count more
    if times:
        s.comment_times = times
        s.used.append(f"Timestamped comments ({len(times)})")
    return s


def fetch_chat_times(url: str, info: dict, cancel: threading.Event, tmp_dir: str) -> list[float]:
    """Live-chat replay (YouTube streams, Twitch VODs) via yt-dlp subtitles."""
    subs = info.get("subtitles") or {}
    key = next((k for k in ("live_chat", "rechat") if k in subs), None)
    if not key:
        return []
    import yt_dlp
    opts = _ydl_base(cancel) | {"skip_download": True, "writesubtitles": True,
                                "subtitleslangs": [key], "outtmpl": os.path.join(tmp_dir, "chat.%(ext)s")}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.process_ie_result(dict(info), download=True)
    except Exception:
        return []
    times: list[float] = []
    for p in Path(tmp_dir).glob("chat*"):
        try:
            times += parse_chat_file(p.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            pass
    return times


def parse_chat_file(text: str) -> list[float]:
    times: list[float] = []
    stripped = text.lstrip()
    if stripped.startswith("["):  # twitch rechat json array
        try:
            for c in json.loads(stripped):
                t = c.get("content_offset_seconds")
                if t is not None:
                    times.append(float(t))
            return times
        except Exception:
            pass
    for line in text.splitlines():  # youtube live_chat json lines
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        ms = (d.get("replayChatItemAction") or {}).get("videoOffsetTimeMsec")
        if ms is None:
            ms = d.get("videoOffsetTimeMsec")
        if ms is not None:
            times.append(int(ms) / 1000.0)
        elif "content_offset_seconds" in d:
            times.append(float(d["content_offset_seconds"]))
    return times
