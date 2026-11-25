"""Learn from your views: links uploaded clips to the signals that picked them, checks
their views after 48 hours, and gently re-weights how future moments are scored.

Safety first: nothing changes until 8 uploads have views, each part's weight can move at
most 40 %, and the change grows slowly with more data (12 videos give about half of it)."""
from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Optional

from .paths import config_dir

MIN_VIDEOS = 8
WAIT = 48 * 3600
RECHECK = 24 * 3600
MAX_CHANGE = 0.40
_LOCK = threading.Lock()


def _file() -> Path:
    return config_dir() / "learning.json"


def load() -> dict:
    try:
        return json.loads(_file().read_text(encoding="utf-8"))
    except Exception:
        return {"channels": {}}


def save(d: dict) -> None:
    p = _file()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def register(channel_id: str, video_id: str, parts: dict, title: str = "", when: Optional[float] = None) -> None:
    """Remember an uploaded clip and the score parts that chose it."""
    if not (channel_id and video_id and parts):
        return
    with _LOCK:
        d = load()
        ch = d.setdefault("channels", {}).setdefault(channel_id, {"videos": {}, "weights": {}, "n": 0})
        ch["videos"][video_id] = {"parts": {k: float(v) for k, v in parts.items()}, "title": title[:100],
                                  "uploaded": when or time.time(), "views": None, "checked": 0}
        save(d)


def due(channel_id: str, now: Optional[float] = None) -> list[str]:
    now = now or time.time()
    ch = load().get("channels", {}).get(channel_id, {})
    return [vid for vid, v in ch.get("videos", {}).items()
            if now - v.get("uploaded", now) >= WAIT and now - v.get("checked", 0) >= RECHECK]


def update_views(channel_id: str, views: dict[str, int], now: Optional[float] = None) -> dict:
    """Store fresh view counts, refit, return the channel's weights."""
    now = now or time.time()
    with _LOCK:
        d = load()
        ch = d.setdefault("channels", {}).setdefault(channel_id, {"videos": {}, "weights": {}, "n": 0})
        for vid, n in views.items():
            if vid in ch["videos"]:
                ch["videos"][vid]["views"] = int(n)
                ch["videos"][vid]["checked"] = now
        w, n = fit(list(ch["videos"].values()))
        ch["weights"], ch["n"] = w, n
        save(d)
        return w


def _corr(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx < 1e-9 or sy < 1e-9:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def fit(videos: list[dict]) -> tuple[dict[str, float], int]:
    """Per-part multipliers from how each part relates to log(views).
    Rank-based (Spearman), so one viral video can't swing everything."""
    rows = [v for v in videos if v.get("views") is not None and v.get("parts")]
    n = len(rows)
    if n < MIN_VIDEOS:
        return {}, n

    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    y = ranks([math.log1p(max(0, r["views"])) for r in rows])
    keys = sorted({k for r in rows for k in r["parts"]})
    trust = n / (n + 12.0)
    out: dict[str, float] = {}
    for k in keys:
        xs = [float(r["parts"].get(k, 0.0)) for r in rows]
        if sum(1 for x in xs if x != 0.0) < MIN_VIDEOS // 2:
            continue                                 # too few videos used this signal
        rho = _corr(ranks(xs), y)
        out[k] = round(1.0 + max(-MAX_CHANGE, min(MAX_CHANGE, 0.8 * rho)) * trust, 3)
    return out, n


def weights(channel_id: str) -> dict[str, float]:
    return dict(load().get("channels", {}).get(channel_id, {}).get("weights", {}))


def summary(channel_id: str) -> tuple[int, int, dict]:
    """(uploads remembered, uploads with views, weights)."""
    ch = load().get("channels", {}).get(channel_id, {})
    vids = ch.get("videos", {})
    return len(vids), sum(1 for v in vids.values() if v.get("views") is not None), dict(ch.get("weights", {}))


def reset(channel_id: str) -> None:
    with _LOCK:
        d = load()
        ch = d.get("channels", {}).get(channel_id)
        if ch:
            ch["weights"], ch["n"] = {}, 0
            for v in ch.get("videos", {}).values():
                v["views"], v["checked"] = None, 0
            save(d)


def apply_for(channel_id: str) -> int:
    """Load the channel's weights into the scorer. Returns how many videos they came from."""
    from . import moments
    ch = load().get("channels", {}).get(channel_id, {})
    moments.LEARNED.clear()
    moments.LEARNED.update(ch.get("weights", {}))
    return int(ch.get("n", 0)) if moments.LEARNED else 0


def find_parts(clip_file: str) -> tuple[dict, str]:
    """Score parts and title of a rendered clip, from its job's clips.json."""
    p = Path(clip_file)
    m = p.parent / "clips.json"
    try:
        d = json.loads(m.read_text(encoding="utf-8"))
    except Exception:
        return {}, ""
    for c in d.get("clips", []):
        if c.get("file") == p.name or os.path.normcase(str(p.parent / c.get("file", ""))) == os.path.normcase(str(p)):
            return dict(c.get("parts") or {}), str(c.get("title", ""))
    return {}, ""


def mark_uploaded(clip_file: str, channel_id: str, video_id: str) -> None:
    """Write the YouTube id into clips.json and remember the clip for learning."""
    p = Path(clip_file)
    m = p.parent / "clips.json"
    try:
        d = json.loads(m.read_text(encoding="utf-8"))
        for c in d.get("clips", []):
            if c.get("file") == p.name:
                c.setdefault("uploads", []).append({"channel": channel_id, "video_id": video_id, "at": time.time()})
        tmp = m.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, m)
    except Exception:
        pass
    parts, title = find_parts(clip_file)
    register(channel_id, video_id, parts, title)
