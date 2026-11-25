"""Finding and scoring moments.

Signals are turned into per-time curves on a common grid, combined with
robust normalisation, then candidate windows are built around peaks so the
payoff lands late in the clip, starts on a sentence and never cuts a word."""
from __future__ import annotations

import math
import re
from typing import Optional

import numpy as np

from .models import Clip, Signals, Transcript
from .translit import has_devanagari, roman_text

HYPE = [
    "oh my god", "omg", "no way", "let's go", "lets go", "let's goo", "clutch", "insane", "what the",
    "are you kidding", "holy", "wtf", "oh no", "headshot", "ace", "did you see", "crazy",
    "unbelievable", "bro what", "how did", "i can't believe", "noo", "yesss", "run run",
    "bhai", "kya baat", "arre", "abe", "ye kya", "pagal", "mast", "kya hua", "ho gaya", "maar diya",
    "bhag", "bachao", "kya scene",
]
_HYPE = re.compile(r"\b(" + "|".join(re.escape(h) for h in HYPE) + r")\b", re.I)
_SENT_END = re.compile(r"[.!?…]$")


def hype_count(text: str) -> int:
    if has_devanagari(text):
        text = roman_text(text)          # "भाई ये क्या" counts like "bhai ye kya"
    return len(_HYPE.findall(text or ""))


def robust_z(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    if mad < 1e-6:
        sd = float(x.std())
        mad = sd if sd > 1e-6 else 1.0
    return ((x - med) / mad).astype(np.float32)


def smooth(x: np.ndarray, seconds: float, hop: float) -> np.ndarray:
    k = max(1, int(round(seconds / hop)))
    if k <= 1 or x.size < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same").astype(np.float32)


def events_to_curve(times: list[float], n: int, hop: float, spread: float = 3.0) -> np.ndarray:
    c = np.zeros(n, dtype=np.float32)
    for t in times:
        i = int(t / hop)
        if 0 <= i < n:
            c[i] += 1
    return smooth(c, spread, hop)


def heatmap_curve(hm: list[tuple[float, float, float]], n: int, hop: float) -> np.ndarray:
    c = np.zeros(n, dtype=np.float32)
    for a, b, v in hm:
        c[max(0, int(a / hop)): min(n, int(math.ceil(b / hop)))] = v
    return c


def speech_curves(tr: Transcript, n: int, hop: float) -> tuple[np.ndarray, np.ndarray]:
    density = np.zeros(n, dtype=np.float32)
    hype = np.zeros(n, dtype=np.float32)
    for w in tr.words():
        i = int(((w.start + w.end) / 2) / hop)
        if 0 <= i < n:
            density[i] += 1
    for s in tr.segments:
        h = hype_count(s.text)
        if h:
            i = int(s.end / hop)
            if 0 <= i < n:
                hype[i] += h
    return smooth(density, 4.0, hop), smooth(hype, 3.0, hop)


WEIGHTS = {"loud": 1.0, "heat": 1.6, "chat": 1.3, "comments": 1.1, "hype": 0.9, "speech": 0.35, "scene": 0.3}


def fuse(duration: float, sig: Signals, tr: Transcript) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Combined interest curve (z-ish units) and the per-signal curves."""
    hop = sig.hop
    n = max(1, int(math.ceil(duration / hop)))
    curves: dict[str, np.ndarray] = {}
    if sig.loudness is not None and len(sig.loudness):
        lz = np.zeros(n, dtype=np.float32)
        l = np.asarray(sig.loudness, dtype=np.float32)[:n]
        lz[: len(l)] = robust_z(smooth(l, 1.5, hop))
        curves["loud"] = np.clip(lz, -3, 6)
    if sig.heatmap:
        curves["heat"] = np.clip(robust_z(heatmap_curve(sig.heatmap, n, hop)), -3, 6)
    if sig.chat_times:
        curves["chat"] = np.clip(robust_z(events_to_curve(sig.chat_times, n, hop, 8.0)), -3, 6)
    if sig.comment_times:
        curves["comments"] = np.clip(events_to_curve(sig.comment_times, n, hop, 6.0) /
                                     (max(1.0, len(set(sig.comment_times))) ** 0.5), 0, 6)
    if not tr.empty:
        dens, hype = speech_curves(tr, n, hop)
        curves["speech"] = np.clip(robust_z(dens), -3, 3)
        if hype.any():
            curves["hype"] = np.clip(hype * 1.5, 0, 6)
    if sig.scene_cuts:
        curves["scene"] = np.clip(robust_z(events_to_curve(sig.scene_cuts, n, hop, 4.0)), -2, 3)
    total = np.zeros(n, dtype=np.float32)
    wsum = 0.0
    for k, c in curves.items():
        w = WEIGHTS.get(k, 0.5)
        total += w * c
        wsum += w
    if wsum:
        total /= wsum ** 0.5
    return total, curves


def peaks(curve: np.ndarray, hop: float, min_gap: float, limit: int, thresh: float) -> list[int]:
    order = np.argsort(-curve)
    gap = max(1, int(min_gap / hop))
    kept: list[int] = []
    for i in order:
        if curve[i] < thresh or len(kept) >= limit:
            break
        if all(abs(int(i) - k) >= gap for k in kept):
            kept.append(int(i))
    return kept


# ---------- boundaries ----------

def sentence_starts(tr: Transcript) -> list[float]:
    out, prev_end, prev_text = [], -10.0, ""
    for s in tr.segments:
        if s.start - prev_end > 0.6 or _SENT_END.search(prev_text.strip()) or not prev_text:
            out.append(s.words[0].start if s.words else s.start)
        prev_end, prev_text = s.end, s.text
    return out


def sentence_ends(tr: Transcript) -> list[float]:
    out = []
    segs = tr.segments
    for i, s in enumerate(segs):
        nxt = segs[i + 1].start if i + 1 < len(segs) else 1e9
        if _SENT_END.search(s.text.strip()) or nxt - s.end > 0.6:
            out.append(s.words[-1].end if s.words else s.end)
    return out


def snap_start(t: float, tr: Transcript, cuts: list[float], lo: float, hi: float) -> float:
    opts = [x for x in sentence_starts(tr) if lo <= x <= hi]
    if opts:
        return min(opts, key=lambda x: abs(x - t)) - 0.12
    opts = [x for x in cuts if lo <= x <= hi]
    if opts:
        return min(opts, key=lambda x: abs(x - t)) + 0.05
    return _not_mid_word(t, tr, prefer="before")


def snap_end(t: float, tr: Transcript, cuts: list[float], lo: float, hi: float) -> float:
    opts = [x for x in sentence_ends(tr) if lo <= x <= hi]
    if opts:
        return min(opts, key=lambda x: abs(x - t)) + 0.35
    opts = [x for x in cuts if lo <= x <= hi]
    if opts:
        return min(opts, key=lambda x: abs(x - t)) - 0.05
    return _not_mid_word(t, tr, prefer="after")


def _not_mid_word(t: float, tr: Transcript, prefer: str) -> float:
    for w in tr.words():
        if w.start < t < w.end:
            return (w.start - 0.05) if prefer == "before" else (w.end + 0.15)
    return t


def make_window(anchor: float, length: float, duration: float, tr: Transcript, cuts: list[float],
                min_len: float, max_len: float) -> tuple[float, float]:
    start = anchor - 0.68 * length              # payoff about two thirds in
    start = min(max(0.0, start), max(0.0, duration - length))
    end = start + length
    s = snap_start(start, tr, cuts, start - 5.0, start + 3.0)
    e = snap_end(end, tr, cuts, end - 4.0, end + 5.0)
    s = max(0.0, s)
    e = min(duration, e)
    if e - s > max_len:
        e = snap_end(s + max_len - 2, tr, cuts, s + max_len - 6, s + max_len) if not tr.empty else s + max_len
        e = min(e, s + max_len)
    if e - s < min_len:
        e = min(duration, s + min_len)
        s = max(0.0, min(s, e - min_len))
    return round(s, 2), round(e, 2)


# ---------- candidates + scoring ----------

def candidates(duration: float, sig: Signals, tr: Transcript, min_len: float, max_len: float,
               want: int) -> list[Clip]:
    min_len = min(min_len, duration)
    max_len = max(min(max_len, duration), min_len)
    curve, parts = fuse(duration, sig, tr)
    hop = sig.hop
    idx = peaks(curve, hop, min_gap=max(8.0, min_len * 0.6), limit=max(want * 6, 40), thresh=0.4)
    anchors = [i * hop for i in idx]
    if len(anchors) < want * 2:   # flat videos: fall back to an even scan
        step = max(min_len * 0.5, duration / max(want * 4, 1))
        anchors += list(np.arange(min_len * 0.6, duration, step))
    lengths = sorted({round(x, 1) for x in (min_len, (min_len + max_len) / 2, max_len, 30.0, 45.0)
                      if min_len <= x <= max_len})
    out: list[Clip] = []
    seen: set[tuple[int, int]] = set()
    for a in anchors:
        for L in lengths:
            s, e = make_window(a, L, duration, tr, sig.scene_cuts, min_len, max_len)
            if e - s < min_len * 0.9:
                continue
            k = (int(s), int(e))
            if k in seen:
                continue
            seen.add(k)
            c = Clip(s, e)
            c.features = features(c, curve, parts, hop, tr)
            c.signal_score = signal_score(c)
            c.score = c.signal_score
            out.append(c)
    out.sort(key=lambda c: c.score, reverse=True)
    return out[:200]


def features(c: Clip, curve: np.ndarray, parts: dict[str, np.ndarray], hop: float, tr: Transcript) -> dict:
    a, b = int(c.start / hop), max(int(c.end / hop), int(c.start / hop) + 1)
    win = curve[a:b]
    f: dict[str, float] = {}
    if win.size:
        f["peak"] = float(win.max())
        f["mean"] = float(win.mean())
        f["peak_pos"] = float(win.argmax() / max(len(win) - 1, 1))
        f["hook"] = float(win[: max(1, int(3 / hop))].mean())
        f["tail"] = float(win[-max(1, int(2 / hop)):].mean())
    for k in ("heat", "chat", "comments"):
        if k in parts:
            w = parts[k][a:b]
            f[k] = float(w.max()) if w.size else 0.0
    c.text = tr.text_between(c.start, c.end)
    words = len(c.text.split())
    f["wps"] = words / max(c.duration, 1.0)
    f["hype"] = float(hype_count(c.text))
    f["starts_clean"] = 1.0 if any(abs(c.start + 0.12 - x) < 0.3 for x in sentence_starts(tr)) else 0.0
    return f


def _sig(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def signal_score(c: Clip) -> float:
    f = c.features
    s = 26 * _sig(f.get("peak", 0) - 1.2)
    s += 12 * _sig(f.get("mean", 0))
    s += 10 * _sig(f.get("hook", 0) + 0.3)
    s += 10 * max(0.0, 1 - abs(f.get("peak_pos", 0.5) - 0.68) / 0.68)
    s += 8 * min(f.get("hype", 0), 3) / 3
    s += 8 * min(f.get("wps", 0) / 2.2, 1.0)
    s += 6 * f.get("starts_clean", 0)
    s += 10 * _sig(f.get("heat", -3) - 0.5) if "heat" in f else 0
    s += 6 * _sig(f.get("chat", -3) - 0.5) if "chat" in f else 0
    s += 4 * min(f.get("comments", 0), 2) / 2 if "comments" in f else 0
    if f.get("tail", 0) < -1.0:            # ends in dead air
        s -= 5
    if not ("heat" in f or "chat" in f or "comments" in f):
        s *= 100 / 80                       # rescale when no external signals
    return round(max(0.0, min(s, 100.0)), 1)


def overlap(a: Clip, b: Clip) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    return inter / max(min(a.duration, b.duration), 1e-6)


def _similar_text(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def select(clips: list[Clip], count: int, max_overlap: float = 0.2) -> list[Clip]:
    picked: list[Clip] = []
    for c in sorted(clips, key=lambda c: c.score, reverse=True):
        if any(overlap(c, p) > max_overlap for p in picked):
            continue
        if c.text and any(_similar_text(c.text, p.text) > 0.7 for p in picked):
            continue
        picked.append(c)
        if len(picked) >= count:
            break
    return picked


def merge_llm(signal_clips: list[Clip], llm_clips: list[Clip], tr: Transcript, sig: Signals,
              duration: float, min_len: float, max_len: float) -> list[Clip]:
    """LLM-proposed moments are scored with the same signals; where an LLM
    clip overlaps a signal clip, they reinforce each other."""
    curve, parts = fuse(duration, sig, tr)
    merged = list(signal_clips)
    for lc in llm_clips:
        s = max(0.0, lc.start)
        e = min(duration, lc.end)
        if e - s < min_len:
            e = min(duration, s + min_len)
        if e - s > max_len:
            e = s + max_len
        s = snap_start(s, tr, sig.scene_cuts, s - 2, s + 2)
        e = snap_end(e, tr, sig.scene_cuts, e - 2, e + 3)
        lc.start, lc.end = round(max(0.0, s), 2), round(min(duration, e), 2)
        lc.features = features(lc, curve, parts, sig.hop, tr)
        lc.signal_score = signal_score(lc)
        lc.source = "llm"
        best = max(merged, key=lambda m: overlap(m, lc), default=None)
        if best is not None and overlap(best, lc) > 0.6 and best.llm_score is None:
            best.llm_score = lc.llm_score
            best.title, best.hook, best.reason = lc.title, lc.hook, lc.reason
            best.description, best.hashtags = lc.description, lc.hashtags
            best.source = "both"
        merged.append(lc)
    for c in merged:
        if c.llm_score is not None:
            c.score = round(0.55 * c.llm_score + 0.45 * c.signal_score + (4 if c.source == "both" else 0), 1)
        else:
            c.score = c.signal_score
        c.score = min(c.score, 100.0)
    return sorted(merged, key=lambda c: c.score, reverse=True)


def fallback_title(c: Clip) -> str:
    text = re.sub(r"\s+", " ", c.text or "").strip()
    if text:
        # prefer the most "hype" sentence in the clip
        parts = re.split(r"(?<=[.!?])\s+", text)
        best = max(parts, key=lambda p: (hype_count(p), -abs(len(p.split()) - 7)))
        words = best.split()[:8]
        t = " ".join(words).strip(" ,.;:-")
        if len(t) >= 8:
            return t[0].upper() + t[1:]
    m, s = divmod(int(c.start), 60)
    return f"Moment at {m}:{s:02d}"
