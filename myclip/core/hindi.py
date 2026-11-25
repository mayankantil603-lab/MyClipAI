"""Hindi captions with AI4Bharat's IndicWhisper (Whisper medium fine-tuned on Hindi).

Following the model card: feed speech pieces of 8 s or less split at pauses, greedy
decoding without timestamp tokens, and drop the short "धन्यवाद"/"सही" tails it adds
after a piece and a last word stretched over silence. English words come out in
Hindi letters, so the regular Whisper English pass (codeswitch) restores them."""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from . import helpers
from .codeswitch import restore_english
from .models import Cancelled, Segment, Word, run_cancellable

SR = 16000
TAILS = {"धन्यवाद", "सही", "शुक्रिया", "ठीक", "हाँ", "हां", "जी"}
_MODEL: dict = {}
_LOCK = threading.Lock()


def available() -> bool:
    return helpers.installed("hindi")


def speech_pieces(audio: np.ndarray, sr: int = SR, max_len: float = 8.0, min_len: float = 0.35,
                  pad: float = 0.15) -> list[tuple[float, float]]:
    """Split at pauses into pieces of at most max_len seconds (energy based, no extra model)."""
    hop = int(sr * 0.03)
    n = len(audio) // hop
    if n < 3:
        return [(0.0, len(audio) / sr)] if len(audio) > sr * min_len else []
    fr = audio[: n * hop].reshape(n, hop).astype(np.float32)
    rms = np.sqrt((fr * fr).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    floor = float(np.percentile(db, 5))
    top = float(np.percentile(db, 95))
    if top < -45.0:                                  # nothing louder than background hiss
        return []
    thr = max(min(floor + 0.35 * (top - floor), top - 14.0), -55.0)   # works when a clip is all talking
    on = db >= thr
    k = 5                                            # close short gaps (150 ms)
    sm = np.convolve(on.astype(np.float32), np.ones(k), mode="same") > 0
    pieces, i = [], 0
    while i < n:
        if not sm[i]:
            i += 1
            continue
        j = i
        while j < n and sm[j]:
            j += 1
        pieces.append((i, j))
        i = j
    out: list[tuple[float, float]] = []
    t = hop / sr
    for a, b in pieces:
        while (b - a) * t > max_len:                 # long run: cut at the quietest frame in the last 40 %
            lo = a + int(max_len * 0.6 / t)
            hi = a + int(max_len / t)
            cut = lo + int(np.argmin(db[lo:hi])) if hi > lo else hi
            out.append((a * t, cut * t))
            a = cut
        out.append((a * t, b * t))
    res = []
    for a, b in out:
        if b - a < min_len:
            continue
        res.append((max(0.0, a - pad), min(len(audio) / sr, b + pad)))
    merged: list[tuple[float, float]] = []
    for a, b in res:                                 # join tiny neighbours while staying under max_len
        if merged and a - merged[-1][1] < 0.25 and b - merged[-1][0] <= max_len:
            merged[-1] = (merged[-1][0], b)
        else:
            if merged and a < merged[-1][1]:         # padding must not make pieces overlap (double words)
                mid = (a + merged[-1][1]) / 2
                merged[-1] = (merged[-1][0], mid)
                a = mid
            merged.append((a, b))
    return [(round(a, 3), round(b, 3)) for a, b in merged if b - a >= min_len]


def drop_tails(segs: list[Segment]) -> list[Segment]:
    out: list[Segment] = []
    for s in segs:
        words = list(s.words)
        if words and len(words) > 1:                 # a tail word stretched over silence
            last = words[-1]
            span = last.end - last.start
            if (span > 1.5 and last.text.strip(" .,!?।") in TAILS) or span > 2.5:
                words = words[:-1]
        text = " ".join(w.text for w in words) if words else s.text
        core = text.strip(" .,!?।")
        short = (words[-1].end - words[0].start) if words else (s.end - s.start)
        if out and core in TAILS and short < 0.5 and abs(s.start - out[-1].end) < 0.35:
            continue
        if not core:
            continue
        out.append(Segment(words[0].start if words else s.start, words[-1].end if words else s.end, text, words))
    return out


def _load(cancel: Optional[threading.Event] = None):
    from faster_whisper import WhisperModel
    from .gpu import cuda_info, gpu_failure
    ok, _, _ = cuda_info()
    dev, ct = ("cuda", "int8_float16") if (ok and not gpu_failure()) else ("cpu", "int8")
    with _LOCK:
        if (dev, ct) in _MODEL:
            return _MODEL[(dev, ct)], dev
        _MODEL.clear()
    path = str(helpers.hindi_dir())
    try:
        m = run_cancellable(lambda: WhisperModel(path, device=dev, compute_type=ct), cancel)
    except Cancelled:
        raise
    except Exception:
        if dev != "cuda":
            raise
        dev, ct = "cpu", "int8"
        m = run_cancellable(lambda: WhisperModel(path, device=dev, compute_type=ct), cancel)
    with _LOCK:
        _MODEL[(dev, ct)] = m
    return m, dev


def release() -> None:
    with _LOCK:
        _MODEL.clear()


def transcribe_audio(audio: np.ndarray, offset: float, cancel: threading.Event) -> tuple[list[Segment], str]:
    """Hindi segments (Devanagari) in video time. Returns (segments, device)."""
    m, dev = _load(cancel)
    segs: list[Segment] = []
    for a, b in speech_pieces(audio):
        if cancel.is_set():
            raise Cancelled()
        piece = audio[int(a * SR): int(b * SR)]
        kw = dict(language="hi", beam_size=1, without_timestamps=True, word_timestamps=True, vad_filter=False,
                  condition_on_previous_text=False)
        it, _ = run_cancellable(lambda: m.transcribe(piece, **kw), cancel)
        for s in run_cancellable(lambda: list(it), cancel):
            words = [Word(offset + a + float(w.start), offset + a + float(w.end), str(w.word).strip())
                     for w in (s.words or []) if str(w.word).strip()]
            text = (s.text or "").strip()
            if not text:
                continue
            st = words[0].start if words else offset + a + float(s.start)
            en = words[-1].end if words else offset + a + float(s.end)
            segs.append(Segment(st, en, text, words))
    return drop_tails(segs), dev


def transcribe_clip(media: str, a: float, b: float, cancel: threading.Event, english_pass=None
                    ) -> tuple[list[Segment], str]:
    """IndicWhisper for one clip, then English words restored with the regular Whisper English pass.
    english_pass(audio, offset) -> list[Segment] is supplied by transcribe.py."""
    from .transcribe import decode
    a0 = max(0.0, a - 0.5)
    audio = decode(media, a0, (b - a) + 1.0, cancel, clean=True)
    segs, dev = transcribe_audio(audio, a0, cancel)
    desc = f"IndicWhisper Hindi ({dev.upper()})"
    if english_pass is not None and segs:
        release()
        try:
            en = english_pass(audio, a0)
            segs, n = restore_english(segs, en)
            if n:
                desc += f", {n} English word{'s' if n != 1 else ''} kept"
        except Cancelled:
            raise
        except Exception as exc:
            print(f"[MyClip] English word check skipped: {exc}", flush=True)
    return segs, desc
