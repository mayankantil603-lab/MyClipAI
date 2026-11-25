"""Whisper transcription (faster-whisper).

First pass: the whole video with a fast model, to find moments.
Caption pass: each chosen clip again with the most accurate model, on cleaned
audio, so burned-in captions are as right as possible.

The spoken language is detected from several parts of the video (not just the
first 30 seconds, which in gaming videos is often an intro or music), Hindi and
Urdu are always written in Devanagari, and common Whisper hallucinations
("thanks for watching", repeated loops, text over game sound) are removed."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import subprocess
import threading
import traceback
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import ffmpeg
from .gpu import cuda_info, explain_gpu_error, gpu_failure, mark_failed, missing_cuda_libs, pick_whisper, register_cuda_dlls
from .codeswitch import restore_english
from .models import Cancelled, Segment, Transcript, Word, run_cancellable
from .translit import has_devanagari

_MODELS: dict[tuple, object] = {}
_LOCK = threading.Lock()
WHISPER_CHOICES = ["auto", "large-v3-turbo", "large-v3", "medium", "small", "base", "tiny"]
CACHE_VERSION = "v2"
SR = 16000
LANG_NAMES = {"hi": "Hindi", "en": "English", "ur": "Urdu", "pa": "Punjabi", "bn": "Bengali", "ta": "Tamil",
              "te": "Telugu", "mr": "Marathi", "gu": "Gujarati", "ml": "Malayalam", "kn": "Kannada",
              "es": "Spanish", "pt": "Portuguese", "ja": "Japanese", "ko": "Korean", "de": "German",
              "fr": "French", "ru": "Russian", "ar": "Arabic", "id": "Indonesian", "tr": "Turkish", "it": "Italian"}


def lang_name(code: str) -> str:
    return LANG_NAMES.get(code or "", code or "unknown")


def norm_lang(code: str) -> str:
    """Spoken Hindi and Urdu are the same language; we always want Devanagari."""
    code = (code or "").lower()
    return "hi" if code == "ur" else code


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def _cache_path(cache_dir: Path, media: str, model: str, lang: str) -> Path:
    st = os.stat(media)
    key = f"{CACHE_VERSION}|{os.path.abspath(media)}|{st.st_size}|{int(st.st_mtime)}|{model}|{lang}"
    return cache_dir / f"tr_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:16]}.json"


def _load(model: str, device: str, ctype: str, cancel: Optional[threading.Event] = None):
    """Load (and on first use download) a Whisper model. Cancel works during the download."""
    from faster_whisper import WhisperModel
    k = (model, device, ctype)
    with _LOCK:
        if k in _MODELS:
            return _MODELS[k]
        _MODELS.clear()              # one model in VRAM at a time (8 GB card)
        gc.collect()
    m = run_cancellable(lambda: WhisperModel(model, device=device, compute_type=ctype), cancel)
    with _LOCK:
        _MODELS.clear()
        _MODELS[k] = m
    return m


def release() -> None:
    """Drop the Whisper model so its VRAM goes back to the AI model."""
    with _LOCK:
        _MODELS.clear()
    gc.collect()


# ---------------------------------------------------------------- audio

def decode(media: str, start: float, dur: float, cancel: threading.Event, clean: bool = False) -> np.ndarray:
    """16 kHz mono float32, read in place so a 3-hour video doesn't need double the RAM.
    Cancel is checked every megabyte. `clean` cuts rumble and hiss outside the voice band."""
    af = ["-af", "highpass=f=80,lowpass=f=8000"] if clean else []
    cmd = [ffmpeg.ffmpeg_path(), "-hide_banner", "-v", "error", "-ss", f"{max(0.0, start):.3f}", "-t", f"{dur:.3f}",
           "-i", media, "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(SR), *af, "-f", "f32le", "-"]
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         creationflags=ffmpeg.NO_WINDOW)
    nbytes = (int(max(1.0, dur) * SR) + SR) * 4
    buf = bytearray(nbytes)
    view = memoryview(buf)
    pos, extra = 0, b""
    try:
        while pos < nbytes:
            n = p.stdout.readinto(view[pos:min(nbytes, pos + (1 << 20))])
            if not n:
                break
            pos += n
            if cancel.is_set():
                raise Cancelled()
        if pos >= nbytes:
            extra = p.stdout.read()
    finally:
        view.release()
        if p.poll() is None:
            p.kill()
        p.wait()
    arr = np.frombuffer(buf, dtype=np.float32, count=pos // 4)
    if extra:
        arr = np.concatenate([arr, np.frombuffer(extra[: len(extra) // 4 * 4], dtype=np.float32)])
    return arr


# ---------------------------------------------------------------- language

def vote_language(windows: list[list[tuple[str, float]]]) -> tuple[str, float]:
    """Combine per-window language probabilities. Urdu counts as Hindi, and a
    clear Hindi share wins over English (Hinglish speakers mix both; forcing
    English on Hindi speech makes Whisper translate instead of transcribe)."""
    score: dict[str, float] = {}
    for probs in windows:
        for lang, p in sorted(probs, key=lambda x: -x[1])[:5]:
            lang = norm_lang(lang)
            score[lang] = score.get(lang, 0.0) + float(p)
    if not score:
        return "", 0.0
    total = sum(score.values()) or 1.0
    best = max(score, key=score.get)
    if best == "en" and score.get("hi", 0.0) / total >= 0.3:
        best = "hi"
    return best, score[best] / total


def _window_probs(model, audio: np.ndarray) -> list[tuple[str, float]]:
    try:
        if hasattr(model, "detect_language"):
            try:
                _, _, probs = model.detect_language(audio=audio, vad_filter=True)
            except TypeError:
                _, _, probs = model.detect_language(audio)
        else:
            _, info = model.transcribe(audio, beam_size=1, vad_filter=True)
            probs = getattr(info, "all_language_probs", None) or [(info.language, info.language_probability)]
        return [(str(l), float(p)) for l, p in probs]
    except Exception:
        return []


def detect_language(model, audio: np.ndarray, duration: float, cancel: threading.Event,
                    on_progress: Optional[Callable[[int, int], None]] = None) -> tuple[str, float]:
    if duration <= 45:
        starts, length = [0.0], max(1.0, duration)
    else:
        fr = [0.08, 0.25, 0.42, 0.58, 0.75, 0.92] if duration >= 300 else [0.15, 0.5, 0.85]
        length = 30.0
        starts = [max(0.0, min(duration - length, f * duration - length / 2)) for f in fr]
    wins = []
    for k, s in enumerate(starts, 1):
        if cancel.is_set():
            raise Cancelled()
        a = audio[int(s * SR): int((s + length) * SR)]
        if on_progress:
            on_progress(k, len(starts))
        if a.size < SR * 2 or float(np.sqrt(np.mean(a * a))) < 1e-3:
            continue
        probs = run_cancellable(lambda a=a: _window_probs(model, a), cancel)
        if probs:
            wins.append(probs)
    return vote_language(wins)


# ---------------------------------------------------------------- cleaning

STRONG = {"thanks for watching", "thank you for watching", "thank you so much for watching",
          "please subscribe", "like and subscribe", "please like and subscribe", "subscribe to my channel",
          "see you in the next video", "subtitles by the amaraorg community", "transcribed by",
          "हमारे चैनल को सब्सक्राइब करें", "चैनल को सब्सक्राइब करें", "वीडियो देखने के लिए धन्यवाद",
          "देखने के लिए धन्यवाद", "सब्सक्राइब करें"}
WEAK = {"thank you", "thanks", "you", "bye", "subscribe", "धन्यवाद", "शुक्रिया", "बहुत बहुत धन्यवाद"}
_PUNCT = re.compile(r"[\s.,!?।॥'\"\-–—…:;()\[\]]+")


def _norm(t: str) -> str:
    return _PUNCT.sub(" ", (t or "").lower()).strip()


def _collapse(words: list[Word], keep: int = 3) -> list[Word]:
    out: list[Word] = []
    run = 0
    for w in words:
        if out and _norm(out[-1].text) == _norm(w.text) and _norm(w.text):
            run += 1
            if run >= keep:
                continue
        else:
            run = 0
        out.append(w)
    return out


def clean_segments(raw: list, offset: float = 0.0) -> list[Segment]:
    out: list[Segment] = []
    recent: list[str] = []
    for s in raw:
        text = (getattr(s, "text", "") or "").strip()
        nt = _norm(text)
        if not nt:
            continue
        nsp = float(getattr(s, "no_speech_prob", 0.0) or 0.0)
        alp = float(getattr(s, "avg_logprob", 0.0) or 0.0)
        cr = float(getattr(s, "compression_ratio", 1.0) or 1.0)
        if (nsp > 0.6 and alp < -0.8) or cr > 2.6 or alp < -1.2:
            continue
        if nt in STRONG or (nt in WEAK and (nsp > 0.2 or alp < -0.5)):
            continue
        if len(recent) >= 2 and recent[-1] == nt and recent[-2] == nt:
            continue                                   # hallucination loop
        words = [Word(offset + float(w.start), offset + float(w.end), str(w.word).strip())
                 for w in (getattr(s, "words", None) or []) if str(w.word).strip()]
        words = _collapse(words)
        start = words[0].start if words else offset + float(s.start)
        end = words[-1].end if words else offset + float(s.end)
        out.append(Segment(start, end, " ".join(w.text for w in words) if words else text, words))
        recent.append(nt)
    return out


# ---------------------------------------------------------------- running

def _run(m, audio, language: str, cancel: threading.Event, on_seg=None, offset: float = 0.0,
         **extra) -> tuple[list[Segment], str]:
    kw = dict(language=language or None, word_timestamps=True, beam_size=5, vad_filter=True,
              vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 400},
              condition_on_previous_text=False, hallucination_silence_threshold=2.0)
    kw.update(extra)
    try:                                               # the first call scans the whole audio (VAD)
        it, info = run_cancellable(lambda: m.transcribe(audio, **kw), cancel)
    except TypeError:                                  # older faster-whisper
        kw.pop("hallucination_silence_threshold", None)
        it, info = run_cancellable(lambda: m.transcribe(audio, **kw), cancel)
    raw = []
    for s in it:
        if cancel.is_set():
            raise Cancelled()
        raw.append(s)
        if on_seg:
            on_seg(s)
    return clean_segments(raw, offset), (getattr(info, "language", "") or language)


def _gpu_failed(exc: BaseException) -> str:
    """Remember why the GPU failed, print the details for the console, return a readable reason."""
    traceback.print_exception(type(exc), exc, exc.__traceback__)
    try:
        missing = missing_cuda_libs()
    except Exception:
        missing = []
    reason = explain_gpu_error(str(exc), missing)
    mark_failed(reason)
    print(f"[MyClip] GPU transcription failed: {exc}\n[MyClip] {reason}", flush=True)
    return reason


def _load_or_cpu(model: str, device: str, ctype: str, cpu_model: str, cancel: Optional[threading.Event] = None,
                 on_note: Optional[Callable[[str], None]] = None):
    try:
        return _load(model, device, ctype, cancel), model, device
    except Cancelled:
        raise
    except Exception as exc:
        if device != "cuda":
            raise
        reason = _gpu_failed(exc)
        if on_note:
            on_note(f"GPU failed, using CPU (slower). {reason}")
        return _load(cpu_model, "cpu", "int8", cancel), cpu_model, "cpu"


def transcribe(media: str, duration: float, choice: str, device_pref: str, language: str,
               cache_dir: Path, on_progress: Callable[[float, str], None],
               cancel: threading.Event) -> tuple[Transcript, str]:
    """Whole video. Returns (transcript, description of what was used)."""
    register_cuda_dlls()
    model, device, ctype = pick_whisper(choice, device_pref)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = _cache_path(cache_dir, media, model, language)
    if cp.exists():
        try:
            tr = Transcript.from_dict(json.loads(cp.read_text(encoding="utf-8")))
            on_progress(1.0, "Using saved transcript")
            return tr, f"Whisper {model} (cached), {lang_name(tr.language)}"
        except Exception:
            pass

    on_progress(0.0, f"Loading Whisper {model} on {device.upper()}")
    small = "small" if model in ("large-v3", "large-v3-turbo", "medium") else model
    m, model, device = _load_or_cpu(model, device, ctype, small, cancel,
                                    lambda msg: on_progress(0.0, msg))

    on_progress(0.002, "Reading the audio")
    audio = decode(media, 0.0, duration + 1.0, cancel)

    lang = norm_lang(language)
    if not lang:
        on_progress(0.005, "Detecting the spoken language")
        lang, _ = detect_language(m, audio, duration, cancel,
                                  lambda k, n: on_progress(0.005, f"Detecting the spoken language ({k} of {n})"))
        if lang:
            on_progress(0.01, f"Spoken language: {lang_name(lang)}")

    def seg(s) -> None:
        if duration:
            on_progress(min(s.end / duration, 1.0), f"Transcribing {_fmt(s.end)} of {_fmt(duration)}")

    try:
        segs, got = _run(m, audio, lang, cancel, seg)
    except Cancelled:
        raise
    except Exception as exc:
        if device != "cuda":
            raise RuntimeError(f"Transcription failed: {exc}") from None
        reason = _gpu_failed(exc)
        on_progress(0.0, f"GPU failed, continuing on CPU (slower). {reason}")
        release()
        model, device = small, "cpu"
        m = _load(model, "cpu", "int8", cancel)
        segs, got = _run(m, audio, lang, cancel, seg)
    if not lang and got == "ur":
        on_progress(0.0, "Writing Hindi in Devanagari")
        segs, got = _run(m, audio, "hi", cancel, seg)
    del audio
    tr = Transcript(segs, norm_lang(got) or lang)
    try:
        cp.write_text(json.dumps(tr.to_dict(), ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    return tr, f"Whisper {model} ({device.upper()}), {lang_name(tr.language)}"


def caption_model(choice: str, device_pref: str) -> tuple[str, str, str]:
    ok, _, vram = cuda_info()
    use_cuda = ok and device_pref != "cpu" and not gpu_failure()
    if choice == "large-v3":
        model = "large-v3"
    elif use_cuda:
        model = "large-v3" if (vram == 0 or vram >= 5500) else "medium"
    else:
        model = "medium"
    return model, ("cuda" if use_cuda else "cpu"), ("float16" if use_cuda else "int8")


def transcribe_clip(media: str, a: float, b: float, choice: str, device_pref: str, language: str,
                    cancel: threading.Event, prompt: Optional[str] = None) -> tuple[list[Segment], str, str]:
    """One clip, most accurate settings. Returns (segments in video time, description, language)."""
    register_cuda_dlls()
    model, device, ctype = caption_model(choice, device_pref)
    a0 = max(0.0, a - 1.0)
    audio = decode(media, a0, (b - a) + 2.0, cancel, clean=True)
    if audio.size < SR // 2:
        return [], f"Whisper {model}", norm_lang(language)
    m, model, device = _load_or_cpu(model, device, ctype, "medium", cancel)
    lang = norm_lang(language)
    kw = dict(best_of=5, temperature=[0.0, 0.2, 0.4, 0.6], initial_prompt=prompt or None)
    try:
        segs, got = _run(m, audio, lang, cancel, None, a0, **kw)
    except Cancelled:
        raise
    except Exception as exc:
        if device != "cuda":
            raise
        _gpu_failed(exc)
        release()
        model, device = "medium", "cpu"
        m = _load(model, "cpu", "int8", cancel)
        segs, got = _run(m, audio, lang, cancel, None, a0, **kw)
    if not lang and got == "ur":
        segs, got = _run(m, audio, "hi", cancel, None, a0, **kw)
    final = norm_lang(got) or lang
    desc = f"Whisper {model} ({device.upper()})"
    if final == "hi" and any(has_devanagari(w.text) for s in segs for w in s.words):
        # Whisper writes English words in Hindi letters when told the speech is Hindi.
        # Listen once more as English and keep the words both passes heard alike.
        try:
            en_segs, _ = _run(m, audio, "en", cancel, None, a0)
            segs, n = restore_english(segs, en_segs)
            if n:
                desc += f", {n} English word{'s' if n != 1 else ''} kept"
        except Cancelled:
            raise
        except Exception as exc:
            print(f"[MyClip] English word check skipped: {exc}", flush=True)
    return segs, desc, final


def _fmt(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"
