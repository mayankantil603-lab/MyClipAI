"""Subtitle language.

Auto (default): Hindi speech becomes Hinglish, English stays English, and a
word from any other language gets its English meaning in brackets.
Every AI answer is keyed by word number and checked against what was actually
said, so a confused model can't shift or translate words; anything rejected
falls back to the built-in rules."""
from __future__ import annotations

import difflib
import json
import re
import threading
from typing import Callable, Optional

from . import llm
from .models import Cancelled, Clip, JobConfig, Segment, Transcript, Word
from .translit import has_devanagari, non_latin, roman_text, roman_word

MODES = [("auto", "Auto", "Hindi as Hinglish, English as English, other-language words get (English) next to them"),
         ("roman", "Hinglish", "Always Hindi in English letters: ye kya ho gaya"),
         ("english", "English", "Always translated to English"),
         ("spoken", "As spoken", "Exactly what Whisper wrote, same script")]

STYLE = {
    "roman": "Write titles, hooks, descriptions and hashtags in Hinglish using Roman letters "
             "(like 'ye kya ho gaya bhai'), never in Devanagari or any other script.",
    "english": "Write titles, hooks, descriptions and hashtags in English.",
}

_KEYED = (' Input is {"words": [{"i": number, "w": word}]}. Keep every "i" exactly as given and return '
          'one object per input word, in the same order.')

ROMAN_PROMPT = ("You prepare burned-in captions. Rewrite each word in casual Hinglish the way Indian creators "
                "type in Roman letters ('क्या' -> 'kya', 'गया' -> 'gaya', 'वहाँ' -> 'vaha'). An English word "
                "written in another script becomes the English word ('गेम' -> 'game'). Never translate Hindi "
                "into English." + _KEYED + ' JSON only: {"items": [{"i": 0, "t": "kya"}]}')

AUTO_PROMPT = {
    "hinglish": (
        "You prepare burned-in captions. The speaker mainly speaks Hindi (Hinglish). For each word:\n"
        "- Hindi or Urdu word, any script: write it in casual Hinglish Roman letters ('क्या' -> 'kya'). "
        "Never translate it.\n"
        "- English word, any script: normal English spelling ('गेम' -> 'game'). No translation.\n"
        "- A word from any OTHER language (Spanish, Japanese, Punjabi, Tamil, Arabic, ...): Roman letters in "
        "\"t\" and a short English meaning in \"en\". If several words in a row form one foreign phrase, put "
        "the meaning only on the last one.\n"
        "Otherwise \"en\" is \"\"." + _KEYED + ' JSON only: {"items": [{"i": 0, "t": "", "en": ""}]}'),
    "english": (
        "You prepare burned-in captions. The speaker mainly speaks English. For each word:\n"
        "- English word: keep it exactly (only fix an obvious misspelling). \"en\" is \"\".\n"
        "- A word from any other language, Hindi included ('bhai', 'arigato', 'gracias'): Roman letters in "
        "\"t\" and a short English meaning in \"en\". If several words in a row form one foreign phrase, put "
        "the meaning only on the last one.\n"
        "Names, brands and game terms are not foreign words." + _KEYED +
        ' JSON only: {"items": [{"i": 0, "t": "", "en": ""}]}'),
}

ENGLISH_PROMPT = ('Translate each subtitle line into short, natural English for a gaming clip. Keep slang '
                  'casual, keep names and game terms. Input is {"lines": [{"i": number, "w": line}]}. Keep every '
                  '"i". JSON only: {"items": [{"i": 0, "en": "..."}]}')


# ---------------------------------------------------------------- helpers

def base_for(tr: Transcript) -> str:
    """What the speaker mainly speaks: hinglish | english | other."""
    lang = (tr.language or "").lower()[:2]
    sample = " ".join(s.text for s in tr.segments[:200])
    if lang in ("hi", "ur") or has_devanagari(sample):
        return "hinglish"
    if lang == "en" or (not lang and not non_latin(sample)):
        return "english"
    return "other"


def title_style(mode: str, tr: Optional[Transcript] = None) -> str:
    if mode == "auto":
        return STYLE["roman"] if (tr is not None and base_for(tr) == "hinglish") else STYLE["english"]
    return STYLE.get(mode, "")


def _copy(tr: Transcript) -> Transcript:
    return Transcript([Segment(s.start, s.end, s.text, [Word(w.start, w.end, w.text) for w in s.words])
                       for s in tr.segments], tr.language)


def _letters(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def similar(a: str, b: str) -> float:
    a, b = _letters(a), _letters(b)
    return difflib.SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def accept_word(orig: str, rule: str, t: str, base: str) -> bool:
    """Is the AI's caption word a faithful rendering of what was said?"""
    if not t or non_latin(t):
        return False
    if non_latin(rule):                 # a script we can't convert ourselves: trust the AI
        return True
    ref = rule if (base == "hinglish" or non_latin(orig)) else orig
    return similar(ref, t) >= (0.34 if base == "hinglish" else 0.5)


def _keyed(conn, key: str, model: str, instruction: str, field: str, values: list[str],
           cancel: threading.Event, size: int, per_item: int) -> dict[int, dict]:
    got: dict[int, dict] = {}
    for s0 in range(0, len(values), size):
        if cancel.is_set():
            raise Cancelled()
        chunk = values[s0:s0 + size]
        payload = {field: [{"i": s0 + k, "w": v} for k, v in enumerate(chunk)]}
        try:
            text = llm.chat(conn, key, model, [{"role": "system", "content": instruction},
                                               {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                            max_tokens=min(4000, 200 + per_item * len(chunk)), cancel=cancel)
            data = llm.parse_json(text)
        except Cancelled:
            raise
        except Exception:
            continue
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            try:
                i = int(it.get("i"))
            except (TypeError, ValueError):
                continue
            if s0 <= i < s0 + len(chunk) and i not in got:
                got[i] = it
    return got


def _clean(x) -> str:
    return " ".join(str(x or "").split())


def _word_index(tr: Transcript, c: Clip) -> list[tuple[int, int]]:
    return [(si, wi) for si, s in enumerate(tr.segments) if s.end > c.start and s.start < c.end
            for wi, w in enumerate(s.words) if w.end > c.start and w.start < c.end]


# ---------------------------------------------------------------- entry

def prepare(tr: Transcript, clips: list[Clip], cfg: JobConfig, media: str, conn, key: str, model: str,
            log: Callable[[str], None], cancel: threading.Event) -> tuple[list[Transcript], str, str]:
    """Returns (caption transcript per clip, note for the user, what was used)."""
    mode = cfg.caption_lang
    same = [tr] * len(clips)
    if mode not in ("roman", "english", "auto") or tr.empty or not clips:
        return same, "", ""
    if mode == "auto":
        base = base_for(tr)
        if base == "other":
            return to_english(tr, clips, cfg, media, conn, key, model, log, cancel)
        if conn:
            return to_auto(tr, clips, base, conn, key, model, log, cancel)
        if base == "hinglish":
            return to_roman(tr, clips, None, key, model, log, cancel)
        return same, "", ""
    if mode == "english":
        if (tr.language or "").lower().startswith("en") and not any(non_latin(s.text) for s in tr.segments[:80]):
            return same, "", ""
        return to_english(tr, clips, cfg, media, conn, key, model, log, cancel)
    return to_roman(tr, clips, conn, key, model, log, cancel)


def _rule_roman(tr: Transcript) -> Transcript:
    out = _copy(tr)
    for s in out.segments:
        s.text = roman_text(s.text)
        for w in s.words:
            w.text = roman_word(w.text)
    return out


def to_roman(tr: Transcript, clips: list[Clip], conn, key: str, model: str,
             log: Callable[[str], None], cancel: threading.Event) -> tuple[list[Transcript], str, str]:
    out = _rule_roman(tr)
    used_ai = False
    if conn:
        for n, c in enumerate(clips, 1):
            idx = _word_index(tr, c)
            src = [tr.segments[si].words[wi].text for si, wi in idx]
            if not any(non_latin(x) for x in src):
                continue
            log(f"Writing Hinglish subtitles, clip {n} of {len(clips)}")
            got = _keyed(conn, key, model, ROMAN_PROMPT, "words", src, cancel, 50, 14)
            for k, (si, wi) in enumerate(idx):
                it = got.get(k)
                if not it or not non_latin(src[k]):       # already English: leave it exactly as heard
                    continue
                t = _clean(it.get("t"))
                if accept_word(src[k], out.segments[si].words[wi].text, t, "hinglish"):
                    out.segments[si].words[wi].text = t
                    used_ai = True
    left = any(non_latin(w.text) for c in clips for w in out.words_between(c.start, c.end))
    note = ("Some words use a script MyClip can only convert with AI. Pick an AI model for complete Hinglish "
            "subtitles.") if left else ""
    return [out] * len(clips), note, "Hinglish subtitles" + (" (AI)" if used_ai else "")


def to_auto(tr: Transcript, clips: list[Clip], base: str, conn, key: str, model: str,
            log: Callable[[str], None], cancel: threading.Event) -> tuple[list[Transcript], str, str]:
    out = _rule_roman(tr) if base == "hinglish" else _copy(tr)
    if base == "english":
        for s in out.segments:
            for w in s.words:
                if has_devanagari(w.text):
                    w.text = roman_word(w.text)
    used_ai, n_tr = False, 0
    for n, c in enumerate(clips, 1):
        idx = _word_index(tr, c)
        if not idx:
            continue
        log(f"Checking languages in the subtitles, clip {n} of {len(clips)}")
        src = [tr.segments[si].words[wi].text for si, wi in idx]
        got = _keyed(conn, key, model, AUTO_PROMPT[base], "words", src, cancel, 50, 24)
        changes = []
        for k, (si, wi) in enumerate(idx):
            it = got.get(k)
            if not it or (base == "hinglish" and not non_latin(src[k])):
                continue                                   # English words in Hindi speech stay as heard
            t, en = _clean(it.get("t")), _clean(it.get("en")).strip("()[] ")
            if not accept_word(src[k], out.segments[si].words[wi].text, t, base):
                continue
            ok_en = bool(en) and en.lower() != t.lower() and len(en) <= 40 and not non_latin(en)
            changes.append((si, wi, t, en if ok_en else ""))
        brackets = sum(1 for ch in changes if ch[3])
        allow = brackets <= max(2, int(0.3 * len(idx)))      # a model bracketing everything misunderstood
        for si, wi, t, en in changes:
            out.segments[si].words[wi].text = f"{t} ({en})" if (en and allow) else t
            used_ai = True
            n_tr += 1 if (en and allow) else 0
    left = any(non_latin(w.text) for c in clips for w in out.words_between(c.start, c.end))
    note = "Some words are in a script MyClip couldn't convert." if left else ""
    used = ("Subtitles: Hinglish" if base == "hinglish" else "Subtitles: English") + \
        (f" with {n_tr} translation{'s' if n_tr != 1 else ''}" if n_tr else "") + (" (AI)" if used_ai else "")
    return [out] * len(clips), note, used


def to_english(tr: Transcript, clips: list[Clip], cfg: JobConfig, media: str, conn, key: str, model: str,
               log: Callable[[str], None], cancel: threading.Event) -> tuple[list[Transcript], str, str]:
    outs: list[Transcript] = []
    used, note = "", ""
    unloaded = False
    for n, c in enumerate(clips, 1):
        segs = [s for s in tr.segments if s.end > c.start and s.start < c.end]
        done: Optional[list[Segment]] = None
        if conn and segs:
            log(f"Translating subtitles, clip {n} of {len(clips)}")
            got = _keyed(conn, key, model, ENGLISH_PROMPT, "lines", [s.text for s in segs], cancel, 40, 30)
            lines = [_clean(got.get(k, {}).get("en")) for k in range(len(segs))]
            if all(lines) and not any(non_latin(x) for x in lines):
                done = [Segment(s.words[0].start if s.words else s.start, s.words[-1].end if s.words else s.end, t, [])
                        for s, t in zip(segs, lines)]
                used = "English subtitles (AI)"
        if done is None and segs:
            log(f"Translating subtitles with Whisper, clip {n} of {len(clips)}")
            try:
                if conn and conn.provider == "Ollama" and not unloaded:
                    llm.unload_ollama(conn.base_url)
                    unloaded = True
                done = whisper_translate(media, c.start, c.end, cfg, tr.language, cancel) or None
                if done:
                    used = used or "English subtitles (Whisper)"
            except Cancelled:
                raise
            except Exception as exc:
                note = f"Couldn't translate some subtitles ({str(exc)[:90]}), so they stay in the spoken language."
        outs.append(Transcript(done, "en") if done else tr)
    return outs, note, used


def whisper_translate(media: str, a: float, b: float, cfg: JobConfig, lang: str,
                      cancel: threading.Event) -> list[Segment]:
    from .transcribe import _load_or_cpu, _run, caption_model, decode, norm_lang, register_cuda_dlls
    register_cuda_dlls()
    model, device, ctype = caption_model(cfg.whisper_model, cfg.device)
    a0 = max(0.0, a - 0.5)
    audio = decode(media, a0, (b - a) + 1.0, cancel, clean=True)
    m, _, _ = _load_or_cpu(model, device, ctype, "medium", cancel)
    segs, _ = _run(m, audio, norm_lang(lang), cancel, None, a0, task="translate", best_of=5)
    return segs
