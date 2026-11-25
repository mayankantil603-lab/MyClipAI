"""Bleeping swear words (Hindi and English, Roman and Devanagari).

Matching is per caption word. An entry ending in * matches any word starting with it.
The audio gets a beep or silence over the word, and the caption shows the first letter
followed by asterisks. The list is editable in Settings."""
from __future__ import annotations

import json
import re
from typing import Optional

from .models import Segment, Transcript, Word
from .paths import config_dir
from .translit import has_devanagari, roman_word

DEFAULT_WORDS = [
    # English
    "fuck*", "motherfuck*", "fck*", "fuk*", "shit*", "bullshit*", "bitch*", "asshole*", "bastard*", "dick", "dicks", "dickhead*",
    "pussy*", "cunt*", "slut*", "whore*", "wtf", "stfu", "mf",
    # Hindi in Roman letters
    "bc", "mc", "bsdk", "bhenchod*", "behenchod*", "bhencho*", "benchod*", "bahenchod*", "madarchod*",
    "maderchod*", "madarchot*", "chutiya*", "chutiye*", "chutia*", "chutya*", "chootiya*", "gaandu*", "gandu*",
    "gaand", "gand", "bhosdi*", "bhosda*", "bhosad*", "bhosri*", "lund", "lauda*", "lavda*", "lawda*", "loda",
    "lode", "randi*", "harami*", "haramkhor*", "bhadwa*", "bhadve*", "jhaat*", "jhant*", "tatte", "chod",
    "chodu*", "chodna*",
    # Devanagari forms (also matched after conversion to Roman letters)
    "चूतिया", "चुतिया", "चूतिये", "चुतिये", "मादरचोद", "भेनचोद", "बहनचोद", "बहनचोद", "भोसड़ीके", "भोसडीके",
    "भोसड़ी", "गांडू", "गांड", "लौड़ा", "लौडा", "लोडा", "लंड", "रंडी", "हरामी", "हरामखोर", "भड़वा", "भड़वे", "चोद",
]
_PUNCT = re.compile(r"^[^\w\u0900-\u097F*]+|[^\w\u0900-\u097F*]+$")
_BRACKET = re.compile(r"^(.*?)(\s*\(.+\))$")


def words_file():
    return config_dir() / "bleep_words.json"


def load_words() -> list[str]:
    try:
        d = json.loads(words_file().read_text(encoding="utf-8"))
        w = [str(x).strip() for x in d.get("words", []) if str(x).strip()]
        if w:
            return w
    except Exception:
        pass
    return list(DEFAULT_WORDS)


def save_words(words: list[str]) -> None:
    clean = []
    for w in words:
        w = w.strip().lower() if not has_devanagari(w) else w.strip()
        if w and w not in clean:
            clean.append(w)
    words_file().write_text(json.dumps({"words": clean}, ensure_ascii=False, indent=1), encoding="utf-8")


def _norm(t: str) -> str:
    t = _PUNCT.sub("", t or "")
    if has_devanagari(t):
        return t
    t = t.lower()
    return re.sub(r"(.)\1{2,}", r"\1\1", t)            # fuuuuck -> fuuck


class Matcher:
    def __init__(self, words: Optional[list[str]] = None) -> None:
        self.exact: set[str] = set()
        self.prefix: list[str] = []
        for w in (words if words is not None else load_words()):
            w = w.strip()
            if not w:
                continue
            w = w if has_devanagari(w) else w.lower()
            if w.endswith("*"):
                self.prefix.append(w[:-1])
            else:
                self.exact.add(w)

    def _hit(self, t: str) -> bool:
        if not t:
            return False
        if t in self.exact:
            return True
        if any(t.startswith(p) for p in self.prefix if p):
            return True
        # repeated-letter variants: fuuck -> fuck
        t2 = re.sub(r"(.)\1+", r"\1", t)
        return t2 != t and (t2 in self.exact or any(t2.startswith(p) for p in self.prefix if p))

    def is_swear(self, word: str) -> bool:
        m = _BRACKET.match(word or "")
        core = m[1] if m else (word or "")
        t = _norm(core)
        if not t:
            return False
        if "*" in t and re.search(r"\w\*+", t):          # Whisper's own censoring: f***
            return True
        if self._hit(t):
            return True
        if has_devanagari(t):
            return self._hit(_norm(roman_word(t)))
        return False


def mask(word: str) -> str:
    """'fucking' -> 'f******', 'chutiya!' -> 'c******!', keeps a trailing (meaning)."""
    m = _BRACKET.match(word)
    core, br = (m[1], m[2]) if m else (word, "")
    lead = re.match(r"^[^\w\u0900-\u097F]*", core)[0]
    tail = re.search(r"[^\w\u0900-\u097F*]*$", core)[0]
    mid = core[len(lead): len(core) - len(tail) if tail else len(core)]
    if not mid:
        return word
    if has_devanagari(mid):
        mid = roman_word(mid)
    return lead + mid[0] + "*" * max(2, len(mid) - 1) + tail + br


def apply(tr: Transcript, start: float, end: float, matcher: Matcher, pad: float = 0.04
          ) -> tuple[Transcript, list[tuple[float, float]]]:
    """Masks swear words in a copy of tr; returns (masked transcript, intervals in clip time)."""
    out_segs, hits = [], []
    for s in tr.segments:
        ws = []
        for w in s.words:
            nw = Word(w.start, w.end, w.text)
            if w.end > start and w.start < end and matcher.is_swear(w.text):
                nw.text = mask(w.text)
                a, b = max(0.0, w.start - start - pad), min(end - start, w.end - start + pad)
                if b > a:
                    hits.append((round(a, 3), round(b, 3)))
            ws.append(nw)
        out_segs.append(Segment(s.start, s.end, " ".join(x.text for x in ws) if ws else s.text, ws))
    hits.sort()
    merged: list[tuple[float, float]] = []
    for a, b in hits:
        if merged and a <= merged[-1][1] + 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return Transcript(out_segs, tr.language), merged


def audio_filter(intervals: list[tuple[float, float]], mode: str, duration: float, post: str) -> str:
    """filter_complex audio chain from [0:a:0] to [aout]. post = the normal afade/loudnorm chain."""
    if not intervals or mode not in ("beep", "mute"):
        return f"[0:a:0]{post}[aout]"
    on = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in intervals)
    chain = f"[0:a:0]aresample=48000,aformat=channel_layouts=stereo,volume=0:enable='{on}'"
    if mode == "mute":
        return f"{chain},{post}[aout]"
    return (f"{chain}[am];sine=frequency=1000:sample_rate=48000:duration={duration + 0.5:.3f},"
            f"volume=0.18,volume=0:enable='not({on})',aformat=channel_layouts=stereo[bp];"
            f"[am][bp]amix=inputs=2:duration=first:normalize=0,{post}[aout]")
