"""Keeping English words English in Hindi speech.

With language="hi", Whisper writes English words in Devanagari ("गेम"), and
turning that back into Roman letters by sound gives "gem". So each caption
clip is also transcribed with language="en". Where both passes heard the
same-sounding word at the same moment, the English pass's spelling is used.
A Hindi word is never replaced by its English translation, because a
translation doesn't sound like the original ("मारा" vs "killed")."""
from __future__ import annotations

import difflib
import re

from .models import Segment, Word
from .translit import has_devanagari, roman_word

PUNCT = ".,!?।॥…:;\"'“”‘’()[]-–—"

# Very common Hindi words. Never replaced, whatever the English pass heard there.
HINDI_COMMON = set("""
hai hain ho hoga hogi honge hua hui hue tha thi the ka ki ke ko se me mein main mai tu tum hum ham aap
ye yeh vo wo woh voh is us in un to toh na nahi nahin mat kya kyu kyun kyon kaise kab kaha kahan kidhar
jab tab ab sab kuch bhi hi par pe aur ya jo jis jise koi kaun kitna kitni itna utna aisa aise vaisa waisa
raha rahi rahe gaya gayi gaye gya gyi kar karo karna kiya kiye diya liya de do le lo dekh dekho dekha chal
chalo chala aa aao aaja ja jao jaa ek bas bahut bohot accha acha theek thik haan han ji yaar bhai arre are
abe mera meri mere tera teri tere uska uski uske unka unki apna apni apne isko usko mujhe tujhe hume tumhe
use ise inko unko idhar udhar yaha yahan vaha vahan wahan upar niche neeche andar bahar pehle baad phir fir
abhi kabhi sirf bilkul matlab lekin magar kyunki isliye agar warna varna sath saath liye wala wali wale log
din raat baat kaam naam ghar pani paani khana maar mar mara maro bol bolo bola sun suno ruk ruko bhag bhago
aage peeche piche kal aaj sach jaldi dheere zara jara bata batao pata laga lag lagta lagi lage sakta sakte
sakti chahiye milega mila mili kaha kahi samajh socha dekhte karte karta karti hota hote hoti jaata jaate
""".split())


def _split(text: str) -> tuple[str, str, str]:
    i, j = 0, len(text)
    while i < j and text[i] in PUNCT:
        i += 1
    while j > i and text[j - 1] in PUNCT:
        j -= 1
    return text[:i], text[i:j], text[j:]


_CONS = (("tion", "shan"), ("sion", "shan"), ("tch", "C"), ("chh", "C"), ("ch", "C"), ("sh", "s"),
         ("ph", "f"), ("wh", "v"), ("th", "t"), ("dh", "d"), ("bh", "b"), ("kh", "k"), ("gh", "g"),
         ("ck", "k"), ("ce", "se"), ("ci", "si"), ("cy", "si"), ("dge", "j"), ("q", "k"), ("x", "ks"),
         ("c", "k"), ("w", "v"), ("ng", "n"), ("z", "j"))


def _common(s: str) -> str:
    for a, b in _CONS:
        s = s.replace(a, b)
    s = s.replace("ai", "e").replace("au", "o")
    return re.sub(r"(.)\1+", r"\1", s)


def norm_hi(s: str) -> str:
    """Hinglish spelling (from Hindi letters) to a rough sound spelling."""
    return _common(re.sub(r"[^a-z]", "", (s or "").lower()))


def norm_en(s: str) -> str:
    """English spelling to the same rough sound spelling: game -> gem, nice -> nes, here -> hir."""
    s = re.sub(r"[^a-z]", "", (s or "").lower())
    s = re.sub(r"igh", "ai", s)
    s = re.sub(r"ere$", "ir", s)
    s = re.sub(r"([^aeiou])([aeiou])([^aeiouwy])e(s?)$",
               lambda m: m[1] + {"a": "e", "i": "ai", "o": "o", "u": "u", "e": "i"}[m[2]] + m[3] + m[4], s)
    s = re.sub(r"ea(?=d)", "e", s)                # head, dead, ready
    for a_, b_ in (("ee", "i"), ("ea", "i"), ("oo", "u"), ("ay", "e"), ("ey", "e"), ("ou", "au"), ("ow", "au")):
        s = s.replace(a_, b_)
    s = re.sub(r"er$", "ar", s)
    s = re.sub(r"y$", "i", s)
    s = re.sub(r"le$", "al", s)
    return _common(s)


norm = norm_hi                                   # kept for callers that only need one side


def skeleton(n: str) -> str:
    return re.sub(r"(.)\1+", r"\1", re.sub(r"[aeiouy]", "", n))


def _vowels(n: str) -> str:
    """Vowel pattern, with the vowels Hindi and English writers swap merged:
    English short u is Hindi a ("luck" / "lak"), an unstressed last vowel is a or e ("open" / "opan")."""
    v = re.sub(r"[^aeiou]", "", n.replace("y", "i")).replace("u", "a")
    if len(v) >= 2 and v[-1] == "e":
        v = v[:-1] + "a"
    return re.sub(r"(.)\1+", r"\1", v)


def sounds_same(hindi_roman: str, english: str, strict: bool = False) -> float:
    """0 when they don't match, otherwise a score (higher is a closer match)."""
    if hindi_roman.lower() == english.lower():
        return 2.0
    score = _sounds_same(norm_hi(hindi_roman), norm_en(english), strict)
    if hindi_roman.isascii() and re.fullmatch(r"[A-Za-z]+", hindi_roman or ""):
        # already English from the word list (e.g. "game"): compare English to English too
        score = max(score, _sounds_same(norm_en(hindi_roman), norm_en(english), strict))
    return score


def _sounds_same(a: str, b: str, strict: bool) -> float:
    if not a or not b:
        return 0.0
    if abs(len(a) - len(b)) > max(2, int(0.5 * max(len(a), len(b)))):
        return 0.0
    ka, kb = skeleton(a), skeleton(b)
    full = difflib.SequenceMatcher(None, a, b).ratio()
    if not ka or not kb:
        return 1.0 + full if a == b else 0.0
    if ka == kb:
        if len(kb) <= 2 or strict:                 # short words: the vowels must agree too (haar is not here)
            va, vb = _vowels(a), _vowels(b)
            if va != vb and difflib.SequenceMatcher(None, va, vb).ratio() < (0.8 if strict else 0.67):
                return 0.0
        return 1.0 + full if full >= (0.6 if strict else 0.5) else 0.0
    if strict or len(kb) < 4:
        return 0.0
    if difflib.SequenceMatcher(None, ka, kb).ratio() >= 0.85 and full >= 0.62:
        return full
    return 0.0


def forms(word: str) -> list[tuple[str, float]]:
    """The English word plus its base form ("pushing" -> "push", "minutes" -> "minute"),
    because Hindi speakers usually use the plain form inside a Hindi sentence."""
    out = [(word, 0.0)]
    w = word.lower()
    if len(w) > 4 and w.endswith("ing"):
        stem = w[:-3]
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiouls":
            stem = stem[:-1]                                          # running -> run
            out.append((word[:len(stem)], 0.03))
        elif len(stem) >= 2 and stem[-1] not in "aeiouwy" and stem[-2] in "aeiou" and \
                (len(stem) < 3 or stem[-3] not in "aeiou"):
            out.append((word[:len(stem)] + "e", 0.04))               # making -> make
            out.append((word[:len(stem)], 0.25))
        else:
            out.append((word[:len(stem)], 0.03))                     # pushing -> push
    elif len(w) > 3 and w.endswith("ed") and not w.endswith("eed"):
        out += [(word[:-2], 0.03), (word[:-1], 0.04)]
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        out.append((word[:-1], 0.02))
    return out


def _english_form(e: Word, first_in_segment: bool) -> str:
    _, core, _ = _split(e.text)
    core = core.strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", core or ""):
        return ""
    if core == "I" or (len(core) >= 2 and core.isupper()):
        return core                                   # I, GTA, PUBG
    if core[0].isupper() and not first_in_segment:
        return core                                   # names: Valorant, Minecraft
    return core.lower()


def best_form(r: str, english: str, strict: bool = False) -> tuple[float, str]:
    best, pick = 0.0, ""
    for f, pen in forms(english):
        sc = sounds_same(r, f, strict)
        if sc > 0:
            exact = skeleton(norm_hi(r)) == skeleton(norm_en(f))
            sc = sc + (0.3 if exact else 0.0) - pen
            if sc > best:
                best, pick = sc, f
    return best, pick


def restore_english(hi: list[Segment], en: list[Segment], tol: float = 0.9) -> tuple[list[Segment], int]:
    """Returns (segments with audio-confirmed English words, how many words were fixed)."""
    E: list[tuple[Word, str]] = []
    for s in en:
        for k, w in enumerate(s.words):
            f = _english_form(w, k == 0)
            if f:
                E.append((w, f))
    if not E:
        return hi, 0

    def near(a: float, b: float) -> list[int]:
        return [i for i, (w, _) in enumerate(E) if w.end > a - tol and w.start < b + tol]

    fixed = 0
    out: list[Segment] = []
    for seg in hi:
        words = [Word(w.start, w.end, w.text) for w in seg.words]
        res: list[Word] = []
        i = 0
        while i < len(words):
            w = words[i]
            pre, core, post = _split(w.text)
            if not has_devanagari(core):
                res.append(w)
                i += 1
                continue
            post = post.replace("।", ".").replace("॥", ".")
            r = roman_word(core)
            best, how = 0.0, None
            if r not in HINDI_COMMON:
                cand = near(w.start, w.end)
                for j in cand:
                    sc, f = best_form(r, E[j][1])
                    if sc > best:
                        best, how = sc, ("one", f)
                for j in cand:                                   # "गेमप्ले" vs "game play"
                    if j + 1 < len(E) and E[j + 1][0].start < w.end + tol:
                        joined = E[j][1] + " " + E[j + 1][1]
                        sc = sounds_same(r, E[j][1] + E[j + 1][1], strict=True) - 0.05
                        if sc > best:
                            best, how = sc, ("one", joined)
            if i + 1 < len(words):                               # "हेड शॉट" vs "headshot"
                w2 = words[i + 1]
                p2, c2, post2 = _split(w2.text)
                if has_devanagari(c2) and not post:
                    r2 = roman_word(c2)
                    if r not in HINDI_COMMON and r2 not in HINDI_COMMON:
                        for j in near(w.start, w2.end):
                            sc, f = best_form(r + r2, E[j][1], strict=True)
                            sc -= 0.02
                            if sc > best and best_form(r, E[j][1])[0] < sc:
                                best, how = sc, ("two_hi", f, post2.replace("।", ".").replace("॥", "."))
            if how and best > 0:
                if how[0] == "one":
                    res.append(Word(w.start, w.end, pre + how[1] + post))
                    i += 1
                else:
                    res.append(Word(w.start, words[i + 1].end, pre + how[1] + how[2]))
                    i += 2
                fixed += 1
                continue
            res.append(w)
            i += 1
        text = " ".join(x.text for x in res) if res else seg.text
        out.append(Segment(res[0].start if res else seg.start, res[-1].end if res else seg.end, text, res))
    return out, fixed
