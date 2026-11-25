"""Devanagari -> casual Roman ("Hinglish"), the way people type it:
"हम वहाँ चलेंगे ये क्या हो गया" -> "hum vaha chalenge ye kya ho gaya".

Rule based (schwa deletion, casual vowel lengths, nasal handling) plus a small
list of very common words. One word in, one word out, so word timings are kept."""
from __future__ import annotations

import re
import unicodedata

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_RUN = re.compile(r"[\u0900-\u0963\u0966-\u097F]+")

CONS = {"क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh",
        "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d",
        "ध": "dh", "न": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r",
        "ल": "l", "ळ": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h"}
NUKTA = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "d", "ढ": "dh", "फ": "f", "य": "y"}
VOWELS = {"अ": "a", "आ": "aa", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri", "ए": "e", "ऐ": "ai",
          "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e"}
MATRAS = {"ा": "A", "ि": "i", "ी": "I", "ु": "u", "ू": "U", "ृ": "ri", "े": "e", "ै": "ai", "ो": "o",
          "ौ": "au", "ॉ": "o", "ॅ": "e"}
LONG = {"A": "aa", "I": "ee", "U": "oo"}
VIRAMA, NUKTA_SIGN, VISARGA = "\u094d", "\u093c", "\u0903"
NASAL = {"\u0902", "\u0901"}
DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}

WORDS = {
    "हम": "hum", "तुम": "tum", "है": "hai", "हैं": "hain", "नहीं": "nahi", "नही": "nahi", "में": "mein",
    "मैं": "main", "क्या": "kya", "यह": "ye", "वह": "vo", "ये": "ye", "वो": "vo", "क्यों": "kyun",
    "क्यूं": "kyun", "और": "aur", "भी": "bhi", "तो": "toh", "हाँ": "haan", "हां": "haan", "अच्छा": "accha",
    "यार": "yaar", "भाई": "bhai", "कुछ": "kuch", "कैसे": "kaise", "ऐसे": "aise", "वहाँ": "vaha",
    "वहां": "vaha", "यहाँ": "yaha", "यहां": "yaha", "मुझे": "mujhe", "तुझे": "tujhe", "हूँ": "hoon",
    "हूं": "hoon", "माँ": "maa", "मां": "maa", "ठीक": "theek", "कहाँ": "kahan", "कहां": "kahan",
    "अरे": "arre", "अबे": "abe", "लिए": "liye", "मैंने": "maine", "आप": "aap", "जी": "ji",
}


_PAIRS = "गेम game|गेम्स games|गेमिंग gaming|गेमर gamer|फोन phone|वीडियो video|वीडियोस videos|चैनल channel|सब्सक्राइब subscribe|लाइक like|कमेंट comment|कमेंट्स comments|शेयर share|लाइव live|स्ट्रीम stream|ओके ok|सॉरी sorry|थैंक thank|थैंक्स thanks|थैंक्यू thankyou|प्लीज please|बेस्ट best|नाइस nice|वाओ wow|वॉव wow|ब्रो bro|गाइज guys|गाइस guys|टीम team|मैच match|किल kill|किल्स kills|हेडशॉट headshot|शॉट shot|गन gun|राउंड round|लेवल level|बॉस boss|मिशन mission|प्लेयर player|प्लेयर्स players|सर्वर server|लैग lag|पिंग ping|रैंक rank|रैंक्ड ranked|क्लच clutch|स्नाइपर sniper|ग्रेनेड grenade|लूट loot|जोन zone|सेफ safe|मैप map|पुश push|बैक back|लेफ्ट left|राइट right|रेडी ready|स्टार्ट start|विन win|फ्री free|फायर fire|ऑनलाइन online|वेट wait|सीरियसली seriously|एक्चुअली actually|बेसिकली basically|ऑसम awesome|क्रेजी crazy|फनी funny|सीन scene|मूड mood|टाइम time|लास्ट last|फर्स्ट first|नेक्स्ट next|न्यू new|फ्रेंड friend|फ्रेंड्स friends|मोबाइल mobile|पीसी PC|लैपटॉप laptop|इंटरनेट internet|ड्रॉप drop|नूब noob|प्रो pro|हैक hack|हैकर hacker|चीटर cheater|रिवाइव revive|नॉक knock|एनिमी enemy|डैमेज damage|हेल्थ health|अपडेट update|स्किन skin|इवेंट event|म्यूजिक music|गुड good|वेरी very|येस yes|ओह oh|ऑप op|बग bug|स्पीड speed|यूट्यूब YouTube|यूट्यूबर YouTuber|इंस्टाग्राम Instagram|वैलोरेंट Valorant|वैलोरंट Valorant|माइनक्राफ्ट Minecraft|पबजी PUBG|बीजीएमआई BGMI|जीटीए GTA|ओएमजी OMG|डेड dead|बूम boom|ब्लास्ट blast|सपोर्ट support|कमिंग coming|गोइंग going|डन done|ओवर over|बाय bye|हेलो hello|हाय hi|वेलकम welcome|गेमप्ले gameplay|स्ट्रीमर streamer"
ENGLISH = {k: v for k, v in (p.split(" ", 1) for p in _PAIRS.split("|")) if " " not in v}
ENGLISH["थैंक्यू"] = "thank you"
_NUKTA_PLAIN = re.compile("([\u091c\u092b\u0915\u0916\u0917])\u093c")   # ज़ फ़ क़ ख़ ग़ (not ड़ ढ़)


def _english(run: str) -> str:
    return ENGLISH.get(_NUKTA_PLAIN.sub(r"\1", run), "")


def has_devanagari(text: str) -> bool:
    return bool(DEVANAGARI.search(text or ""))


def non_latin(text: str) -> bool:
    return any(ord(ch) > 0x024F and ch.isalpha() for ch in text or "")


def _translit(core: str) -> str:
    core = core.replace("ज्ञ", "ग्य")
    units: list[dict] = []
    i, n = 0, len(core)
    while i < n:
        ch = core[i]
        if ch in CONS:
            r = CONS[ch]
            i += 1
            if i < n and core[i] == NUKTA_SIGN:
                r = NUKTA.get(ch, r)
                i += 1
            u = {"c": r, "v": None, "nas": False}           # v None = inherent "a"
            if i < n and core[i] in MATRAS:
                u["v"] = MATRAS[core[i]]
                i += 1
            elif i < n and core[i] == VIRAMA:
                u["v"] = ""
                i += 1
            units.append(u)
        elif ch in VOWELS or ch in MATRAS:
            units.append({"c": None, "v": VOWELS.get(ch) or MATRAS[ch].lower(), "nas": False})
            i += 1
        elif ch in NASAL:
            if units and "raw" not in units[-1]:
                units[-1]["nas"] = True
            i += 1
        elif ch == VISARGA:
            units.append({"c": "h", "v": "", "nas": False})
            i += 1
        elif ch in (NUKTA_SIGN, VIRAMA):
            i += 1
        else:
            units.append({"raw": DIGITS.get(ch, ch)})
            i += 1

    letters = [u for u in units if "raw" not in u]

    def vow(u: dict) -> bool:
        return u["c"] is None or u["v"] != ""

    L = len(letters)
    if L > 1 and letters[-1]["c"] is not None and letters[-1]["v"] is None:
        letters[-1]["v"] = ""                               # kar, ham, bahut
    for j in range(L - 2, 0, -1):                           # samajhna, kamra, karna
        u, p, nx = letters[j], letters[j - 1], letters[j + 1]
        if u["c"] is not None and u["v"] is None and not u["nas"] and vow(p) and nx["c"] is not None and vow(nx):
            u["v"] = ""
    nuclei = sum(1 for u in letters if vow(u))
    last_v = max((k for k, u in enumerate(letters) if vow(u)), default=-1)

    out: list[str] = []
    li = -1
    for u in units:
        if "raw" in u:
            out.append(u["raw"])
            continue
        li += 1
        if u["c"] is not None:
            out.append(u["c"])
        v = "a" if u["v"] is None else u["v"]
        if v in LONG:                                       # kaam, teen, yaar  but  kya, bhi, jana
            v = LONG[v] if (nuclei == 1 and li < L - 1) else v.lower()
        elif v == "aa" and u["c"] is None and li > 0:       # hua, not huaa
            v = "a"
        out.append(v)
        if u["nas"] and v:
            nxt = letters[li + 1] if li + 1 < L else None
            if nxt is None:
                if v in ("e", "ai", "o", "au", "oo"):
                    out.append("n")                         # hain, kyon; vaha, nahi drop it
            elif nxt["c"] and nxt["c"][0] in "pbm":
                out.append("m")
            else:
                out.append("n")
    s = "".join(out)
    return s.replace("chchh", "cch").replace("chch", "cch")


def roman_word(word: str) -> str:
    if not has_devanagari(word):
        return word
    word = unicodedata.normalize("NFC", word)
    s = _RUN.sub(lambda m: WORDS.get(m[0]) or _english(m[0]) or _translit(m[0]), word)
    return s.replace("।", ".").replace("॥", ".")


def roman_text(text: str) -> str:
    if not has_devanagari(text):
        return text
    return re.sub(r"\S+", lambda m: roman_word(m[0]), text)
