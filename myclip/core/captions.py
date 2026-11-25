"""Burned-in captions as ASS: short word groups, current word highlighted."""
from __future__ import annotations

import re

from .models import CaptionStyle, Word


def ass_time(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def bgr(hex_rgb: str) -> str:
    h = hex_rgb.strip("#").upper().rjust(6, "0")
    return f"&H00{h[4:6]}{h[2:4]}{h[0:2]}"


def clean(text: str, upper: bool) -> str:
    t = re.sub(r"[{}\\]", "", text).strip()
    return t.upper() if upper else t


TRANSLATION = bgr("C8C8C8")
_BRACKET = re.compile(r"^(.*?)\s*(\(.+\))$")


def fmt_word(text: str, color: str, base: str) -> str:
    """A caption word; a trailing "(meaning)" is drawn smaller and dimmer and never highlighted."""
    m = _BRACKET.match(text)
    main, br = (m[1], m[2]) if (m and m[1]) else (text, "")
    s = f"{{\\c{color}}}{main}{{\\c{base}}}" if color != base else main
    if br:
        s += f" {{\\fscx72\\fscy72\\c{TRANSLATION}}}{br}{{\\fscx100\\fscy100\\c{base}}}"
    return s


def group_words(words: list[Word], per_line: int, max_gap: float = 0.7, max_span: float = 2.2) -> list[list[Word]]:
    groups: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        brk = cur and (len(cur) >= per_line or w.start - cur[-1].end > max_gap
                       or w.end - cur[0].start > max_span or re.search(r"[.!?,]$", cur[-1].text)
                       or "(" in cur[-1].text)
        if brk:
            groups.append(cur)
            cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    return groups


def build_ass(words: list[Word], clip_start: float, clip_end: float, W: int, H: int,
              style: CaptionStyle, position: float | None = None) -> str:
    fs = max(18, int(H * style.size * (W / H if W > H else 1) ** 0.5))
    outline = max(2, int(fs * style.outline))
    margin_v = int(H * (style.position if position is None else position))
    head = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {W}\nPlayResY: {H}\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Cap,{style.font},{fs},{bgr(style.color)},{bgr(style.color)},&H00000000,&H64000000,"
        f"-1,0,0,0,100,100,0,0,1,{outline},{max(1, outline // 3)},2,{int(W * .07)},{int(W * .07)},{margin_v},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    local: list[Word] = []
    for w in words:
        if w.end <= clip_start or w.start >= clip_end:
            continue
        t = clean(w.text, style.uppercase)
        if t:
            local.append(Word(max(0.0, w.start - clip_start), min(clip_end, w.end) - clip_start, t))
    hl = bgr(style.highlight)
    base = bgr(style.color)
    cur = hl if style.highlight.upper() != style.color.upper() else base
    lines = []
    for g in group_words(local, style.words_per_line):
        for i, w in enumerate(g):
            a = w.start if i else g[0].start
            b = g[i + 1].start if i + 1 < len(g) else max(w.end, w.start + 0.25) + 0.1
            text = " ".join(fmt_word(x.text, cur if j == i else base, base) for j, x in enumerate(g))
            pop = "{\\fscx106\\fscy106\\t(0,90,\\fscx100\\fscy100)}" if i == 0 else ""
            lines.append(f"Dialogue: 0,{ass_time(a)},{ass_time(max(b, a + 0.05))},Cap,,0,0,0,,{pop}{text}")
    return head + "\n".join(lines) + "\n"
