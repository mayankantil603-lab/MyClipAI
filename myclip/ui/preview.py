"""Live caption preview: a frame of the video in the chosen format with sample captions
drawn the way the final clip will look (font, size, colours, outline, position,
highlighted word, the smaller "(meaning)", the hook line and Fit's blurred bars)."""
from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from ..core.models import CAPTION_STYLES, CaptionStyle
from . import anim, theme

SAMPLES = {
    "auto": [["bhai", "ye", "game", "toh"], ["crazy", "hai", "yaar"], ["gracias (thanks)", "bro"]],
    "roman": [["bhai", "ye", "game", "toh"], ["crazy", "hai", "yaar"], ["kya", "clutch", "tha"]],
    "english": [["no", "way", "that"], ["was", "so", "clean"], ["let's", "go"]],
    "spoken": [["भाई", "ये", "गेम", "तो"], ["क्रेज़ी", "है", "यार"], ["क्या", "क्लच", "था"]],
}
ASPECT = {"9:16": (9, 16), "1:1": (1, 1), "4:5": (4, 5), "16:9": (16, 9)}


def layout_for(aspect: str, box_w: float, box_h: float, custom: tuple[int, int] = (1080, 1920),
               src: tuple[int, int] = (1920, 1080)) -> tuple[float, float]:
    """Largest frame of the target aspect that fits the box."""
    if aspect == "Custom":
        aw, ah = custom
    elif aspect == "Original":
        aw, ah = src
    else:
        aw, ah = ASPECT.get(aspect, (9, 16))
    k = min(box_w / aw, box_h / ah)
    return aw * k, ah * k


def groups_for(words: list[str], per_line: int) -> list[list[str]]:
    out, cur = [], []
    for w in words:
        if cur and (len(cur) >= per_line or "(" in cur[-1]):
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return out


class CaptionPreview(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(220, 330)
        self.style_name, self.lang, self.aspect, self.framing = "Bold", "auto", "9:16", "Auto"
        self.hook, self.custom, self.cam = True, (1080, 1920), None
        self.frame: Optional[QPixmap] = None
        self.blur: Optional[QPixmap] = None
        self.src_size = (1920, 1080)
        self.seq: list[list[str]] = []
        self.g, self.w = 0, 0
        self.tick = 0
        self.t = QTimer(self, interval=380, timeout=self._step)
        self._rebuild()

    # ---------------- inputs
    def set_options(self, style: str, lang: str, aspect: str, framing: str, hook: bool,
                    custom: tuple[int, int] = (1080, 1920)) -> None:
        self.style_name = style if style in CAPTION_STYLES or style == "Off" else "Bold"
        self.lang, self.aspect, self.framing, self.hook, self.custom = lang or "auto", aspect, framing, hook, custom
        self._rebuild()

    def set_frame(self, path_or_bytes, cam: Optional[list] = None) -> None:
        img = QImage()
        if isinstance(path_or_bytes, (bytes, bytearray)):
            img.loadFromData(bytes(path_or_bytes))
        elif path_or_bytes and os.path.exists(path_or_bytes):
            img.load(path_or_bytes)
        if img.isNull():
            self.frame = self.blur = None
        else:
            self.frame = QPixmap.fromImage(img)
            self.src_size = (img.width(), img.height())
            small = img.scaled(max(8, img.width() // 28), max(8, img.height() // 28),
                               Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.blur = QPixmap.fromImage(small.scaled(img.width() // 4, img.height() // 4,
                                                      Qt.AspectRatioMode.IgnoreAspectRatio,
                                                      Qt.TransformationMode.SmoothTransformation))
        self.cam = cam
        self.update()

    def _rebuild(self) -> None:
        st = CAPTION_STYLES.get(self.style_name) or CAPTION_STYLES["Bold"]
        words = [w for line in SAMPLES.get(self.lang, SAMPLES["auto"]) for w in line]
        self.seq = groups_for(words, st.words_per_line)
        self.g = self.w = self.tick = 0
        if anim.on():
            self.t.start()
        else:
            self.t.stop()
        self.update()

    def _step(self) -> None:
        self.tick += 1
        self.w += 1
        if self.seq and self.w >= len(self.seq[self.g]):
            self.w = 0
            self.g = (self.g + 1) % len(self.seq)
        self.update()

    def showEvent(self, e) -> None:  # noqa: N802
        if anim.on():
            self.t.start()

    def hideEvent(self, e) -> None:  # noqa: N802
        self.t.stop()

    # ---------------- painting
    def _mode(self, W: float, H: float) -> str:
        f = (self.framing or "Auto").lower()
        sw, sh = self.src_size
        src_ar, dst_ar = sw / max(sh, 1), W / max(H, 1)
        if f == "facecam" and src_ar > dst_ar * 1.2:
            return "facecam"
        if abs(src_ar - dst_ar) < 0.02:
            return "scale"
        if f in ("auto", "facecam"):
            return "fit" if src_ar / dst_ar > 1.6 else "fill"
        return "fill" if f in ("fill", "smart") else "fit"

    def _cover(self, p: QPainter, pm: QPixmap, r: QRectF, src: Optional[QRectF] = None, cx: float = 0.5) -> None:
        s = src or QRectF(0, 0, pm.width(), pm.height())
        k = max(r.width() / s.width(), r.height() / s.height())
        vw, vh = r.width() / k, r.height() / k
        x = s.x() + (s.width() - vw) * cx
        y = s.y() + (s.height() - vh) / 2
        p.drawPixmap(r, pm, QRectF(x, y, vw, vh))

    def paintEvent(self, e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        fw, fh = layout_for(self.aspect, self.width() - 2, self.height() - 2, self.custom, self.src_size)
        r = QRectF((self.width() - fw) / 2, (self.height() - fh) / 2, fw, fh)
        clip = QPainterPath(); clip.addRoundedRect(r, 12, 12)
        p.setClipPath(clip)
        p.fillRect(r, QColor("#16191F"))
        mode = self._mode(fw, fh)
        cap_pos = None
        hook_y = r.y() + fh * 0.09
        if self.frame is None:
            self._placeholder(p, r)
        elif mode == "fit":
            if self.blur is not None:
                self._cover(p, self.blur, r)
                p.fillRect(r, QColor(0, 0, 0, 70))
            sw, sh = self.src_size
            k = min(fw / sw, fh / sh)
            fgw, fgh = sw * k, sh * k
            fg = QRectF(r.x() + (fw - fgw) / 2, r.y() + (fh - fgh) / 2, fgw, fgh)
            p.drawPixmap(fg, self.frame, QRectF(0, 0, sw, sh))
            gap = r.bottom() - fg.bottom()
            if fgh < fh * 0.9:
                cap_pos = max(0.05, min(0.24, gap * 0.55 / fh))
                band_top = fg.y() - r.y()
                if band_top > fh * 0.14:
                    hook_y = r.y() + max(fh * 0.05, band_top * 0.35)
        elif mode == "facecam":
            sw, sh = self.src_size
            cam = self.cam or [0.70, 0.04, 0.28, 0.30]
            cw, ch = cam[2] * sw, cam[3] * sh
            top = min(max(fw * ch / max(cw, 1), fh * 0.25), fh * 0.45)
            self._cover(p, self.frame, QRectF(r.x(), r.y(), fw, top), QRectF(cam[0] * sw, cam[1] * sh, cw, ch))
            self._cover(p, self.frame, QRectF(r.x(), r.y() + top, fw, fh - top))
            p.fillRect(QRectF(r.x(), r.y() + top - 1, fw, 2), QColor(0, 0, 0, 120))
            cap_pos = max(0.1, (fh - top) / fh - 0.10)
            hook_y = r.y() + top + fh * 0.03
        else:
            self._cover(p, self.frame, r)
        if self.style_name != "Off":
            self._captions(p, r, cap_pos)
        if self.hook and (self.tick % 12) < 7:
            self._hook(p, r, hook_y)
        p.setClipping(False)
        p.setPen(QPen(QColor(theme.BORDER_HI), 1)); p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), 12, 12)
        p.end()

    def _placeholder(self, p: QPainter, r: QRectF) -> None:
        p.setPen(QPen(QColor(255, 255, 255, 14), 1))
        step = max(14.0, r.width() / 10)
        x = r.x() - r.height()
        while x < r.right():
            p.drawLine(QPointF(x, r.bottom()), QPointF(x + r.height(), r.y()))
            x += step
        f = QFont(self.font()); f.setPixelSize(11); p.setFont(f)
        p.setPen(QColor(theme.TEXT_3))
        p.drawText(QRectF(r.x(), r.y() + r.height() * 0.42, r.width(), 20), Qt.AlignmentFlag.AlignCenter,
                   "Add a video to preview on a real frame")

    def _text_path(self, text: str, font: QFont, x: float, y: float) -> QPainterPath:
        path = QPainterPath()
        path.addText(QPointF(x, y), font, text)
        return path

    def _captions(self, p: QPainter, r: QRectF, pos: Optional[float]) -> None:
        st: CaptionStyle = CAPTION_STYLES.get(self.style_name) or CAPTION_STYLES["Bold"]
        if not self.seq:
            return
        group = self.seq[self.g % len(self.seq)]
        H = r.height()
        size = max(9.0, H * st.size * ((r.width() / H) if r.width() > H else 1) ** 0.5)
        font = QFont(st.font.replace(" Semibold", ""))
        font.setPixelSize(int(size))
        font.setWeight(QFont.Weight.Black if "Black" in st.font else
                       QFont.Weight.DemiBold if "Semibold" in st.font else QFont.Weight.Bold)
        small = QFont(font); small.setPixelSize(max(7, int(size * 0.72)))
        fm, fms = QFontMetricsF(font), QFontMetricsF(small)
        pieces = []                                  # (text, font, is_bracket, word_index)
        for i, wd in enumerate(group):
            main, br = wd, ""
            if "(" in wd and wd.endswith(")"):
                main, br = wd[:wd.index("(")].strip(), wd[wd.index("("):]
            main = main.upper() if st.uppercase else main
            pieces.append((main, font, False, i))
            if br:
                pieces.append((br.upper() if st.uppercase else br, small, True, i))
        space = fm.horizontalAdvance(" ")
        widths = [(fm if not b else fms).horizontalAdvance(t) for t, _, b, _ in pieces]
        total = sum(widths) + space * (len(pieces) - 1)
        x = r.x() + (r.width() - total) / 2
        base_y = r.bottom() - H * (st.position if pos is None else pos)
        outline = max(1.5, size * st.outline)
        color, hl = QColor("#" + st.color), QColor("#" + st.highlight)
        for (t, f, br, wi), wdt in zip(pieces, widths):
            path = self._text_path(t, f, x, base_y)
            pen = QPen(QColor(0, 0, 0, 235), outline * 2 * (0.7 if br else 1.0))
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.strokePath(path, pen)
            if br:
                fill = QColor("#C8C8C8")
            elif wi == self.w and st.highlight.upper() != st.color.upper():
                fill = hl
            else:
                fill = color
            p.fillPath(path, fill)
            x += wdt + space

    def _hook(self, p: QPainter, r: QRectF, y: float) -> None:
        st: CaptionStyle = CAPTION_STYLES.get(self.style_name) or CAPTION_STYLES["Bold"]
        text = "1V4 CLUTCH" if self.lang != "spoken" else "क्या क्लच था"
        H = r.height()
        size = max(9.0, H * 0.052 * ((r.width() / H) if r.width() > H else 1) ** 0.5)
        font = QFont(st.font.replace(" Semibold", "")); font.setPixelSize(int(size)); font.setWeight(QFont.Weight.Black)
        fm = QFontMetricsF(font)
        tw = fm.horizontalAdvance(text)
        pad = size * 0.32
        box = QRectF(r.x() + (r.width() - tw) / 2 - pad, y, tw + 2 * pad, fm.height() + pad)
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(16, 16, 16, 175))
        p.drawRoundedRect(box, 6, 6)
        p.setPen(QColor("#FFFFFF")); p.setFont(font)
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
