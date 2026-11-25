from __future__ import annotations

import os
from typing import Callable, Iterable, Optional

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QApplication, QButtonGroup, QFrame, QGraphicsOpacityEffect, QHBoxLayout,
                               QLabel, QLayout, QPushButton, QSizePolicy, QVBoxLayout, QWidget)

from . import theme
from .icons import icon, pixmap


class SectionLabel(QLabel):
    """Small uppercase heading with an accent dot in front (repaints with the current accent)."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setObjectName("section")
        self.setContentsMargins(13, 0, 0, 0)

    def paintEvent(self, e) -> None:  # noqa: N802
        super().paintEvent(e)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(theme.ACCENT))
        p.drawEllipse(QPointF(3.5, self.height() / 2), 3.0, 3.0)
        p.end()


def label(text: str, obj: str = "", wrap: bool = False) -> QLabel:
    if obj == "section":
        return SectionLabel(text)
    l = QLabel(text)
    if obj:
        l.setObjectName(obj)
    l.setWordWrap(wrap)
    return l


def pill(text: str, color: str, tip: str = "") -> QLabel:
    """Small rounded badge: tinted background, colored text."""
    l = QLabel(text)
    l.setStyleSheet(f"background: {theme.rgba(color, 0.16)}; color: {color}; border: 1px solid {theme.rgba(color, 0.35)};"
                    f"border-radius: 9px; padding: 1px 8px; font-size: 12px; font-weight: 700;")
    if tip:
        l.setToolTip(tip)
    return l


def dot_text(color: str, text: str) -> str:
    """Rich text: a colored dot, then text."""
    return f'<span style="color:{color}">&#9679;</span>&nbsp; {text}'


def hline() -> QFrame:
    f = QFrame()
    f.setObjectName("line")
    return f


def button(text: str, obj: str = "", ic: str = "", tip: str = "", on: Optional[Callable] = None) -> QPushButton:
    b = QPushButton(text)
    if obj:
        b.setObjectName(obj)
    if ic:
        b.setIcon(icon(ic))
        b.setIconSize(QSize(16, 16))
    if tip:
        b.setToolTip(tip)
    if on:
        b.clicked.connect(on)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


def open_path(path: str) -> None:
    if path and os.path.exists(path):
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))


def reveal(path: str) -> None:
    """Open Explorer with the file selected."""
    if not path:
        return
    if os.name == "nt" and os.path.isfile(path):
        import subprocess
        subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
    else:
        open_path(os.path.dirname(path) if os.path.isfile(path) else path)


class FlowLayout(QLayout):
    """Wrapping row layout for chips."""

    def __init__(self, parent=None, spacing: int = 6) -> None:
        super().__init__(parent)
        self._items = []
        self._sp = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:  # noqa: N802
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, i):  # noqa: N802
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):  # noqa: N802
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, w: int) -> int:  # noqa: N802
        return self._do(QRectF(0, 0, w, 0).toRect(), True)

    def setGeometry(self, rect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._do(rect, False)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        s = QSize()
        for it in self._items:
            s = s.expandedTo(it.minimumSize())
        return s

    def _do(self, rect, test: bool) -> int:
        x, y, line_h = rect.x(), rect.y(), 0
        for it in self._items:
            if it.widget() is not None and not it.widget().isVisible() and not test:
                pass
            hint = it.sizeHint()
            nx = x + hint.width() + self._sp
            if nx - self._sp > rect.right() + 1 and line_h > 0:
                x = rect.x()
                y += line_h + self._sp
                nx = x + hint.width() + self._sp
                line_h = 0
            if not test:
                it.setGeometry(QRectF(x, y, hint.width(), hint.height()).toRect())
            x = nx
            line_h = max(line_h, hint.height())
        return y + line_h - rect.y()


class Chips(QWidget):
    """Single-select chip row. Emits the selected value."""
    changed = Signal(object)

    def __init__(self, options: Iterable = (), wrap: bool = True, parent=None) -> None:
        super().__init__(parent)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.lay = FlowLayout(self) if wrap else QHBoxLayout(self)
        if not wrap:
            self.lay.setContentsMargins(0, 0, 0, 0)
            self.lay.setSpacing(6)
        self.values: list = []
        self.group.idClicked.connect(lambda i: self.changed.emit(self.values[i]))
        self.set_options(options)

    def set_options(self, options: Iterable, keep: bool = True) -> None:
        cur = self.value() if keep else None
        for b in self.group.buttons():
            self.group.removeButton(b)
            b.deleteLater()
        while self.lay.count():
            self.lay.takeAt(0)
        self.values = []
        for i, opt in enumerate(options):
            if isinstance(opt, tuple):
                value, text, tip = (opt + ("",))[:3]
            else:
                value, text, tip = opt, str(opt), ""
            b = QPushButton(text)
            b.setObjectName("chip")
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            if tip:
                b.setToolTip(tip)
            self.group.addButton(b, i)
            self.lay.addWidget(b)
            self.values.append(value)
        if cur is not None and cur in self.values:
            self.set_value(cur)
        self.updateGeometry()

    def value(self):
        i = self.group.checkedId()
        return self.values[i] if 0 <= i < len(self.values) else None

    def set_value(self, v, emit: bool = False) -> None:
        if v in self.values:
            b = self.group.button(self.values.index(v))
            b.setChecked(True)
            if emit:
                self.changed.emit(v)
        else:
            chk = self.group.checkedButton()
            if chk:
                self.group.setExclusive(False)
                chk.setChecked(False)
                self.group.setExclusive(True)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return isinstance(self.lay, FlowLayout)

    def heightForWidth(self, w: int) -> int:  # noqa: N802
        return self.lay.heightForWidth(w) if isinstance(self.lay, FlowLayout) else super().heightForWidth(w)


class Field(QWidget):
    """Label on the left/top, control on the right/bottom."""

    def __init__(self, title: str, control: QWidget, hint: str = "", vertical: bool = True) -> None:
        super().__init__()
        lay = QVBoxLayout(self) if vertical else QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(7 if vertical else 12)
        t = label(title, "muted")
        if not vertical:
            t.setMinimumWidth(150)
        lay.addWidget(t)
        lay.addWidget(control, 1)
        if hint:
            h = label(hint, "faint", wrap=True)
            lay.addWidget(h)


class DropZone(QFrame):
    """The big target. Click to browse; drop handled app-wide by MainWindow."""
    clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(176)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.active = False
        self.file_text = ""
        self.sub_text = ""

    def set_file(self, title: str, sub: str = "") -> None:
        self.file_text, self.sub_text = title, sub
        self.update()

    def set_active(self, on: bool) -> None:
        self.active = on
        self.update()

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

    def enterEvent(self, e) -> None:  # noqa: N802
        self._hover = True
        self.update()

    def leaveEvent(self, e) -> None:  # noqa: N802
        self._hover = False
        self.update()

    def paintEvent(self, e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        hover = getattr(self, "_hover", False)
        acc = QColor(theme.ACCENT)
        if self.active:
            fill = QColor(acc); fill.setAlpha(34)
        elif hover:
            fill = QColor(acc); fill.setAlpha(16)
        else:
            fill = QColor(theme.SURFACE)
        p.setBrush(fill)
        if self.active:
            line = acc
        elif hover:
            line = QColor(acc); line.setAlpha(190)
        else:
            line = QColor(theme.BORDER_HI) if self.file_text else QColor(acc.red(), acc.green(), acc.blue(), 110)
        pen = QPen(line, 1.6 if (self.active or hover) else 1.3)
        if not self.file_text:
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setDashPattern([5, 4])
        p.setPen(pen)
        p.drawRoundedRect(r, 12, 12)
        cx, cy = r.center().x(), r.center().y()
        if self.file_text:
            f = QFont(self.font()); f.setPixelSize(15); f.setWeight(QFont.Weight.DemiBold)
            p.setFont(f); p.setPen(QColor(theme.TEXT))
            p.drawText(QRectF(24, cy - 26, r.width() - 48, 24), Qt.AlignmentFlag.AlignCenter,
                       p.fontMetrics().elidedText(self.file_text, Qt.TextElideMode.ElideMiddle, int(r.width() - 60)))
            f.setPixelSize(12); f.setWeight(QFont.Weight.Normal); p.setFont(f); p.setPen(QColor(theme.TEXT_2))
            p.drawText(QRectF(24, cy + 2, r.width() - 48, 20), Qt.AlignmentFlag.AlignCenter, self.sub_text)
            p.setPen(QColor(theme.TEXT_3))
            p.drawText(QRectF(24, cy + 24, r.width() - 48, 20), Qt.AlignmentFlag.AlignCenter,
                       "Click or drop to replace")
        else:
            ring = QColor(theme.ACCENT); ring.setAlpha(40 if (self.active or hover) else 26)
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(ring)
            p.drawEllipse(QPointF(cx, cy - 32), 22, 22)
            p.drawPixmap(int(cx - 12), int(cy - 44), pixmap("upload", theme.ACCENT, 24))
            f = QFont(self.font()); f.setPixelSize(15); f.setWeight(QFont.Weight.DemiBold)
            p.setFont(f); p.setPen(QColor(theme.TEXT))
            p.drawText(QRectF(0, cy - 10, r.width(), 24), Qt.AlignmentFlag.AlignCenter,
                       "Release to add" if self.active else "Drop a video here")
            f.setPixelSize(12); f.setWeight(QFont.Weight.Normal); p.setFont(f); p.setPen(QColor(theme.TEXT_3))
            p.drawText(QRectF(0, cy + 16, r.width(), 20), Qt.AlignmentFlag.AlignCenter,
                       "or click to browse  ·  MP4, MKV, MOV, WEBM")
        p.end()


class Thumb(QLabel):
    """Rounded thumbnail with duration badge."""

    def __init__(self, path: str, duration: float, w: int = 200, h: int = 356) -> None:
        super().__init__()
        self.setFixedSize(w, h)
        self.src = QPixmap(path) if path and os.path.exists(path) else QPixmap()
        self.duration = duration

    def paintEvent(self, e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        r = QRectF(self.rect())
        path = QPainterPath()
        path.addRoundedRect(r, 7, 7)
        p.setClipPath(path)
        p.fillRect(r, QColor(theme.SURFACE_2))
        if not self.src.isNull():
            s = self.src.scaled(self.size() * self.devicePixelRatioF(), Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                Qt.TransformationMode.SmoothTransformation)
            s.setDevicePixelRatio(self.devicePixelRatioF())
            sw, sh = s.width() / s.devicePixelRatio(), s.height() / s.devicePixelRatio()
            p.drawPixmap(int((r.width() - sw) / 2), int((r.height() - sh) / 2), s)
        m, sec = divmod(int(round(self.duration)), 60)
        txt = f"{m}:{sec:02d}"
        f = QFont(self.font()); f.setPixelSize(11); f.setWeight(QFont.Weight.DemiBold); p.setFont(f)
        tw = p.fontMetrics().horizontalAdvance(txt) + 12
        badge = QRectF(r.right() - tw - 8, r.bottom() - 26, tw, 18)
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(0, 0, 0, 170)); p.drawRoundedRect(badge, 4, 4)
        p.setPen(QColor(theme.TEXT)); p.drawText(badge, Qt.AlignmentFlag.AlignCenter, txt)
        p.end()


class ClipCard(QFrame):
    upload_handler: Optional[Callable[[list], None]] = None     # set by MainWindow

    def __init__(self, clip: dict, folder: str, on_toast: Callable[[str], None]) -> None:
        super().__init__()
        self.setObjectName("card")
        self.clip = clip
        file = clip.get("file", "")
        self.path = file if os.path.isabs(file) else os.path.join(folder, file)
        thumb = clip.get("thumb", "")
        thumb = thumb if (not thumb or os.path.isabs(thumb)) else os.path.join(folder, thumb)
        self.setFixedWidth(226)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)
        th = Thumb(thumb, float(clip.get("end", 0)) - float(clip.get("start", 0)), 200, 300)
        th.setCursor(Qt.CursorShape.PointingHandCursor)
        th.mouseReleaseEvent = lambda e: open_path(self.path)
        th.setToolTip("Play")
        lay.addWidget(th)
        top = QHBoxLayout(); top.setSpacing(6)
        score = clip.get("score")
        if score is not None:
            top.addWidget(pill(f"{float(score):.0f}", theme.score_color(score), self._score_tip()))
        st = int(float(clip.get("start", 0)))
        top.addWidget(label(f"at {st // 60}:{st % 60:02d}", "faint"))
        top.addStretch(1)
        lay.addLayout(top)
        t = label(clip.get("title") or "Untitled", wrap=True)
        t.setStyleSheet("font-weight: 600;")
        t.setMaximumHeight(40)
        lay.addWidget(t)
        if clip.get("reason"):
            r = label(clip["reason"], "faint", wrap=True)
            r.setMaximumHeight(48)
            lay.addWidget(r)
        lay.addStretch(1)
        row = QHBoxLayout(); row.setSpacing(4)
        row.addWidget(button("Play", "ghost", "play", on=lambda: open_path(self.path)))
        row.addWidget(button("", "ghost", "folder", "Show in folder", on=lambda: reveal(self.path)))
        row.addWidget(button("", "ghost", "copy", "Copy title, description and hashtags",
                             on=lambda: self._copy(on_toast)))
        if ClipCard.upload_handler:
            row.addWidget(button("", "ghost", "upload", "Upload to YouTube",
                                 on=lambda: ClipCard.upload_handler([self.as_upload()])))
        row.addStretch(1)
        lay.addLayout(row)
        if not os.path.exists(self.path):
            self.setEnabled(False)
            self.setToolTip("File was moved or deleted")

    def as_upload(self) -> dict:
        return {"path": self.path, "title": self.clip.get("title", ""), "description": self.clip.get("description", ""),
                "hashtags": self.clip.get("hashtags") or []}

    def _score_tip(self) -> str:
        c = self.clip
        parts = [f"Score {float(c.get('score', 0)):.0f} / 100"]
        if c.get("signal_score") is not None:
            parts.append(f"Signals {float(c['signal_score']):.0f}")
        if c.get("llm_score") is not None:
            parts.append(f"AI {float(c['llm_score']):.0f}")
        return "  ·  ".join(parts)

    def _copy(self, toast: Callable[[str], None]) -> None:
        c = self.clip
        text = "\n\n".join(x for x in [c.get("title", ""), c.get("description", ""),
                                         " ".join(c.get("hashtags") or [])] if x)
        QApplication.clipboard().setText(text)
        toast("Copied")


class Toast(QLabel):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._style(False)
        self.hide()
        self._t = QTimer(self, singleShot=True, timeout=self.hide)

    def _style(self, error: bool) -> None:
        edge = theme.RED if error else theme.ACCENT
        self.setStyleSheet(f"background: {theme.SURFACE_2}; border: 1px solid {theme.BORDER_HI};"
                           f"border-left: 3px solid {edge}; border-radius: 7px; padding: 9px 15px;"
                           f"color: {theme.RED if error else theme.TEXT};")

    def show_text(self, text: str, ms: int = 2600, error: bool = False) -> None:
        self.setText(text)
        self._style(error)
        self.adjustSize()
        pw = self.parentWidget()
        self.move((pw.width() - self.width()) // 2, pw.height() - self.height() - 28)
        self.raise_()
        self.show()
        self._t.start(ms)


class Steps(QWidget):
    """Numbered stage tracker: done = check, current = filled, skipped = dash, failed = red."""

    def __init__(self, names: list[str]) -> None:
        super().__init__()
        self.names = names
        self.current = -1
        self.skipped: set[int] = set()
        self.failed = False
        self.setMinimumHeight(26)

    def reset(self) -> None:
        self.current, self.skipped, self.failed = -1, set(), False
        self.update()

    def set_current(self, i: int) -> None:
        self.current = i
        self.update()

    def skip(self, i: int) -> None:
        self.skipped.add(i)
        self.update()

    def paintEvent(self, e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        n = len(self.names)
        slot = self.width() / n
        r, cy = 9.0, 13.0
        for i, name in enumerate(self.names):
            x = i * slot
            cur = i == self.current
            skipped = i in self.skipped and i <= self.current and not (cur and not self.current >= n)
            done = i < self.current and i not in self.skipped
            f = QFont(self.font()); f.setPixelSize(12)
            f.setWeight(QFont.Weight.DemiBold if cur else QFont.Weight.Normal)
            p.setFont(f)
            fm = p.fontMetrics()
            avail = int(slot - 2 * r - 20)
            text = fm.elidedText(name, Qt.TextElideMode.ElideRight, max(avail, 10))
            tw = fm.horizontalAdvance(text)
            if i < n - 1:
                x1, x2 = x + 2 * r + 8 + tw + 8, (i + 1) * slot - 6
                if x2 > x1:
                    lc = QColor(theme.TEAL) if i < self.current else QColor(theme.BORDER)
                    if i < self.current:
                        lc.setAlpha(120)
                    p.setPen(QPen(lc, 1.2))
                    p.drawLine(QPointF(x1, cy), QPointF(x2, cy))
            c = QRectF(x + 1, cy - r, 2 * r, 2 * r)
            nf = QFont(self.font()); nf.setPixelSize(10); nf.setWeight(QFont.Weight.DemiBold)
            if cur and self.failed:
                p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(theme.RED)); p.drawEllipse(c)
                p.setPen(QColor(theme.BG)); p.setFont(nf); p.drawText(c, Qt.AlignmentFlag.AlignCenter, "!")
            elif cur and i not in self.skipped:
                p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(theme.ACCENT)); p.drawEllipse(c)
                p.setPen(QColor(theme.ACCENT_TEXT)); p.setFont(nf); p.drawText(c, Qt.AlignmentFlag.AlignCenter, str(i + 1))
            elif i in self.skipped and i <= self.current:
                pen = QPen(QColor(theme.TEXT_3), 1); pen.setStyle(Qt.PenStyle.DotLine)
                p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush); p.drawEllipse(c)
                p.setPen(QColor(theme.TEXT_3)); p.setFont(nf); p.drawText(c, Qt.AlignmentFlag.AlignCenter, "–")
            elif done:
                tc = QColor(theme.TEAL); tc.setAlpha(46)
                p.setPen(QPen(QColor(theme.TEAL), 1)); p.setBrush(tc); p.drawEllipse(c)
                pen = QPen(QColor(theme.TEAL), 1.7); pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.setPen(pen)
                p.drawLine(QPointF(c.x() + 5.5, cy + 0.3), QPointF(c.x() + 8, cy + 2.8))
                p.drawLine(QPointF(c.x() + 8, cy + 2.8), QPointF(c.x() + 12.8, cy - 2.8))
            else:
                p.setPen(QPen(QColor(theme.BORDER_HI), 1)); p.setBrush(Qt.BrushStyle.NoBrush); p.drawEllipse(c)
                p.setPen(QColor(theme.TEXT_3)); p.setFont(nf); p.drawText(c, Qt.AlignmentFlag.AlignCenter, str(i + 1))
            p.setFont(f)
            p.setPen(QColor(theme.TEXT if cur else (theme.TEXT_2 if done else theme.TEXT_3)))
            p.drawText(QRectF(x + 2 * r + 8, 0, tw + 2, 2 * cy), Qt.AlignmentFlag.AlignVCenter, text)
        p.end()
