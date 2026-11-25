"""Results grid: Shorts-style cards with hover preview, multi-select, sort and filter."""
from __future__ import annotations

import os
from typing import Callable, Optional

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QMenu, QMessageBox,
                               QVBoxLayout, QWidget)

from . import anim, theme
from .icons import icon, pixmap
from .widgets import Chips, FlowLayout, button, empty_state, label, open_path, reveal

CARD_W = 208
THUMB_H = int(CARD_W * 16 / 9)


def clip_path(clip: dict, folder: str, key: str = "file") -> str:
    f = clip.get(key, "") or ""
    if not f:
        return ""
    return f if os.path.isabs(f) else os.path.join(folder, f)


def as_upload(clip: dict, folder: str) -> dict:
    return {"path": clip_path(clip, folder), "title": clip.get("title", ""), "description": clip.get("description", ""),
            "hashtags": clip.get("hashtags") or []}


def sort_items(items: list, mode: str) -> list:
    if mode == "time":
        return sorted(items, key=lambda it: (it[0].get("source") == "compilation", float(it[0].get("start", 0))))
    return sorted(items, key=lambda it: (it[0].get("source") != "compilation",
                                         -(it[0].get("score") if it[0].get("score") is not None else -1)))


def filter_items(items: list, q: str) -> list:
    words = [w for w in (q or "").lower().split() if w]
    if not words:
        return list(items)
    out = []
    for c, f in items:
        hay = " ".join([c.get("title", ""), c.get("description", ""), c.get("reason", ""), c.get("hook", ""),
                        " ".join(c.get("hashtags") or []), " ".join(w[2] for w in c.get("words", [])[:400])]).lower()
        if all(w in hay for w in words):
            out.append((c, f))
    return out


def _fmt(sec: float) -> str:
    s = int(round(max(0.0, sec)))
    return f"{s // 60}:{s % 60:02d}"


class ThumbView(QWidget):
    """9:16 thumbnail with score pill, duration badge and a play mark; cycles hover frames."""
    clicked = Signal()

    def __init__(self, thumb: str, hover: list[str], duration: float, score, w: int = CARD_W, h: int = THUMB_H) -> None:
        super().__init__()
        self.setFixedSize(w, h)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.thumb_path, self.hover_paths = thumb, [p for p in hover if p]
        self.duration, self.score = duration, score
        self.frames: list[QPixmap] = []
        self.cur: Optional[QPixmap] = None
        self.k = 0
        self.hover = False
        self.base = QPixmap(thumb) if thumb and os.path.exists(thumb) else QPixmap()
        self.timer = QTimer(self, interval=420, timeout=self._next)

    def enterEvent(self, e) -> None:  # noqa: N802
        self.hover = True
        if self.hover_paths and not self.frames:
            self.frames = [QPixmap(p) for p in self.hover_paths if os.path.exists(p)]
            self.frames = [f for f in self.frames if not f.isNull()]
        if self.frames and anim.on():
            self.k = 0
            self.timer.start()
        self.update()

    def leaveEvent(self, e) -> None:  # noqa: N802
        self.hover = False
        self.timer.stop()
        self.cur = None
        self.update()

    def _next(self) -> None:
        if not self.frames:
            return
        self.cur = self.frames[self.k % len(self.frames)]
        self.k += 1
        self.update()

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

    def paintEvent(self, e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        r = QRectF(self.rect())
        path = QPainterPath()
        path.addRoundedRect(r, 10, 10)
        p.setClipPath(path)
        p.fillRect(r, QColor(theme.SURFACE_2))
        src = self.cur if (self.cur is not None and not self.cur.isNull()) else self.base
        if not src.isNull():
            dpr = self.devicePixelRatioF()
            s = src.scaled(self.size() * dpr, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                           Qt.TransformationMode.SmoothTransformation)
            s.setDevicePixelRatio(dpr)
            sw, sh = s.width() / dpr, s.height() / dpr
            p.drawPixmap(int((r.width() - sw) / 2), int((r.height() - sh) / 2), s)
        else:
            p.drawPixmap(int(r.center().x() - 14), int(r.center().y() - 14), pixmap("film", theme.TEXT_3, 28))
        # soft bottom shade so the badge and the play mark read on bright frames
        shade = QColor(0, 0, 0, 90 if self.hover else 60)
        p.fillRect(QRectF(0, r.height() - 46, r.width(), 46), shade)
        f = QFont(self.font()); f.setPixelSize(11); f.setWeight(QFont.Weight.DemiBold); p.setFont(f)
        txt = _fmt(self.duration)
        tw = p.fontMetrics().horizontalAdvance(txt) + 12
        badge = QRectF(r.right() - tw - 8, r.bottom() - 26, tw, 18)
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(0, 0, 0, 175)); p.drawRoundedRect(badge, 5, 5)
        p.setPen(QColor(theme.TEXT)); p.drawText(badge, Qt.AlignmentFlag.AlignCenter, txt)
        if self.score is not None:
            try:
                sc = float(self.score)
                col = QColor(theme.score_color(sc))
                st = f"{sc:.0f}"
                sw2 = p.fontMetrics().horizontalAdvance(st) + 16
                pr = QRectF(8, 8, sw2, 20)
                bg = QColor(12, 13, 16, 200)
                p.setBrush(bg); p.setPen(QPen(col, 1)); p.drawRoundedRect(pr, 10, 10)
                p.setPen(col); p.drawText(pr, Qt.AlignmentFlag.AlignCenter, st)
            except (TypeError, ValueError):
                pass
        if self.hover:
            c = r.center()
            ring = QColor(theme.ACCENT); ring.setAlpha(235)
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(ring)
            p.drawEllipse(QPointF(c.x(), c.y()), 22, 22)
            tri = QPainterPath(QPointF(c.x() - 6, c.y() - 9))
            tri.lineTo(c.x() + 10, c.y())
            tri.lineTo(c.x() - 6, c.y() + 9)
            tri.closeSubpath()
            p.setBrush(QColor(theme.ACCENT_TEXT)); p.drawPath(tri)
        p.end()


class ClipCard(QFrame):
    toggled = Signal(object, bool)
    opened = Signal(object)

    def __init__(self, clip: dict, folder: str, toast: Callable, upload: Optional[Callable] = None) -> None:
        super().__init__()
        self.setObjectName("card")
        self.setProperty("selected", False)
        self.clip, self.folder, self.toast, self.upload = clip, folder, toast, upload
        self.path = clip_path(clip, folder)
        self.setFixedWidth(CARD_W + 2)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(1, 1, 1, 10)
        lay.setSpacing(8)
        dur = float(clip.get("end", 0)) - float(clip.get("start", 0))
        hover = [clip_path({"f": h}, folder, "f") for h in (clip.get("hover") or [])]
        self.thumb = ThumbView(clip_path(clip, folder, "thumb"), hover, dur, clip.get("score"))
        self.thumb.clicked.connect(lambda: self.opened.emit(self))
        self.thumb.setToolTip("Watch and edit")
        lay.addWidget(self.thumb)
        self.check = QCheckBox(self.thumb)
        self.check.setObjectName("cardcheck")
        self.check.setCursor(Qt.CursorShape.PointingHandCursor)
        self.check.move(CARD_W - 30, 8)
        self.check.hide()
        self.check.toggled.connect(self._toggled)
        body = QVBoxLayout(); body.setContentsMargins(10, 0, 10, 0); body.setSpacing(3)
        t = QLabel(clip.get("title") or "Untitled")
        t.setWordWrap(True)
        t.setStyleSheet("font-weight: 600;")
        fm = t.fontMetrics()
        t.setMaximumHeight(fm.lineSpacing() * 2 + 2)
        t.setToolTip(clip.get("title") or "")
        body.addWidget(t)
        meta = []
        if clip.get("source") == "compilation":
            meta.append("Compilation")
        else:
            meta.append(f"at {_fmt(float(clip.get('start', 0)))}")
        if clip.get("uploads"):
            meta.append("uploaded")
        if clip.get("edited"):
            meta.append("edited")
        m = label("  ·  ".join(meta), "faint")
        body.addWidget(m)
        lay.addLayout(body)
        self.actions = QWidget()
        row = QHBoxLayout(self.actions); row.setContentsMargins(6, 0, 6, 0); row.setSpacing(2)
        row.addWidget(self._ib("play", "Watch and edit", lambda: self.opened.emit(self)))
        row.addWidget(self._ib("copy", "Copy title, description and hashtags", self._copy))
        if upload:
            row.addWidget(self._ib("upload", "Upload to YouTube", lambda: upload([as_upload(self.clip, self.folder)])))
        row.addWidget(self._ib("folder", "Show in folder", lambda: reveal(self.path)))
        row.addStretch(1)
        lay.addWidget(self.actions)
        self._fx_actions(False)
        if not os.path.exists(self.path):
            self.setEnabled(False)
            self.setToolTip("File was moved or deleted")

    def _ib(self, name: str, tip: str, on) -> QWidget:
        b = button("", "icon", name, tip, on)
        b.setIconSize(QSize(16, 16))
        return b

    def _fx_actions(self, on: bool) -> None:
        # keep the space so the grid doesn't jump; only hide the buttons
        for i in range(self.actions.layout().count()):
            w = self.actions.layout().itemAt(i).widget()
            if w:
                w.setVisible(True)
        self.actions.setEnabled(True)
        eff_opacity = 1.0 if on else 0.0
        from PySide6.QtWidgets import QGraphicsOpacityEffect
        eff = self.actions.graphicsEffect()
        if eff is None:
            eff = QGraphicsOpacityEffect(self.actions)
            self.actions.setGraphicsEffect(eff)
        eff.setOpacity(eff_opacity)

    def enterEvent(self, e) -> None:  # noqa: N802
        self.check.show()
        self._fx_actions(True)

    def leaveEvent(self, e) -> None:  # noqa: N802
        if not self.check.isChecked() and not getattr(self, "select_mode", False):
            self.check.hide()
        self._fx_actions(False)

    def set_select_mode(self, on: bool) -> None:
        self.select_mode = on
        self.check.setVisible(on or self.check.isChecked() or self.underMouse())

    def _toggled(self, on: bool) -> None:
        self.setProperty("selected", on)
        self.style().unpolish(self); self.style().polish(self)
        self.toggled.emit(self, on)

    def _copy(self) -> None:
        c = self.clip
        text = "\n\n".join(x for x in [c.get("title", ""), c.get("description", ""),
                                         " ".join(c.get("hashtags") or [])] if x)
        QApplication.clipboard().setText(text)
        self.toast("Copied")


class SkeletonCard(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(CARD_W + 2, THUMB_H + 70)
        self.phase = 0.0
        self.t = QTimer(self, interval=60, timeout=self._tick)
        if anim.on():
            self.t.start()

    def _tick(self) -> None:
        self.phase = (self.phase + 0.04) % 1.0
        self.update()

    def paintEvent(self, e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        base = QColor(theme.SURFACE)
        hi = QColor(theme.SURFACE_2)
        k = 0.5 + 0.5 * abs(self.phase * 2 - 1)
        col = QColor(int(base.red() + (hi.red() - base.red()) * k), int(base.green() + (hi.green() - base.green()) * k),
                     int(base.blue() + (hi.blue() - base.blue()) * k))
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(col)
        p.drawRoundedRect(QRectF(1, 1, CARD_W, THUMB_H), 10, 10)
        p.drawRoundedRect(QRectF(11, THUMB_H + 14, CARD_W * 0.8, 12), 6, 6)
        p.drawRoundedRect(QRectF(11, THUMB_H + 34, CARD_W * 0.45, 10), 5, 5)
        p.end()


class ClipGrid(QWidget):
    """items: list of (clip dict, job folder)."""
    open_clip = Signal(object, str, list)        # clip, folder, all (clip, folder) in view order
    changed = Signal()

    def __init__(self, toast: Callable, upload: Optional[Callable] = None, toolbar: bool = True,
                 empty_text: str = "No clips yet.") -> None:
        super().__init__()
        self.toast, self.upload = toast, upload
        self.items: list = []
        self.cards: list[ClipCard] = []
        self.selected: list[ClipCard] = []
        self.empty_text = empty_text
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(14)
        self.bar = QWidget()
        bl = QHBoxLayout(self.bar); bl.setContentsMargins(0, 0, 0, 0); bl.setSpacing(10)
        self.sort = Chips([("best", "Best first"), ("time", "Timeline")], wrap=False)
        self.sort.set_value("best")
        self.sort.changed.connect(lambda _: self._render())
        self.filter = QLineEdit(); self.filter.setObjectName("search"); self.filter.setPlaceholderText("Filter clips")
        self.filter.setClearButtonEnabled(True); self.filter.setMaximumWidth(260)
        self.filter.textChanged.connect(lambda _: self._render())
        bl.addWidget(self.sort); bl.addStretch(1); bl.addWidget(self.filter)
        v.addWidget(self.bar)
        self.bar.setVisible(toolbar)
        self.sel = QFrame(); self.sel.setObjectName("selbar")
        sl = QHBoxLayout(self.sel); sl.setContentsMargins(14, 8, 8, 8); sl.setSpacing(6)
        self.sel_lbl = label("", "h3"); sl.addWidget(self.sel_lbl); sl.addStretch(1)
        if upload:
            sl.addWidget(button("Upload selected", "ghost", "upload", on=self._upload_sel))
        sl.addWidget(button("Open folder", "ghost", "folder", on=self._open_sel))
        sl.addWidget(button("Delete", "ghost", "trash", on=self._delete_sel))
        sl.addWidget(button("Clear", "ghost", "close", on=self.clear_selection))
        v.addWidget(self.sel); self.sel.hide()
        self.host = QWidget()
        self.flow = FlowLayout(self.host, spacing=16)
        v.addWidget(self.host)

    # ---------------- data
    def set_items(self, items: list, animate: bool = True) -> None:
        self.items = list(items)
        self._render(animate)

    def add_item(self, clip: dict, folder: str) -> None:
        self.items.append((clip, folder))
        self._render(False)
        if self.cards:
            anim.fade_in(self.cards[-1], 220)

    def show_skeleton(self, n: int) -> None:
        self._clear()
        for _ in range(max(1, min(n, 10))):
            self.flow.addWidget(SkeletonCard())
        self.host.updateGeometry()

    def view_items(self) -> list:
        return filter_items(sort_items(self.items, self.sort.value() or "best"), self.filter.text())

    def _clear(self) -> None:
        while self.flow.count():
            it = self.flow.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.cards, self.selected = [], []
        self._sel_changed()

    def _render(self, animate: bool = False) -> None:
        self._clear()
        view = self.view_items()
        if not view:
            self.flow.addWidget(empty_state("film", "No clips match." if self.filter.text().strip() else self.empty_text))
        for k, (c, f) in enumerate(view):
            card = ClipCard(c, f, self.toast, self.upload)
            card.opened.connect(lambda cd, view=view: self.open_clip.emit(cd.clip, cd.folder, view))
            card.toggled.connect(self._card_toggled)
            self.flow.addWidget(card)
            self.cards.append(card)
            if animate:
                anim.fade_in(card, 200, min(400, 45 * k))
        self.host.updateGeometry()

    # ---------------- selection
    def _card_toggled(self, card: ClipCard, on: bool) -> None:
        if on and card not in self.selected:
            self.selected.append(card)
        elif not on and card in self.selected:
            self.selected.remove(card)
        self._sel_changed()

    def _sel_changed(self) -> None:
        n = len(self.selected)
        self.sel_lbl.setText(f"{n} selected")
        if n and not self.sel.isVisible():
            self.sel.show(); anim.fade_in(self.sel, 140)
        elif not n:
            self.sel.hide()
        for c in self.cards:
            c.set_select_mode(bool(n))

    def clear_selection(self) -> None:
        for c in list(self.selected):
            c.check.setChecked(False)

    def _upload_sel(self) -> None:
        if self.upload and self.selected:
            self.upload([as_upload(c.clip, c.folder) for c in self.selected])

    def _open_sel(self) -> None:
        folders = []
        for c in self.selected:
            if c.folder not in folders:
                folders.append(c.folder)
        for f in folders[:3]:
            open_path(f)

    def _delete_sel(self) -> None:
        n = len(self.selected)
        if not n:
            return
        if QMessageBox.question(self, "Delete clips", f"Delete {n} clip{'s' if n != 1 else ''} from your PC? "
                                "The original video isn't touched.") != QMessageBox.StandardButton.Yes:
            return
        from ..core.editing import delete_clip
        gone = 0
        for c in list(self.selected):
            try:
                delete_clip(c.folder, os.path.basename(c.path))
                self.items = [it for it in self.items if it[0] is not c.clip]
                gone += 1
            except Exception as exc:
                self.toast(f"Couldn't delete {os.path.basename(c.path)}: {exc}", 5000, True)
        self._render()
        self.changed.emit()
        if gone:
            self.toast(f"Deleted {gone} clip{'s' if gone != 1 else ''}")
