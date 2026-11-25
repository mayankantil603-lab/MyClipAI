"""In-app player: watch, trim, fix captions, change the hook, move the facecam box,
compare Whisper with the Hindi model. Edits re-make just that one clip."""
from __future__ import annotations

import os
from typing import Callable, Optional

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QDialog, QFrame, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QSizePolicy,
                               QSlider, QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit, QVBoxLayout, QWidget)

from ..core import editing, helpers
from . import theme, workers
from .grid import as_upload, clip_path
from .icons import icon
from .widgets import button, label, open_path, pill, reveal

TEXT_INPUTS = (QLineEdit, QTextEdit, QPlainTextEdit)


def fmt(ms: float) -> str:
    s = max(0, int(ms // 1000))
    return f"{s // 60}:{s % 60:02d}.{int(ms % 1000) // 100}"


class FacecamEditor(QWidget):
    """A frame of the original video with a box you drag (move) or pull by its corners (resize)."""
    changed = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(320, 180)
        self.pm: Optional[QPixmap] = None
        self.box = [0.70, 0.04, 0.28, 0.30]
        self.drag: Optional[str] = None
        self.start = QPointF()
        self.orig = list(self.box)
        self.setMouseTracking(True)

    def set_frame(self, data: bytes) -> None:
        pm = QPixmap()
        if data and pm.loadFromData(data):
            self.pm = pm
        self.update()

    def set_box(self, box: Optional[list]) -> None:
        if box and len(box) == 4:
            self.box = [float(x) for x in box]
        self.update()

    def _frame_rect(self) -> QRectF:
        if self.pm is None:
            return QRectF(0, 0, self.width(), self.width() * 9 / 16)
        k = min(self.width() / self.pm.width(), self.height() / self.pm.height())
        w, h = self.pm.width() * k, self.pm.height() * k
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def _box_rect(self) -> QRectF:
        f = self._frame_rect()
        x, y, w, h = self.box
        return QRectF(f.x() + x * f.width(), f.y() + y * f.height(), w * f.width(), h * f.height())

    def _hit(self, pos: QPointF) -> Optional[str]:
        b = self._box_rect()
        for name, pt in (("tl", b.topLeft()), ("tr", b.topRight()), ("bl", b.bottomLeft()), ("br", b.bottomRight())):
            if (pos - pt).manhattanLength() < 14:
                return name
        return "move" if b.contains(pos) else None

    def mousePressEvent(self, e) -> None:  # noqa: N802
        self.drag = self._hit(e.position())
        self.start = e.position()
        self.orig = list(self.box)

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        h = self._hit(e.position())
        self.setCursor(Qt.CursorShape.SizeAllCursor if h == "move" else
                       Qt.CursorShape.SizeFDiagCursor if h in ("tl", "br") else
                       Qt.CursorShape.SizeBDiagCursor if h in ("tr", "bl") else Qt.CursorShape.ArrowCursor)
        if not self.drag:
            return
        f = self._frame_rect()
        dx = (e.position().x() - self.start.x()) / max(f.width(), 1)
        dy = (e.position().y() - self.start.y()) / max(f.height(), 1)
        x, y, w, h0 = self.orig
        if self.drag == "move":
            x, y = x + dx, y + dy
        else:
            if "l" in self.drag:
                x, w = x + dx, w - dx
            if "r" in self.drag:
                w = w + dx
            if "t" in self.drag:
                y, h0 = y + dy, h0 - dy
            if "b" in self.drag:
                h0 = h0 + dy
        w, h0 = max(0.06, min(1.0, w)), max(0.06, min(1.0, h0))
        x, y = max(0.0, min(1.0 - w, x)), max(0.0, min(1.0 - h0, y))
        self.box = [round(x, 4), round(y, 4), round(w, 4), round(h0, 4)]
        self.update()

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if self.drag:
            self.changed.emit(list(self.box))
        self.drag = None

    def paintEvent(self, e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        f = self._frame_rect()
        p.fillRect(f, QColor(theme.SURFACE_2))
        if self.pm is not None:
            p.drawPixmap(f, self.pm, QRectF(self.pm.rect()))
        b = self._box_rect()
        shade = QColor(0, 0, 0, 120)
        p.fillRect(QRectF(f.x(), f.y(), f.width(), b.y() - f.y()), shade)
        p.fillRect(QRectF(f.x(), b.bottom(), f.width(), f.bottom() - b.bottom()), shade)
        p.fillRect(QRectF(f.x(), b.y(), b.x() - f.x(), b.height()), shade)
        p.fillRect(QRectF(b.right(), b.y(), f.right() - b.right(), b.height()), shade)
        p.setPen(QPen(QColor(theme.ACCENT), 2)); p.setBrush(Qt.BrushStyle.NoBrush); p.drawRect(b)
        p.setBrush(QColor(theme.ACCENT))
        for pt in (b.topLeft(), b.topRight(), b.bottomLeft(), b.bottomRight()):
            p.drawEllipse(pt, 5, 5)
        p.end()


class PlayerDialog(QDialog):
    changed = Signal(str)                    # job folder whose clips.json changed

    def __init__(self, parent, clip: dict, folder: str, siblings: list, toast: Callable,
                 upload: Optional[Callable] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("MyClip AI")
        self.resize(1180, 780)
        self.setMinimumSize(980, 640)
        self.toast, self.upload = toast, upload
        self.items = [it for it in siblings if os.path.exists(clip_path(it[0], it[1]))] or [(clip, folder)]
        self.idx = next((i for i, (c, f) in enumerate(self.items) if c is clip), 0)
        self.mark_a: Optional[float] = None
        self.mark_b: Optional[float] = None
        self.busy: Optional[workers.TaskWorker] = None
        self.player = None
        root = QHBoxLayout(self); root.setContentsMargins(18, 18, 18, 18); root.setSpacing(20)

        # ---------------- left: video + controls
        left = QVBoxLayout(); left.setSpacing(10)
        self.video_host = QFrame(); self.video_host.setObjectName("soft")
        vh = QVBoxLayout(self.video_host); vh.setContentsMargins(0, 0, 0, 0)
        try:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
            from PySide6.QtMultimediaWidgets import QVideoWidget
            self.video = QVideoWidget()
            self.video.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.audio = QAudioOutput(self)
            self.player = QMediaPlayer(self)
            self.player.setAudioOutput(self.audio)
            self.player.setVideoOutput(self.video)
            self.player.positionChanged.connect(self._pos)
            self.player.durationChanged.connect(self._dur)
            self.player.playbackStateChanged.connect(self._state)
            self.player.errorOccurred.connect(self._error)
            vh.addWidget(self.video)
        except Exception as exc:                  # QtMultimedia missing: system player instead
            print(f"[MyClip] in-app player unavailable: {exc}", flush=True)
            self.video = label("The built-in player isn't available on this PC. Clips open in your video app.",
                               "muted", wrap=True)
            self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
            vh.addWidget(self.video)
        left.addWidget(self.video_host, 1)
        self.seek = QSlider(Qt.Orientation.Horizontal); self.seek.setRange(0, 0)
        self.seek.sliderMoved.connect(lambda v: self.player and self.player.setPosition(v))
        self.seek.sliderPressed.connect(lambda: self.player and self.player.setPosition(self.seek.value()))
        left.addWidget(self.seek)
        ctl = QHBoxLayout(); ctl.setSpacing(4)
        self.prev_btn = button("", "icon", "prev", "Previous clip (↑)", self.prev_clip)
        self.play_btn = button("", "icon", "play", "Play / pause (Space)", self.toggle)
        self.next_btn = button("", "icon", "next", "Next clip (↓)", self.next_clip)
        for b_ in (self.prev_btn, self.play_btn, self.next_btn):
            b_.setIconSize(QSize(18, 18)); ctl.addWidget(b_)
        self.time_lbl = label("0:00.0 / 0:00.0", "muted"); ctl.addSpacing(8); ctl.addWidget(self.time_lbl)
        ctl.addStretch(1)
        ctl.addWidget(button("", "icon", "volume", "Volume"))
        self.vol = QSlider(Qt.Orientation.Horizontal); self.vol.setRange(0, 100); self.vol.setValue(80)
        self.vol.setFixedWidth(110)
        self.vol.valueChanged.connect(lambda v: getattr(self, "audio", None) and self.audio.setVolume(v / 100))
        ctl.addWidget(self.vol)
        left.addLayout(ctl)
        trim = QFrame(); trim.setObjectName("soft")
        tl = QHBoxLayout(trim); tl.setContentsMargins(12, 8, 8, 8); tl.setSpacing(6)
        tl.addWidget(button("", "icon", "scissors", "Trim"))
        tl.addWidget(button("Set start", "ghost", tip="[", on=lambda: self._mark("a")))
        tl.addWidget(button("Set end", "ghost", tip="]", on=lambda: self._mark("b")))
        self.trim_lbl = label("Trim: whole clip", "faint"); tl.addWidget(self.trim_lbl, 1)
        self.trim_btn = button("Save trimmed copy", on=self.save_trim); self.trim_btn.setEnabled(False)
        tl.addWidget(self.trim_btn)
        left.addWidget(trim)
        root.addLayout(left, 3)

        # ---------------- right: details and editors
        right = QVBoxLayout(); right.setSpacing(10)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._details_tab(), "Details")
        self.tabs.addTab(self._captions_tab(), "Captions")
        self.tabs.addTab(self._hook_tab(), "Hook")
        self.tabs.addTab(self._facecam_tab(), "Facecam")
        self.tabs.addTab(self._compare_tab(), "Compare")
        self.tabs.currentChanged.connect(self._tab_changed)
        right.addWidget(self.tabs, 1)
        self.status = label("", "faint", wrap=True); right.addWidget(self.status)
        root.addLayout(right, 2)
        self.load(self.idx)

    # ---------------- tabs
    def _details_tab(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(4, 14, 4, 4); v.setSpacing(10)
        top = QHBoxLayout(); top.setSpacing(8)
        self.score_box = QHBoxLayout(); top.addLayout(self.score_box)
        self.when_lbl = label("", "faint"); top.addWidget(self.when_lbl); top.addStretch(1)
        v.addLayout(top)
        self.title_lbl = label("", "h2", wrap=True); v.addWidget(self.title_lbl)
        self.reason_lbl = label("", "muted", wrap=True); v.addWidget(self.reason_lbl)
        self.visual_lbl = label("", "faint", wrap=True); v.addWidget(self.visual_lbl)
        v.addWidget(label("DESCRIPTION", "section"))
        self.desc = QPlainTextEdit(); self.desc.setReadOnly(True); self.desc.setMaximumHeight(120); v.addWidget(self.desc)
        self.tags_lbl = label("", "info", wrap=True); v.addWidget(self.tags_lbl)
        v.addStretch(1)
        row = QHBoxLayout(); row.setSpacing(6)
        row.addWidget(button("Copy", ic="copy", on=self._copy))
        if self.upload:
            row.addWidget(button("Upload", ic="upload", on=lambda: self.upload([as_upload(*self.items[self.idx])])))
        row.addWidget(button("Show in folder", "ghost", "folder", on=lambda: reveal(self.path)))
        row.addStretch(1)
        v.addLayout(row)
        return w

    def _captions_tab(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(4, 14, 4, 4); v.setSpacing(8)
        v.addWidget(label("Double-click a word to change it. Click a time to jump there.", "faint", wrap=True))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["START", "END", "WORD"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed)
        self.table.cellClicked.connect(self._seek_word)
        self.table.itemChanged.connect(lambda _: self._dirty("caps"))
        v.addWidget(self.table, 1)
        row = QHBoxLayout(); row.setSpacing(6)
        row.addWidget(button("Merge with next", "ghost", on=self._merge))
        row.addWidget(button("Split", "ghost", tip="Split a word that has a space in it into two", on=self._split))
        row.addWidget(button("Delete", "ghost", "trash", on=self._del_word))
        row.addStretch(1)
        self.caps_save = button("Save and re-make clip", "primary", on=self.save_captions); self.caps_save.setEnabled(False)
        row.addWidget(self.caps_save)
        v.addLayout(row)
        self.caps_note = label("", "faint", wrap=True); v.addWidget(self.caps_note)
        return w

    def _hook_tab(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(4, 14, 4, 4); v.setSpacing(10)
        v.addWidget(label("The big line at the top for the first seconds. Six words or fewer works best.",
                          "faint", wrap=True))
        self.hook_edit = QLineEdit(); self.hook_edit.setMaxLength(60)
        self.hook_edit.setPlaceholderText("No hook")
        self.hook_edit.textEdited.connect(lambda _: self._dirty("hook"))
        v.addWidget(self.hook_edit)
        row = QHBoxLayout()
        row.addWidget(button("Remove hook", "ghost", on=lambda: (self.hook_edit.clear(), self._dirty("hook"))))
        row.addStretch(1)
        self.hook_save = button("Save and re-make clip", "primary", on=self.save_hook); self.hook_save.setEnabled(False)
        row.addWidget(self.hook_save)
        v.addLayout(row)
        v.addStretch(1)
        return w

    def _facecam_tab(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(4, 14, 4, 4); v.setSpacing(10)
        v.addWidget(label("Drag the box onto your facecam. It goes on top and the gameplay below. "
                          "The box is remembered for this video.", "faint", wrap=True))
        self.cam_edit = FacecamEditor()
        self.cam_edit.changed.connect(lambda _: self._dirty("cam"))
        v.addWidget(self.cam_edit, 1)
        row = QHBoxLayout()
        self.cam_detect = button("Find it for me", "ghost", "person", on=self._detect_cam)
        row.addWidget(self.cam_detect); row.addStretch(1)
        self.cam_save = button("Save and re-make clip", "primary", on=self.save_cam); self.cam_save.setEnabled(False)
        row.addWidget(self.cam_save)
        v.addLayout(row)
        self.cam_note = label("", "faint", wrap=True); v.addWidget(self.cam_note)
        return w

    def _compare_tab(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(4, 14, 4, 4); v.setSpacing(10)
        v.addWidget(label("Hear this clip with Whisper and with the Hindi model side by side, then keep the better "
                          "one.", "faint", wrap=True))
        self.cmp_btn = button("Compare", on=self._compare)
        v.addWidget(self.cmp_btn, 0, Qt.AlignmentFlag.AlignLeft)
        v.addWidget(label("WHISPER", "section"))
        self.cmp_w = QPlainTextEdit(); self.cmp_w.setReadOnly(True); v.addWidget(self.cmp_w, 1)
        v.addWidget(label("HINDI MODEL", "section"))
        self.cmp_h = QPlainTextEdit(); self.cmp_h.setReadOnly(True); v.addWidget(self.cmp_h, 1)
        row = QHBoxLayout(); row.addStretch(1)
        self.cmp_use = button("Use the Hindi model's words", "primary", on=self._use_hindi); self.cmp_use.setEnabled(False)
        row.addWidget(self.cmp_use)
        v.addLayout(row)
        self.cmp_note = label("", "faint", wrap=True); v.addWidget(self.cmp_note)
        return w

    # ---------------- loading a clip
    @property
    def clip(self) -> dict:
        return self.items[self.idx][0]

    @property
    def folder(self) -> str:
        return self.items[self.idx][1]

    @property
    def path(self) -> str:
        return clip_path(self.clip, self.folder)

    def load(self, i: int, autoplay: bool = True) -> None:
        self.idx = max(0, min(i, len(self.items) - 1))
        try:
            d = editing.load_job(self.folder)
            fresh = d["clips"][editing.find_clip(d, os.path.basename(self.path))]
            self.items[self.idx] = (fresh, self.folder)
            self.job = d
        except Exception:
            self.job = {}
        c = self.clip
        self.setWindowTitle(f"{c.get('title') or 'Clip'}  ·  {self.idx + 1} of {len(self.items)}")
        self.prev_btn.setEnabled(self.idx > 0)
        self.next_btn.setEnabled(self.idx < len(self.items) - 1)
        while self.score_box.count():
            it = self.score_box.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        if c.get("score") is not None:
            self.score_box.addWidget(pill(f"{float(c['score']):.0f}", theme.score_color(c["score"])))
        st = float(c.get("start", 0))
        self.when_lbl.setText("Compilation" if c.get("source") == "compilation" else f"from {int(st // 60)}:{int(st % 60):02d}")
        self.title_lbl.setText(c.get("title") or "Untitled")
        self.reason_lbl.setText(c.get("reason") or "")
        self.visual_lbl.setText(f"On screen: {c['visual']}" if c.get("visual") else "")
        self.desc.setPlainText(c.get("description") or "")
        self.tags_lbl.setText(" ".join(c.get("hashtags") or []))
        self._fill_table()
        self.hook_edit.setText(c.get("hook") or (c.get("plan") or {}).get("hook", ""))
        cam = (c.get("plan") or {}).get("cam") or []
        self.cam_edit.set_box(cam or None)
        self.cam_frame_for = None
        why = editing.can_rerender(self.job, c) if self.job else "This clip's job file is missing."
        for b_ in (self.caps_save, self.hook_save, self.cam_save):
            b_.setEnabled(False)
        self.caps_note.setText(why)
        self.cam_note.setText(why or ("" if helpers.installed("faces") else
                                      "Install Face tracking in Settings → AI helpers to find the facecam for you."))
        self.cam_detect.setEnabled(not why and helpers.installed("faces"))
        hi = helpers.installed("hindi")
        self.cmp_btn.setEnabled(hi and not why)
        self.cmp_note.setText("" if hi else "Install the Hindi speech model in Settings → AI helpers to compare.")
        self.cmp_w.clear(); self.cmp_h.clear(); self.cmp_use.setEnabled(False); self.cmp_result = None
        self.mark_a = self.mark_b = None
        self._trim_label()
        self._dirty_flags = set()
        self.status.setText("")
        if self.tabs.currentIndex() == 3:
            self._load_cam_frame()
        self._set_source(autoplay)

    def _set_source(self, autoplay: bool) -> None:
        if self.player is None:
            return
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(self.path))
        if autoplay:
            self.player.play()

    def release_file(self) -> None:
        """Windows locks a playing file; let go of it before the clip is re-made."""
        if self.player is not None:
            self.player.stop()
            self.player.setSource(QUrl())

    def _fill_table(self) -> None:
        self.table.blockSignals(True)
        words = self.clip.get("words") or []
        st = float(self.clip.get("start", 0))
        self.table.setRowCount(len(words))
        for r, (a, b, t) in enumerate(words):
            for col, val in enumerate((f"{max(0.0, float(a) - st):.2f}", f"{max(0.0, float(b) - st):.2f}", str(t))):
                it = QTableWidgetItem(val)
                if col < 2:
                    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    it.setForeground(QColor(theme.TEXT_3))
                self.table.setItem(r, col, it)
        self.table.blockSignals(False)

    def _table_words(self) -> list:
        st = float(self.clip.get("start", 0))
        out = []
        for r in range(self.table.rowCount()):
            try:
                a = float(self.table.item(r, 0).text()) + st
                b = float(self.table.item(r, 1).text()) + st
            except (AttributeError, ValueError):
                continue
            t = (self.table.item(r, 2).text() if self.table.item(r, 2) else "").strip()
            if t:
                out.append([round(a, 3), round(b, 3), t])
        return out

    # ---------------- playback
    def toggle(self) -> None:
        if self.player is None:
            open_path(self.path)
            return
        from PySide6.QtMultimedia import QMediaPlayer
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _pos(self, ms: int) -> None:
        if not self.seek.isSliderDown():
            self.seek.setValue(ms)
        self.time_lbl.setText(f"{fmt(ms)} / {fmt(self.seek.maximum())}")

    def _dur(self, ms: int) -> None:
        self.seek.setRange(0, ms)

    def _state(self, st) -> None:
        from PySide6.QtMultimedia import QMediaPlayer
        playing = st == QMediaPlayer.PlaybackState.PlayingState
        self.play_btn.setIcon(icon("pause" if playing else "play"))

    def _error(self, err, msg: str = "") -> None:
        from PySide6.QtMultimedia import QMediaPlayer
        if err != QMediaPlayer.Error.NoError:
            self.status.setText(f"The built-in player couldn't play this file ({msg}). Opening it in your video app.")
            open_path(self.path)

    def skip(self, sec: float) -> None:
        if self.player is not None:
            self.player.setPosition(max(0, min(self.seek.maximum(), self.player.position() + int(sec * 1000))))

    def prev_clip(self) -> None:
        if self.idx > 0 and self._leave_ok():
            self.load(self.idx - 1)

    def next_clip(self) -> None:
        if self.idx < len(self.items) - 1 and self._leave_ok():
            self.load(self.idx + 1)

    def _leave_ok(self) -> bool:
        if not getattr(self, "_dirty_flags", None):
            return True
        from PySide6.QtWidgets import QMessageBox
        return QMessageBox.question(self, "Unsaved changes", "Leave this clip without saving your changes?") \
            == QMessageBox.StandardButton.Yes

    def keyPressEvent(self, e) -> None:  # noqa: N802
        fw = QApplication.focusWidget()
        if isinstance(fw, TEXT_INPUTS):              # typing a word or the hook: keys belong to the editor
            return super().keyPressEvent(e)
        k = e.key()
        if k == Qt.Key.Key_Space:
            self.toggle()
        elif k == Qt.Key.Key_Left:
            self.skip(-5)
        elif k == Qt.Key.Key_Right:
            self.skip(5)
        elif k == Qt.Key.Key_Up:
            self.prev_clip()
        elif k == Qt.Key.Key_Down:
            self.next_clip()
        elif k == Qt.Key.Key_BracketLeft:
            self._mark("a")
        elif k == Qt.Key.Key_BracketRight:
            self._mark("b")
        elif k == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(e)

    # ---------------- trim
    def _mark(self, which: str) -> None:
        if self.player is None:
            return
        t = self.player.position() / 1000.0
        if which == "a":
            self.mark_a = t
            if self.mark_b is not None and self.mark_b <= t + 0.3:
                self.mark_b = None
        else:
            self.mark_b = t
            if self.mark_a is not None and self.mark_a >= t - 0.3:
                self.mark_a = None
        self._trim_label()

    def _trim_label(self) -> None:
        dur = self.seek.maximum() / 1000.0 if self.seek.maximum() else (float(self.clip.get("end", 0)) - float(self.clip.get("start", 0)))
        a = self.mark_a or 0.0
        b = self.mark_b if self.mark_b is not None else dur
        if self.mark_a is None and self.mark_b is None:
            self.trim_lbl.setText("Trim: whole clip  ·  [ sets the start, ] the end")
            self.trim_btn.setEnabled(False)
            return
        self.trim_lbl.setText(f"Trim: {fmt(a * 1000)} → {fmt(b * 1000)}  ({b - a:.1f} s)")
        self.trim_btn.setEnabled(b - a >= 1.0)

    def save_trim(self) -> None:
        dur = self.seek.maximum() / 1000.0
        a = self.mark_a or 0.0
        b = self.mark_b if self.mark_b is not None else dur
        name = os.path.basename(self.path)
        folder = self.folder
        self._run(lambda cancel, prog: editing.trim_copy(folder, name, a, b, cancel), "Saving the trimmed copy",
                  lambda rec: self._added(rec, folder), release=False)

    def _added(self, rec: dict, folder: str) -> None:
        self.items.insert(self.idx + 1, (rec, folder))
        self.changed.emit(folder)
        self.toast("Trimmed copy saved")
        self.load(self.idx + 1)

    # ---------------- captions editing
    def _dirty(self, what: str) -> None:
        self._dirty_flags = getattr(self, "_dirty_flags", set()) | {what}
        ok = not editing.can_rerender(self.job, self.clip) if self.job else False
        {"caps": self.caps_save, "hook": self.hook_save, "cam": self.cam_save}[what].setEnabled(ok)

    def _seek_word(self, row: int, col: int) -> None:
        if col < 2 and self.player is not None:
            try:
                self.player.setPosition(int(float(self.table.item(row, 0).text()) * 1000))
            except (AttributeError, ValueError):
                pass

    def _merge(self) -> None:
        r = self.table.currentRow()
        if r < 0 or r + 1 >= self.table.rowCount():
            return
        t = f"{self.table.item(r, 2).text()} {self.table.item(r + 1, 2).text()}".strip()
        self.table.blockSignals(True)
        self.table.item(r, 1).setText(self.table.item(r + 1, 1).text())
        self.table.item(r, 2).setText(t)
        self.table.removeRow(r + 1)
        self.table.blockSignals(False)
        self._dirty("caps")

    def _split(self) -> None:
        r = self.table.currentRow()
        if r < 0:
            return
        parts = self.table.item(r, 2).text().split()
        if len(parts) < 2:
            self.caps_note.setText("Type a space in the word where it should be split, then click Split.")
            return
        a, b = float(self.table.item(r, 0).text()), float(self.table.item(r, 1).text())
        step = (b - a) / len(parts)
        self.table.blockSignals(True)
        self.table.item(r, 1).setText(f"{a + step:.2f}")
        self.table.item(r, 2).setText(parts[0])
        for k, wd in enumerate(parts[1:], 1):
            self.table.insertRow(r + k)
            for col, val in enumerate((f"{a + step * k:.2f}", f"{a + step * (k + 1):.2f}", wd)):
                it = QTableWidgetItem(val)
                if col < 2:
                    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    it.setForeground(QColor(theme.TEXT_3))
                self.table.setItem(r + k, col, it)
        self.table.blockSignals(False)
        self._dirty("caps")

    def _del_word(self) -> None:
        r = self.table.currentRow()
        if r >= 0:
            self.table.removeRow(r)
            self._dirty("caps")

    def save_captions(self) -> None:
        words = self._table_words()
        self._rerender(words=words, label_="Re-making the clip with your captions")

    def save_hook(self) -> None:
        self._rerender(hook=self.hook_edit.text().strip(), label_="Re-making the clip with the new hook")

    def save_cam(self) -> None:
        self._rerender(cam=list(self.cam_edit.box), label_="Re-making the clip with the facecam on top")

    def _rerender(self, label_: str, **kw) -> None:
        name = os.path.basename(self.path)
        folder = self.folder

        def fn(cancel, prog):
            return editing.rerender(folder, name, cancel, progress=prog, **kw)

        def done(rec: dict) -> None:
            self.items[self.idx] = (rec, folder)
            self._dirty_flags = set()
            self.changed.emit(folder)
            self.toast("Clip re-made")
            self.load(self.idx)

        self._run(fn, label_, done, release=True)

    # ---------------- facecam
    def _tab_changed(self, i: int) -> None:
        if i == 3:
            self._load_cam_frame()

    def _load_cam_frame(self) -> None:
        if getattr(self, "cam_frame_for", None) == self.path:
            return
        self.cam_frame_for = self.path
        media = editing.media_of(self.job) if self.job else ""
        if not media:
            return
        t = float(self.clip.get("start", 0)) + 1.0
        from ..core import ffmpeg
        workers.run_task(self, lambda c, p: ffmpeg.grab_jpeg(media, t, 960, 3), self.cam_edit.set_frame)

    def _detect_cam(self) -> None:
        media = editing.media_of(self.job) if self.job else ""
        if not media:
            return

        def fn(cancel, prog):
            from ..core import faces, ffmpeg
            return faces.detect_facecam(media, ffmpeg.probe(media), cancel)

        def done(box) -> None:
            if box:
                self.cam_edit.set_box(list(box))
                self._dirty("cam")
                self.cam_note.setText("Found it. Adjust the box if needed, then save.")
            else:
                self.cam_note.setText("No facecam found. Drag the box onto it yourself.")
        self._run(fn, "Looking for your facecam", done, release=False)

    # ---------------- compare
    def _compare(self) -> None:
        name = os.path.basename(self.path)
        folder = self.folder

        def done(res: dict) -> None:
            self.cmp_result = res
            self.cmp_w.setPlainText(res.get("whisper", ""))
            self.cmp_h.setPlainText(res.get("hindi", ""))
            self.cmp_use.setEnabled(bool(res.get("hindi_words")) and not editing.can_rerender(self.job, self.clip))
        self._run(lambda cancel, prog: editing.compare_hindi(folder, name, cancel, prog), "Comparing", done,
                  release=False)

    def _use_hindi(self) -> None:
        res = getattr(self, "cmp_result", None) or {}
        words = res.get("hindi_words") or []
        lang = (self.job.get("settings") or {}).get("caption_lang", "auto") if self.job else "auto"
        if lang in ("auto", "roman"):
            words = editing.roman_words(words)
        self._rerender(words=words, label_="Re-making the clip with the Hindi model's words")

    # ---------------- background work
    def _run(self, fn, text: str, on_done, release: bool) -> None:
        if self.busy is not None:
            self.toast("Wait for the current step to finish.", 3000, True)
            return
        if release:
            self.release_file()
        self.status.setText(text + "...")
        self.setCursor(Qt.CursorShape.BusyCursor)

        def ok(res) -> None:
            self.busy = None
            self.unsetCursor()
            self.status.setText("")
            on_done(res)

        def bad(msg: str) -> None:
            self.busy = None
            self.unsetCursor()
            self.status.setText("" if msg == "Cancelled" else f"Couldn't finish: {msg}")
            if release:
                self._set_source(False)

        self.busy = workers.run_task(self, fn, ok, bad,
                                     lambda f, m: self.status.setText(f"{text}  ·  {m}  {int(f * 100)}%" if m else text))

    def _copy(self) -> None:
        c = self.clip
        QApplication.clipboard().setText("\n\n".join(x for x in [c.get("title", ""), c.get("description", ""),
                                                               " ".join(c.get("hashtags") or [])] if x))
        self.toast("Copied")

    def closeEvent(self, e) -> None:  # noqa: N802
        if self.busy is not None:
            self.busy.cancel.set()
        self.release_file()
        super().closeEvent(e)


def open_player(parent, clip: dict, folder: str, siblings: list, toast: Callable, upload=None,
                on_changed: Optional[Callable[[str], None]] = None) -> None:
    dlg = PlayerDialog(parent, clip, folder, siblings, toast, upload)
    if on_changed:
        dlg.changed.connect(on_changed)
    dlg.setModal(False)
    dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dlg.show()
