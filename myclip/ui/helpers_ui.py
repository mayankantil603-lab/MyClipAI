"""Settings -> AI helpers (one-click installs) and the swear-word list editor."""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QFrame, QHBoxLayout, QMessageBox, QPlainTextEdit, QProgressBar, QVBoxLayout,
                               QWidget)

from ..core import bleep, helpers
from . import theme, workers
from .widgets import button, dot_text, label


class HelperRow(QFrame):
    def __init__(self, h: helpers.Helper, toast: Callable, ollama_url: Callable[[], str], changed: Callable) -> None:
        super().__init__()
        self.setObjectName("panel")
        self.h, self.toast, self.ollama_url, self.changed = h, toast, ollama_url, changed
        self.worker = None
        v = QVBoxLayout(self); v.setContentsMargins(16, 14, 12, 14); v.setSpacing(8)
        top = QHBoxLayout(); top.setSpacing(10)
        col = QVBoxLayout(); col.setSpacing(3)
        self.name = label(h.name, "h3"); col.addWidget(self.name)
        col.addWidget(label(h.what, "muted", wrap=True))
        self.size = label(h.size, "faint", wrap=True); col.addWidget(self.size)
        top.addLayout(col, 1)
        btns = QVBoxLayout(); btns.setSpacing(6)
        self.state = label("", "faint"); self.state.setAlignment(Qt.AlignmentFlag.AlignRight); btns.addWidget(self.state)
        row = QHBoxLayout(); row.setSpacing(6); row.addStretch(1)
        self.remove_btn = button("Remove", "ghost", on=self.remove); row.addWidget(self.remove_btn)
        self.install_btn = button("Install", "primary", on=self.install); row.addWidget(self.install_btn)
        btns.addLayout(row); btns.addStretch(1)
        top.addLayout(btns)
        v.addLayout(top)
        self.bar = QProgressBar(); self.bar.setRange(0, 100); self.bar.setTextVisible(False); self.bar.hide()
        v.addWidget(self.bar)
        self.msg = label("", "faint", wrap=True); self.msg.hide(); v.addWidget(self.msg)
        self.refresh()

    def refresh(self) -> None:
        ok = helpers.installed(self.h.id)
        txt = helpers.status_text(self.h.id)
        self.state.setText(dot_text(theme.TEAL if ok else theme.TEXT_3, txt))
        busy = self.worker is not None
        self.install_btn.setText("Cancel" if busy else ("Reinstall" if ok else ("Find model" if self.h.id == "vision" else "Install")))
        self.install_btn.setObjectName("" if (ok or busy) else "primary")
        self.install_btn.style().unpolish(self.install_btn); self.install_btn.style().polish(self.install_btn)
        self.remove_btn.setVisible(ok and not busy)

    def install(self) -> None:
        if self.worker is not None:
            self.worker.cancel.set()
            return
        hid = self.h.id
        url = self.ollama_url()
        self.bar.setValue(0); self.bar.show(); self.msg.setText("Starting..."); self.msg.show()

        def prog(f: float, m: str) -> None:
            if f >= 0:
                self.bar.setValue(int(f * 100))
            if m:
                self.msg.setText(m)

        def done(res) -> None:
            self.worker = None
            self.bar.hide()
            self.msg.setText(str(res or "Ready"))
            self.refresh(); self.changed()
            self.toast(f"{self.h.name}: {res or 'ready'}")

        def failed(m: str) -> None:
            self.worker = None
            self.bar.hide()
            self.msg.setText("Cancelled" if m == "Cancelled" else m)
            self.msg.setObjectName("" if m == "Cancelled" else "error")
            self.msg.style().unpolish(self.msg); self.msg.style().polish(self.msg)
            self.refresh()

        self.worker = workers.run_task(self, lambda cancel, p: helpers.install(hid, cancel, p, url), done, failed, prog)
        self.refresh()

    def remove(self) -> None:
        if QMessageBox.question(self, "Remove", f"Remove {self.h.name}? You can install it again any time.") \
                != QMessageBox.StandardButton.Yes:
            return
        hid = self.h.id

        def done(_) -> None:
            self.worker = None
            self.msg.setText("Removed"); self.msg.show()
            self.refresh(); self.changed()

        self.worker = workers.run_task(self, lambda cancel, p: helpers.remove(hid, cancel), done,
                                       lambda m: (setattr(self, "worker", None), self.refresh()))
        self.refresh()


class HelpersSection(QWidget):
    def __init__(self, toast: Callable, ollama_url: Callable[[], str], changed: Callable) -> None:
        super().__init__()
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(10)
        v.addWidget(label("Optional extras that make clips better. Each one is used only when it's installed; "
                          "you can turn them off per job under Advanced on the Create page.", "muted", wrap=True))
        self.rows = [HelperRow(h, toast, ollama_url, changed) for h in helpers.HELPERS]
        for r in self.rows:
            v.addWidget(r)

    def refresh(self) -> None:
        for r in self.rows:
            r.refresh()


class BleepWordsDialog(QDialog):
    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.setWindowTitle("Words to bleep")
        self.setMinimumSize(520, 560)
        v = QVBoxLayout(self); v.setContentsMargins(22, 20, 22, 18); v.setSpacing(12)
        v.addWidget(label("Words to bleep", "h2"))
        v.addWidget(label("One word per line, in English letters or Hindi script. End a word with * to also catch "
                          "longer forms (fuck* catches fucking). Hindi script words are also matched after "
                          "they're written in English letters.", "muted", wrap=True))
        self.edit = QPlainTextEdit()
        self.edit.setPlainText("\n".join(bleep.load_words()))
        v.addWidget(self.edit, 1)
        row = QHBoxLayout()
        row.addWidget(button("Reset to the built-in list", "ghost",
                             on=lambda: self.edit.setPlainText("\n".join(bleep.DEFAULT_WORDS))))
        row.addStretch(1)
        row.addWidget(button("Cancel", "ghost", on=self.reject))
        row.addWidget(button("Save", "primary", on=self._save))
        v.addLayout(row)

    def _save(self) -> None:
        bleep.save_words([l for l in self.edit.toPlainText().splitlines()])
        self.accept()
