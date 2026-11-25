from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QFormLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                               QVBoxLayout)

from ..core.store import LOCAL_PROVIDERS, PROVIDERS, Connection, ConnectionStore
from . import workers
from .widgets import button, label


class ConnectionDialog(QDialog):
    def __init__(self, parent, store: ConnectionStore, conn: Connection | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit connection" if conn else "Add connection")
        self.setMinimumWidth(460)
        self.store = store
        self.orig = conn
        self.result_conn: Connection | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 20, 22, 18)
        lay.setSpacing(14)
        lay.addWidget(label(self.windowTitle(), "h2"))
        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.provider = QComboBox()
        self.provider.addItems(list(PROVIDERS))
        self.name = QLineEdit()
        self.url = QLineEdit()
        self.model = QLineEdit()
        self.model.setPlaceholderText("Pick later from the list")
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        has_key = bool(conn and store.keys.get(conn.id))
        self.key.setPlaceholderText("Saved. Leave empty to keep it." if has_key else "Paste your API key")
        form.addRow("Provider", self.provider)
        form.addRow("Name", self.name)
        form.addRow("Endpoint", self.url)
        form.addRow("Model", self.model)
        form.addRow("API key", self.key)
        lay.addLayout(form)
        where = ("Keys are stored in Windows Credential Manager." if store.keys.secure else
                 "Secure storage isn't available, so the key is kept only until you close the app.")
        self.note = label(where, "faint", wrap=True)
        lay.addWidget(self.note)
        self.status = label("", "faint", wrap=True)
        lay.addWidget(self.status)
        row = QHBoxLayout()
        self.test_btn = button("Test", on=self.test)
        row.addWidget(self.test_btn)
        row.addStretch(1)
        row.addWidget(button("Cancel", "ghost", on=self.reject))
        row.addWidget(button("Save", "primary", on=self.save))
        lay.addLayout(row)
        self.provider.currentTextChanged.connect(self._prov)
        if conn:
            self.provider.setCurrentText(conn.provider)
            self.name.setText(conn.name)
            self.url.setText(conn.base_url)
            self.model.setText(conn.model)
        else:
            self._prov(self.provider.currentText())

    def _prov(self, p: str) -> None:
        base, env, rec = PROVIDERS.get(p, ("", "", []))
        if not self.orig or self.orig.provider != p:
            self.url.setText(base)
            self.name.setText(f"{p} (local)" if p in LOCAL_PROVIDERS else p)
            self.model.setText(rec[0] if rec else "")
        local = p in LOCAL_PROVIDERS
        self.key.setEnabled(not local)
        hint = "No key needed. Local apps are usually found automatically." if local else (
            f"You can also set the {env} environment variable instead." if env else "")
        self.status.setText(hint)

    def _build(self) -> Connection:
        c = Connection(self.name.text().strip() or self.provider.currentText(), self.provider.currentText(),
                       self.url.text().strip().rstrip("/"), self.model.text().strip())
        if self.orig:
            c.id = self.orig.id
            if self.orig.provider == c.provider:
                c.models_cache = self.orig.models_cache
        return c

    def _key(self, c: Connection) -> str:
        return self.key.text().strip() or self.store.api_key(c)

    def test(self) -> None:
        c = self._build()
        if not c.base_url:
            self.status.setText("Enter an endpoint first.")
            return
        if not c.model:
            self.status.setText("Enter a model name to test with.")
            return
        self.test_btn.setEnabled(False)
        self.status.setText("Testing...")
        w = workers.TestWorker(c, self._key(c), c.model)
        w.done.connect(self._tested)
        workers.start(self, w, w.done)

    def _tested(self, ok: bool, msg: str) -> None:
        self.test_btn.setEnabled(True)
        self.status.setText(("Works. " if ok else "Failed: ") + ("" if ok else msg))

    def save(self) -> None:
        c = self._build()
        if not c.base_url:
            self.status.setText("Enter an endpoint.")
            return
        key = self.key.text().strip()
        secure = self.store.upsert(c, key if key else None)
        if key and not secure:
            self.status.setText("Saved for this session only.")
        self.result_conn = c
        self.accept()


def ask_text(parent, title: str, prompt: str, text: str = "") -> str:
    val, ok = QInputDialog.getText(parent, title, prompt, text=text)
    return val.strip() if ok else ""
