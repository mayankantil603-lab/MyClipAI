"""Studio: connect YouTube channels, upload queue, videos, comments."""
from __future__ import annotations

import os
import threading
import time
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QDateTime, QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit, QDialog, QFileDialog, QFrame, QGridLayout,
                               QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMenu, QMessageBox, QProgressBar,
                               QPushButton, QSpinBox, QTextEdit, QVBoxLayout, QWidget)

from ..core import youtube as yt
from ..core.models import Cancelled
from ..core.paths import config_dir
from . import theme
from .widgets import Chips, Field, button, dot_text, hline, label, open_path

PRIVACY = [("private", "Private"), ("unlisted", "Unlisted"), ("public", "Public")]


# ---------------------------------------------------------------- background

class Call(QObject):
    """Run fn() on a thread; deliver result or error on the GUI thread."""
    ok = Signal(object)
    err = Signal(str)

    def __init__(self, fn: Callable, on_ok: Callable, on_err: Callable) -> None:
        super().__init__()
        self.fn = fn
        self.ok.connect(on_ok)
        self.err.connect(on_err)
        _keep.add(self)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        try:
            self.ok.emit(self.fn())
        except Cancelled:
            self.err.emit("Cancelled")
        except Exception as exc:
            self.err.emit(str(exc) or exc.__class__.__name__)
        finally:
            QTimer.singleShot(0, lambda: _keep.discard(self))


_keep: set = set()


class Uploader(QObject):
    """Works through the queue one file at a time. Pauses on daily limit / sign-in problems."""
    progress = Signal(str, float)
    changed = Signal()
    paused_msg = Signal(str)

    def __init__(self, store: yt.YouTubeStore, queue: yt.UploadQueue) -> None:
        super().__init__()
        self.store, self.queue = store, queue
        self.stop = threading.Event()
        self.cancel = threading.Event()
        self.wake = threading.Event()
        self.paused = False
        self.current = ""
        threading.Thread(target=self._loop, daemon=True, name="uploader").start()

    def resume(self) -> None:
        self.paused = False
        self.wake.set()

    def pause(self) -> None:
        self.paused = True
        self.cancel.set()

    def _loop(self) -> None:
        while not self.stop.is_set():
            self.wake.wait(2.0)
            self.wake.clear()
            if self.paused or not self.store.channels:
                continue
            item = self.queue.next_pending()
            if not item:
                continue
            self.cancel.clear()
            self.current = item.id
            self.changed.emit()
            try:
                yt.process(item, self.store, self.queue, self.progress.emit, self.cancel)
            except Cancelled:
                pass
            except yt.YTError as e:
                self.paused = True
                self.paused_msg.emit(str(e))
            except Exception as exc:
                self.queue.set(item, status="failed", error=str(exc)[:200])
            self.current = ""
            self.changed.emit()
            self.wake.set()

    def shutdown(self) -> None:
        self.stop.set()
        self.cancel.set()
        self.wake.set()


# ---------------------------------------------------------------- dialogs

class UploadDialog(QDialog):
    """Upload one or many clips. Each clip keeps its own AI title/description/hashtags."""

    def __init__(self, parent, store: yt.YouTubeStore, clips: list[dict]) -> None:
        super().__init__(parent)
        self.store, self.clips = store, clips
        self.items: list[yt.UploadItem] = []
        many = len(clips) > 1
        self.setWindowTitle(f"Upload {len(clips)} clips" if many else "Upload to YouTube")
        self.setMinimumWidth(560)
        v = QVBoxLayout(self); v.setContentsMargins(22, 20, 22, 18); v.setSpacing(14)
        v.addWidget(label(self.windowTitle(), "h2"))
        self.channel = QComboBox()
        for c in store.channels:
            self.channel.addItem(c.title + (f"  {c.handle}" if c.handle else ""), c.id)
        self.channel.setCurrentIndex(max(0, self.channel.findData(store.active_id)))
        v.addWidget(Field("Channel", self.channel))
        if not many:
            c = clips[0]
            self.title = QLineEdit(c.get("title", "")); self.title.setMaxLength(100)
            self.desc = QTextEdit(); self.desc.setPlainText(c.get("description", "")); self.desc.setFixedHeight(90)
            self.tags = QLineEdit(", ".join(h.lstrip("#") for h in c.get("hashtags") or []))
            self.tags.setPlaceholderText("valorant, clutch, gaming")
            self.count = label("", "faint")
            self.title.textChanged.connect(lambda t: self.count.setText(f"{len(t)}/100"))
            self.count.setText(f"{len(self.title.text())}/100")
            row = QHBoxLayout(); row.addWidget(label("Title", "muted")); row.addStretch(1); row.addWidget(self.count)
            v.addLayout(row); v.addWidget(self.title)
            v.addWidget(Field("Description", self.desc)); v.addWidget(Field("Tags", self.tags))
        else:
            v.addWidget(label("Each clip uses its own title, description and hashtags. "
                              "You can edit them later in Studio → Videos.", "faint", wrap=True))
        self.privacy = Chips(PRIVACY, wrap=False); self.privacy.set_value("private")
        v.addWidget(Field("Visibility", self.privacy))
        self.when = Chips([("now", "Now"), ("at", "Schedule")] + ([("spread", "Spread out")] if many else []), wrap=False)
        self.when.set_value("now")
        v.addWidget(Field("Publish", self.when))
        self.sched = QWidget(); sh = QHBoxLayout(self.sched); sh.setContentsMargins(0, 0, 0, 0); sh.setSpacing(8)
        self.dt = QDateTimeEdit(QDateTime.currentDateTime().addSecs(3600 * 3)); self.dt.setCalendarPopup(True)
        self.dt.setDisplayFormat("ddd d MMM yyyy  HH:mm"); self.dt.setMinimumDateTime(QDateTime.currentDateTime().addSecs(900))
        sh.addWidget(label("First at", "muted")); sh.addWidget(self.dt)
        self.every = QSpinBox(); self.every.setRange(1, 168); self.every.setValue(24); self.every.setSuffix(" h")
        self.every_lbl = label("then every", "muted")
        sh.addWidget(self.every_lbl); sh.addWidget(self.every); sh.addStretch(1)
        v.addWidget(self.sched)
        self.when.changed.connect(self._when); self._when("now")
        self.kids = QCheckBox("Made for kids"); v.addWidget(self.kids)
        self.shorts = QCheckBox("Add #Shorts"); self.shorts.setChecked(True); v.addWidget(self.shorts)
        self.err = label("", "error", wrap=True); v.addWidget(self.err)
        b = QHBoxLayout(); b.addStretch(1)
        b.addWidget(button("Cancel", "ghost", on=self.reject))
        b.addWidget(button("Add to upload queue", "primary", on=self._ok))
        v.addLayout(b)

    def _when(self, w: str) -> None:
        self.sched.setVisible(w != "now")
        self.every.setVisible(w == "spread"); self.every_lbl.setVisible(w == "spread")
        self.privacy.setEnabled(w == "now")
        if w != "now":
            self.privacy.set_value("public")

    def _ok(self) -> None:
        cid = self.channel.currentData()
        if not cid:
            self.err.setText("Connect a channel first.")
            return
        mode = self.when.value()
        first = self.dt.dateTime().toPython()
        times = [""] * len(self.clips)
        if mode == "at":
            times = [first.strftime("%Y-%m-%dT%H:%M")] * len(self.clips)
        elif mode == "spread":
            times = yt.spread_schedule(len(self.clips), first, self.every.value())
        for i, c in enumerate(self.clips):
            if len(self.clips) == 1:
                title, desc = self.title.text(), self.desc.toPlainText()
                tags = [t.strip() for t in self.tags.text().split(",") if t.strip()]
            else:
                title, desc, tags = c.get("title", ""), c.get("description", ""), [h.lstrip("#") for h in c.get("hashtags") or []]
            if not os.path.isfile(c["path"]):
                continue
            self.items.append(yt.UploadItem(c["path"], title or Path(c["path"]).stem, desc, tags,
                                            self.privacy.value() or "private", times[i], "20", self.kids.isChecked(),
                                            self.shorts.isChecked(), cid, ""))
        if not self.items:
            self.err.setText("Those files are no longer on disk.")
            return
        self.accept()


class EditDialog(QDialog):
    def __init__(self, parent, video: dict) -> None:
        super().__init__(parent)
        self.video = video
        self.setWindowTitle("Edit video"); self.setMinimumWidth(540)
        v = QVBoxLayout(self); v.setContentsMargins(22, 20, 22, 18); v.setSpacing(14)
        v.addWidget(label("Edit video", "h2"))
        self.title = QLineEdit(video["title"]); self.title.setMaxLength(100)
        self.desc = QTextEdit(); self.desc.setPlainText(video["description"]); self.desc.setFixedHeight(120)
        self.tags = QLineEdit(", ".join(video.get("tags") or []))
        self.privacy = Chips(PRIVACY, wrap=False); self.privacy.set_value(video["privacy"])
        v.addWidget(Field("Title", self.title)); v.addWidget(Field("Description", self.desc))
        v.addWidget(Field("Tags", self.tags)); v.addWidget(Field("Visibility", self.privacy))
        self.thumb_path = ""
        tr = QHBoxLayout(); self.thumb_lbl = label("No new thumbnail", "faint")
        tr.addWidget(button("Choose thumbnail", on=self._thumb)); tr.addWidget(self.thumb_lbl); tr.addStretch(1)
        v.addLayout(tr)
        v.addWidget(label("Custom thumbnails need a verified channel. JPG or PNG under 2 MB, ideally 1280×720.",
                          "faint", wrap=True))
        b = QHBoxLayout(); b.addStretch(1)
        b.addWidget(button("Cancel", "ghost", on=self.reject)); b.addWidget(button("Save", "primary", on=self.accept))
        v.addLayout(b)

    def _thumb(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Thumbnail", "", "Images (*.jpg *.jpeg *.png)")
        if p:
            if os.path.getsize(p) > 2 * 1024 * 1024:
                self.thumb_lbl.setText("Too big. Keep it under 2 MB.")
                return
            self.thumb_path = p
            self.thumb_lbl.setText(os.path.basename(p))

    def values(self) -> tuple:
        return (self.title.text(), self.desc.toPlainText(),
                [t.strip() for t in self.tags.text().split(",") if t.strip()], self.privacy.value() or "private")


class SetupDialog(QDialog):
    def __init__(self, parent, store: yt.YouTubeStore) -> None:
        super().__init__(parent)
        self.store = store
        self.setWindowTitle("Set up YouTube"); self.setMinimumWidth(560)
        v = QVBoxLayout(self); v.setContentsMargins(24, 22, 24, 20); v.setSpacing(12)
        v.addWidget(label("Set up YouTube (one time, about 5 minutes)", "h2"))
        steps = [
            "Open Google Cloud Console and create a project (any name).",
            "Enable “YouTube Data API v3” for that project.",
            "OAuth consent screen: choose External, fill in the app name and your email, "
            "and add your own Google account under Test users.",
            "Credentials → Create credentials → OAuth client ID → Application type: Desktop app.",
            "Download the JSON file and choose it below.",
        ]
        for i, s in enumerate(steps, 1):
            v.addWidget(label(f"{i}.  {s}", "muted", wrap=True))
        v.addWidget(button("Open Google Cloud Console", ic="chevron", on=lambda: webbrowser.open(yt.CONSOLE_URL)),
                    0, Qt.AlignmentFlag.AlignLeft)
        v.addSpacing(6)
        v.addWidget(label("Your client file stays on this PC. Its secret is kept in Windows Credential Manager.",
                          "faint", wrap=True))
        v.addWidget(label("New Google projects allow about 6 uploads a day and treat the app as "
                          "“in testing”, so you may need to sign in again every 7 days. You can ask "
                          "Google for more in the console.", "faint", wrap=True))
        self.err = label("", "error", wrap=True); v.addWidget(self.err)
        b = QHBoxLayout(); b.addStretch(1)
        b.addWidget(button("Cancel", "ghost", on=self.reject))
        b.addWidget(button("Choose client file", "primary", on=self._choose))
        v.addLayout(b)

    def _choose(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Google client file", str(Path.home() / "Downloads"), "JSON (*.json)")
        if not p:
            return
        try:
            cid, sec = yt.parse_client_file(Path(p).read_text(encoding="utf-8"))
        except Exception as exc:
            self.err.setText(str(exc))
            return
        self.store.set_client(cid, sec)
        self.accept()


# ---------------------------------------------------------------- page

def _num(n: Optional[int]) -> str:
    if n is None:
        return "hidden"
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= div:
            return f"{n / div:.1f}".rstrip("0").rstrip(".") + suf
    return str(n)


def _ago(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        s = (datetime.now(dt.tzinfo) - dt).total_seconds()
    except Exception:
        return ""
    for div, name in ((86400 * 365, "y"), (86400 * 30, "mo"), (86400, "d"), (3600, "h"), (60, "m")):
        if s >= div:
            return f"{int(s // div)}{name} ago"
    return "just now"


class Stat(QWidget):
    def __init__(self, name: str) -> None:
        super().__init__()
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(2)
        self.val = label("–"); self.val.setStyleSheet(f"font-size: 26px; font-weight: 700; color: {theme.TEXT};")
        v.addWidget(self.val); v.addWidget(label(name.upper(), "stat"))


class StudioPage(QWidget):
    def __init__(self, toast: Callable[..., None], keys) -> None:
        super().__init__()
        self.toast = toast
        self.store = yt.YouTubeStore(config_dir() / "youtube.json", keys)
        self.queue = yt.UploadQueue(config_dir() / "uploads.json")
        self.uploader = Uploader(self.store, self.queue)
        self.uploader.changed.connect(self._render_queue)
        self.uploader.progress.connect(self._progress)
        self.uploader.paused_msg.connect(self._paused)
        self.net = QNetworkAccessManager(self)
        self.overview: dict = {}
        self.videos: list[dict] = []
        self.bars: dict[str, QProgressBar] = {}
        self._login_cancel = threading.Event()

        self.setObjectName("page")
        v = QVBoxLayout(self); v.setContentsMargins(48, 38, 48, 48); v.setSpacing(0)
        head = QHBoxLayout(); head.addWidget(label("Studio", "h1")); head.addStretch(1)
        self.yt_btn = button("Open YouTube Studio", "ghost", "chevron", on=lambda: webbrowser.open("https://studio.youtube.com"))
        head.addWidget(self.yt_btn)
        v.addLayout(head); v.addSpacing(4)
        self.sub = label("Upload clips and manage your channel without leaving the app.", "muted")
        v.addWidget(self.sub); v.addSpacing(22)

        ch = QHBoxLayout(); ch.addWidget(label("CHANNEL", "section")); ch.addStretch(1)
        self.connect_btn = button("Connect channel", "ghost", "plus", on=self.connect_channel)
        ch.addWidget(self.connect_btn)
        v.addLayout(ch); v.addSpacing(8)
        self.channel_chips = Chips()
        self.channel_chips.changed.connect(self._switch)
        self.channel_chips.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.channel_chips.customContextMenuRequested.connect(self._channel_menu)
        v.addWidget(self.channel_chips)
        self.empty = label("", "muted", wrap=True); v.addWidget(self.empty)
        v.addSpacing(22)

        self.stats_w = QWidget(); sg = QHBoxLayout(self.stats_w); sg.setContentsMargins(0, 0, 0, 0); sg.setSpacing(48)
        self.s_subs, self.s_views, self.s_videos = Stat("Subscribers"), Stat("Views"), Stat("Videos")
        for s in (self.s_subs, self.s_views, self.s_videos):
            sg.addWidget(s)
        sg.addStretch(1)
        self.refresh_btn = button("Refresh", "ghost", on=self.refresh)
        sg.addWidget(self.refresh_btn, 0, Qt.AlignmentFlag.AlignBottom)
        v.addWidget(self.stats_w); v.addSpacing(26)

        tabs = QHBoxLayout()
        self.tabs = Chips([("queue", "Uploads"), ("videos", "Videos"), ("comments", "Comments")], wrap=False)
        self.tabs.set_value("queue"); self.tabs.changed.connect(self._tab)
        tabs.addWidget(self.tabs); tabs.addStretch(1)
        self.tab_actions = QHBoxLayout(); self.tab_actions.setSpacing(6)
        tabs.addLayout(self.tab_actions)
        v.addLayout(tabs); v.addSpacing(14)

        self.body = QVBoxLayout(); self.body.setSpacing(8)
        v.addLayout(self.body)
        v.addStretch(1)
        self._refresh_channels()
        self._tab("queue")
        QTimer.singleShot(1200, self.refresh)

    # ---------------- channels
    def _refresh_channels(self) -> None:
        self.channel_chips.set_options([(c.id, c.title, c.handle) for c in self.store.channels], keep=False)
        self.channel_chips.set_value(self.store.active_id)
        has = bool(self.store.channels)
        self.channel_chips.setVisible(has)
        self.stats_w.setVisible(has)
        if not self.store.configured:
            self.empty.setText("Connect your YouTube channel to upload clips, schedule posts, edit videos and "
                               "reply to comments. You need a free Google Cloud client once; Connect walks you through it.")
        elif not has:
            self.empty.setText("Click Connect channel and sign in with Google.")
        else:
            self.empty.setText("")
        self.empty.setVisible(not has)

    def connect_channel(self) -> None:
        if not self.store.configured:
            if not SetupDialog(self, self.store).exec():
                return
        self.connect_btn.setEnabled(False); self.connect_btn.setText("Waiting for Google sign-in...")
        self._login_cancel.clear()
        Call(lambda: yt.login(self.store, self._login_cancel), self._connected, self._login_failed)

    def _connected(self, ch) -> None:
        self.connect_btn.setEnabled(True); self.connect_btn.setText("Connect channel")
        self._refresh_channels()
        self.toast(f"Connected {ch.title}")
        self.uploader.resume()
        self.refresh()

    def _login_failed(self, msg: str) -> None:
        self.connect_btn.setEnabled(True); self.connect_btn.setText("Connect channel")
        if "isn't enabled" in msg:
            msg += " Open the console and enable it, then try again."
        self.toast(msg, 6000, error=True)

    def _switch(self, cid: str) -> None:
        self.store.set_active(cid)
        self.refresh()

    def _channel_menu(self, pos) -> None:
        btn = self.channel_chips.childAt(pos)
        if not isinstance(btn, QPushButton):
            return
        cid = self.channel_chips.values[self.channel_chips.group.id(btn)]
        c = self.store.get(cid)
        m = QMenu(self)
        m.addAction("Reconnect", self.connect_channel)
        m.addAction("Disconnect", lambda: self._disconnect(c))
        m.addAction("Change Google client file", lambda: SetupDialog(self, self.store).exec())
        m.exec(self.channel_chips.mapToGlobal(pos))

    def _disconnect(self, c) -> None:
        if QMessageBox.question(self, "Disconnect", f"Disconnect {c.title} from MyClip?") == QMessageBox.StandardButton.Yes:
            self.store.remove(c.id)
            self._refresh_channels()
            self.refresh()

    # ---------------- data
    def refresh(self) -> None:
        ch = self.store.active
        if not ch:
            self._tab(self.tabs.value() or "queue")
            return
        self.refresh_btn.setEnabled(False)
        api = yt.YouTube(self.store, ch.id)

        def load():
            ov = api.overview()
            return ov, api.videos(ov["uploads"], 50)
        Call(load, self._loaded, self._load_failed)

    def _loaded(self, res) -> None:
        self.refresh_btn.setEnabled(True)
        self.overview, self.videos = res
        self.s_subs.val.setText(_num(self.overview["subs"]))
        self.s_views.val.setText(_num(self.overview["views"]))
        self.s_videos.val.setText(_num(self.overview["videos"]))
        if self.tabs.value() == "videos":
            self._render_videos()

    def _load_failed(self, msg: str) -> None:
        self.refresh_btn.setEnabled(True)
        self.toast(msg, 5000, error=True)

    # ---------------- tabs
    def _clear(self, layout) -> None:
        while layout.count():
            it = layout.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
            elif it.layout():
                self._clear(it.layout())

    def _tab(self, t: str) -> None:
        self._clear(self.tab_actions)
        if t == "queue":
            self.tab_actions.addWidget(button("Add videos", "ghost", "plus", on=self.add_files))
            self.pause_btn = button("Resume" if self.uploader.paused else "Pause", "ghost", on=self._toggle_pause)
            self.tab_actions.addWidget(self.pause_btn)
            self.tab_actions.addWidget(button("Clear finished", "ghost", on=lambda: (self.queue.clear_done(), self._render_queue())))
            self._render_queue()
        elif t == "videos":
            self._render_videos()
        else:
            self.tab_actions.addWidget(button("Refresh", "ghost", on=self._load_comments))
            self._load_comments()

    # uploads
    def add_files(self) -> None:
        if not self.store.channels:
            self.toast("Connect a channel first.", error=True)
            return
        files, _ = QFileDialog.getOpenFileNames(self, "Videos to upload", str(Path.home() / "Videos" / "MyClipAI"),
                                                "Videos (*.mp4 *.mov *.mkv *.webm *.avi)")
        if files:
            self.upload_clips([{"path": f, "title": Path(f).stem.split(" - ", 1)[-1]} for f in files])

    def upload_clips(self, clips: list[dict]) -> None:
        """Entry point used by clip cards, results and the library."""
        if not self.store.channels:
            self.toast("Connect your channel in Studio first.", error=True)
            return
        d = UploadDialog(self, self.store, clips)
        if d.exec() and d.items:
            self.queue.add(d.items)
            self.uploader.resume()
            self.tabs.set_value("queue", emit=True)
            n = len(d.items)
            self.toast(f"{n} video{'s' if n > 1 else ''} added to the upload queue")

    def _toggle_pause(self) -> None:
        if self.uploader.paused:
            self.uploader.resume()
        else:
            self.uploader.pause()
        self.pause_btn.setText("Resume" if self.uploader.paused else "Pause")
        self._render_queue()

    def _paused(self, msg: str) -> None:
        self.toast(msg, 7000, error=True)
        if self.tabs.value() == "queue":
            self._tab("queue")

    def _progress(self, item_id: str, f: float) -> None:
        b = self.bars.get(item_id)
        if b:
            b.setValue(int(f * 100))

    def _render_queue(self) -> None:
        if self.tabs.value() != "queue":
            return
        self._clear(self.body)
        self.bars = {}
        items = sorted(self.queue.items, key=lambda i: ({"uploading": 0, "queued": 1, "failed": 2, "done": 3}[i.status], -i.added))
        if not items:
            self.body.addWidget(label("Nothing in the queue. Use Upload on any clip, or Add videos.", "muted"))
            return
        if self.uploader.paused and self.queue.pending:
            self.body.addWidget(label("Uploads are paused.", "faint"))
        for it in items[:100]:
            f = QFrame(); f.setObjectName("panel")
            h = QHBoxLayout(f); h.setContentsMargins(14, 10, 10, 10); h.setSpacing(12)
            col = QVBoxLayout(); col.setSpacing(4)
            col.addWidget(label(it.title or os.path.basename(it.file)))
            ch = self.store.get(it.channel_id)
            bits = [ch.title if ch else "", it.privacy.capitalize() if not it.publish_at else
                    "Publishes " + it.publish_at.replace("T", " ")]
            state = {"queued": "Waiting", "uploading": f"Uploading {int(it.progress * 100)}%", "done": "Uploaded",
                     "failed": "Failed"}[it.status]
            color = {"queued": theme.TEXT_3, "uploading": theme.ACCENT, "done": theme.TEAL, "failed": theme.RED}[it.status]
            rest = "  ·  ".join(b for b in bits if b)
            meta = label(dot_text(color, f'<span style="color:{color}">{state}</span>' +
                                  (f"  ·  {rest}" if rest else "")), "faint")
            col.addWidget(meta)
            if it.status == "uploading" or (it.status == "queued" and it.progress > 0):
                bar = QProgressBar(); bar.setRange(0, 100); bar.setValue(int(it.progress * 100)); bar.setTextVisible(False)
                if it.status == "queued":
                    bar.setObjectName("chat")          # paused part-way: sky, not accent
                col.addWidget(bar); self.bars[it.id] = bar
            if it.error:
                e = label(it.error, "error" if it.status == "failed" else "faint", wrap=True); col.addWidget(e)
            h.addLayout(col, 1)
            if it.status == "done" and it.video_id:
                h.addWidget(button("Open", "ghost", on=lambda _, i=it: webbrowser.open(f"https://youtu.be/{i.video_id}")))
            if it.status == "failed":
                h.addWidget(button("Retry", "ghost", on=lambda _, i=it: self._retry(i)))
            if it.status != "uploading":
                h.addWidget(button("Remove", "ghost", on=lambda _, i=it: (self.queue.remove(i.id), self._render_queue())))
            self.body.addWidget(f)

    def _retry(self, it) -> None:
        if it.publish_at:
            try:
                if datetime.fromisoformat(it.publish_at) <= datetime.now() + timedelta(minutes=5):
                    val, ok = QInputDialog.getText(self, "New time", "Publish at (YYYY-MM-DD HH:MM)",
                                                   text=(datetime.now() + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M"))
                    if not ok:
                        return
                    it.publish_at = datetime.strptime(val.strip(), "%Y-%m-%d %H:%M").strftime("%Y-%m-%dT%H:%M")
            except ValueError:
                self.toast("Use the format 2026-11-26 18:00", error=True)
                return
        self.queue.set(it, status="queued", error="")
        self.uploader.resume()
        self._render_queue()

    # videos
    def _render_videos(self) -> None:
        self._clear(self.body)
        if not self.store.active:
            self.body.addWidget(label("Connect a channel to see your videos.", "muted"))
            return
        if not self.videos:
            self.body.addWidget(label("No videos yet, or still loading.", "muted"))
            return
        for vd in self.videos:
            f = QFrame(); f.setObjectName("panel")
            h = QHBoxLayout(f); h.setContentsMargins(10, 10, 10, 10); h.setSpacing(14)
            th = QLabel(); th.setFixedSize(128, 72); th.setStyleSheet(f"background: {theme.SURFACE_2}; border-radius: 4px;")
            th.setScaledContents(True)
            if vd["thumb"]:
                self._load_img(vd["thumb"], th)
            h.addWidget(th)
            col = QVBoxLayout(); col.setSpacing(4)
            t = label(vd["title"]); t.setStyleSheet("font-weight: 600;"); col.addWidget(t)
            priv = vd["privacy"].capitalize() + (f" until {vd['publish_at'][:16].replace('T', ' ')} UTC" if vd["publish_at"] else "")
            if vd["upload_status"] in ("failed", "rejected"):
                priv = f"Problem: {vd['failure'] or vd['upload_status']}"
            elif vd["upload_status"] == "uploaded":
                priv += "  ·  processing"
            col.addWidget(label(f"{priv}  ·  {_ago(vd['published'])}", "faint"))
            col.addWidget(label(f"{_num(vd['views'])} views  ·  {_num(vd['likes'])} likes  ·  {_num(vd['comments'])} comments", "muted"))
            h.addLayout(col, 1)
            h.addWidget(button("Edit", "ghost", on=lambda _, x=vd: self._edit(x)))
            h.addWidget(button("Open", "ghost", on=lambda _, x=vd: webbrowser.open(f"https://youtu.be/{x['id']}")))
            more = button("", "ghost", "down", "More")
            more.clicked.connect(lambda _, x=vd, b=more: self._video_menu(x, b))
            h.addWidget(more)
            self.body.addWidget(f)

    def _load_img(self, url: str, target: QLabel) -> None:
        reply = self.net.get(QNetworkRequest(QUrl(url)))

        def done():
            pm = QPixmap()
            if pm.loadFromData(reply.readAll().data()):
                try:
                    target.setPixmap(pm)
                except RuntimeError:
                    pass
            reply.deleteLater()
        reply.finished.connect(done)

    def _video_menu(self, vd: dict, btn) -> None:
        m = QMenu(self)
        m.addAction("Copy link", lambda: (QApplication.clipboard().setText(f"https://youtu.be/{vd['id']}"),
                                          self.toast("Copied")))
        m.addAction("Open in YouTube Studio", lambda: webbrowser.open(f"https://studio.youtube.com/video/{vd['id']}/edit"))
        m.addSeparator()
        m.addAction("Delete from YouTube", lambda: self._delete(vd))
        m.exec(btn.mapToGlobal(btn.rect().bottomLeft()))

    def _edit(self, vd: dict) -> None:
        d = EditDialog(self, vd)
        if not d.exec():
            return
        title, desc, tags, priv = d.values()
        api = yt.YouTube(self.store, self.store.active_id)
        thumb = d.thumb_path

        def save():
            api.update(vd, title, desc, tags, priv)
            if thumb:
                api.set_thumbnail(vd["id"], thumb)
        Call(save, lambda _: (self.toast("Saved"), self.refresh()), lambda m: self.toast(m, 6000, error=True))

    def _delete(self, vd: dict) -> None:
        if QMessageBox.question(self, "Delete video", f"Delete “{vd['title']}” from YouTube? This can't be undone.") \
                != QMessageBox.StandardButton.Yes:
            return
        api = yt.YouTube(self.store, self.store.active_id)
        Call(lambda: api.delete(vd["id"]), lambda _: (self.toast("Deleted"), self.refresh()),
             lambda m: self.toast(m, 6000, error=True))

    # comments
    def _load_comments(self) -> None:
        self._clear(self.body)
        if not self.store.active:
            self.body.addWidget(label("Connect a channel to see comments.", "muted"))
            return
        self.body.addWidget(label("Loading...", "faint"))
        api = yt.YouTube(self.store, self.store.active_id)
        Call(lambda: api.comments(40), self._render_comments, lambda m: (self._clear(self.body), self.body.addWidget(label(m, "error", wrap=True))))

    def _render_comments(self, items: list[dict]) -> None:
        if self.tabs.value() != "comments":
            return
        self._clear(self.body)
        if not items:
            self.body.addWidget(label("No comments yet.", "muted"))
            return
        titles = {v["id"]: v["title"] for v in self.videos}
        for c in items:
            f = QFrame(); f.setObjectName("panel")
            h = QHBoxLayout(f); h.setContentsMargins(14, 10, 10, 10); h.setSpacing(12)
            col = QVBoxLayout(); col.setSpacing(4)
            top = label(f"{c['author']}  ·  {_ago(c['published'])}" + (f"  ·  on {titles[c['video_id']]}" if c['video_id'] in titles else ""), "faint")
            col.addWidget(top)
            col.addWidget(label(c["text"], wrap=True))
            if c["replies"]:
                col.addWidget(label(f"{c['replies']} replies", "faint"))
            h.addLayout(col, 1)
            h.addWidget(button("Reply", "ghost", on=lambda _, x=c: self._reply(x)))
            h.addWidget(button("Open", "ghost", on=lambda _, x=c: webbrowser.open(
                f"https://www.youtube.com/watch?v={x['video_id']}&lc={x['id']}")))
            self.body.addWidget(f)

    def _reply(self, c: dict) -> None:
        text, ok = QInputDialog.getMultiLineText(self, "Reply", f"Reply to {c['author']}")
        if not ok or not text.strip():
            return
        api = yt.YouTube(self.store, self.store.active_id)
        Call(lambda: api.reply(c["id"], text.strip()), lambda _: (self.toast("Reply posted"), self._load_comments()),
             lambda m: self.toast(m, 6000, error=True))

    def shutdown(self) -> None:
        self._login_cancel.set()
        self.uploader.shutdown()
