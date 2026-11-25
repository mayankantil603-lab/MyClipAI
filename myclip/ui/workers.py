from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QObject, QThread, Signal, Slot

from ..core import discovery, engine, ffmpeg, llm, sources
from ..core.gpu import cuda_info
from ..core.models import Cancelled, JobConfig
from ..core.store import Connection, ConnectionStore


class JobWorker(QObject):
    """Every signal carries the job's token, so the window can ignore a job it already let go of."""
    progress = Signal(int, str, float, str)   # token, stage, fraction (-1 = note), message
    finished = Signal(int, object)            # token, JobResult
    failed = Signal(int, str)
    cancelled = Signal(int)

    def __init__(self, cfg: JobConfig, store: ConnectionStore, api_key: str = "", token: int = 0) -> None:
        super().__init__()
        self.cfg, self.store, self.key, self.token = cfg, store, api_key, token
        self.cancel = threading.Event()

    @Slot()
    def run(self) -> None:
        t = self.token
        try:
            res = engine.run(self.cfg, self.store, lambda st, f, m: self.progress.emit(t, st, float(f), m),
                             self.cancel, self.key)
            if self.cancel.is_set():
                self.cancelled.emit(t)
            else:
                self.finished.emit(t, res)
        except Cancelled:
            self.cancelled.emit(t)
        except Exception as exc:
            if self.cancel.is_set():
                self.cancelled.emit(t)          # errors while stopping are just part of stopping
                return
            traceback.print_exc()
            self.failed.emit(t, sources.friendly_error(str(exc)) or exc.__class__.__name__)


class ProbeWorker(QObject):
    """Quick metadata for the drop zone (duration, title) without blocking."""
    done = Signal(str, str, str)   # source, title, subtitle

    def __init__(self, source: str) -> None:
        super().__init__()
        self.source = source

    @Slot()
    def run(self) -> None:
        try:
            if sources.is_url(self.source):
                info = sources.fetch_info(self.source, threading.Event())
                d = int(info.get("duration") or 0)
                sub = " · ".join(x for x in [info.get("uploader") or info.get("extractor_key") or "",
                                              f"{d // 60}:{d % 60:02d}" if d else "",
                                              "most replayed data" if info.get("heatmap") else ""] if x)
                self.done.emit(self.source, info.get("title") or self.source, sub)
            else:
                mi = ffmpeg.probe(self.source)
                d = int(mi.duration)
                t = f"{d // 3600}:{d % 3600 // 60:02d}:{d % 60:02d}" if d >= 3600 else f"{d // 60}:{d % 60:02d}"
                self.done.emit(self.source, "", f"{mi.width}x{mi.height} · {mi.fps:.0f} fps · {t}")
        except Exception as exc:
            self.done.emit(self.source, "", f"Couldn't read: {str(exc)[:90]}")


class ModelsWorker(QObject):
    done = Signal(str, list, str)   # connection id, models, error

    def __init__(self, conn: Connection, key: str) -> None:
        super().__init__()
        self.conn, self.key = conn, key

    @Slot()
    def run(self) -> None:
        try:
            self.done.emit(self.conn.id, llm.list_models(self.conn, self.key), "")
        except Exception as exc:
            self.done.emit(self.conn.id, [], str(exc)[:160])


class TestWorker(QObject):
    done = Signal(bool, str)

    def __init__(self, conn: Connection, key: str, model: str) -> None:
        super().__init__()
        self.conn, self.key, self.model = conn, key, model

    @Slot()
    def run(self) -> None:
        try:
            llm.test(self.conn, self.key, self.model)
            self.done.emit(True, "Connected")
        except Exception as exc:
            self.done.emit(False, str(exc)[:200])


def start(parent: QObject, worker: QObject, *finish_signals) -> QThread:
    """Run worker.run on a new thread; thread and worker clean themselves up."""
    th = QThread(parent)
    worker.moveToThread(th)
    th.started.connect(worker.run)
    for s in finish_signals:
        s.connect(th.quit)
    th.finished.connect(worker.deleteLater)
    th.finished.connect(th.deleteLater)
    parent._threads = getattr(parent, "_threads", set())
    parent._threads.add(th)
    th.finished.connect(lambda: parent._threads.discard(th))
    worker._thread_ref = th
    th._worker_ref = worker          # a cancelled job can outlive the window's reference to it
    th.start()
    return th


class DetectWorker(QObject):
    done = Signal(list, list)       # connections, notes

    @Slot()
    def run(self) -> None:
        try:
            ok, _, vram = cuda_info()
            found, notes = discovery.detect(vram if ok else 0)
            self.done.emit(found, notes)
        except Exception as exc:
            self.done.emit([], [f"Detection failed: {exc}"])


class LiveWorker(QObject):
    status = Signal(object)        # LiveStatus
    clip = Signal(dict, str)       # clip record, folder
    note = Signal(str)
    finished = Signal()
    failed = Signal(str)

    def __init__(self, cfg: JobConfig, opts, store: ConnectionStore, api_key: str = "") -> None:
        super().__init__()
        from ..core.live import LiveSession
        self.session = LiveSession(cfg, opts, store, api_key, self.status.emit, self.clip.emit, self.note.emit)

    @Slot()
    def run(self) -> None:
        try:
            self.session.run()
            self.finished.emit()
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(sources.friendly_error(str(exc)) or exc.__class__.__name__)
