from __future__ import annotations

import os
import sys
import traceback


def main() -> int:
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("MyClipAI.Desktop")
        except Exception:
            pass
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication, QMessageBox
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("MyClip AI")
    app.setOrganizationName("MyClipAI")
    app.setStyle("Fusion")
    from PySide6.QtCore import QSettings
    from .ui import theme
    saved = QSettings("MyClipAI", "MyClipAI").value("accent", theme.DEFAULT_ACCENT)
    app.setStyleSheet(theme.build_qss(str(saved or theme.DEFAULT_ACCENT)))

    def hook(et, ev, tb):
        traceback.print_exception(et, ev, tb)
        QMessageBox.critical(None, "MyClip AI", f"Unexpected error:\n{ev}")

    sys.excepthook = hook

    # one window only: opening the icon again brings the running app to the front
    from PySide6.QtNetwork import QLocalServer, QLocalSocket
    key = "MyClipAI-" + (os.environ.get("USERNAME") or os.environ.get("USER") or "user")
    probe = QLocalSocket()
    probe.connectToServer(key)
    if probe.waitForConnected(300):
        probe.write(b"show")
        probe.flush()
        probe.waitForBytesWritten(300)
        probe.disconnectFromServer()
        return 0
    QLocalServer.removeServer(key)                  # left over from a crash
    server = QLocalServer()
    server.listen(key)

    from PySide6.QtGui import QIcon
    ico = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "myclip.ico")
    if os.path.isfile(ico):
        app.setWindowIcon(QIcon(ico))

    from .ui.main_window import MainWindow
    w = MainWindow()

    def bring_front() -> None:
        while server.hasPendingConnections():
            c = server.nextPendingConnection()
            c.disconnectFromServer()
        if w.isMinimized():
            w.showNormal()
        w.show()
        w.raise_()
        w.activateWindow()

    server.newConnection.connect(bring_front)
    w.show()
    _update_downloader_later()
    return app.exec()


def _update_downloader_later() -> None:
    """Keep yt-dlp current (sites change often). Once a day, in the background, no window."""
    import threading

    def run() -> None:
        try:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys.path.insert(0, root)
            import update_tools
            update_tools.main()
        except Exception:
            traceback.print_exc()

    threading.Thread(target=run, daemon=True, name="yt-dlp-update").start()
