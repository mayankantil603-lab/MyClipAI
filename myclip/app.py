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
    from .ui.main_window import MainWindow
    w = MainWindow()
    w.show()
    return app.exec()
