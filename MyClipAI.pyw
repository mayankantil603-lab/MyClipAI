r"""MyClip AI launcher: opens the app with no console window.
The desktop and Start menu icons run this with .venv\Scripts\pythonw.exe.
Output goes to %APPDATA%\MyClipAI\logs\myclip.log."""
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)


def log_path() -> str:
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    d = os.path.join(base, "MyClipAI", "logs")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "myclip.log")


def open_log():
    p = log_path()
    try:
        if os.path.getsize(p) > 4 * 1024 * 1024:          # keep one old log
            old = p + ".old"
            if os.path.exists(old):
                os.remove(old)
            os.replace(p, old)
    except OSError:
        pass
    f = open(p, "a", encoding="utf-8", buffering=1, errors="replace")
    f.write(f"\n===== MyClip AI started {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    return f


def fatal(msg: str) -> None:
    text = f"MyClip AI couldn't start.\n\n{msg}\n\nDetails are in:\n{log_path()}\n\nIf this keeps happening, run run.bat once to repair the setup."
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, text, "MyClip AI", 0x10)
    except Exception:
        pass


if sys.stdout is None or sys.stderr is None or "pythonw" in os.path.basename(sys.executable).lower():
    _log = open_log()
    sys.stdout = sys.stderr = _log

try:
    from myclip.app import main
except Exception as exc:                                    # missing packages, broken venv, ...
    traceback.print_exc()
    fatal(f"{type(exc).__name__}: {exc}")
    sys.exit(1)

try:
    sys.exit(main())
except SystemExit:
    raise
except Exception as exc:
    traceback.print_exc()
    fatal(f"{type(exc).__name__}: {exc}")
    sys.exit(1)
