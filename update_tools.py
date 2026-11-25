"""Keeps yt-dlp up to date (YouTube and Twitch change often). Runs from run.bat,
at most once a day, and never blocks the app for long."""
import os
import subprocess
import sys
import time

STAMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "ytdlp_checked")


def main() -> None:
    force = "--force" in sys.argv
    try:
        if not force and time.time() - os.path.getmtime(STAMP) < 20 * 3600:
            return
    except OSError:
        pass
    print("Checking for downloader updates...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "-U", "yt-dlp"],
                       timeout=180, stdin=subprocess.DEVNULL, capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:
        return
    try:
        os.makedirs(os.path.dirname(STAMP), exist_ok=True)
        with open(STAMP, "w") as f:
            f.write(str(time.time()))
    except OSError:
        pass


if __name__ == "__main__":
    main()
