from __future__ import annotations

import os
import re
from pathlib import Path


def videos_root() -> Path:
    p = Path(os.environ.get("MYCLIP_HOME") or (Path.home() / "Videos" / "MyClipAI"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def downloads_dir() -> Path:
    p = videos_root() / "Downloads"
    p.mkdir(parents=True, exist_ok=True)
    return p


def shorts_dir() -> Path:
    p = videos_root() / "Shorts"
    p.mkdir(parents=True, exist_ok=True)
    return p


def config_dir() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / ".config")
    p = Path(base) / "MyClipAI"
    p.mkdir(parents=True, exist_ok=True)
    return p


_BAD = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def safe_name(text: str, limit: int = 60) -> str:
    """Windows-safe file/folder name. Keeps unicode letters (Hindi etc.)."""
    s = _BAD.sub("", text or "").strip()
    s = re.sub(r"\s+", " ", s).strip(" .")
    s = s[:limit].rstrip(" .")
    if not s or s.lower() in _RESERVED:
        s = "clip"
    return s


def unique_dir(parent: Path, name: str) -> Path:
    base = parent / safe_name(name)
    p, i = base, 2
    while p.exists():
        p = parent / f"{base.name} ({i})"
        i += 1
    p.mkdir(parents=True)
    return p
