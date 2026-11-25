"""Finds AI models already on this PC.

Checks the usual local servers (Ollama, LM Studio, Jan, llama.cpp, KoboldCpp,
vLLM) on their default ports, lists their models, and picks a sensible default
that fits the GPU. If Ollama is installed but not running, it is started."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import requests

from .store import PROVIDERS, Connection

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
SKIP = re.compile(r"embed|bge|nomic|minilm|e5-|rerank|whisper|clip|llava-projector|mmproj|tts", re.I)


def _get(url: str, timeout: float = 1.5) -> Optional[dict]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None


def params_b(name: str, details: dict | None = None) -> float:
    """Parameter count in billions, from Ollama details or the name ("qwen3:8b", "12B")."""
    ps = (details or {}).get("parameter_size", "")
    for src in (ps, name):
        m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", src or "")
        if m:
            return float(m[1])
    return 0.0


def _gb(n: float) -> str:
    return f"{n / 1024**3:.1f} GB"


def _ollama(base: str) -> Optional[Connection]:
    root = base.rsplit("/v1", 1)[0]
    d = _get(root + "/api/tags")
    if d is None:
        return None
    names, info = [], {}
    for m in d.get("models", []):
        n = m.get("name") or m.get("model")
        if not n or SKIP.search(n):
            continue
        det = m.get("details") or {}
        p = params_b(n, det)
        bits = [f"{p:g}B" if p else "", det.get("quantization_level", ""), _gb(m["size"]) if m.get("size") else ""]
        names.append(n)
        info[n] = " · ".join(b for b in bits if b)
    return Connection("Ollama", "Ollama", base, "", models_cache=names, auto=True, model_info=info)


def _openai_compat(provider: str, base: str) -> Optional[Connection]:
    d = _get(base.rstrip("/") + "/models")
    if d is None:
        return None
    items = d.get("data") or d.get("models") or []
    names, info = [], {}
    for m in items:
        n = str(m.get("id") or m.get("name") or "")
        if not n or SKIP.search(n) or (m.get("type") and m.get("type") not in ("llm", "vlm")):
            continue
        p = params_b(n)
        names.append(n)
        info[n] = f"{p:g}B" if p else ""
    if not names:
        return None
    return Connection(provider, provider, base, "", models_cache=names, auto=True, model_info=info)


def best_default(names: list[str], info: dict, vram_mb: int) -> str:
    """Biggest model that fits fully in VRAM (fast); else the smallest one."""
    if not names:
        return ""
    budget = (vram_mb / 1024) if vram_mb else 8.0
    def size_gb(n: str) -> float:
        m = re.search(r"(\d+(?:\.\d+)?) GB", info.get(n, ""))
        if m:
            return float(m[1])
        p = params_b(n)
        return p * 0.62 if p else 99.0      # ~Q4 size estimate
    fits = [n for n in names if size_gb(n) <= budget * 0.9]
    if fits:
        return max(fits, key=lambda n: (params_b(n), -size_gb(n)))
    return min(names, key=size_gb)


def ollama_installed() -> Optional[str]:
    exe = shutil.which("ollama")
    if exe:
        return exe
    local = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe")
    return local if os.path.isfile(local) else None


def start_ollama(wait: float = 8.0) -> bool:
    exe = ollama_installed()
    if not exe:
        return False
    try:
        subprocess.Popen([exe, "serve"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, creationflags=NO_WINDOW | getattr(subprocess, "DETACHED_PROCESS", 0))
    except Exception:
        return False
    end = time.time() + wait
    while time.time() < end:
        if _get("http://localhost:11434/api/tags", 1.0) is not None:
            return True
        time.sleep(0.5)
    return False


def detect(vram_mb: int = 0, start_if_needed: bool = True) -> tuple[list[Connection], list[str]]:
    """Returns (connections found, notes for the user)."""
    notes: list[str] = []
    checks = [("Ollama", PROVIDERS["Ollama"][0])] + \
             [(p, PROVIDERS[p][0]) for p in ("LM Studio", "Jan", "llama.cpp", "KoboldCpp", "vLLM")]
    extra = os.environ.get("MYCLIP_LOCAL_LLM_URLS", "")   # e.g. http://192.168.1.5:11434/v1;http://...
    for i, u in enumerate(x.strip() for x in extra.split(";") if x.strip()):
        checks.append(("Ollama" if ":11434" in u else "llama.cpp", u))

    def probe(pair):
        prov, base = pair
        return _ollama(base) if prov == "Ollama" else _openai_compat(prov, base)

    with ThreadPoolExecutor(max_workers=len(checks)) as ex:
        found = [c for c in ex.map(probe, checks) if c]
    if not any(c.provider == "Ollama" for c in found) and start_if_needed and ollama_installed():
        notes.append("Started Ollama")
        if start_ollama():
            c = _ollama(PROVIDERS["Ollama"][0])
            if c:
                found.insert(0, c)
    # same port answered twice (e.g. llama.cpp vs another on 8080): keep first
    seen, uniq = set(), []
    for c in found:
        k = c.base_url.rstrip("/")
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    for c in uniq:
        c.model = best_default(c.models_cache, c.model_info, vram_mb)
        if len(uniq) > 1 or c.provider != "Ollama":
            c.name = c.provider
        c.name = f"{c.name} (local)"
    if not uniq:
        notes.append("No local AI found. Install Ollama or LM Studio and download a model.")
    elif all(not c.models_cache for c in uniq):
        notes.append("Ollama is running but has no models. Run: ollama pull qwen3:8b")
    return uniq, notes
