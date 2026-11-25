"""Vision: a model looks at frames of each shortlisted clip and rates the action on
screen (kills, wins, deaths, funny moments, kill feed). The rating nudges the clip's
score and the description goes to the AI that writes titles and hooks.

Prefers a vision model already in Ollama (gemma3, qwen2.5vl, llava, ...). If the
selected cloud model can see images (gpt-4o, gemini, ...), that works too."""
from __future__ import annotations

import base64
import json
import re
import threading
import time
from typing import Callable, Optional

import requests

from . import ffmpeg
from .models import Cancelled, Clip, run_cancellable

NAME_HINTS = ("vl", "vision", "llava", "gemma3", "minicpm-v", "moondream", "bakllava", "pixtral",
              "mistral-small3", "llama4", "granite3.2-vision")
CLOUD_HINTS = ("gpt-4o", "gpt-4.1", "gpt-5", "o3", "o4", "gemini", "claude", "llama-4", "pixtral", "qwen-vl",
               "qwen2.5-vl", "qwen3-vl", "vision")
_CACHE: dict = {}

PROMPT = ("These are frames from one moment of a gaming video, in order. Rate how much happens on screen for a "
          "short-form clip, 0 (nothing, menus, loading) to 10 (a big play: multi-kill, clutch, win screen, huge "
          "fail, very funny reaction). Notice kill feeds, 'Victory'/'Eliminated' banners, health bars, explosions "
          "and faces reacting. JSON only: {\"action\": 0-10, \"events\": [\"kill\", \"win\", ...], "
          "\"on_screen\": \"one short sentence of what happens\"}")


def root(url: str) -> str:
    return (url or "http://localhost:11434").rstrip("/").rsplit("/v1", 1)[0]


def _caps(r: str, name: str) -> Optional[set]:
    try:
        d = requests.post(r + "/api/show", json={"model": name}, timeout=8).json()
        caps = d.get("capabilities")
        return set(caps) if isinstance(caps, list) else None
    except Exception:
        return None


def find_local_models(url: str = "http://localhost:11434") -> list[tuple[str, int]]:
    """Vision-capable models in Ollama as (name, size in bytes). Cached for a minute."""
    r = root(url)
    hit = _CACHE.get(r)
    if hit and time.time() - hit[0] < 60:
        return hit[1]
    try:
        tags = requests.get(r + "/api/tags", timeout=4).json().get("models", [])
    except Exception:
        return []
    out = []
    for m in tags:
        name = m.get("name") or m.get("model") or ""
        if not name or "embed" in name:
            continue
        caps = _caps(r, name)
        ok = ("vision" in caps) if caps is not None else any(h in name.lower() for h in NAME_HINTS)
        if ok:
            out.append((name, int(m.get("size") or 0)))
    _CACHE[r] = (time.time(), out)
    return out


def choose(found: list[tuple[str, int]], text_model: str, vram_mb: int = 0) -> str:
    """The text model if it can see (no extra model load), else the biggest that fits the GPU, else the smallest."""
    if not found:
        return ""
    names = [n for n, _ in found]
    if text_model in names:
        return text_model
    budget = (vram_mb or 8192) * 1024 * 1024 * 0.95
    fits = [(n, s) for n, s in found if 0 < s <= budget]
    if fits:
        return max(fits, key=lambda x: x[1])[0]
    return min(found, key=lambda x: x[1] or 1 << 60)[0]


def cloud_can_see(model: str) -> bool:
    m = (model or "").lower()
    return any(h in m for h in CLOUD_HINTS) and "embed" not in m and "audio" not in m


def frame_times(c: Clip, n: int = 3) -> list[float]:
    d = c.duration
    peak = float(c.features.get("peak_pos", 0.68)) if c.features else 0.68
    ts = [0.2, min(0.95, max(0.05, peak)), 0.88][:n]
    return [round(c.start + f * d, 2) for f in ts]


def parse_result(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    m = re.search(r"\{.*\}", text, re.S)
    d = json.loads(m.group(0) if m else text)
    try:
        action = float(d.get("action", 5))
    except (TypeError, ValueError):
        action = 5.0
    ev = d.get("events") or []
    return {"action": max(0.0, min(10.0, action)),
            "events": [str(e)[:24] for e in ev][:6] if isinstance(ev, list) else [],
            "on_screen": " ".join(str(d.get("on_screen", "")).split())[:160]}


def _ask_ollama(r: str, model: str, images: list[bytes], context: str, cancel: threading.Event) -> dict:
    body = {"model": model, "stream": False, "format": "json", "keep_alive": "5m",
            "options": {"temperature": 0.1, "num_ctx": 4096, "num_predict": 220},
            "messages": [{"role": "user", "content": PROMPT + (f"\nWhat is said: {context[:400]}" if context else ""),
                          "images": [base64.b64encode(b).decode() for b in images]}]}
    resp = run_cancellable(lambda: requests.post(r + "/api/chat", json=body, timeout=240), cancel)
    resp.raise_for_status()
    return parse_result((resp.json().get("message") or {}).get("content", ""))


def _ask_cloud(conn, key: str, model: str, images: list[bytes], context: str, cancel: threading.Event) -> dict:
    content = [{"type": "text", "text": PROMPT + (f"\nWhat is said: {context[:400]}" if context else "")}]
    content += [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode()}}
                for b in images]
    body = {"model": model, "temperature": 0.1, "max_tokens": 220, "messages": [{"role": "user", "content": content}]}
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    url = conn.base_url.rstrip("/") + "/chat/completions"
    resp = run_cancellable(lambda: requests.post(url, json=body, headers=h, timeout=180), cancel)
    resp.raise_for_status()
    return parse_result(resp.json()["choices"][0]["message"]["content"] or "")


def pick_target(store, text_conn, text_key: str, text_model: str, vram_mb: int = 0):
    """('ollama', root, model) or ('cloud', conn, key, model) or None."""
    urls = []
    if text_conn is not None and text_conn.provider == "Ollama":
        urls.append(text_conn.base_url)
    if store is not None:
        urls += [c.base_url for c in store.items if c.provider == "Ollama"]
    urls.append("http://localhost:11434")
    seen = set()
    for u in urls:
        r = root(u)
        if r in seen:
            continue
        seen.add(r)
        found = find_local_models(r)
        if found:
            tm = text_model if (text_conn is not None and text_conn.provider == "Ollama") else ""
            return ("ollama", r, choose(found, tm, vram_mb))
    if text_conn is not None and not text_conn.is_local and text_model and cloud_can_see(text_model) \
            and (text_key or not text_conn.needs_key):
        return ("cloud", text_conn, text_key, text_model)
    return None


def review(clips: list[Clip], media: str, target, cancel: threading.Event,
           log: Optional[Callable[[str], None]] = None) -> int:
    """Score each clip's visuals in place. Returns how many were scored."""
    from .moments import vision_adjust
    n = 0
    for i, c in enumerate(clips, 1):
        if cancel.is_set():
            raise Cancelled()
        if log:
            log(f"Watching clip {i} of {len(clips)}")
        imgs = [b for b in (ffmpeg.grab_jpeg(media, t, 640, 5) for t in frame_times(c)) if b]
        if not imgs:
            continue
        try:
            if target[0] == "ollama":
                res = _ask_ollama(target[1], target[2], imgs, c.text, cancel)
            else:
                res = _ask_cloud(target[1], target[2], target[3], imgs, c.text, cancel)
        except Cancelled:
            raise
        except Exception as exc:
            print(f"[MyClip] vision skipped for one clip: {exc}", flush=True)
            continue
        c.features["vision"] = res["action"] / 10.0
        c.visual = res["on_screen"]
        c.score = vision_adjust(c.score, res["action"])
        n += 1
    return n
