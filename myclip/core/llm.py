"""LLM: lists models, proposes moments from the full transcript, and writes
titles/hooks/hashtags. Works with any OpenAI-compatible chat endpoint."""
from __future__ import annotations

import json
import re
import threading
from typing import Callable, Optional

import requests

from .models import Cancelled, Clip, Signals, Transcript, run_cancellable
from .store import Connection

TIMEOUT = 240


def _headers(key: str) -> dict:
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def list_models(conn: Connection, key: str) -> list[str]:
    base = conn.base_url.rstrip("/")
    names: list[str] = []
    try:
        r = requests.get(base + "/models", headers=_headers(key), timeout=15)
        r.raise_for_status()
        data = r.json()
        items = data.get("data") or data.get("models") or []
        names = [str(m.get("id") or m.get("name")) for m in items if (m.get("id") or m.get("name"))]
    except Exception:
        if conn.provider == "Ollama":
            r = requests.get(base.rsplit("/v1", 1)[0] + "/api/tags", timeout=10)
            r.raise_for_status()
            names = [m["name"] for m in r.json().get("models", [])]
        else:
            raise
    names = [n.removeprefix("models/") for n in names]
    if conn.provider == "OpenAI":
        names = [n for n in names if n.startswith(("gpt-", "o1", "o3", "o4", "chatgpt"))
                 and not any(x in n for x in ("audio", "realtime", "transcribe", "tts", "image", "search"))]
    if conn.provider == "Gemini":
        names = [n for n in names if "gemini" in n and "embedding" not in n]
    rec = conn.recommended
    return sorted(set(names), key=lambda n: (n not in rec, rec.index(n) if n in rec else 0, n))


def test(conn: Connection, key: str, model: str) -> str:
    out = chat(conn, key, model, [{"role": "user", "content": "Reply with the single word: ok"}],
               max_tokens=5, json_mode=False)
    return out.strip()[:40] or "ok"


def ctx_for(messages: list[dict], max_tokens: int) -> int:
    """Smallest context that holds the prompt + answer (about 3.2 chars/token),
    rounded up to 2048. Smaller context = less VRAM = more layers on the GPU."""
    chars = sum(len(m.get("content", "")) for m in messages)
    need = int(chars / 3.2) + max_tokens + 512
    return min(32768, max(4096, -(-need // 2048) * 2048))


def _ollama_root(conn: Connection) -> str:
    return conn.base_url.rstrip("/").rsplit("/v1", 1)[0]


def _ollama_chat(conn: Connection, model: str, messages: list[dict], max_tokens: int, json_mode: bool,
                 cancel: Optional[threading.Event]) -> str:
    """Streams the answer. On Cancel the connection is closed, which makes Ollama stop generating."""
    body = {"model": model, "messages": messages, "stream": True, "keep_alive": "10m",
            "think": "low" if model.startswith("gpt-oss") else False,
            "options": {"temperature": 0.2, "num_ctx": ctx_for(messages, max_tokens), "num_predict": max_tokens}}
    if json_mode:
        body["format"] = "json"
    url = _ollama_root(conn) + "/api/chat"
    live: dict = {}

    def once() -> str:
        r = requests.post(url, json=body, timeout=(10, TIMEOUT * 4), stream=True)
        live["r"] = r
        try:
            if r.status_code == 400 and "think" in body and "think" in r.text.lower():
                return "\x00retry-no-think"
            if r.status_code == 404:
                raise RuntimeError(f"Model {model} isn't installed in Ollama.")
            r.raise_for_status()
            parts: list[str] = []
            for line in r.iter_lines():
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("error"):
                    raise RuntimeError(f"Ollama: {d['error']}")
                parts.append((d.get("message") or {}).get("content") or "")
                if d.get("done"):
                    break
            return "".join(parts)
        finally:
            r.close()

    for attempt in range(3):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        try:
            out = run_cancellable(once, cancel)
            if out == "\x00retry-no-think":
                body.pop("think")            # model or Ollama version without thinking control
                continue
            return out
        except Cancelled:
            r = live.get("r")
            if r is not None:
                try:
                    r.close()
                except Exception:
                    pass
            raise
        except requests.RequestException as exc:
            if attempt == 2:
                raise RuntimeError(f"Ollama request failed: {exc}") from None
            (cancel or threading.Event()).wait(3)
    return ""


def unload_ollama(base_url: str = "http://localhost:11434") -> None:
    """Ask Ollama to free VRAM now (before Whisper, and after the job)."""
    root = base_url.rstrip("/").rsplit("/v1", 1)[0]
    try:
        loaded = requests.get(root + "/api/ps", timeout=3).json().get("models", [])
        for m in loaded:
            requests.post(root + "/api/generate", json={"model": m.get("name") or m.get("model"), "keep_alive": 0},
                          timeout=10)
    except Exception:
        pass


def chat(conn: Connection, key: str, model: str, messages: list[dict], max_tokens: int = 4000,
         json_mode: bool = True, cancel: Optional[threading.Event] = None) -> str:
    if conn.provider == "Ollama":
        return _ollama_chat(conn, model, messages, max_tokens, json_mode, cancel)
    body = {"model": model, "messages": messages, "temperature": 0.3, "max_tokens": max_tokens}
    if json_mode and conn.provider in ("OpenAI", "Groq", "OpenRouter", "Ollama", "Gemini"):
        body["response_format"] = {"type": "json_object"}
    url = conn.base_url.rstrip("/") + "/chat/completions"
    for attempt in range(3):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        try:
            r = run_cancellable(lambda: requests.post(url, json=body, headers=_headers(key), timeout=TIMEOUT), cancel)
            if r.status_code == 400 and "response_format" in body:
                body.pop("response_format")
                continue
            if r.status_code in (429, 500, 502, 503) and attempt < 2:
                (cancel or threading.Event()).wait(4 * (attempt + 1))
                continue
            if r.status_code == 401:
                raise RuntimeError("The API key was rejected (401).")
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"] or ""
        except requests.RequestException as exc:
            if attempt == 2:
                raise RuntimeError(f"LLM request failed: {exc}") from None
    raise RuntimeError("LLM request failed.")


def parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def transcript_lines(tr: Transcript, a: float = 0, b: float = 1e12) -> list[str]:
    out = []
    for s in tr.segments:
        if s.end < a or s.start > b:
            continue
        out.append(f"[{s.start:.1f}-{s.end:.1f}] {s.text.strip()}")
    return out


def chunks(lines: list[str], max_chars: int = 24000, overlap: int = 8) -> list[list[str]]:
    res, cur, size = [], [], 0
    for ln in lines:
        if size + len(ln) > max_chars and cur:
            res.append(cur)
            cur = cur[-overlap:]
            size = sum(len(x) for x in cur)
        cur.append(ln)
        size += len(ln) + 1
    if cur:
        res.append(cur)
    return res


SYSTEM = """You are an experienced short-form editor (YouTube Shorts, Reels, TikTok).
You pick moments from long videos that work as standalone clips.
A good clip: a hook in the first 2-3 seconds, a clear payoff (a play, a fail, a joke,
a reveal, a strong opinion), understandable without the rest of the video, no dead air.
Use only timestamps that exist in the transcript. Keep each clip between {min_len:.0f}
and {max_len:.0f} seconds. Write titles like a human creator would: short, specific,
no clickbait words like "insane" unless they are said in the clip, no emoji.
Answer in the language of the transcript. Return JSON only."""

PROPOSE = """Video: {title}
{context}
Transcript with timestamps in seconds:
{lines}

Propose up to {n} clips from this part. JSON:
{{"clips":[{{"start":float,"end":float,"score":0-100,"title":"max 60 chars",
"hook":"first line viewers hear or a 1-line on-screen hook","reason":"why it works, 1 sentence",
"description":"1-2 sentences for the post","hashtags":["#tag", "..."]}}]}}"""


def _system(min_len: float, max_len: float, style: str = "") -> str:
    s = SYSTEM.format(min_len=min_len, max_len=max_len)
    return s.replace("Answer in the language of the transcript.", style) if style else s


def propose(conn: Connection, key: str, model: str, tr: Transcript, sig: Signals, want: int,
            min_len: float, max_len: float, log: Callable[[str], None],
            cancel: threading.Event, style: str = "") -> list[Clip]:
    ctx = []
    if sig.description:
        ctx.append("Description: " + sig.description[:800].replace("\n", " "))
    if sig.tags:
        ctx.append("Tags: " + ", ".join(sig.tags[:15]))
    if sig.chapters:
        ctx.append("Chapters: " + "; ".join(f"{a:.0f}s {t}" for a, _, t in sig.chapters[:30]))
    hot = sorted(sig.heatmap, key=lambda h: -h[2])[:6]
    if hot:
        ctx.append("Most replayed by viewers: " + ", ".join(f"{a:.0f}-{b:.0f}s" for a, b, _ in hot))
    parts = chunks(transcript_lines(tr), max_chars=12000 if conn.is_local else 24000)
    per = max(3, -(-want * 2 // max(len(parts), 1)))
    system = _system(min_len, max_len, style)
    out: list[Clip] = []
    for i, part in enumerate(parts, 1):
        if cancel.is_set():
            raise Cancelled()
        log(f"Reading transcript part {i} of {len(parts)}")
        msg = PROPOSE.format(title=sig.title or "(untitled)", context="\n".join(ctx),
                             lines="\n".join(part), n=per)
        try:
            data = parse_json(chat(conn, key, model, [{"role": "system", "content": system},
                                                      {"role": "user", "content": msg}], cancel=cancel))
        except Cancelled:
            raise
        except Exception as exc:
            log(f"Part {i} skipped: {exc}")
            continue
        for d in data.get("clips", []):
            try:
                c = Clip(float(d["start"]), float(d["end"]))
            except Exception:
                continue
            if c.end <= c.start:
                continue
            c.llm_score = max(0.0, min(100.0, float(d.get("score", 60))))
            c.title = str(d.get("title", "")).strip()[:80]
            c.hook = str(d.get("hook", "")).strip()[:140]
            c.reason = str(d.get("reason", "")).strip()[:240]
            c.description = str(d.get("description", "")).strip()[:400]
            c.hashtags = [("#" + h.lstrip("#")).replace(" ", "") for h in d.get("hashtags", []) if str(h).strip()][:8]
            out.append(c)
    return out


REVIEW = """These are the final candidate clips. For each, rate it 0-100 as a standalone short
and write the title, hook, description and hashtags. Penalise clips that need context,
start mid-thought, or end before the payoff.
{items}
JSON: {{"clips":[{{"id":int,"score":0-100,"title":"","hook":"","reason":"","description":"","hashtags":[]}}]}}"""


def review(conn: Connection, key: str, model: str, clips: list[Clip], sig: Signals,
           log: Callable[[str], None], cancel: threading.Event, style: str = "") -> None:
    if not clips:
        return
    items = "\n".join(f"[{i}] {c.start:.0f}-{c.end:.0f}s: {(c.text or '(no speech)')[:700]}"
                      for i, c in enumerate(clips))
    try:
        data = parse_json(chat(conn, key, model, [
            {"role": "system", "content": _system(0, 999, style)},
            {"role": "user", "content": f"Video: {sig.title}\n" + REVIEW.format(items=items)}], cancel=cancel))
    except Cancelled:
        raise
    except Exception as exc:
        log(f"Final review skipped: {exc}")
        return
    for d in data.get("clips", []):
        try:
            c = clips[int(d.get("id", -1))]
        except (IndexError, ValueError, TypeError):
            continue
        sc = d.get("score")
        if sc is not None:
            c.llm_score = float(sc) if c.llm_score is None else round((c.llm_score + float(sc)) / 2, 1)
            c.score = round(min(100.0, 0.55 * c.llm_score + 0.45 * c.signal_score), 1)
        for attr in ("title", "hook", "reason", "description"):
            v = str(d.get(attr, "") or "").strip()
            if v:
                setattr(c, attr, v[:400])
        tags = d.get("hashtags") or []
        if tags:
            c.hashtags = [("#" + str(h).lstrip("#")).replace(" ", "") for h in tags][:8]
