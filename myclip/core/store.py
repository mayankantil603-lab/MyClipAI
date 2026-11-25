"""Saved connections (LLM APIs) and presets.

Non-secret data lives in %APPDATA%/MyClipAI/*.json. API keys live in Windows
Credential Manager through `keyring`. If keyring is unavailable, keys are kept
in memory for this session only. Keys are never written to disk by us."""
from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional

KEYRING_SERVICE = "MyClipAI"

PROVIDERS = {
    #  name:        (base_url, env var, recommended models)
    "Ollama":      ("http://localhost:11434/v1", "", []),
    "LM Studio":   ("http://localhost:1234/v1", "", []),
    "Jan":         ("http://localhost:1337/v1", "", []),
    "llama.cpp":   ("http://localhost:8080/v1", "", []),
    "KoboldCpp":   ("http://localhost:5001/v1", "", []),
    "vLLM":        ("http://localhost:8000/v1", "", []),
    "OpenAI":      ("https://api.openai.com/v1", "MYCLIP_OPENAI_API_KEY", ["gpt-4.1", "gpt-4.1-mini", "gpt-4o"]),
    "Gemini":      ("https://generativelanguage.googleapis.com/v1beta/openai", "MYCLIP_GEMINI_API_KEY",
                    ["gemini-2.5-pro", "gemini-2.5-flash"]),
    "Groq":        ("https://api.groq.com/openai/v1", "MYCLIP_GROQ_API_KEY",
                    ["llama-3.3-70b-versatile", "qwen/qwen3-32b"]),
    "OpenRouter":  ("https://openrouter.ai/api/v1", "MYCLIP_OPENROUTER_API_KEY",
                    ["anthropic/claude-sonnet-4", "google/gemini-2.5-pro", "openai/gpt-4.1"]),
    "Custom":      ("", "MYCLIP_LLM_API_KEY", []),
}


LOCAL_PROVIDERS = {"Ollama", "LM Studio", "Jan", "llama.cpp", "KoboldCpp", "vLLM"}


@dataclass
class Connection:
    name: str
    provider: str
    base_url: str
    model: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    models_cache: list[str] = field(default_factory=list)
    auto: bool = False              # created by local detection
    model_info: dict = field(default_factory=dict)   # name -> "8B · 5.2 GB"

    @property
    def is_local(self) -> bool:
        return self.provider in LOCAL_PROVIDERS

    @property
    def needs_key(self) -> bool:
        return not self.is_local

    @property
    def recommended(self) -> list[str]:
        return PROVIDERS.get(self.provider, ("", "", []))[2]


class _Keys:
    def __init__(self) -> None:
        self._mem: dict[str, str] = {}
        self._lock = threading.Lock()
        try:
            import keyring  # noqa: F401
            self.backend = keyring
        except Exception:
            self.backend = None

    @property
    def secure(self) -> bool:
        return self.backend is not None

    def get(self, cid: str) -> str:
        with self._lock:
            if cid in self._mem:
                return self._mem[cid]
        if self.backend:
            try:
                return self.backend.get_password(KEYRING_SERVICE, cid) or ""
            except Exception:
                return ""
        return ""

    def set(self, cid: str, key: str) -> bool:
        """Returns True if stored securely, False if session-only."""
        if self.backend:
            try:
                if key:
                    self.backend.set_password(KEYRING_SERVICE, cid, key)
                else:
                    self.delete(cid)
                with self._lock:
                    self._mem.pop(cid, None)
                return True
            except Exception:
                pass
        with self._lock:
            self._mem[cid] = key
        return False

    def delete(self, cid: str) -> None:
        with self._lock:
            self._mem.pop(cid, None)
        if self.backend:
            try:
                self.backend.delete_password(KEYRING_SERVICE, cid)
            except Exception:
                pass


class ConnectionStore:
    def __init__(self, path: Path, keys: Optional[_Keys] = None) -> None:
        self.path = path
        self.keys = keys or _Keys()
        self.items: list[Connection] = []
        self.active_id = ""
        self.load()

    def load(self) -> None:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            known = {f.name for f in fields(Connection)}
            self.items = [Connection(**{k: v for k, v in c.items() if k in known}) for c in d.get("connections", [])]
            self.active_id = d.get("active", "")
        except Exception:
            self.items, self.active_id = [], ""

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"active": self.active_id, "connections": [asdict(c) for c in self.items]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def get(self, cid: str) -> Optional[Connection]:
        return next((c for c in self.items if c.id == cid), None)

    @property
    def active(self) -> Optional[Connection]:
        return self.get(self.active_id)

    def set_active(self, cid: str) -> None:
        self.active_id = cid if self.get(cid) or cid == "" else self.active_id
        self.save()

    def upsert(self, c: Connection, key: Optional[str] = None) -> bool:
        secure = True
        if key is not None:
            secure = self.keys.set(c.id, key)
        existing = self.get(c.id)
        if existing:
            self.items[self.items.index(existing)] = c
        else:
            self.items.append(c)
        self.save()
        return secure

    def remove(self, cid: str) -> None:
        self.items = [c for c in self.items if c.id != cid]
        self.keys.delete(cid)
        if self.active_id == cid:
            self.active_id = ""
        self.save()

    def merge_detected(self, found: list["Connection"]) -> tuple[list[str], list[str]]:
        """Add or refresh auto-detected local servers. Returns (added names, updated names).
        Servers that went offline are kept (models stay listed) so choices aren't lost."""
        added, updated = [], []
        for f in found:
            ex = next((c for c in self.items if c.provider == f.provider and
                       c.base_url.rstrip("/") == f.base_url.rstrip("/")), None)
            if ex:
                ex.models_cache, ex.model_info = f.models_cache, f.model_info
                if not ex.model or ex.model not in f.models_cache:
                    ex.model = f.model
                updated.append(ex.name)
            else:
                self.items.append(f)
                added.append(f.name)
        if not self.active and found and not self.active_id:
            self.active_id = found[0].id
        self.save()
        return added, updated

    def api_key(self, c: Connection) -> str:
        key = self.keys.get(c.id)
        if key:
            return key
        env = PROVIDERS.get(c.provider, ("", "", []))[1]
        return os.environ.get(env, "") if env else os.environ.get("MYCLIP_LLM_API_KEY", "")


@dataclass
class Preset:
    name: str
    values: dict
    builtin: bool = False


BUILTIN_PRESETS = [
    Preset("Best quality", {"whisper_model": "large-v3-turbo", "device": "auto", "quality": "1080p",
                            "use_external": True, "use_scenes": True, "framing": "Smart",
                            "use_llm": True, "caption_quality": "best"}, True),
    Preset("Fast", {"whisper_model": "small", "device": "auto", "quality": "1080p",
                    "use_external": False, "use_scenes": False, "framing": "Fit",
                    "use_llm": False, "caption_quality": "fast"}, True),
    Preset("Offline", {"whisper_model": "auto", "device": "auto", "use_external": False,
                       "use_scenes": True, "framing": "Smart", "use_llm": "local", "caption_quality": "best"}, True),
]
PRESET_KEYS = {"whisper_model", "device", "quality", "use_external", "use_scenes", "framing",
               "use_llm", "aspect", "captions", "caption_lang", "caption_quality", "clip_count", "min_len", "max_len"}


class PresetStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.custom: list[Preset] = []
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            self.custom = [Preset(p["name"], p["values"]) for p in d]
        except Exception:
            pass

    @property
    def all(self) -> list[Preset]:
        return BUILTIN_PRESETS + self.custom

    def find(self, name: str) -> Optional[Preset]:
        return next((p for p in self.all if p.name == name), None)

    def save_custom(self, name: str, values: dict) -> Preset:
        name = name.strip() or "My preset"
        if any(p.name == name for p in BUILTIN_PRESETS):
            name += " (custom)"
        clean = {k: v for k, v in values.items() if k in PRESET_KEYS}
        self.custom = [p for p in self.custom if p.name != name] + [Preset(name, clean)]
        self._write()
        return self.custom[-1]

    def delete(self, name: str) -> None:
        self.custom = [p for p in self.custom if p.name != name]
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([{"name": p.name, "values": p.values} for p in self.custom],
                                        indent=2), encoding="utf-8")


def resolve_llm_for_preset(preset: Preset, store: ConnectionStore) -> Optional[str]:
    """Which connection id a preset should activate. None = leave as is, "" = off."""
    mode = preset.values.get("use_llm")
    if mode is False:
        return ""
    if mode == "local":
        local = next((c for c in store.items if c.is_local and c.models_cache), None) or \
            next((c for c in store.items if c.is_local), None)
        return local.id if local else ""
    if mode is True:
        if store.active:
            return store.active.id
        rank = ["OpenAI", "Gemini", "OpenRouter", "Groq", "Custom", *sorted(LOCAL_PROVIDERS)]
        for prov in rank:
            for c in store.items:
                if c.provider == prov and (store.api_key(c) if c.needs_key else c.models_cache):
                    return c.id
        return ""
    return None
