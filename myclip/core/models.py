from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional


class Cancelled(Exception):
    """Raised inside workers when the user cancels."""


def run_cancellable(fn, cancel: "Optional[threading.Event]" = None, poll: float = 0.15):
    """Run a blocking call (model download, AI request, ...) so Cancel works at once.
    On cancel we stop waiting and raise Cancelled; the call finishes on its own in the
    background and its result is thrown away."""
    if cancel is None:
        return fn()
    box: dict = {}

    def target() -> None:
        try:
            box["v"] = fn()
        except BaseException as e:      # noqa: BLE001 - re-raised in the caller
            box["e"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    while t.is_alive():
        if cancel.is_set():
            raise Cancelled()
        t.join(poll)
    if "e" in box:
        raise box["e"]
    return box.get("v")


ASPECTS = {
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
    "16:9": (1920, 1080),
}
QUALITY_SCALE = {"1080p": 1.0, "1440p": 4 / 3, "4K": 2.0}


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    segments: list[Segment] = field(default_factory=list)
    language: str = ""

    @property
    def empty(self) -> bool:
        return not self.segments

    def words(self) -> list[Word]:
        out: list[Word] = []
        for s in self.segments:
            if s.words:
                out.extend(s.words)
                continue
            toks = s.text.split()
            if toks:
                step = max(s.end - s.start, 0.05) / len(toks)
                out += [Word(s.start + i * step, s.start + (i + 1) * step, t) for i, t in enumerate(toks)]
        return out

    def words_between(self, a: float, b: float) -> list[Word]:
        return [w for w in self.words() if w.end > a and w.start < b]

    def text_between(self, a: float, b: float) -> str:
        return " ".join(w.text.strip() for w in self.words_between(a, b)).strip()

    def to_dict(self) -> dict:
        return {"language": self.language, "segments": [
            {"start": s.start, "end": s.end, "text": s.text,
             "words": [[w.start, w.end, w.text] for w in s.words]} for s in self.segments]}

    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        return cls([Segment(s["start"], s["end"], s["text"], [Word(*w) for w in s.get("words", [])])
                    for s in d.get("segments", [])], d.get("language", ""))


@dataclass
class MediaInfo:
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool


@dataclass
class Signals:
    """Everything we know about the source beyond the transcript.
    Curves are sampled every `hop` seconds."""
    hop: float = 0.5
    loudness: Any = None            # np.ndarray dBFS
    scene_cuts: list[float] = field(default_factory=list)
    heatmap: list[tuple[float, float, float]] = field(default_factory=list)  # (start, end, 0..1)
    chat_times: list[float] = field(default_factory=list)
    comment_times: list[float] = field(default_factory=list)
    chapters: list[tuple[float, float, str]] = field(default_factory=list)
    title: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    used: list[str] = field(default_factory=list)


@dataclass
class Clip:
    start: float
    end: float
    text: str = ""
    features: dict[str, float] = field(default_factory=dict)
    signal_score: float = 0.0
    llm_score: Optional[float] = None
    score: float = 0.0
    title: str = ""
    hook: str = ""
    reason: str = ""
    description: str = ""
    hashtags: list[str] = field(default_factory=list)
    source: str = "signals"         # signals | llm | both
    file: str = ""
    thumb: str = ""
    visual: str = ""                # what the vision model saw on screen
    hover: list[str] = field(default_factory=list)   # small frames for the card's hover preview
    plan: dict = field(default_factory=dict)         # how it was rendered (see render.RenderPlan)
    words: list = field(default_factory=list)        # caption words [[start, end, text]] in video time

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class CaptionStyle:
    name: str
    font: str = "Arial Black"
    size: float = 0.050          # fraction of output height
    color: str = "FFFFFF"        # RRGGBB
    highlight: str = "FFD84D"
    outline: float = 0.12        # fraction of font size
    position: float = 0.24       # distance from bottom, fraction of height
    uppercase: bool = True
    words_per_line: int = 3


CAPTION_STYLES = {
    "Bold": CaptionStyle("Bold"),
    "Clean": CaptionStyle("Clean", font="Segoe UI Semibold", size=0.040, highlight="FFFFFF",
                          outline=0.08, uppercase=False, words_per_line=5, position=0.20),
    "Pop": CaptionStyle("Pop", size=0.060, highlight="5CE1E6", words_per_line=2, position=0.30),
    "Minimal": CaptionStyle("Minimal", font="Segoe UI", size=0.034, highlight="FFFFFF",
                            outline=0.06, uppercase=False, words_per_line=7, position=0.10),
}


@dataclass
class JobConfig:
    source: str                      # file path or URL
    clip_count: int = 8
    min_len: float = 20.0
    max_len: float = 60.0
    aspect: str = "9:16"             # 9:16 | 1:1 | 4:5 | 16:9 | Original | Custom
    custom_w: int = 1080
    custom_h: int = 1920
    framing: str = "Auto"            # Auto | Fit | Fill | Smart | Facecam
    quality: str = "1080p"
    captions: str = "Bold"           # style name or "Off"
    caption_lang: str = "auto"       # auto | roman (Hinglish) | english | spoken
    caption_quality: str = "best"    # best = listen again to each clip with the most accurate model
    whisper_model: str = "auto"
    device: str = "auto"             # auto | cuda | cpu
    language: str = ""
    connection_id: str = ""          # LLM connection; "" = none
    llm_model: str = ""
    use_external: bool = True        # heatmap/chat/comments
    use_scenes: bool = True
    output_dir: str = ""
    # AI helpers (each is used only when installed in Settings -> AI helpers)
    use_voice: bool = True           # Demucs voice cleanup before the caption pass
    use_vision: bool = True          # a vision model rates what happens on screen
    use_faces: bool = True           # crop follows faces, facecam detection
    hindi_asr: bool = False          # IndicWhisper Hindi for Hindi captions
    sharp_thumbs: bool = True        # sharpest frame + Real-ESRGAN
    # extras
    hook_text: bool = True           # big hook line at the top for the first seconds
    bleep: str = "off"               # off | beep | mute
    learn_channel: str = ""          # YouTube channel whose views tune the scoring ("" = defaults)

    def target_size(self, info: Optional[MediaInfo] = None) -> tuple[int, int]:
        if self.aspect == "Custom":
            w, h = self.custom_w, self.custom_h
        elif self.aspect == "Original" and info:
            w, h = info.width, info.height
            if max(w, h) > 1920:
                k = 1920 / max(w, h)
                w, h = w * k, h * k
            return _even(w), _even(h)
        else:
            w, h = ASPECTS.get(self.aspect, ASPECTS["9:16"])
        k = QUALITY_SCALE.get(self.quality, 1.0)
        return _even(w * k), _even(h * k)


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)
