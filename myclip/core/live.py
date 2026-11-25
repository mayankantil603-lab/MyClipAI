"""Live mode.

Records a live stream into a rolling buffer of 2-second segments, listens to
the streamer's voice and the chat speed, and when either suddenly rises above
that stream's own normal level it marks a moment (with pre-roll so the build-up
is included). When things calm down the moment ends, the segments are joined,
and the clip is finished like any other: transcript, captions, framing, title."""
from __future__ import annotations

import collections
import json
import math
import os
import queue
import re
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import ffmpeg, llm, moments, render, subtitles, transcribe
from .models import Cancelled, Clip, JobConfig, Signals
from .paths import config_dir, safe_name, shorts_dir, unique_dir
from .store import ConnectionStore

HOP = 0.5
SENSITIVITY = {"Low": 2.8, "Normal": 2.1, "High": 1.6}


# ---------------------------------------------------------------- detection

@dataclass
class Moment:
    start: float
    end: float
    peak: float = 0.0
    reason: str = ""
    manual: bool = False


class SpikeDetector:
    """Feed one sample every HOP seconds: feed(t, loudness_db, chat_rate).
    Baselines are robust (median/MAD over the last 5 minutes) and frozen while
    a moment is running, so the spike itself doesn't become the new normal."""

    def __init__(self, sensitivity: str = "Normal", min_len: float = 15.0, max_len: float = 60.0,
                 pre: float = 18.0, post: float = 3.0, calm: float = 6.0, cooldown: float = 12.0,
                 warmup: float = 40.0, min_hold: float = 1.5, hop: float = HOP) -> None:
        self.on = SENSITIVITY.get(sensitivity, 2.1)
        self.off = max(1.2, self.on * 0.6)
        self.max_len = max(max_len, 8.0)
        self.min_len = min(min_len, self.max_len)
        self.pre = min(pre, self.max_len * 0.35)
        self.post, self.calm, self.cooldown = post, calm, cooldown
        self.warmup, self.min_hold = warmup, min_hold
        n = int(300 / hop)
        self.loud: collections.deque = collections.deque(maxlen=n)
        self.chat: collections.deque = collections.deque(maxlen=n)
        self.t0: Optional[float] = None
        self.hot_since: Optional[float] = None
        self.next_ok = 0.0
        self.last_end = -1e9
        self.active: Optional[dict] = None
        self.last = (0.0, 0.0, 0.0)
        self.recent: collections.deque = collections.deque(maxlen=max(1, int(2.0 / hop)))

    @staticmethod
    def _z(buf, x: float, floor: float) -> float:
        if len(buf) < 20:
            return 0.0
        a = np.fromiter(buf, dtype=np.float64)
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med))) * 1.4826
        return (x - med) / max(mad, floor)

    def score(self, loud: Optional[float], chat: Optional[float]) -> tuple[float, float, float, str]:
        sl = self._z(self.loud, loud, 2.5) if loud is not None else 0.0
        sc = self._z(self.chat, chat, 1.0) if chat is not None else 0.0
        hi, lo = max(sl, sc), min(sl, sc)
        s = hi + (0.4 * lo if lo > 0.5 else 0.0)        # both together count extra
        reason = "voice and chat" if lo > 1.0 else ("voice" if sl >= sc else "chat")
        return s, sl, sc, reason

    @property
    def warm(self) -> bool:
        return self.t0 is not None and len(self.loud) + len(self.chat) > 0 and \
            (self._t - self.t0) >= self.warmup

    def feed(self, t: float, loud: Optional[float], chat: Optional[float]) -> list[Moment]:
        if self.t0 is None:
            self.t0 = t
        self._t = t
        s, sl, sc, reason = self.score(loud, chat)
        self.last = (s, sl, sc)
        self.recent.append(s)
        out: list[Moment] = []
        if self.active is None:
            if loud is not None:
                self.loud.append(loud)
            if chat is not None:
                self.chat.append(chat)
            if t - self.t0 >= self.warmup and t >= self.next_ok and s >= self.on:
                if self.hot_since is None:
                    self.hot_since = t
                if t - self.hot_since >= self.min_hold:
                    self.active = {"start": max(self.t0, self.last_end, self.hot_since - self.pre), "last_hot": t,
                                   "peak": s, "reason": reason}
            else:
                self.hot_since = None
        else:
            a = self.active
            if sum(self.recent) / len(self.recent) >= self.off:   # smoothed: noise can't keep it open
                a["last_hot"] = t
            a["peak"] = max(a["peak"], s)
            end = None
            if t - a["last_hot"] >= self.calm:
                end = a["last_hot"] + self.post
            elif t - a["start"] >= self.max_len:
                end = a["start"] + self.max_len
            if end is not None:
                out.append(self._close(end))
        return out

    def _close(self, end: float) -> Moment:
        a = self.active
        end = min(max(end, a["start"] + self.min_len), a["start"] + self.max_len)
        m = Moment(round(a["start"], 2), round(end, 2), round(a["peak"], 2), a["reason"])
        self.active, self.hot_since = None, None
        self.next_ok = end + self.cooldown
        self.last_end = m.end
        return m

    def flush(self, t: float) -> Optional[Moment]:
        """Stream ended or user stopped: close a running moment at t."""
        if self.active is None:
            return None
        a = self.active
        if t - a["start"] < self.min_len * 0.6:
            self.active = None
            return None
        m = self._close(min(t, a["last_hot"] + self.post))
        m.end = min(m.end, t)
        return m

    def busy_from(self) -> Optional[float]:
        return self.active["start"] if self.active else None


# ---------------------------------------------------------------- chat

class ChatCounter:
    def __init__(self) -> None:
        self.times: collections.deque = collections.deque(maxlen=50000)
        self.lock = threading.Lock()
        self.ok = False
        self.source = ""

    def add(self, t: Optional[float] = None, n: int = 1) -> None:
        t = time.time() if t is None else t
        with self.lock:
            for _ in range(n):
                self.times.append(t)

    def rate(self, window: float = 5.0, now: Optional[float] = None) -> float:
        now = time.time() if now is None else now
        with self.lock:
            return float(sum(1 for x in self.times if now - window <= x <= now))


def twitch_channel(url: str) -> str:
    m = re.search(r"twitch\.tv/(?!videos/|directory)([A-Za-z0-9_]{3,25})(?:[/?#]|$)", url or "", re.I)
    return m[1].lower() if m else ""


def irc_privmsg_count(text: str) -> int:
    return sum(1 for line in text.split("\r\n") if " PRIVMSG #" in line)


class TwitchChat(threading.Thread):
    """Anonymous read-only Twitch chat (no login needed)."""

    def __init__(self, channel: str, counter: ChatCounter, stop: threading.Event) -> None:
        super().__init__(daemon=True, name="twitch-chat")
        self.channel, self.counter, self.stop = channel, counter, stop

    def run(self) -> None:
        self.counter.source = "Twitch chat"
        while not self.stop.is_set():
            try:
                with socket.create_connection(("irc.chat.twitch.tv", 6667), timeout=15) as s:
                    s.settimeout(2.0)
                    s.sendall(f"NICK justinfan{int(time.time()) % 90000 + 10000}\r\nJOIN #{self.channel}\r\n".encode())
                    self.counter.ok = True
                    buf = ""
                    while not self.stop.is_set():
                        try:
                            data = s.recv(65536)
                        except socket.timeout:
                            continue
                        if not data:
                            break
                        buf += data.decode("utf-8", "replace")
                        lines, buf = buf.rsplit("\r\n", 1) if "\r\n" in buf else ("", buf)
                        if "PING" in lines:
                            s.sendall(b"PONG :tmi.twitch.tv\r\n")
                        n = irc_privmsg_count(lines + "\r\n")
                        if n:
                            self.counter.add(n=n)
            except Exception:
                self.counter.ok = False
            self.stop.wait(5)


class YouTubeChat(threading.Thread):
    """YouTube live chat through yt-dlp's live chat downloader, tailed as it grows."""

    def __init__(self, url: str, counter: ChatCounter, stop: threading.Event, folder: str) -> None:
        super().__init__(daemon=True, name="yt-chat")
        self.url, self.counter, self.stop, self.folder = url, counter, stop, folder

    def run(self) -> None:
        self.counter.source = "YouTube chat"
        threading.Thread(target=self._download, daemon=True).start()
        seen: dict[str, int] = {}
        while not self.stop.wait(1.0):
            for p in Path(self.folder).glob("*live_chat*"):
                try:
                    size = p.stat().st_size
                    pos = seen.get(p.name, 0)
                    if size <= pos:
                        continue
                    with open(p, "rb") as f:
                        f.seek(pos)
                        chunk = f.read(size - pos)
                    seen[p.name] = size
                    n = chunk.count(b"addChatItemAction")
                    if n:
                        self.counter.ok = True
                        self.counter.add(n=n)
                except OSError:
                    continue

    def _download(self) -> None:
        try:
            import yt_dlp

            def hook(d: dict) -> None:
                if self.stop.is_set():
                    raise Cancelled()

            opts = {"quiet": True, "no_warnings": True, "skip_download": True, "writesubtitles": True,
                    "subtitleslangs": ["live_chat"], "outtmpl": os.path.join(self.folder, "chat.%(ext)s"),
                    "progress_hooks": [hook]}
            with yt_dlp.YoutubeDL(opts) as y:
                y.download([self.url])
        except Exception:
            pass


def start_chat(url: str, info: dict, counter: ChatCounter, stop: threading.Event, folder: str) -> str:
    ch = twitch_channel(url) or (info.get("uploader_id") or "" if "twitch" in (info.get("extractor") or "").lower() else "")
    if ch:
        TwitchChat(ch.lower(), counter, stop).start()
        return "Twitch chat"
    if "youtube" in (info.get("extractor") or "").lower():
        os.makedirs(folder, exist_ok=True)
        YouTubeChat(url, counter, stop, folder).start()
        return "YouTube chat"
    return ""


# ---------------------------------------------------------------- recording

@dataclass
class Segment:
    path: str
    start: float
    end: float


def parse_segment_csv(text: str, folder: str, base: float) -> list[Segment]:
    out = []
    for line in text.splitlines():
        parts = line.strip().split(",")
        if len(parts) < 3:
            continue
        try:
            out.append(Segment(os.path.join(folder, parts[0]), base + float(parts[1]), base + float(parts[2])))
        except ValueError:
            continue
    return out


def deletable(segs: list[Segment], keep_from: float) -> list[Segment]:
    return [s for s in segs if s.end < keep_from]


def pick_live_formats(formats: list[dict], max_h: int = 1080) -> list[dict]:
    """1 muxed stream if there is one (prefer HLS, highest up to max_h), otherwise
    best video + best audio as two inputs, otherwise anything with a URL.
    A codec of None means "not reported", which on HLS usually means muxed."""
    usable = [f for f in formats or [] if f.get("url") and f.get("ext") not in ("mhtml", "vtt", "json", "srv3")
              and "storyboard" not in str(f.get("format_note", "")).lower()]
    if not usable:
        return []

    def rank(f: dict) -> tuple:
        h = f.get("height") or 0
        return (0 < h <= max_h, h if h <= max_h else -h, "m3u8" in str(f.get("protocol", "")),
                f.get("tbr") or 0, f.get("fps") or 0)

    muxed = [f for f in usable if f.get("vcodec") != "none" and f.get("acodec") != "none"]
    if muxed:
        return [max(muxed, key=rank)]
    video = [f for f in usable if f.get("vcodec") != "none"]
    audio = [f for f in usable if f.get("vcodec") == "none" and f.get("acodec") != "none"]
    if video and audio:
        best_a = max(audio, key=lambda f: (f.get("abr") or f.get("tbr") or 0, "m3u8" in str(f.get("protocol", ""))))
        return [max(video, key=rank), best_a]
    return [max(video or usable, key=rank)]


def resolve(url: str) -> tuple[dict, list[str], list[dict]]:
    """(info, stream URLs, headers per URL). Formats are read without yt-dlp's own
    format selection, so a site offering unusual formats doesn't fail outright."""
    import yt_dlp
    from .sources import friendly_error
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True}
    info: dict = {}
    try:
        with yt_dlp.YoutubeDL(opts) as y:
            target = url
            for _ in range(4):                       # follow /live, channel and short links
                info = y.extract_info(target, download=False, process=False) or {}
                t = info.get("_type")
                if t in ("url", "url_transparent") and info.get("url"):
                    target = info["url"]
                    continue
                if t == "playlist":
                    first = next((e for e in (info.get("entries") or []) if e), None)
                    if not first:
                        raise RuntimeError("This channel isn't live right now.")
                    if first.get("formats"):
                        info = first
                        break
                    target = first.get("url") or first.get("webpage_url") or ""
                    if not target:
                        raise RuntimeError("This channel isn't live right now.")
                    continue
                break
            if not info.get("formats") and not info.get("url"):
                info = y.extract_info(target, download=False) or {}      # let yt-dlp try its own way
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(friendly_error(str(exc))) from None
    if info.get("is_live") is False or info.get("live_status") in ("was_live", "not_live", "post_live"):
        raise RuntimeError("This link isn't live right now. Use Create for recorded videos.")
    if info.get("live_status") == "is_upcoming":
        raise RuntimeError("This stream hasn't started yet.")
    picked = pick_live_formats(info.get("formats") or ([{"url": info["url"], "vcodec": None, "acodec": None}]
                                                        if info.get("url") else []))
    if not picked:
        raise RuntimeError("The site didn't offer a stream MyClip can record. Close MyClip and start run.bat "
                           "again to update the downloader.")
    base_h = dict(info.get("http_headers") or {})
    return info, [f["url"] for f in picked], [dict(f.get("http_headers") or base_h) for f in picked]


class Recorder:
    """ffmpeg copies the stream into 2 s segments and also pipes 8 kHz mono audio
    to us for loudness. Restarts with a fresh address if the stream drops."""

    def __init__(self, url: str, folder: str, stop: threading.Event,
                 on_audio: Callable[[float, float], None], note: Callable[[str], None]) -> None:
        self.url, self.folder, self.stop = url, folder, stop
        self.on_audio, self.note = on_audio, note
        self.segments: list[Segment] = []
        self.lock = threading.Lock()
        self.t_start = time.time()
        self.run_id = 0
        self.proc: Optional[subprocess.Popen] = None
        self.failed: Optional[str] = None
        self.ended = False
        self._csv_pos: dict[str, int] = {}
        self._csv_base: dict[str, float] = {}

    @property
    def covered(self) -> float:
        with self.lock:
            return self.segments[-1].end if self.segments else 0.0

    def start(self, streams: list[str], headers: list[dict]) -> None:
        threading.Thread(target=self._loop, args=(streams, headers), daemon=True, name="recorder").start()

    @staticmethod
    def _cmd(streams: list[str], headers: list[dict], csv: str, pattern: str) -> list[str]:
        """One input (muxed) or two (video + audio). Segments get both; the pipe gets audio for levels."""
        a = [ffmpeg.ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-nostdin"]
        for k, url in enumerate(streams[:2]):
            if url.startswith("http"):
                a += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5"]
            h = headers[k] if k < len(headers) else {}
            if h:
                a += ["-headers", "".join(f"{hk}: {hv}\r\n" for hk, hv in h.items())]
            a += ["-i", url]
        audio_in = 1 if len(streams) > 1 else 0
        a += ["-map", "0:v:0?", "-map", f"{audio_in}:a:0?", "-c", "copy", "-f", "segment", "-segment_time", "2",
              "-segment_format", "mpegts", "-segment_list", csv, "-segment_list_type", "csv",
              "-reset_timestamps", "1", pattern,
              "-map", f"{audio_in}:a:0?", "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "pipe:1"]
        return a

    def _loop(self, stream: list, headers: list) -> None:
        fails: list[float] = []
        while not self.stop.is_set():
            self.run_id += 1
            base = self.covered if self.segments else 0.0
            csv = os.path.join(self.folder, f"r{self.run_id}.csv")
            self._csv_base[csv] = base
            pattern = os.path.join(self.folder, f"r{self.run_id}_%05d.ts")
            err: list[str] = []
            try:
                self.proc = subprocess.Popen(self._cmd(stream, headers, csv, pattern), stdin=subprocess.DEVNULL,
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                             creationflags=ffmpeg.NO_WINDOW)
            except Exception as exc:
                self.failed = f"Couldn't start recording: {exc}"
                return
            threading.Thread(target=lambda p=self.proc: err.extend(
                l.decode("utf-8", "replace") for l in p.stderr), daemon=True).start()
            self._read_audio(self.proc, base)
            self.proc.wait()
            self.poll_segments()
            if self.stop.is_set():
                break
            now = time.time()
            fails = [f for f in fails if now - f < 90] + [now]
            if len(fails) >= 4:
                self.ended = True
                self.note("The stream ended or can't be reached.")
                break
            self.note("Stream dropped, reconnecting")
            self.stop.wait(3)
            try:
                _, stream, headers = resolve(self.url)
            except Exception as exc:
                if "live right now" in str(exc) or "isn't live" in str(exc):
                    self.ended = True
                    self.note("The stream has ended.")
                    break

    def _read_audio(self, proc: subprocess.Popen, base: float) -> None:
        sr = 8000
        block = int(sr * HOP) * 2
        n = 0
        while not self.stop.is_set():
            buf = proc.stdout.read(block)
            if not buf:
                break
            a = np.frombuffer(buf[: len(buf) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0
            if a.size == 0:
                continue
            db = 20 * math.log10(float(np.sqrt(np.mean(a * a))) + 1e-7)
            n += 1
            self.on_audio(base + n * HOP, db)
        if self.stop.is_set() and proc.poll() is None:
            proc.kill()

    def poll_segments(self) -> None:
        for csv, base in list(self._csv_base.items()):
            try:
                with open(csv, "r", encoding="utf-8") as f:
                    f.seek(self._csv_pos.get(csv, 0))
                    text = f.read()
                    self._csv_pos[csv] = f.tell()
            except OSError:
                continue
            new = parse_segment_csv(text, self.folder, base)
            if new:
                with self.lock:
                    self.segments.extend(new)

    def prune(self, keep_from: float) -> None:
        with self.lock:
            old = deletable(self.segments, keep_from)
            self.segments = [s for s in self.segments if s not in old]
        for s in old:
            try:
                os.remove(s.path)
            except OSError:
                pass

    def covering(self, a: float, b: float) -> list[Segment]:
        with self.lock:
            return [s for s in self.segments if s.end > a and s.start < b]

    def kill(self) -> None:
        p = self.proc
        if p and p.poll() is None:
            p.kill()


# ---------------------------------------------------------------- session

@dataclass
class LiveOptions:
    sensitivity: str = "Normal"
    finish_now: bool = True          # False = finish clips after the stream


@dataclass
class LiveStatus:
    state: str = "Starting"
    recorded: float = 0.0
    voice: float = 0.0
    chat_rate: float = 0.0
    chat_source: str = ""
    in_moment: bool = False
    queued: int = 0
    made: int = 0
    note: str = ""


class LiveSession:
    def __init__(self, cfg: JobConfig, opts: LiveOptions, store: Optional[ConnectionStore], api_key: str,
                 on_status: Callable[[LiveStatus], None], on_clip: Callable[[dict, str], None],
                 on_note: Callable[[str], None]) -> None:
        self.cfg, self.opts, self.store, self.key = cfg, opts, store, api_key
        self.on_status, self.on_clip, self.on_note = on_status, on_clip, on_note
        self.stop = threading.Event()        # stop watching
        self.abort = threading.Event()       # also drop unfinished clips
        self.manual = threading.Event()
        self.audio_q: "queue.Queue[tuple[float, float]]" = queue.Queue()
        self.status = LiveStatus()
        self.folder: Optional[Path] = None
        self.clips: list[dict] = []
        self.title = "Live"
        self.used: list[str] = []
        self.lang_votes: dict[str, int] = {}

    def _lang(self) -> str:
        """The stream's language, learned from earlier clips (needs a few lines of speech)."""
        if not self.lang_votes or sum(self.lang_votes.values()) < 4:
            return ""
        return max(self.lang_votes, key=self.lang_votes.get)

    def request_clip(self) -> None:
        self.manual.set()

    def run(self) -> None:
        cfg = self.cfg
        work = Path(config_dir()) / "live" / time.strftime("%Y%m%d-%H%M%S")
        work.mkdir(parents=True, exist_ok=True)
        try:
            self._set(state="Connecting")
            info, stream, headers = resolve(cfg.source)
            if len(stream) > 1:
                self._note("Recording video and audio separately")
            self.title = info.get("title") or info.get("uploader") or "Live"
            for pat in (r"\s*\d{4}-\d{2}-\d{2} \d{2}:\d{2}$",):
                self.title = re.sub(pat, "", self.title)
            chat = ChatCounter()
            src = start_chat(cfg.source, info, chat, self.stop, str(work / "chat"))
            rec = Recorder(cfg.source, str(work), self.stop, lambda t, db: self.audio_q.put((t, db)), self._note)
            rec.start(stream, headers)
            det = SpikeDetector(self.opts.sensitivity, cfg.min_len, cfg.max_len)
            pending: list[Moment] = []
            todo: "queue.Queue[Optional[Moment]]" = queue.Queue()
            finisher = threading.Thread(target=self._finisher, args=(todo, rec), daemon=True)
            hold: list[Moment] = []
            if self.opts.finish_now:
                finisher.start()
            self._set(state="Watching", chat_source=src)
            self.used = ["Voice level"] + ([src] if src else [])
            last_t = 0.0
            last_status = 0.0
            no_audio_since = time.time()
            while not self.stop.is_set() and not rec.ended:
                try:
                    t, db = self.audio_q.get(timeout=0.5)
                    no_audio_since = time.time()
                except queue.Empty:
                    t, db = None, None
                    if time.time() - no_audio_since > 4 and rec.covered > last_t + 1:
                        t = rec.covered                     # stream without audio: drive by video
                if t is not None:
                    last_t = t
                    rate = chat.rate(5.0) if chat.ok else None
                    for m in det.feed(t, db, rate):
                        pending.append(m)
                        self._note(f"Moment at {_hms(m.start)} ({m.reason})")
                    if self.manual.is_set():
                        self.manual.clear()
                        span = min(30.0, cfg.max_len * 0.7)
                        pending.append(Moment(max(0.0, t - span), t + 4.0, 0, "manual", True))
                        self._note(f"Clipping {_hms(t - span)}")
                rec.poll_segments()
                covered = rec.covered
                for m in [m for m in pending if covered >= m.end + 0.5]:
                    pending.remove(m)
                    if self.opts.finish_now:
                        todo.put(m)
                    else:
                        hold.append(m)
                need = [m.start for m in pending + hold] + ([det.busy_from()] if det.busy_from() is not None else [])
                if not self.opts.finish_now:
                    keep_from = min(need) - 2 if need else covered - max(cfg.max_len + 60, 240)
                else:
                    keep_from = min([covered - max(cfg.max_len + 60, 240)] + [x - 2 for x in need])
                rec.prune(keep_from)
                if time.time() - last_status > 0.8:
                    last_status = time.time()
                    s, sl, _ = det.last
                    self._set(recorded=covered, voice=max(0.0, sl), chat_rate=chat.rate(5.0) if chat.ok else 0.0,
                              in_moment=det.active is not None, queued=todo.qsize() + len(pending) + len(hold),
                              state="Warming up" if covered < det.warmup else "Watching")
            if rec.failed:
                raise RuntimeError(rec.failed)
            rec.kill()
            time.sleep(0.5)
            rec.poll_segments()
            last = det.flush(rec.covered)
            for m in pending + ([last] if last else []):
                m.end = min(m.end, rec.covered)
                if m.end - m.start >= 5:
                    hold.append(m) if not self.opts.finish_now else todo.put(m)
            for m in hold:
                todo.put(m)
            todo.put(None)
            if not finisher.is_alive():
                finisher.start()
            self._set(state="Finishing", queued=todo.qsize() - 1)
            while finisher.is_alive():
                finisher.join(0.5)
                if self.abort.is_set():
                    break
            self._set(state="Stopped")
        finally:
            shutil.rmtree(work, ignore_errors=True)

    # ------------------------------------------------------------ finishing
    def _finisher(self, todo: "queue.Queue[Optional[Moment]]", rec: Recorder) -> None:
        while True:
            m = todo.get()
            if m is None or self.abort.is_set():
                return
            try:
                self._finish(m, rec)
            except Cancelled:
                return
            except Exception as exc:
                self._note(f"A clip failed: {str(exc)[:160]}")
            self._set(queued=max(0, todo.qsize() - (1 if self.stop.is_set() else 0)))

    def _finish(self, m: Moment, rec: Recorder) -> None:
        segs = rec.covering(m.start, m.end)
        if not segs:
            return
        cfg = self.cfg
        if self.folder is None:
            base = Path(cfg.output_dir) if cfg.output_dir else shorts_dir()
            base.mkdir(parents=True, exist_ok=True)
            self.folder = unique_dir(base, f"{safe_name(self.title, 50)} - live {time.strftime('%Y-%m-%d %H.%M')}")
            (self.folder / ".thumbs").mkdir(exist_ok=True)
        tmp = Path(segs[0].path).parent / f"join_{int(time.time() * 1000)}"
        tmp.mkdir(exist_ok=True)
        try:
            lst = tmp / "list.txt"
            lst.write_text("".join(f"file '{Path(s.path).as_posix()}'\n" for s in segs), encoding="utf-8")
            raw = str(tmp / "raw.ts")
            ffmpeg.run(["-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                        "-fflags", "+genpts", raw], self.abort)
            info = ffmpeg.probe(raw)
            off = segs[0].start
            clip = Clip(max(0.0, m.start - off), min(info.duration, m.end - off))
            if clip.duration < 3:
                return
            from .models import Transcript
            tr = Transcript()
            if info.has_audio and transcribe.available():
                lang = transcribe.norm_lang(cfg.language) or self._lang()
                if cfg.caption_quality == "best":
                    segs, _, got = transcribe.transcribe_clip(raw, 0.0, info.duration, cfg.whisper_model,
                                                              cfg.device, lang, self.abort)
                    tr = Transcript(segs, got or lang)
                else:
                    tr, _ = transcribe.transcribe(raw, info.duration, cfg.whisper_model, cfg.device, lang,
                                                  tmp, lambda f, msg: None, self.abort)
                if not cfg.language and tr.language and tr.segments:
                    self.lang_votes[tr.language] = self.lang_votes.get(tr.language, 0) + len(tr.segments)
            clip.text = tr.text_between(clip.start, clip.end)
            conn = self.store.get(cfg.connection_id) if (self.store and cfg.connection_id) else None
            key, model, usable = "", "", None
            if conn:
                key = self.key or self.store.api_key(conn)
                model = cfg.llm_model or conn.model
                if model and (key or not conn.needs_key):
                    usable = conn
            if usable and clip.text:
                llm.review(usable, key, model, [clip], Signals(title=self.title), lambda s: None, self.abort,
                           style=subtitles.title_style(cfg.caption_lang, tr))
            cap = tr
            if cfg.caption_lang in ("roman", "english", "auto") and cfg.captions != "Off" and not tr.empty:
                try:
                    caps, _, _ = subtitles.prepare(tr, [clip], cfg, raw, usable, key, model, lambda s: None, self.abort)
                    cap = caps[0]
                except Cancelled:
                    raise
                except Exception:
                    cap = tr
            if not clip.title:
                probe_clip = Clip(clip.start, clip.end)
                probe_clip.text = cap.text_between(clip.start, clip.end)
                clip.title = moments.fallback_title(probe_clip)
            if clip.score == 0:
                clip.score = round(min(100.0, 50 + m.peak * 8), 1) if not m.manual else 0.0
            n = len(self.clips) + 1
            out = self.folder / f"{n:02d} - {safe_name(clip.title, 50)}.mp4"
            try:
                render.render(raw, clip, str(out), cfg, info, cap, self.abort)
            except BaseException:
                render.cleanup_partial(str(out))
                raise
            th = self.folder / ".thumbs" / f"{n:02d}.jpg"
            try:
                render.thumbnail(str(out), str(th), min(clip.duration * 0.35, 3.0), self.abort)
            except Exception:
                th = None
            rec_d = {"file": out.name, "thumb": f".thumbs/{th.name}" if th else "",
                     "start": m.start, "end": m.end, "score": clip.score or None, "signal_score": None,
                     "llm_score": clip.llm_score, "title": clip.title, "hook": clip.hook,
                     "reason": clip.reason or ("Clipped by you" if m.manual else f"Spike in {m.reason}"),
                     "description": clip.description, "hashtags": clip.hashtags,
                     "source": "manual" if m.manual else "live"}
            self.clips.append(rec_d)
            self._write_manifest()
            self._set(made=len(self.clips))
            self.on_clip(rec_d, str(self.folder))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _write_manifest(self) -> None:
        from dataclasses import asdict
        data = {"app": "MyClip AI", "version": 1, "live": True, "created": time.strftime("%Y-%m-%d %H:%M"),
                "title": self.title + " (live)", "source": self.cfg.source, "used": self.used,
                "settings": {k: v for k, v in asdict(self.cfg).items() if k != "source"}
                | {"sensitivity": self.opts.sensitivity}, "clips": self.clips}
        p = self.folder / "clips.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)

    # ------------------------------------------------------------ helpers
    def _set(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self.status, k, v)
        self.on_status(_copy(self.status))

    def _note(self, msg: str) -> None:
        self.status.note = msg
        self.on_note(msg)


def _copy(s: LiveStatus) -> LiveStatus:
    from dataclasses import replace
    return replace(s)


def _hms(t: float) -> str:
    t = int(max(0, t))
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"
