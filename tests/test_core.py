"""Core logic tests. Run:  python -m pytest tests   (or: python tests/test_core.py)"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from myclip.core.captions import ass_time, bgr, build_ass, group_words
from myclip.core.ffmpeg import parse_probe
from myclip.core.models import CAPTION_STYLES, Clip, JobConfig, MediaInfo, Segment, Signals, Transcript, Word
from myclip.core.moments import candidates, fallback_title, merge_llm, overlap, select
from myclip.core.paths import safe_name
from myclip.core.render import caption_position, resolve_framing, video_filter
from myclip.core.sources import parse_chat_file, parse_timestamps, signals_from_info
from myclip.core.store import Connection, ConnectionStore, PresetStore, _Keys, resolve_llm_for_preset


def mk_tr(n=600):
    segs = []
    for i in range(0, n, 5):
        text = "Oh my god no way, that was insane!" if 300 <= i < 305 else "Just walking around looting some stuff."
        words = [Word(i + j * 0.55, i + j * 0.55 + 0.45, w) for j, w in enumerate(text.split())]
        segs.append(Segment(float(i), words[-1].end, text, words))
    return Transcript(segs, "en")


def test_safe_names():
    assert safe_name('a<b>:c"/d\\e|f?g*') == "abcdefg"
    assert safe_name("CON") == "clip" and safe_name("  ...  ") == "clip"
    assert safe_name("भाई क्लच moment!").startswith("भाई")


def test_probe_parse():
    txt = ("  Duration: 01:02:03.50, start: 0.000000, bitrate: 8000 kb/s\n"
           "  Stream #0:0: Video: h264 (High), yuv420p, 1920x1080 [SAR 1:1 DAR 16:9], 59.94 fps\n"
           "  Stream #0:1: Audio: aac (LC), 48000 Hz, stereo")
    m = parse_probe(txt)
    assert abs(m.duration - 3723.5) < 1e-6 and (m.width, m.height) == (1920, 1080) and m.has_audio


def test_target_sizes():
    assert JobConfig("x").target_size() == (1080, 1920)
    assert JobConfig("x", quality="4K").target_size() == (2160, 3840)
    w, h = JobConfig("x", aspect="Custom", custom_w=721, custom_h=1281).target_size()
    assert w % 2 == 0 and h % 2 == 0
    assert JobConfig("x", aspect="Original").target_size(MediaInfo(10, 3840, 2160, 30, True)) == (1920, 1080)


def test_external_parsing():
    assert parse_timestamps("bro 3:42 and 1:02:03, 99:99 no", 4000) == [222.0, 3723.0]
    yt = "\n".join(json.dumps({"replayChatItemAction": {"videoOffsetTimeMsec": str(ms)}}) for ms in (1000, 2500))
    assert parse_chat_file(yt) == [1.0, 2.5]
    s = signals_from_info({"heatmap": [{"start_time": 0, "end_time": 10, "value": 1.0}],
                           "comments": [{"text": "0:05 lol", "like_count": 100}]}, 10)
    assert s.heatmap and s.comment_times.count(5.0) == 5


def test_loud_moment_found_and_no_overlap():
    sig = Signals()
    loud = np.full(1200, -32.0, dtype=np.float32)
    loud[600:615] = -6.0
    sig.loudness = loud
    c = candidates(600.0, sig, mk_tr(), 20, 60, 5)
    assert c[0].start <= 302 <= c[0].end and 20 <= c[0].duration <= 60.6
    pick = select(c, 5)
    assert all(overlap(a, b) <= 0.2 for a in pick for b in pick if a is not b)


def test_heatmap_outweighs_loudness():
    sig = Signals()
    loud = np.full(1200, -30.0, dtype=np.float32)
    loud[200:206] = -12.0
    sig.loudness = loud
    sig.heatmap = [(float(a), float(a + 6), 1.0 if a == 480 else 0.05) for a in range(0, 600, 6)]
    c = candidates(600.0, sig, mk_tr(), 20, 60, 3)
    assert c[0].start <= 482 <= c[0].end + 2


def test_no_mid_word_cuts():
    tr = mk_tr()
    sig = Signals()
    sig.loudness = np.random.RandomState(1).normal(-30, 3, 1200).astype(np.float32)
    for c in candidates(600.0, sig, tr, 20, 60, 8)[:60]:
        for w in tr.words():
            assert not (w.start + 0.02 < c.start < w.end - 0.02)
            assert not (w.start + 0.02 < c.end < w.end - 0.02)


def test_short_silent_video():
    c = candidates(12.0, Signals(), Transcript(), 20, 60, 3)
    assert c and all(x.end <= 12.0 + 1e-6 for x in c)


def test_llm_merge():
    tr = mk_tr()
    sig = Signals()
    sig.loudness = np.full(1200, -30.0, dtype=np.float32)
    lc = Clip(295.0, 330.0)
    lc.llm_score, lc.title = 95, "The clutch"
    merged = merge_llm(candidates(600.0, sig, tr, 20, 60, 4), [lc], tr, sig, 600.0, 20, 60)
    assert merged[0].title == "The clutch"


def test_titles():
    tr = mk_tr()
    c = Clip(295, 330)
    c.text = tr.text_between(295, 330)
    assert "no way" in fallback_title(c).lower() or "oh my god" in fallback_title(c).lower()
    assert fallback_title(Clip(125, 140)) == "Moment at 2:05"


def test_captions():
    ass = build_ass([Word(10, 10.4, "hello"), Word(10.5, 11, "{bad}"), Word(11.1, 11.5, "three")],
                    10, 20, 1080, 1920, CAPTION_STYLES["Bold"])
    assert "HELLO" in ass and "{BAD}" not in ass and ass.count("Dialogue:") == 3
    assert ass_time(3725.456) == "1:02:05.46" and bgr("FFD84D") == "&H004DD8FF"
    assert [len(g) for g in group_words([Word(i * .3, i * .3 + .2, "w") for i in range(7)], 3)] == [3, 3, 1]


def test_framing():
    wide = MediaInfo(100, 1920, 1080, 60, True)
    assert resolve_framing(JobConfig("x"), wide, 1080, 1920) == "fit"
    assert resolve_framing(JobConfig("x", aspect="16:9"), wide, 1920, 1080) == "scale"
    for m in ("fit", "fill", "smart", "scale"):
        assert video_filter(m, 1080, 1920, 0.3).endswith("[v]")
    assert 0.05 <= caption_position("fit", wide, 1080, 1920, 0.24) <= 0.24


class FakeKeyring:
    def __init__(self):
        self.d = {}

    def get_password(self, a, b):
        return self.d.get((a, b))

    def set_password(self, a, b, c):
        self.d[(a, b)] = c

    def delete_password(self, a, b):
        self.d.pop((a, b), None)


def test_connection_store_keeps_keys_off_disk():
    keys = _Keys()
    keys.backend = FakeKeyring()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.json"
        st = ConnectionStore(p, keys)
        c = Connection("OpenAI", "OpenAI", "https://api.openai.com/v1", "gpt-4.1")
        assert st.upsert(c, "sk-secret") is True
        st.set_active(c.id)
        assert "sk-secret" not in p.read_text()
        st2 = ConnectionStore(p, keys)
        assert st2.active.id == c.id and st2.api_key(st2.active) == "sk-secret"
        st2.remove(c.id)
        assert keys.backend.d == {} and st2.active is None


def test_presets():
    with tempfile.TemporaryDirectory() as d:
        ps = PresetStore(Path(d) / "p.json")
        p = ps.save_custom("Fast", {"quality": "4K", "source": "leak"})
        assert p.name == "Fast (custom)" and "source" not in p.values
        keys = _Keys()
        keys.backend = None
        st = ConnectionStore(Path(d) / "c.json", keys)
        assert resolve_llm_for_preset(ps.find("Fast"), st) == ""
        assert resolve_llm_for_preset(ps.find("Offline"), st) == ""   # nothing detected yet
        local = Connection("Ollama (local)", "Ollama", "http://localhost:11434/v1", "qwen3:8b",
                           models_cache=["qwen3:8b"], auto=True)
        st.merge_detected([local])
        assert st.get(resolve_llm_for_preset(ps.find("Offline"), st)).is_local


def test_detection_picks_model_that_fits(monkeypatch=None):
    from myclip.core import discovery
    gb = 1024 ** 3
    data = {"http://localhost:11434/api/tags": {"models": [
        {"name": "qwen3:8b", "size": 5.2 * gb, "details": {"parameter_size": "8.2B"}},
        {"name": "gemma3:12b", "size": 8.1 * gb, "details": {"parameter_size": "12.2B"}},
        {"name": "gpt-oss:20b", "size": 13.8 * gb, "details": {"parameter_size": "20.9B"}},
        {"name": "nomic-embed-text", "size": 0.3 * gb, "details": {}}]}}
    real = discovery._get
    discovery._get = lambda url, timeout=1.5: data.get(url)
    try:
        found, _ = discovery.detect(8188, start_if_needed=False)
    finally:
        discovery._get = real
    assert len(found) == 1 and found[0].models_cache == ["qwen3:8b", "gemma3:12b", "gpt-oss:20b"]
    assert found[0].model == "qwen3:8b"
    assert discovery.best_default(found[0].models_cache, found[0].model_info, 16384) == "gpt-oss:20b"


def test_context_sizing():
    from myclip.core.llm import ctx_for
    assert ctx_for([{"content": "x" * 100}], 1000) == 4096
    big = ctx_for([{"content": "x" * 40000}], 4000)
    assert big % 2048 == 0 and 16000 < big <= 32768


def test_library_search_and_delete():
    from myclip.core import engine
    job = {"title": "Valorant ranked", "source": "https://youtu.be/x", "created": "2026-11-25",
           "clips": [{"title": "1v4 clutch", "hashtags": ["#valorant"]}]}
    assert engine.matches(job, "clutch") and engine.matches(job, "VALORANT #valorant")
    assert not engine.matches(job, "minecraft")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "a").mkdir()
        (root / "a" / "clips.json").write_text(json.dumps(job), encoding="utf-8")
        (root / "not_a_job").mkdir()
        items = engine.read_library([str(root), str(root)])
        assert [i["title"] for i in items if i["folder"].startswith(d)] == ["Valorant ranked"]
        try:
            engine.delete_job(str(root / "not_a_job"))
            assert False
        except RuntimeError:
            pass
        engine.delete_job(str(root / "a"))
        assert not (root / "a").exists()


def _sim(loud, chat=None, **kw):
    from myclip.core.live import HOP, SpikeDetector
    d = SpikeDetector(**kw)
    out = []
    for i, x in enumerate(loud):
        out += d.feed(i * HOP, x, None if chat is None else chat[i])
    last = d.flush(len(loud) * HOP)
    return out + ([last] if last else [])


def test_live_detects_shout_and_chat_and_stops():
    rng = np.random.RandomState(3)
    n = 1200
    loud = rng.normal(-28, 2.0, n)
    loud[400:416] = -6                                   # shout at 200 s
    chat = np.clip(rng.normal(10, 3, n), 0, None)
    chat[800:830] = 60                                   # chat burst at 400 s
    ms = _sim(loud, chat, min_len=15, max_len=60)
    assert len(ms) == 2
    assert 180 <= ms[0].start < 192 and 208 < ms[0].end < 232     # pre-roll, then stops when calm
    assert "chat" in ms[1].reason


def test_live_no_false_positives_and_no_overlap():
    for k in range(10):
        r = np.random.RandomState(100 + k)
        assert _sim(r.normal(-25, r.uniform(1, 5), 1200), np.clip(r.normal(10, r.uniform(1, 5), 1200), 0, None)) == []
    sus = np.random.RandomState(1).normal(-28, 2, 1200)
    sus[200:520] = -5                                    # 160 s of hype
    ms = _sim(sus, max_len=60)
    assert all(m.end - m.start <= 60.01 for m in ms)
    assert all(ms[i].start >= ms[i - 1].end for i in range(1, len(ms)))


def test_live_helpers():
    from myclip.core.live import deletable, irc_privmsg_count, parse_segment_csv, twitch_channel
    assert twitch_channel("https://www.twitch.tv/Shroud") == "shroud"
    assert twitch_channel("https://www.twitch.tv/videos/123") == ""
    assert irc_privmsg_count(":a!a@a PRIVMSG #x :hi\r\n:b!b@b PRIVMSG #x :yo\r\nPING :tmi\r\n") == 2
    segs = parse_segment_csv("r1_00000.ts,0.0,2.0\nr1_00001.ts,2.0,4.0\nbad\n", "C:/x", 100.0)
    assert len(segs) == 2 and segs[1].start == 102.0
    assert deletable(segs, 103) == [segs[0]]


class _Resp:
    def __init__(self, code, body=None, headers=None):
        self.status_code, self._b, self.headers = code, body or {}, headers or {}

    def json(self):
        return self._b


class _FakeGoogle:
    """Token refresh + resumable upload with one dropped connection and one expired token."""

    def __init__(self, size):
        import requests
        self.size, self.got, self.dropped, self.expired, self.body = size, 0, False, True, None
        self.err = requests.RequestException

    def post(self, url, data=None, timeout=0):
        return _Resp(200, {"access_token": "new", "expires_in": 3600})

    def request(self, method, url, params=None, json=None, data=None, headers=None, timeout=0):
        import re
        if self.expired and headers.get("Authorization") == "Bearer old":
            self.expired = False
            return _Resp(401, {"error": {"message": "expired"}})
        if method == "POST" and url.endswith("/videos"):
            self.body = json
            return _Resp(200, {}, {"Location": "https://up/s"})
        cr = headers["Content-Range"]
        if cr.startswith("bytes */"):
            return _Resp(308, {}, {"Range": f"bytes=0-{self.got - 1}"} if self.got else {})
        a, b = map(int, re.match(r"bytes (\d+)-(\d+)/", cr).groups())
        assert a == self.got
        if not self.dropped and b >= self.size // 2:
            self.dropped = True
            raise self.err("reset")
        self.got = b + 1
        return _Resp(201, {"id": "VID"}) if self.got >= self.size else _Resp(308, {}, {"Range": f"bytes=0-{b}"})


def test_youtube_resumable_upload_survives_drop_and_expired_token():
    import threading
    import time
    from myclip.core import youtube as yt
    yt.CHUNK = 1000

    class Keys:
        def __init__(self):
            self.d = {}

        def get(self, k):
            return self.d.get(k, "")

        def set(self, k, v):
            self.d[k] = v
            return True

        def delete(self, k):
            self.d.pop(k, None)

    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "c.mp4")
        open(f, "wb").write(os.urandom(4500))
        st = yt.YouTubeStore(Path(d) / "yt.json", Keys())
        st.set_client("x.apps.googleusercontent.com", "secret")
        st.add_channel(yt.Channel("UC1", "Me"), {"access_token": "old", "refresh_token": "r", "expires_at": time.time() + 999})
        q = yt.UploadQueue(Path(d) / "q.json")
        it = yt.UploadItem(f, "  clutch <x> ", tags=["valorant", "valorant"])
        q.add([it])
        g = _FakeGoogle(4500)
        yt.process(it, st, q, lambda *a: None, threading.Event(), g)
        assert it.status == "done" and it.video_id == "VID" and g.got == 4500
        assert g.body["snippet"]["title"] == "clutch x" and g.body["snippet"]["tags"] == ["valorant"]
        assert "#Shorts" in g.body["snippet"]["description"]
        saved = (Path(d) / "yt.json").read_text()
        assert "secret" not in saved and "old" not in saved


def test_youtube_helpers():
    from datetime import datetime
    from myclip.core import youtube as yt
    assert yt.spread_schedule(2, datetime(2026, 11, 26, 18, 0), 24) == ["2026-11-26T18:00", "2026-11-27T18:00"]
    assert yt.build_body(yt.UploadItem("f", "t", publish_at="2030-01-01T10:00"))["status"]["privacyStatus"] == "private"
    try:
        yt.parse_client_file(json.dumps({"web": {"client_id": "a.apps.googleusercontent.com"}}))
        assert False
    except yt.YTError:
        pass


def test_hinglish_transliteration():
    from myclip.core.translit import roman_text
    assert roman_text("हम वहाँ चलेंगे ये क्या हो गया") == "hum vaha chalenge ye kya ho gaya"
    assert roman_text("आज काम नहीं हुआ।") == "aaj kaam nahi hua."
    assert roman_text("समझना अच्छा दोस्त") == "samajhna accha dost"
    assert roman_text("game में") == "game mein"


def test_subtitles_keep_word_timing():
    import threading
    from myclip.core import subtitles
    words = [Word(10 + i * .5, 10.4 + i * .5, w) for i, w in enumerate(["ये", "क्या", "हो", "गया"])]
    tr = Transcript([Segment(10, 12, "ये क्या हो गया", words)], "hi")
    caps, _, used = subtitles.prepare(tr, [Clip(5, 30)], JobConfig("x", caption_lang="roman"), "m", None, "", "",
                                      lambda m: None, threading.Event())
    got = caps[0].words_between(5, 30)
    assert [w.text for w in got] == ["ye", "kya", "ho", "gaya"]
    assert [w.start for w in got] == [w.start for w in words]
    assert tr.segments[0].words[0].text == "ये"          # original untouched


def test_auto_subtitles_base_language_and_brackets():
    from myclip.core import subtitles
    hi = Transcript([Segment(1, 2, "ये क्या", [Word(1, 1.4, "ये"), Word(1.5, 1.9, "क्या")])], "hi")
    en = Transcript([Segment(1, 2, "no way", [Word(1, 1.4, "no"), Word(1.5, 1.9, "way")])], "en")
    es = Transcript([Segment(1, 2, "muchas gracias", [])], "es")
    assert subtitles.base_for(hi) == "hinglish" and subtitles.base_for(en) == "english"
    assert subtitles.base_for(es) == "other"
    ass = build_ass([Word(1, 1.4, "gracias (thanks)"), Word(1.5, 1.9, "bro")], 0, 5, 1080, 1920,
                    CAPTION_STYLES["Bold"])
    first = [l for l in ass.splitlines() if l.startswith("Dialogue")][0]
    assert "(THANKS)" in first and "\\fscx72" in first and "\x0c" not in ass


def test_language_vote_prefers_hindi_for_hinglish():
    from myclip.core.transcribe import vote_language
    wins = [[("en", .7), ("hi", .2)], [("hi", .8), ("ur", .15)], [("ur", .6), ("hi", .3)], [("hi", .9)]]
    assert vote_language(wins)[0] == "hi"
    assert vote_language([[("en", .9), ("hi", .05)]] * 5)[0] == "en"


def test_hallucinations_removed():
    from myclip.core.transcribe import clean_segments

    class W_:
        def __init__(self, a, b, w):
            self.start, self.end, self.word = a, b, w

    class S_:
        def __init__(self, a, b, text, nsp=.05, alp=-.3, cr=1.3):
            self.start, self.end, self.text = a, b, text
            self.no_speech_prob, self.avg_logprob, self.compression_ratio = nsp, alp, cr
            self.words = [W_(a + i * .4, a + i * .4 + .3, w) for i, w in enumerate(text.split())]

    raw = [S_(0, 2, "bhai ye kya hua"), S_(3, 5, "Thanks for watching!"), S_(6, 7, "x", nsp=.8, alp=-1.0),
           S_(8, 10, "go go go go go go"), S_(11, 12, "junk", alp=-1.5)]
    assert [s.text for s in clean_segments(raw)] == ["bhai ye kya hua", "go go go"]


def test_english_words_in_devanagari():
    from myclip.core.translit import roman_text
    assert roman_text("ये गेम तो फ़ोन पे खेलो") == "ye game toh phone pe khelo"
    assert roman_text("भाई क्लच हेडशॉट मारा") == "bhai clutch headshot mara"


def test_ai_caption_words_cannot_shift():
    import threading
    from myclip.core import subtitles

    class FakeLLM:
        @staticmethod
        def chat(conn, key, model, messages, max_tokens=0, cancel=None, json_mode=True):
            ws = json.loads(messages[1]["content"])["words"]
            items = [{"i": x["i"], "t": {"ये": "ye", "क्या": "what", "हुआ": "hua"}.get(x["w"], x["w"]), "en": ""}
                     for x in ws]
            return json.dumps({"items": items[::-1][:-1]})        # reversed, one missing, one translated

        parse_json = staticmethod(json.loads)

    real = subtitles.llm
    subtitles.llm = FakeLLM
    try:
        ws = [Word(1 + i * .5, 1.4 + i * .5, w) for i, w in enumerate(["ये", "क्या", "हुआ"])]
        tr = Transcript([Segment(1, 3, "ये क्या हुआ", ws)], "hi")

        class Conn:
            provider = "Ollama"
        caps, _, _ = subtitles.prepare(tr, [Clip(0, 10)], JobConfig("x", caption_lang="auto"), "m", Conn(), "", "m",
                                       lambda m: None, threading.Event())
    finally:
        subtitles.llm = real
    assert [w.text for w in caps[0].words_between(0, 10)] == ["ye", "kya", "hua"]


def test_cancel_stops_blocking_steps_at_once():
    import threading
    import time
    from myclip.core.models import Cancelled, run_cancellable
    ev = threading.Event()
    threading.Timer(0.3, ev.set).start()
    t0 = time.time()
    try:
        run_cancellable(lambda: time.sleep(30), ev)
        assert False
    except Cancelled:
        assert time.time() - t0 < 2
    assert run_cancellable(lambda: 42, threading.Event()) == 42


def test_live_format_picking():
    from myclip.core.live import pick_live_formats
    hls = [{"format_id": str(h), "url": f"u{h}", "protocol": "m3u8_native", "height": h, "vcodec": "avc1",
            "acodec": "mp4a"} for h in (360, 720, 1080, 1440)]
    assert pick_live_formats(hls)[0]["format_id"] == "1080"
    dash = [{"format_id": "v", "url": "v", "height": 1080, "vcodec": "avc1", "acodec": "none"},
            {"format_id": "a", "url": "a", "vcodec": "none", "acodec": "mp4a", "abr": 128}]
    assert [f["format_id"] for f in pick_live_formats(dash)] == ["v", "a"]
    assert pick_live_formats([{"format_id": "k", "url": "k", "height": 720}])[0]["format_id"] == "k"


def test_friendly_ytdlp_errors():
    from myclip.core.sources import friendly_error
    msg = friendly_error("\x1b[0;31mERROR:\x1b[0m [youtube] uFk: Requested format is not available.")
    assert "\x1b" not in msg and "[0;31m" not in msg and "run.bat" in msg


def test_theme_helpers():
    import importlib.util
    spec = importlib.util.spec_from_file_location("theme", Path(__file__).resolve().parents[1] / "myclip" / "ui" / "theme.py")
    th = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(th)
    assert th.rgba("#FF6A3D", 0.15) == "rgba(255, 106, 61, 38)"
    assert th.score_color(80) == th.TEAL and th.score_color(60) == th.AMBER and th.score_color(10) == th.TEXT_3
    qss = th.build_qss("#4C8DFF")
    assert th.ACCENT == "#4C8DFF" and "#4C8DFF" in qss and qss.count("{") == qss.count("}")
    assert th.set_accent("not a color") == th.DEFAULT_ACCENT


def _cs_case(hi_words, en_spec, step=0.45):
    from myclip.core.codeswitch import restore_english
    from myclip.core.translit import roman_word
    hw = [Word(i * step, i * step + 0.4, x) for i, x in enumerate(hi_words)]
    ew, last = [], 0.0
    for t, a in en_spec:
        st = a * step + 0.08 if a is not None else last + 0.05
        ew.append(Word(st, st + 0.35, t))
        last = st + 0.35
    out, _ = restore_english([Segment(0, hw[-1].end, "", hw)], [Segment(0, ew[-1].end, "", ew)])
    return " ".join(roman_word(w.text) for w in out[0].words)


def test_english_words_kept_in_hindi_speech():
    assert _cs_case(["भाई", "ये", "गेम", "तो", "बहुत", "क्रेज़ी", "है", "यार"],
                    [("Brother,", 0), ("this", 1), ("game", 2), ("is", 3), ("very", 4), ("crazy", 5), ("man.", 7)]) \
        == "bhai ye game toh bahut crazy hai yaar"
    assert _cs_case(["प्लेयर", "ने", "हेड", "शॉट", "मारा"],
                    [("The", None), ("player", 0), ("got", 1), ("a", None), ("headshot.", 3)]) == "player ne headshot mara"
    assert _cs_case(["वैलोरेंट", "में", "रैंक", "पुश", "कर", "रहा", "हूँ"],
                    [("I'm", 4), ("pushing", 3), ("rank", 2), ("in", 1), ("Valorant.", 0)]) \
        == "Valorant mein rank push kar raha hoon"


def test_hindi_words_never_replaced_by_translation():
    assert _cs_case(["मैंने", "उसको", "मार", "दिया"], [("I", 0), ("killed", 2), ("him.", 1)]) == "maine usko maar diya"
    assert _cs_case(["हार", "गए", "आओ", "मैंने"], [("here", 0), ("way", 2), ("money", 3), ("go", 1)]) == \
        "haar gae aao maine"


# ---------------- v2

def test_bleep_matching_and_mask():
    from myclip.core.bleep import DEFAULT_WORDS, Matcher, apply, mask
    m = Matcher(DEFAULT_WORDS)
    for w in ("fuck", "Fucking!", "fuuuuck", "f***", "bc", "chutiya", "चूतिया", "madarchod"):
        assert m.is_swear(w), w
    for w in ("bhai", "class", "gandhi", "dickens", "assess", "grand", "मज़ा"):
        assert not m.is_swear(w), w
    assert mask("Fucking!") == "F******!"
    tr = Transcript([Segment(0, 3, "", [Word(0, .3, "bhai"), Word(1.0, 1.5, "fucking"), Word(2.3, 2.6, "bc")])], "hi")
    masked, iv = apply(tr, 0.0, 3.0, m)
    assert [w.text for w in masked.words()] == ["bhai", "f******", "b**"] and len(iv) == 2


def test_hindi_speech_pieces():
    from myclip.core.hindi import drop_tails, speech_pieces
    sr = 16000
    rng = np.random.RandomState(0)
    tone = lambda d: (0.3 * np.sin(2 * np.pi * 200 * np.arange(int(d * sr)) / sr)).astype(np.float32)
    sil = lambda d: rng.normal(0, 0.003, int(d * sr)).astype(np.float32)
    p = speech_pieces(np.concatenate([sil(.5), tone(3), sil(.6), tone(19), sil(1)]))
    assert p and all(b - a <= 8.31 for a, b in p) and all(p[i][0] >= p[i - 1][1] for i in range(1, len(p)))
    assert speech_pieces(sil(5)) == []
    segs = [Segment(0, 2, "भाई ये", [Word(0, .5, "भाई"), Word(.6, 1, "ये")]),
            Segment(1.02, 1.3, "धन्यवाद", [Word(1.02, 1.3, "धन्यवाद")])]
    assert [s.text for s in drop_tails(segs)] == ["भाई ये"]


def test_learning_is_bounded_and_waits_for_data():
    import math
    import random
    from myclip.core import learning
    rng = random.Random(1)
    vids = []
    for i in range(20):
        parts = {"hype": rng.uniform(0, 8), "speech": rng.uniform(0, 8), "peak": rng.uniform(5, 25)}
        vids.append({"parts": parts, "views": int(math.exp(6 + 0.4 * parts["hype"] + rng.gauss(0, .5)))})
    w, n = learning.fit(vids)
    assert n == 20 and w["hype"] > 1.05 and all(0.6 <= v <= 1.4 for v in w.values())
    assert learning.fit(vids[:5]) == ({}, 5)


def test_best_of_plan():
    from myclip.core.compile import plan, xfade_graph
    clips = [{"file": f"{i}.mp4", "start": i * 100, "end": i * 100 + d, "score": s, "features": {"peak_pos": .7}}
             for i, (d, s) in enumerate([(30, 88), (45, 92), (20, 40), (60, 75), (25, 81), (35, 66), (15, 99)])]
    p = plan(clips, 5, 60)
    assert len(p) == 5 and sum(x["b"] - x["a"] for x in p) <= 60.01
    assert [x["start"] for x in p] == sorted(x["start"] for x in p)
    g, v, a = xfade_graph(3, [10, 12, 9], 1080, 1920)
    assert "offset=21.500" in g and v == "[vx2]" and a == "[ax2]"


def test_facecam_and_moving_crop_filters():
    from myclip.core.faces import smooth_path, x_expr
    from myclip.core.render import RenderPlan, caption_position, facecam_split, resolve_framing, video_filter
    wide = MediaInfo(100, 1920, 1080, 60, True)
    cam = [0.72, 0.68, 0.26, 0.28]
    assert resolve_framing(JobConfig("x", framing="Facecam"), wide, 1080, 1920) == "facecam"
    top, bot = facecam_split(1080, 1920, cam, wide)
    assert top + bot == 1920 and 0.25 * 1920 <= top <= 0.45 * 1920
    f = video_filter("facecam", 1080, 1920, 0.5, None, cam, wide)
    assert "vstack=inputs=2" in f and f.endswith("[v]")
    assert 0.1 <= caption_position("facecam", wide, 1080, 1920, 0.24, cam) < 0.7
    path = smooth_path([0, 1, 2, 3, 4, 5], [0.5, 0.52, 0.7, 0.8, 0.8, 0.78], 6.0)
    assert "clip(t-" in x_expr(path, 1080)
    plan = RenderPlan(0.4, path, cam, "GG", [(1.0, 1.4)], "beep")
    assert RenderPlan.from_dict(plan.to_dict()).to_dict() == plan.to_dict()


def test_hook_in_ass():
    ass = build_ass([Word(10, 10.4, "bhai")], 10, 30, 1080, 1920, CAPTION_STYLES["Bold"], 0.24,
                    ("1v4 clutch in ranked bro", 700, 2.5))
    assert ass.count("[V4+ Styles]") == 1 and "Style: Hook," in ass and ",Hook,," in ass


def test_grid_sort_and_filter():
    from myclip.ui.grid import filter_items, sort_items
    items = [({"title": "Clutch", "start": 50, "score": 70, "hashtags": ["#valorant"]}, "/f"),
             ({"title": "Funny fail", "start": 10, "score": 90}, "/f")]
    assert [c["title"] for c, _ in sort_items(items, "best")] == ["Funny fail", "Clutch"]
    assert [c["title"] for c, _ in sort_items(items, "time")] == ["Funny fail", "Clutch"]
    assert [c["title"] for c, _ in filter_items(items, "VALORANT")] == ["Clutch"]


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            n += 1
            print("ok", name)
    print(f"{n} tests passed")
