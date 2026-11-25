"""YouTube channel control: sign-in, resumable queued uploads, videos, comments.

Uses the user's own Google OAuth "Desktop app" client. Sign-in is the standard
installed-app flow (browser + loopback + PKCE). Tokens and the client secret
live in Windows Credential Manager; nothing secret is written to files."""
from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import re
import secrets
import threading
import time
import urllib.parse
import uuid
import webbrowser
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import requests

from .models import Cancelled

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
API = "https://www.googleapis.com/youtube/v3"
UPLOAD = "https://www.googleapis.com/upload/youtube/v3"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload",
          "https://www.googleapis.com/auth/youtube",
          "https://www.googleapis.com/auth/youtube.force-ssl"]
CHUNK = 8 * 1024 * 1024
CATEGORIES = {"Gaming": "20", "Entertainment": "24", "People & Blogs": "22", "Comedy": "23",
              "Sports": "17", "Education": "27"}
CONSOLE_URL = "https://console.cloud.google.com/apis/library/youtube.googleapis.com"
QUOTA_REASONS = {"quotaExceeded", "uploadLimitExceeded", "rateLimitExceeded", "dailyLimitExceeded"}


class YTError(RuntimeError):
    def __init__(self, msg: str, reason: str = "", status: int = 0) -> None:
        super().__init__(msg)
        self.reason, self.status = reason, status


def _error(r) -> YTError:
    reason, msg = "", f"HTTP {r.status_code}"
    try:
        d = r.json()
        e = d.get("error", {})
        if isinstance(e, str):                       # token endpoint
            reason, msg = e, d.get("error_description") or e
        else:
            reason = ((e.get("errors") or [{}])[0]).get("reason", "") or e.get("status", "")
            msg = e.get("message") or reason or msg
    except Exception:
        pass
    msg = re.sub(r"<[^>]+>", "", str(msg))
    if reason in QUOTA_REASONS:
        msg = "YouTube's daily limit was reached. Uploads continue tomorrow."
    elif reason == "youtubeSignupRequired":
        msg = "This Google account has no YouTube channel yet. Create one on youtube.com first."
    elif reason == "accessNotConfigured" or "has not been used in project" in msg:
        msg = "YouTube Data API v3 isn't enabled in your Google Cloud project."
    return YTError(msg, reason, r.status_code)


# ---------------------------------------------------------------- oauth helpers

def pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)[:96]
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def auth_url(client_id: str, redirect: str, challenge: str, state: str) -> str:
    q = {"client_id": client_id, "redirect_uri": redirect, "response_type": "code", "scope": " ".join(SCOPES),
         "access_type": "offline", "prompt": "consent select_account", "include_granted_scopes": "true",
         "code_challenge": challenge, "code_challenge_method": "S256", "state": state}
    return AUTH_URL + "?" + urllib.parse.urlencode(q)


def parse_client_file(text: str) -> tuple[str, str]:
    d = json.loads(text)
    if "web" in d and "installed" not in d:
        raise YTError("This is a 'Web application' client. Create one of type 'Desktop app' instead.")
    c = d.get("installed") or d
    cid = c.get("client_id", "")
    if not cid.endswith(".apps.googleusercontent.com"):
        raise YTError("This doesn't look like a Google OAuth client file.")
    return cid, c.get("client_secret", "")


def _tok(d: dict, old: Optional[dict] = None) -> dict:
    return {"access_token": d["access_token"],
            "refresh_token": d.get("refresh_token") or (old or {}).get("refresh_token", ""),
            "expires_at": time.time() + int(d.get("expires_in", 3600))}


# ---------------------------------------------------------------- accounts

@dataclass
class Channel:
    id: str
    title: str
    handle: str = ""
    thumb: str = ""


class YouTubeStore:
    def __init__(self, path: Path, keys) -> None:
        self.path, self.keys = path, keys
        self.client_id = ""
        self.channels: list[Channel] = []
        self.active_id = ""
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            self.client_id = d.get("client_id", "")
            known = {f.name for f in fields(Channel)}
            self.channels = [Channel(**{k: v for k, v in c.items() if k in known}) for c in d.get("channels", [])]
            self.active_id = d.get("active", "")
        except Exception:
            pass

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"client_id": self.client_id, "active": self.active_id,
                                   "channels": [asdict(c) for c in self.channels]}, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    @property
    def configured(self) -> bool:
        return bool(self.client_id)

    @property
    def active(self) -> Optional[Channel]:
        return self.get(self.active_id)

    def get(self, cid: str) -> Optional[Channel]:
        return next((c for c in self.channels if c.id == cid), None)

    def set_client(self, client_id: str, secret: str) -> None:
        self.client_id = client_id
        self.keys.set("yt-client", secret)
        self.save()

    def client_secret(self) -> str:
        return self.keys.get("yt-client")

    def token(self, cid: str) -> dict:
        try:
            return json.loads(self.keys.get(f"yt:{cid}") or "{}")
        except Exception:
            return {}

    def save_token(self, cid: str, tok: dict) -> None:
        self.keys.set(f"yt:{cid}", json.dumps(tok, separators=(",", ":")))

    def add_channel(self, ch: Channel, tok: dict) -> None:
        self.channels = [c for c in self.channels if c.id != ch.id] + [ch]
        self.save_token(ch.id, tok)
        self.active_id = ch.id
        self.save()

    def set_active(self, cid: str) -> None:
        if self.get(cid):
            self.active_id = cid
            self.save()

    def remove(self, cid: str) -> None:
        self.channels = [c for c in self.channels if c.id != cid]
        self.keys.delete(f"yt:{cid}")
        if self.active_id == cid:
            self.active_id = self.channels[0].id if self.channels else ""
        self.save()


def login(store: YouTubeStore, cancel: threading.Event, timeout: float = 300.0, session=requests) -> Channel:
    """Opens the browser for Google sign-in and waits for the redirect."""
    if not store.client_id:
        raise YTError("Choose your Google client file first.")
    got: dict = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if not q:
                self.send_response(204)
                self.end_headers()
                return
            got.update({k: v[0] for k, v in q.items()})
            ok = "code" in q
            html = ("<html><body style=\"font-family:Segoe UI,sans-serif;background:#0e0e10;color:#ececee;"
                    "padding:48px\"><h3 style=\"font-weight:600\">" + ("Connected" if ok else "Not connected") +
                    "</h3><p style=\"color:#a0a0a8\">You can close this tab and go back to MyClip AI.</p></body></html>")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))

        def log_message(self, *args) -> None:
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    srv.timeout = 0.5
    redirect = f"http://127.0.0.1:{srv.server_address[1]}"
    verifier, challenge = pkce()
    state = secrets.token_urlsafe(16)
    webbrowser.open(auth_url(store.client_id, redirect, challenge, state))
    end = time.time() + timeout
    try:
        while "code" not in got and "error" not in got and time.time() < end:
            if cancel.is_set():
                raise Cancelled()
            srv.handle_request()
    finally:
        srv.server_close()
    if not got:
        raise YTError("Sign-in timed out. Try again.")
    if got.get("error"):
        raise YTError("Sign-in was cancelled." if got["error"] == "access_denied" else got["error"])
    if got.get("state") != state:
        raise YTError("Sign-in failed a security check. Try again.")
    r = session.post(TOKEN_URL, data={"code": got["code"], "client_id": store.client_id,
                                      "client_secret": store.client_secret(), "redirect_uri": redirect,
                                      "grant_type": "authorization_code", "code_verifier": verifier}, timeout=30)
    if r.status_code >= 400:
        raise _error(r)
    tok = _tok(r.json())
    if not tok["refresh_token"]:
        raise YTError("Google didn't allow offline access. Remove MyClip from your Google account's "
                      "third-party access and connect again.")
    items = YouTube(store, "", session, token=tok).channels_mine()
    if not items:
        raise YTError("This Google account has no YouTube channel yet. Create one on youtube.com first.")
    first = None
    for it in items:
        sn = it.get("snippet", {})
        ch = Channel(it["id"], sn.get("title", "Channel"), sn.get("customUrl", ""),
                     ((sn.get("thumbnails") or {}).get("default") or {}).get("url", ""))
        store.add_channel(ch, tok)
        first = first or ch
    store.set_active(first.id)
    return first


# ---------------------------------------------------------------- api

def _next_offset(rng: str) -> int:
    m = re.search(r"bytes=0-(\d+)", rng or "")
    return int(m[1]) + 1 if m else 0


def _video(v: dict) -> dict:
    sn, st, stats = v.get("snippet", {}), v.get("status", {}), v.get("statistics", {})
    th = sn.get("thumbnails") or {}
    return {"id": v.get("id", ""), "title": sn.get("title", ""), "description": sn.get("description", ""),
            "tags": sn.get("tags", []), "category": sn.get("categoryId", "20"),
            "published": sn.get("publishedAt", ""), "privacy": st.get("privacyStatus", ""),
            "publish_at": st.get("publishAt", ""), "kids": bool(st.get("selfDeclaredMadeForKids", st.get("madeForKids", False))),
            "upload_status": st.get("uploadStatus", ""), "failure": st.get("failureReason") or st.get("rejectionReason") or "",
            "thumb": (th.get("medium") or th.get("default") or {}).get("url", ""),
            "views": int(stats.get("viewCount", 0) or 0), "likes": int(stats.get("likeCount", 0) or 0),
            "comments": int(stats.get("commentCount", 0) or 0),
            "duration": v.get("contentDetails", {}).get("duration", "")}


class YouTube:
    def __init__(self, store: YouTubeStore, channel_id: str, session=requests, token: Optional[dict] = None) -> None:
        self.store, self.cid, self.http = store, channel_id, session
        self._tok = token

    # auth
    def _get_tok(self) -> dict:
        return self._tok if self._tok is not None else self.store.token(self.cid)

    def _put_tok(self, tok: dict) -> None:
        if self._tok is not None:
            self._tok = tok
        else:
            self.store.save_token(self.cid, tok)

    def access_token(self, force: bool = False) -> str:
        tok = self._get_tok()
        if not tok.get("refresh_token") and not tok.get("access_token"):
            raise YTError("This channel needs to be connected again.", "auth")
        if force or tok.get("expires_at", 0) < time.time() + 60:
            if not tok.get("refresh_token"):
                raise YTError("This channel needs to be connected again.", "auth")
            r = self.http.post(TOKEN_URL, data={"client_id": self.store.client_id,
                                                "client_secret": self.store.client_secret(),
                                                "refresh_token": tok["refresh_token"],
                                                "grant_type": "refresh_token"}, timeout=30)
            if r.status_code >= 400:
                e = _error(r)
                if e.reason in ("invalid_grant", "unauthorized_client"):
                    raise YTError("Access to this channel expired. Right-click it and choose Reconnect.", "auth")
                raise e
            tok = _tok(r.json(), tok)
            self._put_tok(tok)
        return tok["access_token"]

    def req(self, method: str, url: str, params=None, json_body=None, data=None, headers=None, timeout=60):
        r = None
        for attempt in range(2):
            h = {"Authorization": f"Bearer {self.access_token(force=attempt == 1)}"}
            h.update(headers or {})
            r = self.http.request(method, url, params=params, json=json_body, data=data, headers=h, timeout=timeout)
            if r.status_code != 401:
                break
        if r.status_code >= 400:
            raise _error(r)
        return r

    # read
    def channels_mine(self) -> list[dict]:
        r = self.req("GET", f"{API}/channels", params={"part": "snippet,statistics,contentDetails", "mine": "true"})
        return r.json().get("items", [])

    def overview(self) -> dict:
        items = self.channels_mine()
        c = next((i for i in items if i.get("id") == self.cid), items[0] if items else None)
        if not c:
            raise YTError("Channel not found.")
        sn, st = c.get("snippet", {}), c.get("statistics", {})
        return {"id": c["id"], "title": sn.get("title", ""), "handle": sn.get("customUrl", ""),
                "thumb": ((sn.get("thumbnails") or {}).get("medium") or (sn.get("thumbnails") or {}).get("default") or {}).get("url", ""),
                "subs": None if st.get("hiddenSubscriberCount") else int(st.get("subscriberCount", 0) or 0),
                "views": int(st.get("viewCount", 0) or 0), "videos": int(st.get("videoCount", 0) or 0),
                "uploads": c.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads", "")}

    def videos(self, uploads_playlist: str, limit: int = 50) -> list[dict]:
        if not uploads_playlist:
            return []
        r = self.req("GET", f"{API}/playlistItems", params={"part": "contentDetails", "playlistId": uploads_playlist,
                                                            "maxResults": min(50, limit)})
        ids = [i["contentDetails"]["videoId"] for i in r.json().get("items", []) if i.get("contentDetails")]
        if not ids:
            return []
        r = self.req("GET", f"{API}/videos", params={"part": "snippet,statistics,status,contentDetails", "id": ",".join(ids)})
        vids = [_video(v) for v in r.json().get("items", [])]
        order = {v: i for i, v in enumerate(ids)}
        return sorted(vids, key=lambda v: order.get(v["id"], 0))

    def view_counts(self, video_ids: list[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        for k in range(0, len(video_ids), 50):
            r = self.req("GET", f"{API}/videos", params={"part": "statistics", "id": ",".join(video_ids[k:k + 50])})
            for v in r.json().get("items", []):
                out[v["id"]] = int((v.get("statistics") or {}).get("viewCount", 0) or 0)
        return out

    def comments(self, limit: int = 30) -> list[dict]:
        r = self.req("GET", f"{API}/commentThreads", params={"part": "snippet", "allThreadsRelatedToChannelId": self.cid,
                                                             "maxResults": min(100, limit), "order": "time",
                                                             "textFormat": "plainText"})
        out = []
        for t in r.json().get("items", []):
            top = t.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
            out.append({"id": t.get("id", ""), "video_id": t.get("snippet", {}).get("videoId", ""),
                        "author": top.get("authorDisplayName", ""), "text": top.get("textDisplay", ""),
                        "published": top.get("publishedAt", ""), "likes": int(top.get("likeCount", 0) or 0),
                        "replies": int(t.get("snippet", {}).get("totalReplyCount", 0) or 0)})
        return out

    # write
    def update(self, video: dict, title: str, description: str, tags: list[str], privacy: str,
               publish_at: str = "") -> None:
        m = clean_meta(title, description, tags, shorts=False)
        status = {"privacyStatus": "private" if publish_at else privacy, "selfDeclaredMadeForKids": bool(video.get("kids"))}
        if publish_at:
            status["publishAt"] = to_rfc3339(publish_at)
        body = {"id": video["id"], "status": status,
                "snippet": {"title": m["title"], "description": m["description"], "tags": m["tags"],
                            "categoryId": video.get("category") or "20"}}
        self.req("PUT", f"{API}/videos", params={"part": "snippet,status"}, json_body=body)

    def delete(self, video_id: str) -> None:
        self.req("DELETE", f"{API}/videos", params={"id": video_id})

    def reply(self, thread_id: str, text: str) -> None:
        self.req("POST", f"{API}/comments", params={"part": "snippet"},
                 json_body={"snippet": {"parentId": thread_id, "textOriginal": text}})

    def set_thumbnail(self, video_id: str, path: str) -> None:
        ctype = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        with open(path, "rb") as f:
            self.req("POST", f"{UPLOAD}/thumbnails/set", params={"videoId": video_id}, data=f.read(),
                     headers={"Content-Type": ctype}, timeout=120)

    def upload(self, path: str, body: dict, on_progress: Callable[[float], None], cancel: threading.Event,
               session_url: str = "", on_session: Optional[Callable[[str], None]] = None) -> str:
        """Resumable upload. Survives network drops and app restarts (via session_url)."""
        size = os.path.getsize(path)
        if size == 0:
            raise YTError("The file is empty.")
        if not session_url:
            r = self.req("POST", f"{UPLOAD}/videos", params={"uploadType": "resumable", "part": "snippet,status"},
                         json_body=body, headers={"X-Upload-Content-Length": str(size),
                                                  "X-Upload-Content-Type": "video/*"})
            session_url = r.headers.get("Location", "")
            if not session_url:
                raise YTError("YouTube didn't start the upload.")
            if on_session:
                on_session(session_url)
        pos = self._offset(session_url, size)
        if isinstance(pos, str):
            on_progress(1.0)
            return pos
        fails = 0
        with open(path, "rb") as f:
            while True:
                if cancel.is_set():
                    raise Cancelled()
                f.seek(pos)
                chunk = f.read(CHUNK)
                end = pos + len(chunk) - 1
                try:
                    r = self.http.request("PUT", session_url, data=chunk, timeout=180,
                                          headers={"Authorization": f"Bearer {self.access_token()}",
                                                   "Content-Length": str(len(chunk)),
                                                   "Content-Range": f"bytes {pos}-{end}/{size}"})
                except requests.RequestException:
                    r = None
                if r is not None and r.status_code in (200, 201):
                    on_progress(1.0)
                    return r.json()["id"]
                if r is not None and r.status_code == 308:
                    pos = _next_offset(r.headers.get("Range", ""))
                    fails = 0
                    on_progress(pos / size)
                    continue
                if r is not None and r.status_code in (404, 410):
                    raise YTError("The upload session expired. Retry to start it again.", "session")
                if r is not None and 400 <= r.status_code < 500 and r.status_code not in (408, 429):
                    raise _error(r)
                fails += 1
                if fails > 8:
                    raise YTError("The upload keeps failing. Check your internet and retry.")
                if cancel.wait(min(60, 2 ** fails)):
                    raise Cancelled()
                q = self._offset(session_url, size)
                if isinstance(q, str):
                    on_progress(1.0)
                    return q
                pos = q

    def _offset(self, url: str, size: int):
        try:
            r = self.http.request("PUT", url, data=b"", timeout=30,
                                  headers={"Authorization": f"Bearer {self.access_token()}", "Content-Length": "0",
                                           "Content-Range": f"bytes */{size}"})
        except requests.RequestException:
            return 0
        if r.status_code in (200, 201):
            return r.json()["id"]
        if r.status_code == 308:
            return _next_offset(r.headers.get("Range", ""))
        if r.status_code in (404, 410):
            raise YTError("The upload session expired. Retry to start it again.", "session")
        return 0


# ---------------------------------------------------------------- metadata

def clean_meta(title: str, description: str, tags: list, shorts: bool) -> dict:
    t = re.sub(r"\s+", " ", re.sub(r"[<>]", "", title or "")).strip()[:100].strip() or "Untitled"
    clean: list[str] = []
    total = 0
    for raw in tags or []:
        x = re.sub(r"[<>,#]", "", str(raw)).strip()
        if not x or x.lower() in {y.lower() for y in clean}:
            continue
        cost = len(x) + (2 if " " in x else 0) + (1 if clean else 0)
        if total + cost > 500:
            break
        clean.append(x)
        total += cost
    d = re.sub(r"[<>]", "", description or "").strip()
    add = ["#" + re.sub(r"\s+", "", x) for x in clean[:5]]
    add = [h for h in add if h.lower() not in d.lower()]
    if shorts and "#shorts" not in (t + " " + d).lower():
        add.append("#Shorts")
    if add:
        d = (d + "\n\n" + " ".join(add)).strip()
    return {"title": t, "description": d[:5000], "tags": clean}


def to_rfc3339(local_iso: str) -> str:
    dt = datetime.fromisoformat(local_iso)
    if dt.tzinfo is None:
        dt = dt.astimezone()                        # treat as this PC's local time
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def spread_schedule(n: int, first: datetime, every_hours: float) -> list[str]:
    return [(first + timedelta(hours=every_hours * i)).strftime("%Y-%m-%dT%H:%M") for i in range(n)]


# ---------------------------------------------------------------- queue

@dataclass
class UploadItem:
    file: str
    title: str
    description: str = ""
    tags: list = field(default_factory=list)
    privacy: str = "private"
    publish_at: str = ""
    category: str = "20"
    made_for_kids: bool = False
    shorts: bool = True
    channel_id: str = ""
    thumbnail: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    status: str = "queued"            # queued | uploading | done | failed
    progress: float = 0.0
    video_id: str = ""
    error: str = ""
    session_url: str = ""
    added: float = field(default_factory=time.time)


def build_body(it: UploadItem) -> dict:
    m = clean_meta(it.title, it.description, it.tags, it.shorts)
    status = {"privacyStatus": "private" if it.publish_at else it.privacy,
              "selfDeclaredMadeForKids": bool(it.made_for_kids), "embeddable": True}
    if it.publish_at:
        status["publishAt"] = to_rfc3339(it.publish_at)
    return {"snippet": {"title": m["title"], "description": m["description"], "tags": m["tags"],
                        "categoryId": it.category}, "status": status}


class UploadQueue:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.RLock()
        self.items: list[UploadItem] = []
        try:
            known = {f.name for f in fields(UploadItem)}
            for d in json.loads(path.read_text(encoding="utf-8")):
                it = UploadItem(**{k: v for k, v in d.items() if k in known})
                if it.status == "uploading":
                    it.status = "queued"             # resumes from session_url
                self.items.append(it)
        except Exception:
            pass

    def save(self) -> None:
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps([asdict(i) for i in self.items], indent=1, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)

    def add(self, items: list[UploadItem]) -> None:
        with self.lock:
            self.items.extend(items)
            self.save()

    def get(self, item_id: str) -> Optional[UploadItem]:
        with self.lock:
            return next((i for i in self.items if i.id == item_id), None)

    def next_pending(self) -> Optional[UploadItem]:
        with self.lock:
            return next((i for i in self.items if i.status == "queued"), None)

    def set(self, item: UploadItem, **kw) -> None:
        with self.lock:
            for k, v in kw.items():
                setattr(item, k, v)
            self.save()

    def remove(self, item_id: str) -> None:
        with self.lock:
            self.items = [i for i in self.items if i.id != item_id]
            self.save()

    def clear_done(self) -> None:
        with self.lock:
            self.items = [i for i in self.items if i.status != "done"]
            self.save()

    @property
    def pending(self) -> int:
        with self.lock:
            return sum(1 for i in self.items if i.status in ("queued", "uploading"))


def process(item: UploadItem, store: YouTubeStore, queue: UploadQueue,
            on_progress: Callable[[str, float], None], cancel: threading.Event, session=requests) -> None:
    """Upload one item. Raises YTError only for problems that should pause the
    whole queue (daily limit, signed out); everything else marks the item failed."""
    if not os.path.isfile(item.file):
        queue.set(item, status="failed", error="File not found. It may have been moved or deleted.")
        return
    if item.publish_at:
        try:
            when = datetime.fromisoformat(item.publish_at)
        except ValueError:
            when = None
        if when is None or when <= datetime.now() + timedelta(minutes=5):
            queue.set(item, status="failed", error="The scheduled time has passed. Pick a new time and retry.")
            return
    cid = item.channel_id or store.active_id
    if not store.get(cid):
        queue.set(item, status="failed", error="That channel isn't connected any more.")
        return
    yt = YouTube(store, cid, session)
    queue.set(item, status="uploading", error="")
    last = [item.progress]

    def prog(f: float) -> None:
        on_progress(item.id, f)
        if f - last[0] >= 0.02 or f >= 1.0:
            last[0] = f
            queue.set(item, progress=f)

    try:
        vid = yt.upload(item.file, build_body(item), prog, cancel, item.session_url,
                        lambda u: queue.set(item, session_url=u))
    except Cancelled:
        queue.set(item, status="queued")
        raise
    except YTError as e:
        if e.reason == "session":
            queue.set(item, status="failed", session_url="", progress=0.0, error=str(e))
            return
        if e.reason in QUOTA_REASONS or e.reason == "auth":
            queue.set(item, status="queued", error=str(e))
            raise
        queue.set(item, status="failed", error=str(e))
        return
    queue.set(item, status="done", progress=1.0, video_id=vid, session_url="")
    try:                                             # remember it so the views can teach the scoring
        from .learning import mark_uploaded
        mark_uploaded(item.file, cid, vid)
    except Exception as exc:
        print(f"[MyClip] couldn't link the upload for learning: {exc}", flush=True)
    if item.thumbnail and os.path.isfile(item.thumbnail):
        try:
            yt.set_thumbnail(vid, item.thumbnail)
        except Exception as exc:
            queue.set(item, error=f"Uploaded, but the thumbnail wasn't set: {exc}")


def refresh_learning(store: YouTubeStore, channel_id: str, session=requests) -> tuple[int, int]:
    """Fetch views for uploads that are 48 h or older and refit. Returns (checked, videos with views)."""
    from . import learning
    ids = learning.due(channel_id)
    if not ids:
        n_all, n_views, _ = learning.summary(channel_id)
        return 0, n_views
    views = YouTube(store, channel_id, session).view_counts(ids)
    missing = [v for v in ids if v not in views]          # deleted or private: keep, try again later
    learning.update_views(channel_id, views)
    _, n_views, _ = learning.summary(channel_id)
    return len(views), n_views
