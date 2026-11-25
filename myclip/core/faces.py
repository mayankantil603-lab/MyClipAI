"""Faces: a smoothed crop that follows the streamer's face, and facecam detection.

Detection uses OpenCV's YuNet face model when it's installed (Settings -> AI helpers),
with OpenCV's built-in Haar detector as a fallback. All the path maths is plain numpy."""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from . import ffmpeg, helpers
from .models import Cancelled, MediaInfo


def available() -> bool:
    return helpers.installed("faces")


class Detector:
    def __init__(self, w: int, h: int) -> None:
        import cv2
        self.cv2 = cv2
        self.kind = "haar"
        p = helpers.yunet_path()
        self.yn = None
        if p.exists() and hasattr(cv2, "FaceDetectorYN"):
            try:
                self.yn = cv2.FaceDetectorYN.create(str(p), "", (w, h), 0.72, 0.3, 50)
                self.kind = "yunet"
            except Exception:
                self.yn = None
        if self.yn is None:
            self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, bgr: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """Faces as (x, y, w, h, score) in pixels."""
        if self.yn is not None:
            self.yn.setInputSize((bgr.shape[1], bgr.shape[0]))
            _, faces = self.yn.detect(bgr)
            if faces is None:
                return []
            return [(float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[-1])) for f in faces]
        gray = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2GRAY)
        found = self.haar.detectMultiScale(gray, 1.15, 5, minSize=(24, 24))
        return [(float(x), float(y), float(w), float(h), 0.8) for (x, y, w, h) in found]


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def track(media: str, start: float, end: float, info: MediaInfo, cancel: threading.Event,
          fps: float = 3.0, width: int = 480) -> list[tuple[float, Optional[tuple[float, float, float, float]]]]:
    """(time from clip start, face box normalised (x, y, w, h)) per sampled frame; None when no face."""
    h = _even(width * info.height / max(info.width, 1))
    frames = ffmpeg.raw_frames(media, start, end - start, fps, width, h, "bgr24", cancel)
    if not len(frames):
        return []
    det = Detector(width, h)
    out, prev = [], None
    for i, fr in enumerate(frames):
        if cancel.is_set():
            raise Cancelled()
        faces = det.detect(np.ascontiguousarray(fr))
        best = pick_face(faces, prev, width, h)
        if best is not None:
            prev = best
        out.append((i / fps, best))
    return out


def pick_face(faces, prev, w: int, h: int) -> Optional[tuple[float, float, float, float]]:
    """Biggest confident face, preferring one near the previous position (keeps the same person)."""
    if not faces:
        return None
    def norm(f):
        return (f[0] / w, f[1] / h, f[2] / w, f[3] / h)
    def score(f):
        x, y, fw, fh = norm(f)
        s = f[4] * (fw * fh) ** 0.5
        if prev is not None:
            dx = (x + fw / 2) - (prev[0] + prev[2] / 2)
            s *= 1.0 / (1.0 + 6.0 * abs(dx))
        return s
    return norm(max(faces, key=score))


def face_to_center(fx: float, src_w: int, src_h: int, W: int, H: int) -> Optional[float]:
    """Face centre (0..1 of the source width) -> crop position c used by the fill filter
    (crop x = (iw - W) * c, after scaling the source to height H). None if there's nothing to pan."""
    iw = H * src_w / max(src_h, 1)
    if iw <= W + 1:
        return None
    c = (fx * iw - W / 2) / (iw - W)
    return float(min(1.0, max(0.0, c)))


def smooth_path(times: list[float], targets: list[Optional[float]], duration: float, step: float = 0.5,
                dead: float = 0.05, max_speed: float = 0.35) -> list[tuple[float, float]]:
    """Camera-operator style: hold still inside a dead zone, then ease toward the face with a speed limit."""
    valid = [(t, c) for t, c in zip(times, targets) if c is not None]
    if not valid:
        return []
    grid = np.arange(0.0, max(duration, step) + 1e-6, step)
    ts = np.array([t for t, _ in valid])
    cs = np.array([c for _, c in valid])
    raw = np.array([cs[int(np.argmin(np.abs(ts - g)))] for g in grid])
    if len(raw) >= 3:                                   # median of 3 removes single bad detections
        pad = np.concatenate([raw[:1], raw, raw[-1:]])
        raw = np.median(np.stack([pad[:-2], pad[1:-1], pad[2:]]), axis=0)
    cur = float(np.median(raw[: max(1, int(2.0 / step))]))
    out = []
    lim = max_speed * step
    for g, tgt in zip(grid, raw):
        d = float(tgt) - cur
        if abs(d) > dead:
            cur += max(-lim, min(lim, d * 0.5))
        out.append((round(float(g), 3), round(min(1.0, max(0.0, cur)), 4)))
    return out


def simplify(path: list[tuple[float, float]], tol: float = 0.012) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker on (t, c): keeps the ffmpeg expression short."""
    if len(path) <= 2:
        return list(path)
    keep = [False] * len(path)
    keep[0] = keep[-1] = True
    stack = [(0, len(path) - 1)]
    while stack:
        a, b = stack.pop()
        (ta, ca), (tb, cb) = path[a], path[b]
        best, idx = 0.0, -1
        for i in range(a + 1, b):
            t, c = path[i]
            lin = ca + (cb - ca) * ((t - ta) / (tb - ta) if tb > ta else 0.0)
            e = abs(c - lin)
            if e > best:
                best, idx = e, i
        if best > tol and idx > 0:
            keep[idx] = True
            stack += [(a, idx), (idx, b)]
    return [p for p, k in zip(path, keep) if k]


def x_expr(path: list[tuple[float, float]], W: int) -> str:
    """Piecewise-linear crop x for ffmpeg's crop filter: (iw-W) * c(t), clamped to the frame."""
    p = simplify(path)
    c0 = p[0][1]
    terms = []
    for (t0, a), (t1, b) in zip(p, p[1:]):
        dt = t1 - t0
        if dt <= 0 or abs(b - a) < 1e-4:
            continue
        terms.append(f"{(b - a) / dt:+.5f}*clip(t-{t0:.3f},0,{dt:.3f})")
    inner = f"{c0:.4f}" + "".join(terms)
    return f"max(0,min(iw-{W},(iw-{W})*({inner})))"


def crop_path(media: str, start: float, end: float, info: MediaInfo, W: int, H: int,
              cancel: threading.Event) -> Optional[list[tuple[float, float]]]:
    """A smooth crop path that follows the face, or None (no faces / nothing to pan)."""
    samples = track(media, start, end, info, cancel)
    if not samples:
        return None
    hits = [s for s in samples if s[1] is not None]
    if len(hits) < max(2, 0.35 * len(samples)):
        return None
    times, targets = [], []
    for t, box in samples:
        times.append(t)
        targets.append(None if box is None else face_to_center(box[0] + box[2] / 2, info.width, info.height, W, H))
    if all(c is None for c in targets):
        return None
    path = smooth_path(times, targets, end - start)
    return path or None


# ---------------------------------------------------------------- facecam

def persistent_edges(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per column / per row edge strength that stays put across frames (overlay borders).
    gray: [n, h, w] uint8. Returns (vertical-line strength per x, horizontal-line strength per y)."""
    g = gray.astype(np.float32)
    gx = np.abs(np.diff(g, axis=2))                    # [n, h, w-1]
    gy = np.abs(np.diff(g, axis=1))                    # [n, h-1, w]
    vx = gx.mean(axis=1)                               # [n, w-1]
    vy = gy.mean(axis=2)                               # [n, h-1]
    sx = vx.mean(0) / (1.0 + vx.std(0))                # strong and steady
    sy = vy.mean(0) / (1.0 + vy.std(0))
    return sx, sy


def snap_edges(gray: np.ndarray, rect: tuple[int, int, int, int], search: float = 0.35) -> tuple[int, int, int, int]:
    """Move each side of rect (x, y, w, h in pixels) to the strongest steady edge nearby."""
    n, H, W = gray.shape
    x, y, w, h = rect

    def best(profile: np.ndarray, pos: int, span: int, lo: int, hi: int) -> int:
        a, b = max(lo, pos - span), min(hi, pos + span)
        if b <= a:
            return pos
        seg = profile[a:b]
        k = int(np.argmax(seg))
        base = float(np.median(profile[max(0, a - span):min(len(profile), b + span)]) + 1e-6)
        return a + k if float(seg[k]) > 2.2 * base else pos

    y0, y1 = max(0, y), min(H, y + h)
    x0, x1 = max(0, x), min(W, x + w)
    band_v = gray[:, y0:y1, :] if y1 - y0 > 4 else gray
    band_h = gray[:, :, x0:x1] if x1 - x0 > 4 else gray
    sx, _ = persistent_edges(band_v)
    _, sy = persistent_edges(band_h)
    sx_pad = np.concatenate([sx, sx[-1:]])
    sy_pad = np.concatenate([sy, sy[-1:]])
    left = best(sx_pad, x, int(w * search), 0, W - 1)
    right = best(sx_pad, x + w, int(w * search), 1, W)
    top = best(sy_pad, y, int(h * search), 0, H - 1)
    bottom = best(sy_pad, y + h, int(h * search), 1, H)
    if right - left < w * 0.5 or bottom - top < h * 0.5:
        return rect
    return left, top, right - left, bottom - top


def cluster_faces(boxes: list[tuple[float, float, float, float]], min_share: float = 0.55
                  ) -> Optional[tuple[float, float, float, float]]:
    """The face box that appears in the same place in most samples (a facecam), as a median box."""
    if not boxes:
        return None
    centers = np.array([[b[0] + b[2] / 2, b[1] + b[3] / 2] for b in boxes])
    best_i, best_n = 0, 0
    for i, c in enumerate(centers):
        n = int((np.abs(centers - c) < 0.06).all(axis=1).sum())
        if n > best_n:
            best_i, best_n = i, n
    near = (np.abs(centers - centers[best_i]) < 0.06).all(axis=1)
    return tuple(float(v) for v in np.median(np.array(boxes)[near], axis=0)) if best_n else None


def cam_rect_from_face(face: tuple[float, float, float, float], src_ar: float) -> tuple[float, float, float, float]:
    """A first guess of the camera box (normalised) around a facecam face."""
    fx, fy, fw, fh = face
    cw = min(0.6, fw * 3.4)
    ch = min(0.6, cw * src_ar * 9 / 16)                 # most facecams are 16:9
    ch = max(ch, fh * 2.6)
    cx, cy = fx + fw / 2, fy + fh * 0.55
    x = min(max(0.0, cx - cw / 2), 1 - cw)
    y = min(max(0.0, cy - ch / 2), 1 - ch)
    return (round(x, 4), round(y, 4), round(cw, 4), round(ch, 4))


def detect_facecam(media: str, info: MediaInfo, cancel: threading.Event, samples: int = 14
                   ) -> Optional[tuple[float, float, float, float]]:
    """Normalised (x, y, w, h) of the facecam box, or None if there's no clear facecam."""
    width = 640
    h = _even(width * info.height / max(info.width, 1))
    det = Detector(width, h)
    times = [info.duration * (0.08 + 0.84 * i / max(1, samples - 1)) for i in range(samples)]
    boxes, grays = [], []
    for t in times:
        if cancel.is_set():
            raise Cancelled()
        fr = ffmpeg.raw_frames(media, t, 0.2, 5, width, h, "bgr24", cancel)
        if not len(fr):
            continue
        bgr = np.ascontiguousarray(fr[0])
        grays.append(bgr.mean(axis=2).astype(np.uint8))
        faces = det.detect(bgr)
        small = [f for f in faces if (f[2] * f[3]) / (width * h) < 0.05]
        if small:
            f = max(small, key=lambda f: f[4] * f[2] * f[3])
            boxes.append((f[0] / width, f[1] / h, f[2] / width, f[3] / h))
    if len(boxes) < max(3, 0.5 * len(grays)):
        return None
    face = cluster_faces(boxes)
    if face is None:
        return None
    guess = cam_rect_from_face(face, info.width / max(info.height, 1))
    if len(grays) >= 4:
        g = np.stack(grays)
        px = (int(guess[0] * width), int(guess[1] * h), int(guess[2] * width), int(guess[3] * h))
        sx, sy, sw, sh = snap_edges(g, px)
        guess = (round(sx / width, 4), round(sy / h, 4), round(sw / width, 4), round(sh / h, 4))
    return guess
