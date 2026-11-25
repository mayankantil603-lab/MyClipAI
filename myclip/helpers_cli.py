"""Runs heavy helper models in their own process, so PyTorch never loads into the app
and its GPU memory is fully freed when it's done.

    python -m myclip.helpers_cli separate jobs.json
    python -m myclip.helpers_cli selftest voice

Prints one JSON object per line: {"done": i, "total": n} and finally {"ok": true, ...}."""
from __future__ import annotations

import json
import sys
import wave

import numpy as np


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError("expected 16-bit wav")
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return a.reshape(-1, ch).T.copy(), sr


def write_wav(path: str, a: np.ndarray, sr: int) -> None:
    a = np.clip(a, -1.0, 1.0)
    pcm = (a.T * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(a.shape[0])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _say(**kw) -> None:
    print(json.dumps(kw), flush=True)


def _model(device: str):
    import torch
    from demucs.pretrained import get_model
    model = get_model("htdemucs")
    model.eval()
    dev = "cuda" if (device != "cpu" and torch.cuda.is_available()) else "cpu"
    try:
        model.to(dev)
    except Exception:
        dev = "cpu"
        model.to(dev)
    return model, dev


def vocals(model, dev: str, x: np.ndarray) -> np.ndarray:
    import torch
    from demucs.apply import apply_model
    if x.shape[0] == 1:
        x = np.repeat(x, 2, axis=0)
    wav = torch.from_numpy(x[:2])
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    wav = (wav - mean) / std
    with torch.no_grad():
        try:
            out = apply_model(model, wav[None].to(dev), device=dev, split=True, overlap=0.25, progress=False)[0]
        except RuntimeError as exc:
            if dev != "cuda" or "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            out = apply_model(model.to("cpu"), wav[None], device="cpu", split=True, overlap=0.25, progress=False)[0]
    out = out * std + mean
    return out[model.sources.index("vocals")].cpu().numpy()


def separate(jobs_file: str) -> int:
    jobs = json.load(open(jobs_file, encoding="utf-8"))
    items = jobs.get("items", [])
    model, dev = _model(jobs.get("device", "auto"))
    _say(device=dev)
    for i, it in enumerate(items, 1):
        try:
            x, sr = read_wav(it["in"])
            write_wav(it["out"], vocals(model, dev, x), sr)
            _say(done=i, total=len(items))
        except Exception as exc:
            _say(done=i, total=len(items), error=str(exc)[:200])
    _say(ok=True, device=dev)
    return 0


def selftest(kind: str) -> int:
    try:
        if kind == "voice":
            model, dev = _model("auto")
            sr = model.samplerate
            t = np.arange(sr * 2) / sr
            x = np.stack([0.2 * np.sin(2 * np.pi * 220 * t)] * 2).astype(np.float32)
            v = vocals(model, dev, x)
            assert v.shape[-1] == x.shape[-1]
            _say(ok=True, msg=f"Ready ({'GPU' if dev == 'cuda' else 'CPU'})")
            return 0
        _say(ok=False, msg=f"unknown test {kind}")
        return 1
    except Exception as exc:
        _say(ok=False, msg=f"{type(exc).__name__}: {exc}")
        return 1


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "separate":
        return separate(argv[1])
    if len(argv) >= 2 and argv[0] == "selftest":
        return selftest(argv[1])
    print("usage: separate jobs.json | selftest voice")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
