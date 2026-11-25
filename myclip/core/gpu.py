"""CUDA setup for faster-whisper on Windows.

pip's nvidia-* wheels put their DLLs under site-packages/nvidia/*/bin, which
Windows doesn't search. We register those folders before CTranslate2 loads."""
from __future__ import annotations

import functools
import os
import sys
from pathlib import Path

_registered = False


def register_cuda_dlls() -> list[str]:
    global _registered
    if _registered or sys.platform != "win32":
        return []
    _registered = True
    added: list[str] = []
    for base in map(Path, sys.path):
        nv = base / "nvidia"
        if not nv.is_dir():
            continue
        for bin_dir in nv.glob("*/bin"):
            try:
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
                added.append(str(bin_dir))
            except OSError:
                pass
    return added


@functools.lru_cache(maxsize=1)
def cuda_info() -> tuple[bool, str, int]:
    """(available, gpu name, VRAM MB). VRAM via nvidia-smi when present."""
    register_cuda_dlls()
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() <= 0:
            return False, "", 0
    except Exception:
        return False, "", 0
    name, vram = "NVIDIA GPU", 0
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        first = out.strip().splitlines()[0]
        n, m = first.rsplit(",", 1)
        name, vram = n.strip(), int(float(m))
    except Exception:
        pass
    return True, name, vram


_FAILED = {"reason": ""}


def mark_failed(reason: str) -> None:
    """GPU transcription failed once: use the CPU for the rest of this session
    instead of failing (and wasting time) again on every clip."""
    _FAILED["reason"] = reason


def gpu_failure() -> str:
    return _FAILED["reason"]


def missing_cuda_libs() -> list[str]:
    """The cuBLAS / cuDNN DLLs CTranslate2 needs on Windows that can't be found."""
    if sys.platform != "win32":
        return []
    register_cuda_dlls()
    try:
        import ctranslate2
        ver = tuple(int(x) for x in ctranslate2.__version__.split(".")[:2])
    except Exception:
        ver = (4, 5)
    need = ["cublas64_12.dll", "cublasLt64_12.dll"]
    need += ["cudnn_ops64_9.dll", "cudnn_cnn64_9.dll"] if ver >= (4, 5) else ["cudnn_ops_infer64_8.dll"]
    import ctypes
    missing = []
    for name in need:
        try:
            ctypes.WinDLL(name, winmode=0)          # normal Windows search, PATH included
        except OSError:
            missing.append(name)
    return missing


def explain_gpu_error(msg: str, missing: list[str] | None = None) -> str:
    m = (msg or "").lower()
    if missing or "cublas" in m or "cudnn" in m or ".dll" in m or ("library" in m and "not found" in m):
        return ("GPU libraries are missing" + (f" ({', '.join(missing[:2])})" if missing else "") +
                ". Close MyClip and start run.bat again (it installs them), or run install-gpu.bat.")
    if "out of memory" in m or "cudaerrormemoryallocation" in m:
        return "The GPU ran out of memory. Close games or other apps using the GPU and try again."
    if "no kernel image" in m or "sm_" in m or "unsupported gpu" in m or "invalid device function" in m:
        return "This Whisper version doesn't support your GPU yet. Run install-gpu.bat to update it."
    if "driver" in m or "insufficient" in m:
        return "Your NVIDIA driver is too old for GPU transcription. Update it in the NVIDIA app."
    return f"GPU error: {msg.strip()[:180]}"


def pick_whisper(choice: str, device: str) -> tuple[str, str, str]:
    """Resolve (model, device, compute_type) from user choice."""
    ok, _, vram = cuda_info()
    use_cuda = ok and device != "cpu" and not gpu_failure()
    if device == "cuda" and not ok:
        use_cuda = False
    if choice and choice != "auto":
        model = choice
    elif use_cuda:
        model = "large-v3-turbo" if (vram == 0 or vram >= 5500) else "small"
    else:
        model = "small"
    return model, ("cuda" if use_cuda else "cpu"), ("float16" if use_cuda else "int8")
