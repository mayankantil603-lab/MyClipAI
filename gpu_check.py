"""Prints whether Whisper can run on this PC's NVIDIA GPU, and why not if it can't.
Run:  python gpu_check.py"""
import sys
import traceback

sys.path.insert(0, ".")
from myclip.core.gpu import cuda_info, explain_gpu_error, missing_cuda_libs, register_cuda_dlls  # noqa: E402

register_cuda_dlls()
ok, name, vram = cuda_info()
try:
    import ctranslate2
    print(f"CTranslate2 {ctranslate2.__version__}, CUDA devices: {ctranslate2.get_cuda_device_count()}")
except Exception as exc:
    print(f"CTranslate2 not installed: {exc}")
print(f"GPU: {name or 'none found'}" + (f", {vram} MB" if vram else ""))
miss = missing_cuda_libs()
print("Libraries: " + ("all found" if not miss else "missing " + ", ".join(miss)))
if not ok:
    print("\nRESULT: no usable NVIDIA GPU found. MyClip will use the CPU.")
    sys.exit(0)
try:
    import numpy as np
    from faster_whisper import WhisperModel
    print("Running a short test on the GPU (downloads the 75 MB tiny model once)...")
    m = WhisperModel("tiny", device="cuda", compute_type="float16")
    audio = (np.random.RandomState(0).randn(16000 * 4) * 0.05).astype("float32")
    segs, _ = m.transcribe(audio, vad_filter=False, beam_size=1, language="en")
    list(segs)
    print("\nRESULT: GPU transcription works.")
except Exception as exc:
    traceback.print_exc()
    print("\nRESULT: GPU transcription does NOT work.")
    print(explain_gpu_error(str(exc), miss))
