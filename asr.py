#!/usr/bin/env python3
"""Offline speech recognition for dubber (faster-whisper / CTranslate2).

usage: asr.py MODEL LANG|auto AUDIO OUTDIR [PROMPT]

Writes OUTDIR/orig.srt and OUTDIR/orig.json ({"language": ..., ...}).
Device is chosen by DUBBER_DEVICE (auto|cuda|cpu). Models must already be
in the local HuggingFace cache (see fetch_models.sh).
"""
import glob
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")


def cuda_libs():
    """ctranslate2 needs cuBLAS/cuDNN from the pip nvidia-* wheels on the
    loader path; re-exec once with LD_LIBRARY_PATH extended."""
    if os.environ.get("_DUBBER_LIBS"):
        return
    import site
    dirs = []
    for sp in site.getsitepackages():
        dirs += glob.glob(os.path.join(sp, "nvidia", "*", "lib"))
    if dirs:
        os.environ["LD_LIBRARY_PATH"] = ":".join(
            dirs + [os.environ.get("LD_LIBRARY_PATH", "")])
    os.environ["_DUBBER_LIBS"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


def load_16k_mono(path):
    """Read the WAV dubber prepared and hand Whisper raw samples, avoiding
    faster-whisper's PyAV decoder (version-sensitive)."""
    from math import gcd
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        g = gcd(16000, sr)
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    return x


def srt_time(t):
    ms = int(round(t * 1000))
    return "%02d:%02d:%02d,%03d" % (ms // 3600000, ms // 60000 % 60,
                                    ms // 1000 % 60, ms % 1000)


def main():
    if len(sys.argv) not in (5, 6):
        sys.exit(__doc__)
    cuda_libs()
    import ctranslate2
    from faster_whisper import WhisperModel

    model, lang, audio, outdir = sys.argv[1:5]
    prompt = sys.argv[5] if len(sys.argv) == 6 and sys.argv[5] else None
    dev = os.environ.get("DUBBER_DEVICE", "auto")
    if dev == "auto":
        dev = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    compute = "int8_float16" if dev == "cuda" else "int8"
    print(f"  faster-whisper {model} on {dev} ({compute})", file=sys.stderr)

    wm = WhisperModel(model, device=dev, compute_type=compute,
                      cpu_threads=os.cpu_count() or 4)
    segments, info = wm.transcribe(
        load_16k_mono(audio),
        language=None if lang == "auto" else lang,
        task="transcribe",
        beam_size=5,
        vad_filter=True,            # skip music/noise: fewer hallucinations
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
        initial_prompt=prompt,
        # word timing (cross-attention alignment) gives the real start of
        # the first spoken word; segment times alone can be off by ~0.5 s
        word_timestamps=True,
    )

    n = 0
    with open(os.path.join(outdir, "orig.srt"), "w", encoding="utf-8") as f:
        for s in segments:
            text = s.text.strip()
            if not text:
                continue
            start, end = s.start, s.end
            if s.words:
                start, end = s.words[0].start, s.words[-1].end
            n += 1
            f.write(f"{n}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n\n")
            print(f"  [{srt_time(start)[:8]}] {text}", file=sys.stderr,
                  flush=True)
    with open(os.path.join(outdir, "orig.json"), "w") as f:
        json.dump({"language": info.language,
                   "language_probability": info.language_probability,
                   "duration": info.duration}, f)
    print(f"  {n} segments, language {info.language} "
          f"({info.language_probability:.2f})", file=sys.stderr)


if __name__ == "__main__":
    main()
