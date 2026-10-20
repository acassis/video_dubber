#!/usr/bin/env python3
"""Offline speech recognition for dubber (faster-whisper / CTranslate2).

usage: asr.py MODEL LANG|auto AUDIO OUTDIR [PROMPT]

Writes OUTDIR/orig.srt and OUTDIR/orig.json ({"language": ..., ...}).
Device is chosen by DUBBER_DEVICE (auto|cuda|cpu). Models must already be
in the local HuggingFace cache (see fetch_models.sh).

Speech is transcribed in VAD clips of at most 15 s, so that dense speech
never overflows Whisper's 448-token window; speech still left uncovered
is transcribed again, and phrases longer than 10 s are split at a pause.
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


MAX_SEG = 10.0     # longer phrases are split at a pause: more timing anchors
MIN_GAP = 1.0      # speech left untranscribed for this long is redone
LONG_SEG = 20.0    # a phrase this long means Whisper's window overflowed
CLIP_MAX = 15.0    # first-pass clip length: dense speech fits 448 tokens
GAP_CLIP = 12.0    # second-pass clip length (well below Whisper's 30 s)
PUNCT = tuple(".,?!;:\u06d4\u060c\u061f\u0964")   # incl. Urdu/Hindi marks


def phrases(segments, offset=0.0, lo=None, hi=None):
    """Segments as lists of (start, end, word); words whose middle lies
    outside [lo, hi] are dropped (they belong to a neighbouring clip)."""
    out = []
    for s in segments:
        ws = [(w.start + offset, w.end + offset, w.word) for w in s.words or []
              if w.word.strip()] or \
             ([(s.start + offset, s.end + offset, s.text)] if s.text.strip() else [])
        if lo is not None:
            ws = [w for w in ws if lo <= (w[0] + w[1]) / 2 <= hi]
        if ws:
            out.append(ws)
    return out


def words_text(ws):
    return " ".join("".join(w[2] for w in ws).split())


def speech_clips(pcm):
    """Speech regions (Silero VAD) cut at pauses into clips of at most
    CLIP_MAX seconds, as a flat [start, end, start, end, ...] list.
    faster-whisper's own vad_filter packs speech into 30 s windows, which
    dense Urdu/Hindi speech overflows (see redo_gaps)."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    sr = 16000
    speech = get_speech_timestamps(pcm, VadOptions(
        min_silence_duration_ms=500, max_speech_duration_s=CLIP_MAX))
    clips = []
    for r in speech:
        a, b = r["start"] / sr, r["end"] / sr
        if clips and b - clips[-1][0] <= CLIP_MAX and a - clips[-1][1] < 1.0:
            clips[-1][1] = b
        else:
            clips.append([a, b])
    return [t for c in clips for t in c]


def redo_gaps(wm, pcm, segs, lang, opts):
    """Whisper decodes at most 448 tokens per 30 s window. Dense speech in
    a script that costs many tokens per word (Urdu, Hindi) overflows it:
    the window's text is cut off mid-word, its word times are spread
    evenly over the window, and the rest of the window is skipped. Such a
    window shows up as one phrase longer than LONG_SEG followed by speech
    (VAD) that no phrase covers. Both are transcribed again in short clips;
    returns the new list of phrases."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    sr = 16000
    bad = [s for s in segs if s[-1][1] - s[0][0] > LONG_SEG]
    segs = [s for s in segs if s[-1][1] - s[0][0] <= LONG_SEG]
    speech = get_speech_timestamps(pcm, VadOptions(min_silence_duration_ms=500))
    covered = sorted((s[0][0], s[-1][1]) for s in segs)
    holes = []
    for r in speech:
        a, b = r["start"] / sr, r["end"] / sr
        for c0, c1 in covered:
            if c1 <= a or c0 >= b:
                continue
            if c0 - a >= MIN_GAP:
                holes.append((a, c0))
            a = max(a, c1)
        if b - a >= MIN_GAP:
            holes.append((a, b))
    missed = sum(b - a for a, b in holes)
    holes += [(s[0][0], s[-1][1]) for s in bad]
    merged = []                 # an overflowed phrase and the gap after it
    for a, b in sorted(holes):
        if merged and a - merged[-1][1] < 0.5:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    found = []
    for a, b in merged:
        k = max(1, round((b - a) / GAP_CLIP + 0.49))
        for j in range(k):
            lo, hi = a + (b - a) * j / k, a + (b - a) * (j + 1) / k
            # start 3 s early: Whisper loses the first words of a clip cut
            # mid-word; what it repeats from the lead-in is filtered out
            x0 = max(0.0, lo - 3.0)
            clip = pcm[int(x0 * sr):int((hi + 0.5) * sr)]
            s, _ = wm.transcribe(clip, language=lang, vad_filter=False, **opts)
            found += phrases(s, x0, lo, hi)
    if merged:
        print(f"  {len(bad)} overflowed Whisper windows, {missed:.1f} s of "
              f"untranscribed speech: {sum(b - a for a, b in merged):.1f} s "
              f"transcribed again in short clips ({len(found)} segments)",
              file=sys.stderr)
    return segs + found


def no_overlaps(segs):
    """Drop a phrase that lies mostly inside an earlier one (a clip
    boundary repeated, or a short hallucination such as "thank you" at the
    edge of a clip): every phrase must own its time slot in the dub."""
    out = []
    for s in segs:
        a, b = s[0][0], s[-1][1]
        if out and min(b, out[-1][-1][1]) - a > 0.5 * (b - a):
            continue
        out.append(s)
    return out


def split(ws):
    """Split a phrase longer than MAX_SEG at its longest pause (preferring
    one after punctuation), recursively. Each piece then starts on time in
    the dub instead of one long translation drifting over 30 s of speech."""
    if ws[-1][1] - ws[0][0] <= MAX_SEG or len(ws) < 4:
        return [ws]
    t0, t1 = ws[0][0], ws[-1][1]
    best, at = -1e9, 0
    for i in range(2, len(ws) - 1):
        if ws[i - 1][1] - t0 < 3.0 or t1 - ws[i][0] < 3.0:
            continue
        # a long pause or punctuation wins; among equals, the most central
        # cut, so the pieces stay long enough to be translated as phrases
        score = ws[i][0] - ws[i - 1][1] + \
            (0.3 if ws[i - 1][2].strip().endswith(PUNCT) else 0.0) - \
            0.2 * abs(ws[i][0] - (t0 + t1) / 2) / (t1 - t0)
        if score > best:
            best, at = score, i
    if not at:
        return [ws]
    return split(ws[:at]) + split(ws[at:])


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
    pcm = load_16k_mono(audio)
    opts = dict(task="transcribe", beam_size=5,
                condition_on_previous_text=False, initial_prompt=prompt,
                # word timing (cross-attention alignment) gives the real
                # start of the first spoken word; segment times alone can
                # be off by ~0.5 s
                word_timestamps=True)
    # VAD clips instead of vad_filter: skips music/noise (fewer
    # hallucinations) without packing speech into full 30 s windows
    segments, info = wm.transcribe(
        pcm, language=None if lang == "auto" else lang,
        clip_timestamps=speech_clips(pcm), **opts)
    segs = redo_gaps(wm, pcm, phrases(segments), info.language, opts)
    segs = no_overlaps(sorted((p for s in segs for p in split(s)),
                              key=lambda w: w[0][0]))

    with open(os.path.join(outdir, "orig.srt"), "w", encoding="utf-8") as f:
        for n, s in enumerate(segs, 1):
            start, end, text = s[0][0], s[-1][1], words_text(s)
            f.write(f"{n}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n\n")
            print(f"  [{srt_time(start)[:8]}] {text}", file=sys.stderr,
                  flush=True)
    n = len(segs)
    with open(os.path.join(outdir, "orig.json"), "w") as f:
        json.dump({"language": info.language,
                   "language_probability": info.language_probability,
                   "duration": info.duration}, f)
    print(f"  {n} segments, language {info.language} "
          f"({info.language_probability:.2f})", file=sys.stderr)


if __name__ == "__main__":
    main()
