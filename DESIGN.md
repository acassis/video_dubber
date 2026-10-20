# dubber — design and implementation notes

This document explains how `dubber` is built, why it is built that way, and
what was learned while testing it on real videos. For usage, see
[README.md](README.md).

## 1. Goal and constraints

`dubber` takes a video, recognizes the speech, translates it into one or
more languages, speaks the translation with a neural voice and writes a new
video with the dubbed sound track.

The fixed constraints that shaped every decision:

* **Fully offline.** After a one-time setup (`fetch_models.sh`), nothing
  touches the network. The Python helpers force `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1`, and all models are loaded from local files.
* **A C program drives everything.** The main program is C. The
  machine-learning engines that only exist as Python libraries run as small
  helper scripts that the C program starts as child processes.
* **Modest hardware.** It was developed on a laptop with an RTX 3050 Ti
  (4 GB VRAM, about 3.7 GB usable), a 16-core CPU and 62 GB RAM. Every GPU
  stage has to fit into 4 GB, one at a time. It also runs CPU-only.
* **Real-world input.** The test videos are long YouTube workshop videos
  (CNC machine building) in Urdu/Hindi. They have machine noise, music,
  English technical loanwords spoken in Urdu, and YouTube DASH files with
  separate video and audio streams.

## 2. Architecture

```
 input.mp4 (+ -a audio)
     │ ffmpeg: 44.1 kHz stereo WAV
     ▼
 separate.py ── BS-RoFormer ──► vocals.wav + background.wav        GPU
     │
     ▼
 asr.py ── faster-whisper turbo, ≤15 s VAD clips, word timestamps ──► orig.srt  GPU
     │
     ▼  src.txt (one line per segment) + slots.txt (seconds per line)
 translate.py ── Gemma 3 (llama.cpp) [+ LoRA] / NLLB-200 ──► <lang>.txt   GPU
     │
     ▼
 dubber.c ── Piper (N parallel processes) ──► one WAV per segment     CPU
     │       fit to slot: faster Piper, then rubberband
     ▼       place each clip at its original start time
 dub_<lang>.wav
     │
     ▼
 ffmpeg ── video copied + background + voice (amix, limiter) + subtitles
     ▼
 input_<lang>.mp4
```

### Division of labour between C and Python

| Part | Language | Why |
|---|---|---|
| Option parsing, process control, caching, scheduling | C (`dubber.c`) | the requested main program; no dependencies besides libc/libm |
| SRT parsing and writing, WAV reading and writing, silence detection, timeline mixing, fades | C | simple, fast, and exact sample-level control over timing |
| Voice/background separation | Python (`separate.py`) | `audio-separator` / PyTorch |
| Speech recognition | Python (`asr.py`) | `faster-whisper` / CTranslate2 |
| Translation | Python (`translate.py`) | `llama-cpp-python`, `transformers` |
| LoRA training | Python (`tools/train_lora.py`) | PEFT / bitsandbytes; not part of a dubbing run |
| TTS | `piper` binary | started directly from C, one process per segment |
| Decode, time-stretch, mux | `ffmpeg` binary | started directly from C |

The helpers communicate with the C program through files in the work
directory (`<input>.work/`), not through pipes. Every intermediate result can
therefore be inspected, and an interrupted run can resume.

### Python environment

The system Python on the development machine had a CPU-only PyTorch and an
"externally managed" pip. `fetch_models.sh` therefore creates its own venv in
`~/.local/share/dubber/venv`:

* CUDA PyTorch (cu128 index) when `nvidia-smi` exists, otherwise CPU wheels.
  torchvision is reinstalled from the same index, because a mismatched PyPI
  build breaks `import transformers`.
* `llama-cpp-python` as a prebuilt CUDA wheel (cu124 index), so no CUDA
  toolkit or compiler is needed.

`dubber` uses that venv's interpreter automatically, or `$DUBBER_PYTHON` if
set. `asr.py` re-executes itself once with `LD_LIBRARY_PATH` extended to the
pip `nvidia-*` wheels, because CTranslate2 needs cuBLAS/cuDNN on the loader
path.

### Device selection and overlap

`pick_device()` treats the GPU as usable only if `nvidia-smi` reports one
**and** the venv's PyTorch can use CUDA. A GPU that PyTorch cannot reach is
reported and the run falls back to the CPU. `DUBBER_DEVICE` passes the
decision to every helper.

GPU stages run one after another, because two models do not fit in 4 GB
together. Piper runs on the CPU in `jobs` parallel processes (default: cores
/ 2). With several target languages, `translate.py` for language *k+1* is
started in the background **before** the TTS for language *k*, so the GPU
translates while the CPU speaks. The TTS loop uses `wait()`, so it can reap
the background translator. `stash_reaped()` keeps that exit status for
`translate_finish()`. An early version lost it and reported a successful
translation as failed.

### Caching and resumability

Each step writes a small `.key` file holding all of its inputs and reuses its
output when the key is unchanged:

| Step | Key file | Contents |
|---|---|---|
| audio extraction | `source.key` | source path |
| separation | `separate.key` | input WAV |
| ASR | `asr.key` | `words` marker, model, language, audio, prompt |
| translation | `slots.txt`, `translate.key` | per-line slot lengths; translator, prompt, `--best` |
| English pivot | first line of `pivot_en.txt` | sha256 of lines, budgets, source language, model, prompt, prompt version, LoRA path and mtime |

If the transcript or any translation setting changes, every translation is
redone (`retranslate`). `-f` ignores all caches. In practice this made
experiments cheap. For example, a re-run with a new `-p` reused the
26-minute separation of the long video.

## 3. Stage by stage

### 3.1 Audio input

ffmpeg converts the sound to 44.1 kHz stereo 16-bit PCM. YouTube downloads
are often video-only DASH streams, so `-a FILE` takes the sound from a
separate file (`.m4a` or `.webm`/Opus both work). Only the first video stream
of the input is used, and it is copied without re-encoding.

### 3.2 Voice/background separation

**Requirement:** remove the voice but keep machine sounds and music, using a
model that recognizes voice energy, not a band-stop filter.

**Choice:** BS-RoFormer (`model_bs_roformer_ep_317_sdr_12.9755.ckpt`, the
"viperx" checkpoint) through `python-audio-separator`. The RoFormer family
leads the public source-separation benchmarks (MVSEP, Mel-RoFormer paper), and
this checkpoint had the best instrumental score in the audio-separator
registry. Those benchmarks are measured on music, so it was verified on
speech here:

* Whisper finds **no speech at all** in the separated `background.wav`.
* Whisper transcribes `vocals.wav` more accurately than the original mix.

**Tuning:** `overlap=2` with fp16 autocast is about 3× faster than the
default `overlap=4` in fp32, with no audible difference (`overlap=1` risks
audible seams). Separation is still the slowest stage, at about 26 minutes
for a 55-minute video on the 3050 Ti.

**Mix:** the final mix puts the background at its original level (`--bg`
changes it) under the new voice, with `alimiter` at 0.95 against clipping.
The measured loudness of the output matched the original track (about
−17.5 dB). `--no-separate` restores the old behaviour: the voice is
recognized from the full mix and the original track is muted, or mixed in at
`--bg`.

### 3.3 Speech recognition

**Choice:** Whisper `large-v3-turbo` through **faster-whisper** (CTranslate2,
`int8_float16` on the GPU, `int8` on the CPU).

* The first version used the reference `openai-whisper` package. On the
  55-minute video recognition took about **1 h 40 min**. faster-whisper with
  the same model takes about **4 minutes**.
* `small` misheard Hindi badly ("fazer um não depois da máquina"), so
  `turbo` is the default.
* **Silero VAD** (500 ms minimum silence) skips music and noise. Without
  it, Whisper wrote garbage over YouTube's background music: Korean words,
  and "बरे बरे बरे…" repeated.
* **Short clips, not 30 s windows.** faster-whisper's own `vad_filter`
  packs speech into 30 s windows, and Whisper decodes at most 448 tokens
  per window. Urdu in Arabic script costs several tokens per word, so a
  window of fast, continuous speech overflows. Its text is cut off
  mid-word, its word times are spread evenly over the window, and the rest
  of the window is skipped. In the servo-motor video the first 29 s became
  one phrase and the speech from 0:29 to 0:38 was lost, so the dub was
  silent from 0:20 to 0:38 while the speaker talked. The pass was also
  nondeterministic: different runs lost 20 to 160 s of speech. `asr.py`
  now cuts the VAD speech at pauses into clips of at most 15 s
  (`speech_clips()`) and passes them as `clip_timestamps`.
* **Safety net:** `redo_gaps()` treats a phrase longer than 20 s as an
  overflowed window. That phrase, and any VAD speech of 1 s or more that no
  phrase covers, is transcribed again in clips of at most 12 s. Each clip
  starts 3 s early, because Whisper drops the first words of a clip cut
  mid-word; words from that lead-in are discarded.
* **Phrase length:** phrases over 10 s are split at their longest pause,
  with a bonus for punctuation and a preference for the middle (pieces of
  at least 3 s). Each piece gets its own start time in the dub.
  `no_overlaps()` drops a phrase lying mostly inside an earlier one, such
  as a clip boundary decoded twice or a "شکریہ" ("thank you")
  hallucination.
* `condition_on_previous_text=False` prevents one hallucination from
  propagating.
* **Word timestamps** (cross-attention alignment) are enabled. A segment's
  start is its **first word's** start, not the start of the VAD chunk. The
  segment start alone can be off by about 0.5 s, and this is the anchor for
  all timing (§4).
* Audio is decoded with soundfile and scipy rather than faster-whisper's PyAV
  decoder, which is sensitive to version changes.
* `-p/--prompt` is passed as Whisper's `initial_prompt`.

The source language is auto-detected unless `-s` is given. YouTube's
"dubbed-auto" track labels turned out to be wrong: a track labelled Hindi
was Urdu.

### 3.4 Translation

This stage changed the most during development.

#### First version: NLLB-200

`facebook/nllb-200-distilled-600M` translated sentence by sentence with beam
search 4, fp16 on the GPU, batches grouped by length, and splitting of a
batch on CUDA out-of-memory (single long lines go to the CPU). It is fast
and robust, but it has **no context**. Urdu workshop speech is full of
English loanwords written in Urdu script, and NLLB mangled them:
"fiber-lagger cutting machine", "reducer" → "radiator", "tapeei" for
"threaded", "bol" for "bolt". Telling Whisper the topic (`--prompt`) did not
help, because the words were *heard* correctly and *translated* wrongly.
NLLB remains the fallback (`-t nllb`) and fills individual lines the LLM
fails on.

#### Current version: a local LLM with context

The current translator is **Gemma 3 4B instruct** (Q4_K_M GGUF, bartowski),
run with `llama-cpp-python` on the GPU. Settings that fit 4 GB: `n_ctx=4096`,
`n_batch=n_ubatch=256`, flash attention, temperature 0. `n_ctx=8192` ran out
of memory while creating the context.

It translates in three steps:

1. **Study.** An evenly spaced sample of the transcript, up to 1800 tokens,
   is sent with a request for (a) the subject in one or two sentences and
   (b) up to 30 `TERM: source = target` glossary lines. The prompt asks the
   model to map loanwords back to the real technical term. Junk entries such
   as "(No direct equivalent…)" and entries of 40 or more characters are
   dropped.
2. **Windowed translation.** Lines are translated **6 at a time**
   (`WINDOW`). Each request contains:
   * the subject, plus the user's `-p` text
   * only the glossary terms that occur in this window
   * the **translations** (not the sources) of the 3 previous lines
     (`HISTORY`)
   * a character limit per line
   * the numbered source lines

   The prompt insists that each line is translated into its own line, never
   merged or moved, because each translation will be spoken during its own
   line's time.
3. **Shorten.** Lines longer than 1.15 × their budget (`SLACK`) are sent
   back with the source, the current wording and the limit, to be rewritten
   shorter. A rewrite is accepted only if it is actually shorter.

**Character budget (isochrony).** `dubber.c` writes `slots.txt`, which holds
for each line the seconds until the next segment starts. The budget is
`max(8, slot × DUBBER_CPS)`, where `DUBBER_CPS = 17` characters per second
roughly matches Piper's speaking rate. Writing to fit is better than speeding
up afterwards, which §4 still does as a fallback.

**Free text, not JSON.** A JSON grammar would make parsing trivial, but
llama.cpp's grammar sampling runs on the CPU over Gemma's 262k-token
vocabulary. It took 165 s for 40 lines, against 25 s for free text. Answers
are therefore numbered lines (`1. …`). The parser (`LLM.numbered`) groups the
lines of each numbered block and takes the **first line of the block that
passes the plausibility check**. This handles answers where the model
repeats the source after the number and puts the translation on the next
line.

**Plausibility checks** (each found by watching the 4B fail):

| Failure seen | Check |
|---|---|
| Urdu copied back into the "translation" (Piper then read Urdu with an English voice, giving 40-second clips) | at least 80% of letters must be in the target script (`SCRIPTS`, by Unicode character name) |
| Romanized Urdu instead of English ("yeh machine hai…") | count Urdu/Hindi function words (`ROMAN`): 2 or more, or ≥ 20% of the words, rejects the line |
| "source => translation" echoed from the context | reject lines containing `=>`; history shows translations only |
| budget echoed: "… (85)", "[max 23 chars]" | `LIMIT_ECHO` strips it; limits are given on a separate "Character limits:" line |
| lines merged or shifted across numbers | stricter fragment wording, window reduced to 6 |
| a stray word in another script ("InshaAllah انشاءاللہ") | `strip_foreign()` removes words not in the target script |
| loops from Whisper hallucinations ("no, no, no, no…") | `clean()` collapses phrases of 1 to 6 words repeated 3 or more times, before and after translation |

A line that fails is retried **alone and without history**. A line that
still fails is translated by NLLB, after the LLM has been freed, because both
do not fit in 4 GB together.

**English pivot.** For a non-English target, the source is first translated
to English (`pivot_en.txt`, cached and shared by all target languages), then
English to the target. Reasons:

* A small model writes Portuguese/French/… from English much better than it
  reads Urdu. Direct Urdu → Portuguese had many more failed lines.
* The hard step (understanding Urdu) is done once, so every dub uses the
  same terms.
* With `--best`, only the hard step uses the big model.

**CUDA library clash.** The llama.cpp wheel bundles an older `libcudart`. If
it is loaded first, importing torch for the NLLB fallback fails
(`undefined symbol cudaGetDriverEntryPointByVersion`). `LLM.__init__` imports
torch first.

#### `--best`: Gemma 3 12B for the source → English step

The 12B knows the workshop terms the 4B gets wrong:

| 4B | 12B |
|---|---|
| radius guide | reducer gear |
| sixteen inch | sixteen gauge |
| .001 inch | one millimeter |
| refining | manufacturing |

It does not fit in 4 GB. `LLM.__init__` tries GPU layer counts
`-1, 40, 32, 26, 20, 14, 8, 0` and keeps the first that loads (14 layers for
the 12B), so the rest runs on the CPU. It is about 10× slower: 169 s against
about 20 s on a 2-minute clip, and about 43 minutes for the source → English
step of the 55-minute video. English → target stays on the 4B.

#### `-p/--prompt`

`-p` goes to Whisper (initial prompt), to the study step (as the uploader's
description) and into every translation prompt's subject. It is the
manual fix for terms both models miss. In the second test video, "سینسی"
(CNC spoken in Urdu) became "Sense" 11 times. With
`-p "…; سینسی = CNC"` every occurrence came out as CNC.

### 3.5 LoRA distillation (12B → 4B)

**Question asked:** "use LoRA to improve the translation and the alignment".

**Decision on alignment:** LoRA cannot fix timing. Training changes what the
model writes, not where the audio is placed, and the misalignment had
deterministic causes in the pipeline. Timing was fixed in the pipeline
instead (§4).

**Decision on translation:** the user does not read Urdu, so they cannot
correct translations. Instead, the 12B acts as the **teacher**: its checked
answers become training data for the 4B (knowledge distillation with QLoRA).
The goal is close to 12B terminology at 4B speed.

Workflow:

1. `DUBBER_RECORD=teacher.jsonl dubber … --best` appends every study and
   translation request together with its **checked** answer (only windows
   whose lines all passed) as `{model, prompt, answer}` lines.
2. `tools/train_lora.py teacher.jsonl …` keeps the records whose model name
   contains `12b`, trains on the first 90% and scores on the last 10% (the
   end of the video, so the test set is not shuffled into training).

Fitting QLoRA training of a 4B model with a 262k vocabulary into 4 GB:

* Base: `unsloth/gemma-3-4b-it-bnb-4bit` (4-bit NF4 weights, about 3.1 GB on
  disk).
* The **vision tower** and projector are dropped: text only.
* The tied **embedding/output matrix** (262k × 2560, about 1.3 GB) is moved
  to the **CPU**. Embeddings are looked up on the CPU and moved to the GPU.
  The final hidden states of the **answer tokens only** are copied to the
  CPU, the logits and the cross-entropy are computed there in fp32, and the
  gradient is sent back with `hs.backward(hc.grad)`. Full-sequence logits
  on the GPU would need gigabytes.
* Gradient checkpointing; one sequence per step, gradient accumulation of 4.
* LoRA rank 8, alpha 16, dropout 0.05, on all attention and MLP projections
  of the language model.
* AdamW with `foreach=False` (no batched temporaries) and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The first attempt ran
  out of memory at the first optimizer step. Peak usage is now 3.0 GB.
* LR 2e-4, 5-step warm-up, then linear decay, 3 epochs.

**Conversion:** llama.cpp's `convert_lora_to_gguf.py` reads only the base
model's config and refuses a bitsandbytes `quantization_config`. `convert()`
creates `lora/base_cfg/` with a copy of the config without it, plus symlinks
to the tokenizer files. The output is
`~/.local/share/dubber/lora/gemma3-4b-dubber.gguf` (30 MB, f16).

**Use:** `translate.py` loads that adapter automatically, and only on top of
the default 4B GGUF (`lora_for()`). `DUBBER_LORA=none` disables it. The
adapter path and mtime are part of the pivot cache key, so retraining
invalidates old pivots.

**Result:** 104 teacher records from the 55-minute video, 93 for training
and 11 held out. Training took 15 minutes, and the loss went from 2.27 to
0.35. chrF against the 12B on the held-out examples:

| | chrF |
|---|---|
| 4B | 50.2 |
| 4B + LoRA | 59.0 |

On the 2-minute clip, 4B + LoRA produces "reducer gear", "sixteen gauge" and
"one millimeter" in 22 s, where the 12B needs 169 s. That clip is part of the
training video, so the held-out score is the fair measure. On the second,
unseen video, the adapter did **not** know "CNC" in Urdu script. A training
set from a single video is small, and more `--best` recordings make a better
adapter. Neither model knows *pauna inch* (¾″); `-p` covers that.

### 3.6 Text-to-speech

**Choice:** Piper (VITS voices exported to ONNX), one medium-quality voice
per language (`voices[]` in `dubber.c`, downloaded by `fetch_models.sh`).
Reasons:

* fast enough on the CPU to run many processes in parallel while the GPU is
  busy translating, and no GPU memory needed
* a `--length-scale` option for speaking faster without a pitch change
* fully local voices for all the target languages
* a standalone binary that C can start directly

`speakable()` sends only lines with at least one letter or digit to Piper.
Piper once crashed on a translated line that was only "- - - -". A Piper
failure on one segment now leaves a gap and a warning instead of aborting
the run. `--sentence-silence 0` avoids pauses inside a clip.

### 3.7 Mux

ffmpeg copies the video stream without re-encoding. The background and the
voice are mixed with
`amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95`. The audio is
AAC at 192 kbit/s. The translated SRT is embedded as a `mov_text` subtitle
track (`--no-subs` disables this), and the file gets `+faststart`.

## 4. Timing: every phrase starts where the original did

**Requirement:** each dubbed phrase must start at the same moment as the
original phrase.

Originally, a clip that was longer than its slot pushed the following
clips later, and about 1 in 7 phrases started late. The causes and fixes:

| Cause | Fix |
|---|---|
| segment start came from the VAD chunk, up to about 0.5 s early or late | Whisper word timestamps: start = first word |
| translation too long for the slot | per-line character budget plus a "shorten" pass (§3.4) |
| Piper adds silence before the speech | `speech_bounds()` trims everything below `SILENCE = 300` (about −41 dBFS), keeping 5 ms |
| speech still longer than the slot | pass 2: re-synthesize with `length_scale = slot/duration`, down to `MIN_LENGTH_SCALE = 0.80` (1.25×) |
| still too long | pass 3: `ffmpeg rubberband=tempo=…:window=short` (keeps the pitch), up to `MAX_TEMPO = 1.6` |
| still too long (would need more than 2×) | cut at the next phrase's start with an 8 ms fade, and report it |

The track is built in memory. Each clip is added at
`round(start × rate)`, never later, with 8 ms fades at both ends and
clipping protection. The slot is `next.start − start`, so clips never
overlap. A clip is never placed later than its start. The log line `all N
segments start on time: a sped up by Piper, b also time-compressed, c cut
short` makes the result checkable.

**Measured:** on the 2-minute clip, onset error against the original first
word was 6 ms on average and 12 ms at most. On the 55-minute video, all 552
segments start on time and 1 per language was cut short. On the 10-minute
second video, all 141 start on time and 2 per language were cut short.

## 5. Robustness lessons from real videos

* **Whisper hallucinations** on music/noise were caused by YouTube's AI-dub
  background music. VAD, `condition_on_previous_text=False` and `clean()`
  fixed them. The first full English dub had "scattered" 60 times and long
  runs of "No, no, no…".
* **NLLB out-of-memory** on a batch of long sentences: fp16, length-sorted
  batches, splitting on OOM, CPU fallback. Peak usage is 2.4 GB.
* **Lost child exit status** in the TTS `wait()` loop: `stash_reaped()`.
* **Untranslated Urdu passed to Piper** (19 of 26 clips "cut short" in one
  test): the script-share check.
* **Disk space:** the pip cache had grown to 14 GB. Models and venv need
  about 8 GB, plus 7.3 GB for the 12B and 3.1 GB for the training base.
* **YouTube downloads:** the AAC audio stream (format 140) can return HTTP
  403. The Opus stream (format 251) works with `-a`.
* **Whisper window overflow** on dense Urdu: up to 30 s of speech was
  silently lost and the dub went quiet while the speaker talked. Fixed by
  short VAD clips plus re-transcription of gaps (§3.3).
* **Running next to a desktop:** `tools/run_limited.sh` runs a job in its
  own systemd user scope (§6.1).

## 6. Performance (RTX 3050 Ti 4 GB, 16-core CPU)

| Job | Time |
|---|---|
| 2-minute clip, Urdu → en + pt, first GPU version | 2 min 14 s, VRAM peak 3.4 GB |
| 55-minute video, English, first version (openai-whisper, NLLB) | about 1 h 55 min (ASR 1 h 40 min) |
| 55-minute video, en + pt, GPU + separation + NLLB | about 40 min (separation 26 min, ASR 4 min) |
| 55-minute video, en + pt, Gemma 4B | 27.5 min (separation reused) |
| 55-minute video, en + pt, `--best` (12B pivot) | 56 min (pivot about 43 min) |
| LoRA training, 93 examples, 3 epochs | 15 min, 3.0 GB VRAM |
| 10-minute video, en + pt, 4B + LoRA | 13 min first run; 8:52 re-run with `-p` (separation reused) |

### 6.1 Memory: running next to the browser

`--best` maps the 7.3 GB 12B GGUF into memory. Most of it is file-backed
pages, which the kernel can drop and read again, but the job still
competes with the desktop. Ubuntu's `systemd-oomd` kills the cgroup under
`user@1000.service` with the most reclaim activity once memory pressure
stays above 50% for 20 s. `tools/run_limited.sh CMD…` needs no root:

* It waits until 12 GB is available before starting.
* It runs the job in its own scope with `MemoryHigh=18G` (above that the
  job reclaims only its own pages), `MemoryMax=24G` (a runaway job is
  stopped inside its scope, never the browser) and `MemorySwapMax=0`, at
  nice 10.
* A guard freezes the scope (cgroup freezer, nothing is lost) if the
  system drops below 2 GB available, and thaws it at 5 GB.
* It prints the scope's peak memory at the end.

A real reservation (`MemoryLow`) is ignored unless every parent cgroup has
one, and that needs root:
`sudo systemctl set-property user-1000.slice MemoryLow=16G`.

## 7. Known limitations

* Translation quality depends on the 4B. Domain terms it has never seen
  need `-p` or more teacher data.
* Whisper sometimes splits phrases at odd points, which gives fragment
  lines ("this") whose translation sounds clipped.
* Culture-specific units are kept as spoken ("twenty-five lakh").
* One voice per language: speaker identity, gender and emotion are not
  transferred.
* When the translation is much longer than the original, speech gets fast
  (up to 2×). About a quarter of the English lines in the second video
  needed some speed-up.
* No lip-sync.

## 8. Files

| File | Role |
|---|---|
| `dubber.c` | main program: options, device choice, caching, scheduling, TTS fitting, timeline, mux |
| `separate.py` | voice/background separation (BS-RoFormer) |
| `asr.py` | faster-whisper transcription in short VAD clips with word timestamps, gap re-transcription, phrase splitting → `orig.srt`, `orig.json` |
| `translate.py` | LLM translator (study, windows, shorten, checks, pivot, LoRA) and NLLB fallback |
| `tools/train_lora.py` | QLoRA distillation of the 12B into the 4B, GGUF conversion, chrF evaluation |
| `tools/run_limited.sh` | run a job in a memory-limited systemd user scope, paused when RAM runs low |
| `fetch_models.sh` | one-time setup: venv, llama.cpp, Whisper, NLLB, Gemma, RoFormer, Piper voices |
| `Makefile` | `make` builds `dubber` (gcc, C11, `-lm`) |

Work directory (`<input>.work/`): `source.wav`, `vocals.wav`,
`background.wav`, `orig.srt`, `orig.json`, `src.txt`, `slots.txt`,
`pivot_en.txt`, `<lang>.txt`, `<lang>.srt`, `tts_<lang>/`, `dub_<lang>.wav`
and the `.key` files.

Model data (`~/.local/share/dubber/`): `venv/`, `voices/`, `separator/`,
`llm/` (Gemma GGUFs), `lora/` (adapter, training base, converter).

## 9. Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `DUBBER_PYTHON` | dubber | interpreter for the helpers |
| `DUBBER_DEVICE` | all helpers | `cuda` / `cpu`, set by dubber |
| `DUBBER_TRANSLATOR` | translate.py | `auto`, `llm`, `nllb` (set from `-t`) |
| `DUBBER_PROMPT` | translate.py | user context (set from `-p`) |
| `DUBBER_LLM` | translate.py | GGUF of the main LLM |
| `DUBBER_LLM_PIVOT` | translate.py | GGUF for source → English (set by `--best`) |
| `DUBBER_LLM_GPU_LAYERS` | translate.py | force the number of GPU layers |
| `DUBBER_LORA` | translate.py | adapter path, or `none` |
| `DUBBER_CPS` | translate.py | characters per second for budgets (17) |
| `DUBBER_PIVOT` | translate.py | `0` disables the English pivot |
| `DUBBER_RECORD` | translate.py | append teacher data to this `.jsonl` |
| `DUBBER_NLLB_MODEL` | translate.py | NLLB model name or directory |
| `DUBBER_SEP_MODEL`, `DUBBER_SEP_MODELS` | separate.py | separation model file and directory |
| `DUBBER_LORA_EPOCHS`, `DUBBER_LORA_RANK`, `DUBBER_LORA_SKIP_TRAIN` | train_lora.py | training settings |
| `DUBBER_LLM_SIZE` | fetch_models.sh | `4b` (default) or `12b` |
| `DUBBER_VENV`, `DUBBER_VOICES` | fetch_models.sh | install locations |

## 10. Development history

1. **First version:** C driver, openai-whisper, NLLB-200 and Piper. Tested
   on synthetic Hindi, then on a 55-minute YouTube video (about 2 h for the
   English dub). Hallucination loops and a Piper crash were fixed.
2. **GPU, separation and prompt:** CUDA detection, faster-whisper,
   BS-RoFormer separation with the background mixed back, `--prompt`, GPU/CPU
   overlap, NLLB OOM handling.
3. **Translation and timing:** Gemma 3 4B with subject/glossary, windows,
   budgets and checks; English pivot; word-level timing, silence trimming,
   Piper speed-up plus rubberband; `--best` with the 12B.
4. **LoRA:** teacher recording, QLoRA distillation that fits in 4 GB, GGUF
   adapter loaded automatically. chrF 50.2 → 59.0.
5. **Second video** (CNC plasma → laser, 10 min): dubbed with 4B + LoRA;
   "CNC" fixed with `-p`.
6. **Third video** (servo motors, 6:43): the dub was silent from 0:20 to
   0:38 because Whisper's window overflowed. ASR now uses short VAD clips,
   re-transcribes gaps and splits long phrases; `tools/run_limited.sh` was
   added for memory-safe `--best` runs.
