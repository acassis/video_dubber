# dubber — offline automatic video dubbing

```
                        ┌─ background.wav ───────────────────────────────┐
sound ─ffmpeg─> WAV ─RoFormer┤                                                 ├─ffmpeg─> video_<lang>.mp4
                        └─ vocals.wav ─faster-whisper─> Gemma 3 ─> Piper ─┘   (video copied,
                              (GPU)         (GPU)          (GPU)    (CPU)      + subtitles)
```

`dubber` is a C program that drives the pipeline. Once the models are cached,
it never accesses the network.

How it is built and why: [DESIGN.md](DESIGN.md).

| Stage       | Engine                                             | Device |
|-------------|----------------------------------------------------|--------|
| Voice/background split | BS-RoFormer (viperx ep317) via audio-separator | GPU, fp16 |
| ASR         | faster-whisper `turbo` (large-v3-turbo), Silero VAD | GPU, int8_float16 |
| Translation | Gemma 3 4B instruct (llama.cpp, Q4), context-aware; NLLB-200 as fallback | GPU |
| TTS         | Piper neural voices, N parallel workers            | CPU |
| Mux         | ffmpeg (video stream copied, no re-encode)         | CPU |

With several languages (`-l en,pt`), the GPU translates the next language
while the CPU is still voicing the current one.

## Setup (once, needs internet)

```sh
make
./fetch_models.sh turbo en pt fr es zh
```

This creates a Python venv in `~/.local/share/dubber/venv`, with CUDA PyTorch
if `nvidia-smi` is found, and about 8 GB in total. It also downloads the
Whisper, NLLB, RoFormer and Piper models.

## Use (offline)

```sh
./dubber -i talk.mp4 -l pt                     # -> talk_pt.mp4
./dubber -i talk.mp4 -l en,pt -s ur            # two languages, Urdu source
./dubber -i video_only.mp4 -a audio.m4a -l fr  # separate sound file
./dubber -i talk.mp4 -l en --bg 0.6            # quieter background
./dubber -i talk.mp4 -l en --no-separate       # old behaviour: voice only
./dubber -i talk.mp4 -l en -d cpu              # force CPU
```

## Translation

NLLB translates sentence by sentence and doesn't know the subject. Urdu and
Hindi speakers use many English technical loanwords written in Urdu or
Devanagari script, and NLLB mistranslates them ("fiber-lagger cutting
machine", "reducer" becoming "radiator"). The default translator is a local
LLM, Gemma 3 4B run through llama.cpp, and it works in three steps:

1. **Study:** it reads a sample of the whole transcript and writes down the
   subject of the video and a glossary of its technical terms.
2. **Translate:** it translates 6 lines at a time. Each call gets the
   subject, the glossary terms that occur in those lines, what was said just
   before, and a character limit per line derived from the time available.
3. **Shorten:** it rewrites any line that is still too long to be spoken in
   its time slot.

For targets other than English, it translates the source into English once
(cached as `pivot_en.txt` and shared by every language) and then English
into the target. A small model writes other languages from English far
better than it reads Urdu, and every dub then uses the same terms.

Each answer is checked: it must be in the target language's writing system
and must not be a transliteration or a copy of the source. A line that fails
is retried on its own, and NLLB translates any line that still fails.

`-p/--prompt "topic, terms…"` goes to both Whisper and the translator. Use
it to name the subject and fix terms the model gets wrong, e.g.
`-p "CNC fiber laser build; pauna inch = 3/4 inch; gauge = sheet thickness"`.
`-t nllb` selects the old sentence-level engine.

`--best` uses Gemma 3 12B for the source → English step
(`DUBBER_LLM_SIZE=12b ./fetch_models.sh`, 7.3 GB). The 12B knows the shop
terms the 4B gets wrong: "reducer gear" instead of "radius guide", "sixteen
gauge" instead of "sixteen inch", "1 mm" instead of ".001 inch". It doesn't
fit in 4 GB of VRAM, so 14 of its layers run on the GPU and the rest on the
CPU. The step takes about 10× longer: about 3 minutes for a 2-minute clip,
probably about an hour for the 55-minute video (not yet measured). The
English → target step stays on the fast 4B.

### LoRA: teaching the 4B what the 12B knows

`tools/train_lora.py` distils the 12B into the fast 4B for your kind of
videos. No human translation is needed:

```sh
# 1. dub with the 12B and record every request + checked answer
DUBBER_RECORD=teacher.jsonl ./dubber -i video.mp4 -l en,pt --best
# 2. train a LoRA adapter on the 4B (QLoRA, 4-bit, fits a 4 GB GPU)
~/.local/share/dubber/venv/bin/python tools/train_lora.py teacher.jsonl
```

The adapter (30 MB) is written to
`~/.local/share/dubber/lora/gemma3-4b-dubber.gguf`, and from then on
`translate.py` loads it automatically on top of the 4B (`DUBBER_LORA=none`
turns it off). Several recordings can be passed at once; more videos of the
same kind give a better adapter.

Training needs the 4-bit training copy of the 4B
(`unsloth/gemma-3-4b-it-bnb-4bit` in `~/.local/share/dubber/lora/base`,
3.1 GB), `peft`, `bitsandbytes` and llama.cpp's `convert_lora_to_gguf.py`.
It keeps the 1.3 GB embedding/output matrix on the CPU, so it uses 3.0 GB of
VRAM. One run on the 55-minute video's 93 examples took 15 minutes.

Measured on 11 held-out examples from the end of that video (chrF against
the 12B, 100 = identical):

| | chrF |
|---|---|
| 4B | 50.2 |
| 4B + LoRA | 59.0 |

The adapter brings back "reducer gear", "sixteen gauge" and "1 mm" at 4B
speed. On the 2-minute clip, the Urdu → English step took 22 s with the
4B + LoRA, against 169 s with the 12B. The clip is part of the training
video, though, so only the held-out score above shows how well it
generalizes. Neither model gets *pauna inch* (¾″) right; use `-p` for that.

## Timing: every phrase starts where the original did

1. Whisper word timestamps put each segment's start at its first spoken
   word, not at the beginning of the VAD chunk.
2. The translator knows how many seconds each line may take: the time until
   the next phrase starts, at about 17 characters per second.
3. Piper's leading silence is trimmed, so the first word lands on the
   original start.
4. If a phrase is still too long, Piper speaks it faster (down to length
   scale 0.80). Anything still over is time-compressed by rubberband, which
   keeps the pitch, so it ends before the next phrase starts.
5. Nothing is ever pushed later. Only a phrase that would need more than 2×
   speed is cut short with a fade, and the log reports it.

The work directory (`<input>.work/`) keeps every intermediate file: the
stems, the transcript, the translations and the SRT files. Re-running reuses
finished steps and redoes a step only when its inputs changed. Use `-f` to
redo everything.

## Measured on an RTX 3050 Ti (4 GB) + 16-core CPU

2-minute clip, Urdu → English + Portuguese: 2 min 14 s end to end, with a
VRAM peak of 3.4 GB. Whisper found no speech in the separated background.

## Notes

* A YouTube `videoplayback.mp4` is often a **video-only** DASH stream. In that case,
  download the matching audio stream too and pass it with `-a`.
* YouTube's "dubbed-auto" tracks can be mislabeled. Check what language the
  speaker really uses (`-s` auto-detects when omitted).
