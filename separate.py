#!/usr/bin/env python3
"""Offline voice / background separation for dubber.

usage: separate.py AUDIO OUTDIR

Uses a RoFormer source-separation network (python-audio-separator) to split
AUDIO into OUTDIR/vocals.wav (speech) and OUTDIR/background.wav (everything
else: machines, music, room sound). This is a learned separator, not a band
filter: it predicts which time-frequency energy belongs to the voice.

DUBBER_DEVICE=cpu forces CPU, DUBBER_SEP_MODEL picks another model file,
DUBBER_SEP_MODELS is the model directory (filled by fetch_models.sh).
"""
import os
import sys

DEFAULT_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    audio, outdir = sys.argv[1:]
    if os.environ.get("DUBBER_DEVICE") == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import logging
    from audio_separator.separator import Separator

    model = os.environ.get("DUBBER_SEP_MODEL", DEFAULT_MODEL)
    mdir = os.environ.get("DUBBER_SEP_MODELS", os.path.expanduser(
        "~/.local/share/dubber/separator"))
    if not os.path.exists(os.path.join(mdir, model)):
        sys.exit(f"separate.py: model {model} not in {mdir}; "
                 "run fetch_models.sh")

    # overlap 2 + fp16 autocast: ~3x faster than overlap 4 / fp32 on a
    # 4 GB RTX 3050 Ti with no audible difference; overlap 1 risks seams
    gpu = os.environ.get("DUBBER_DEVICE") != "cpu"
    sep = Separator(log_level=logging.WARNING, model_file_dir=mdir,
                    output_dir=outdir, output_format="WAV",
                    use_autocast=gpu,
                    mdxc_params={"segment_size": 256, "batch_size": 1,
                                 "overlap": 2, "override_model_segment_size": False,
                                 "pitch_shift": 0})
    sep.load_model(model_filename=model)
    sep.separate(audio, custom_output_names={"Vocals": "vocals",
                                             "Instrumental": "background"})
    for stem in ("vocals", "background"):
        if not os.path.exists(os.path.join(outdir, stem + ".wav")):
            sys.exit(f"separate.py: {stem}.wav was not produced")
    print("  wrote vocals.wav and background.wav", file=sys.stderr)


if __name__ == "__main__":
    main()
