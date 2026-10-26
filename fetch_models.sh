#!/bin/sh
# One-time setup (needs internet): python venv with CUDA support plus every
# model dubber needs, so that dubber itself can run offline.
#
# usage: ./fetch_models.sh [whisper-model] [lang ...]
#   ./fetch_models.sh turbo en pt fr zh es
#
# After this, dubber never touches the network.
set -e

WMODEL=${1:-turbo}
[ $# -gt 0 ] && shift
LANGS=${*:-"en pt fr es de it zh ru hi"}
DATA=$HOME/.local/share/dubber
VENV=${DUBBER_VENV:-$DATA/venv}
VOICEDIR=${DUBBER_VOICES:-$DATA/voices}
SEPDIR=${DUBBER_SEP_MODELS:-$DATA/separator}
SEPMODEL=${DUBBER_SEP_MODEL:-model_bs_roformer_ep_317_sdr_12.9755.ckpt}
LLMDIR=$DATA/llm
# context-aware translator; DUBBER_LLM_SIZE=12b fetches the bigger, better
# (and ~3x slower on a 4 GB GPU) Gemma 3 12B instead
LLMSIZE=${DUBBER_LLM_SIZE:-4b}
LLMFILE=google_gemma-3-$LLMSIZE-it-Q4_K_M.gguf
LLMURL=https://huggingface.co/bartowski/google_gemma-3-$LLMSIZE-it-GGUF/resolve/main/$LLMFILE
HF=https://huggingface.co/rhasspy/piper-voices/resolve/main

voice_for() {
    case "$1" in
    en) echo en/en_US/lessac/medium/en_US-lessac-medium ;;
    pt) echo pt/pt_BR/faber/medium/pt_BR-faber-medium ;;
    fr) echo fr/fr_FR/siwis/medium/fr_FR-siwis-medium ;;
    es) echo es/es_ES/davefx/medium/es_ES-davefx-medium ;;
    de) echo de/de_DE/thorsten/medium/de_DE-thorsten-medium ;;
    it) echo it/it_IT/paola/medium/it_IT-paola-medium ;;
    zh) echo zh/zh_CN/huayan/medium/zh_CN-huayan-medium ;;
    ru) echo ru/ru_RU/irina/medium/ru_RU-irina-medium ;;
    hi) echo hi/hi_IN/pratham/medium/hi_IN-pratham-medium ;;
    ar) echo ar/ar_JO/kareem/medium/ar_JO-kareem-medium ;;
    nl) echo nl/nl_NL/mls/medium/nl_NL-mls-medium ;;
    pl) echo pl/pl_PL/gosia/medium/pl_PL-gosia-medium ;;
    tr) echo tr/tr_TR/dfki/medium/tr_TR-dfki-medium ;;
    uk) echo uk/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium ;;
    vi) echo vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium ;;
    *) echo "" ;;
    esac
}

if [ ! -x "$VENV/bin/python" ]; then
    echo "== creating python venv $VENV"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q -U pip
    # torchvision must come from the same index as torch, otherwise a
    # mismatched PyPI build breaks "import transformers" models
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "== NVIDIA GPU found: installing CUDA PyTorch"
        TORCH_INDEX=https://download.pytorch.org/whl/cu128
    else
        TORCH_INDEX=https://download.pytorch.org/whl/cpu
    fi
    "$VENV/bin/pip" install -q --resume-retries 20 torch torchaudio \
        torchvision --index-url $TORCH_INDEX
    "$VENV/bin/pip" install -q --resume-retries 20 faster-whisper \
        "audio-separator[gpu]" audioread transformers sentencepiece
    "$VENV/bin/pip" install -q --force-reinstall --no-deps torchvision \
        --index-url $TORCH_INDEX
fi
PY="$VENV/bin/python"

if ! "$PY" -c "import llama_cpp" 2>/dev/null; then
    echo "== llama.cpp (local LLM translator)"
    if command -v nvidia-smi >/dev/null 2>&1; then
        "$VENV/bin/pip" install -q --resume-retries 20 --only-binary=:all: \
            llama-cpp-python \
            --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
    else
        "$VENV/bin/pip" install -q --resume-retries 20 llama-cpp-python
    fi
fi

echo "== faster-whisper model: $WMODEL"
"$PY" -c "
from faster_whisper.utils import download_model
download_model('$WMODEL')
"

echo "== NLLB-200 translation model"
"$PY" -c "
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
n = 'facebook/nllb-200-distilled-600M'
AutoTokenizer.from_pretrained(n); AutoModelForSeq2SeqLM.from_pretrained(n)
"

echo "== translation LLM: $LLMFILE"
mkdir -p "$LLMDIR"
[ -s "$LLMDIR/$LLMFILE" ] || curl -L --retry 5 -C - -o "$LLMDIR/$LLMFILE" "$LLMURL"

echo "== voice/background separation model: $SEPMODEL"
mkdir -p "$SEPDIR"
"$PY" -c "
import logging
from audio_separator.separator import Separator
s = Separator(log_level=logging.WARNING, model_file_dir='$SEPDIR')
s.load_model(model_filename='$SEPMODEL')
"

mkdir -p "$VOICEDIR"
for l in $LANGS; do
    v=$(voice_for "$l")
    if [ -z "$v" ]; then
        echo "== no default Piper voice for '$l' (use dubber --voice)"
        continue
    fi
    f=$(basename "$v")
    echo "== Piper voice for $l: $f"
    [ -s "$VOICEDIR/$f.onnx" ] || curl -fL -o "$VOICEDIR/$f.onnx" "$HF/$v.onnx"
    [ -s "$VOICEDIR/$f.onnx.json" ] || curl -fL -o "$VOICEDIR/$f.onnx.json" "$HF/$v.onnx.json"
done
# --lipsync: mouth tracking, speakers, sex, meaning check, Kokoro voices
echo "== lip-sync models (MediaPipe, ECAPA, age/gender, LaBSE, Kokoro)"
"$VENV/bin/pip" install -q --resume-retries 20 mediapipe praat-parselmouth \
    speechbrain kokoro
mkdir -p "$DATA/vision"
F=$DATA/vision/face_landmarker.task
[ -s "$F" ] || curl -fL -o "$F" https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
"$PY" -c "
from huggingface_hub import snapshot_download as get
get('speechbrain/spkrec-ecapa-voxceleb')
get('audeering/wav2vec2-large-robust-6-ft-age-gender')   # CC BY-NC-SA 4.0
get('MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7',
    allow_patterns=['*.json', '*.model', 'model.safetensors'])
get('sentence-transformers/LaBSE', allow_patterns=['*.json', '*.txt',
    'model.safetensors', '1_Pooling/*', '2_Dense/*'])
v = ['pf_dora', 'pm_alex', 'pm_santa', 'af_heart', 'af_bella', 'am_michael',
     'am_fenrir', 'ef_dora', 'em_alex', 'em_santa', 'ff_siwis', 'if_sara',
     'im_nicola', 'hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi']
get('hexgrad/Kokoro-82M', allow_patterns=['config.json', 'kokoro-v1_0.pth']
    + [f'voices/{x}.pt' for x in v])
"
echo "== done; models are cached locally"
