#!/usr/bin/env python3
"""Offline line-by-line translator used by dubber.

usage: translate.py SRC TGT INFILE OUTFILE [SLOTS]

Reads one segment per line from INFILE and writes exactly the same number
of translated lines to OUTFILE. SLOTS (optional) holds, per line, the
seconds available until the next segment starts; the translation of each
line is kept short enough to be spoken in that time (isochrony).

Two engines, both local:

  llm   a local instruction LLM (GGUF via llama.cpp, default Gemma 3 4B).
        It first reads a sample of the whole transcript to find the
        subject and a glossary of technical terms, then translates in
        windows of neighbouring lines with that context, and finally
        rewrites lines that are too long for their time slot.
  nllb  Meta NLLB-200 (sentence by sentence, no context). Fallback.

Environment:
  DUBBER_TRANSLATOR  auto (llm if its model is present), llm or nllb
  DUBBER_LLM         GGUF file (default ~/.local/share/dubber/llm/...)
  DUBBER_PROMPT      extra context from the user: topic, terms
  DUBBER_CPS         speech rate of the TTS voice, characters/s (17)
  DUBBER_LORA        adapter for the 4B model (default: the one made by
                     tools/train_lora.py if present; "none" disables)
  DUBBER_LLM_PIVOT   GGUF for the SRC -> English step only (dubber --best:
                     Gemma 3 12B; English -> TGT stays on the fast model)
  DUBBER_RECORD      append every request + checked answer to this .jsonl
                     (training data for tools/train_lora.py)
  DUBBER_PIVOT       1 (default): with the LLM, translate SRC -> English
                     once (cached as pivot_en.txt) and English -> TGT
  DUBBER_NLLB_MODEL  NLLB model name or directory
"""
import os
import re
import sys
import json
import time
import unicodedata

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

NLLB_CODES = {
    "ar": "arb_Arab", "bn": "ben_Beng", "de": "deu_Latn", "en": "eng_Latn",
    "es": "spa_Latn", "fa": "pes_Arab", "fr": "fra_Latn", "gu": "guj_Gujr",
    "hi": "hin_Deva", "id": "ind_Latn", "it": "ita_Latn", "ja": "jpn_Jpan",
    "kn": "kan_Knda", "ko": "kor_Hang", "ml": "mal_Mlym", "mr": "mar_Deva",
    "nl": "nld_Latn", "pa": "pan_Guru", "pl": "pol_Latn", "pt": "por_Latn",
    "ru": "rus_Cyrl", "ta": "tam_Taml", "te": "tel_Telu", "th": "tha_Thai",
    "tr": "tur_Latn", "uk": "ukr_Cyrl", "ur": "urd_Arab", "vi": "vie_Latn",
    "zh": "zho_Hans",
}

NAMES = {
    "ar": "Arabic", "bn": "Bengali", "de": "German", "en": "English",
    "es": "Spanish", "fa": "Persian", "fr": "French", "gu": "Gujarati",
    "hi": "Hindi", "id": "Indonesian", "it": "Italian", "ja": "Japanese",
    "kn": "Kannada", "ko": "Korean", "ml": "Malayalam", "mr": "Marathi",
    "nl": "Dutch", "pa": "Punjabi", "pl": "Polish", "pt": "Brazilian Portuguese",
    "ru": "Russian", "ta": "Tamil", "te": "Telugu", "th": "Thai",
    "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese",
    "zh": "Simplified Chinese",
}

# writing system expected in a translation, by target language
SCRIPTS = {"ar": "ARABIC", "fa": "ARABIC", "ur": "ARABIC", "bn": "BENGALI",
           "gu": "GUJARATI", "hi": "DEVANAGARI", "mr": "DEVANAGARI",
           "kn": "KANNADA", "ml": "MALAYALAM", "pa": "GURMUKHI",
           "ta": "TAMIL", "te": "TELUGU", "th": "THAI", "ko": "HANGUL",
           "ru": "CYRILLIC", "uk": "CYRILLIC", "zh": "CJK",
           "ja": ("CJK", "HIRAGANA", "KATAKANA")}

DEFAULT_LLM = "~/.local/share/dubber/llm/google_gemma-3-4b-it-Q4_K_M.gguf"
DEFAULT_LORA = "~/.local/share/dubber/lora/gemma3-4b-dubber.gguf"
MODEL_NAME = ""     # model in use, for DUBBER_RECORD
PIVOT_VERSION = 1   # bump when the prompts change, to drop cached pivots
WINDOW = 6          # lines translated per LLM call
HISTORY = 3         # previous lines (with their translation) shown as context
SLACK = 1.15        # allowed overshoot of the length budget before a rewrite

# A phrase of 1..6 words repeated 3+ times ("no, no, no, no", "बरे बरे बरे").
# Whisper produces these on music/noise and translators turn them into loops.
QUOTES = ' *"\'\u201c\u201d\u2018\u2019'
# the model sometimes repeats the length limit: "... (85)", "... [max 23 chars]"
LIMIT_ECHO = re.compile(r"\s*[\[(][^\])]*\d+[^\])]*[\])]\s*\.?\s*$")
# Romanized Urdu/Hindi function words: an "English" line full of these was
# transliterated instead of translated (small LLMs drift into this).
ROMAN = set("hai hain ke ki ka ko mein aap yeh woh toh jo aur nahin nahi kar "
            "karte raha rahe tha thi hoga gaye liye sath bhi".split())
NUMBERED = re.compile(r"^\s*\**\s*(\d+)\s*[.):]\s*(.*)$")
REPEAT = re.compile(r"(\S+(?:\s+\S+){0,5}?)(?:[\s,.;:!?-]+\1){2,}", re.IGNORECASE)


def clean(s):
    prev = None
    while prev != s:
        prev, s = s, REPEAT.sub(r"\1", s)
    s = " ".join(s.split())
    return s if any(c.isalnum() for c in s) else ""


def log(*a):
    print(" ", *a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- NLLB

def translate_nllb(lines, src, tgt):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if src not in NLLB_CODES or tgt not in NLLB_CODES:
        sys.exit(f"translate.py: language pair not supported: {src}->{tgt}")
    name = os.environ.get("DUBBER_NLLB_MODEL", "facebook/nllb-200-distilled-600M")
    want = os.environ.get("DUBBER_DEVICE", "auto")
    dev = "cuda" if want != "cpu" and torch.cuda.is_available() else "cpu"
    log(f"NLLB on {dev}")
    tok = AutoTokenizer.from_pretrained(name, src_lang=NLLB_CODES[src])
    # fp16 on the GPU halves memory; small GPUs (4 GB) need it
    dtype = torch.float16 if dev == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(name, dtype=dtype).to(dev).eval()
    tgt_id = tok.convert_tokens_to_ids(NLLB_CODES[tgt])
    cpu_model = None

    def run(idx, device, m):
        enc = tok([lines[i] for i in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=256).to(device)
        limit = min(256, 2 * enc["input_ids"].shape[1] + 10)
        with torch.no_grad():
            gen = m.generate(**enc, forced_bos_token_id=tgt_id,
                             num_beams=4, max_new_tokens=limit,
                             no_repeat_ngram_size=4, repetition_penalty=1.2)
        for i, t in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            out[i] = clean(t)

    def run_safe(idx):
        """On CUDA OOM split the batch; a single line that still does not
        fit is translated on the CPU."""
        nonlocal cpu_model
        try:
            run(idx, dev, model)
            return
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
        if len(idx) > 1:
            h = len(idx) // 2
            run_safe(idx[:h])
            run_safe(idx[h:])
            return
        if cpu_model is None:
            log("(GPU memory full: falling back to CPU for long lines)")
            cpu_model = AutoModelForSeq2SeqLM.from_pretrained(name).eval()
        run(idx, "cpu", cpu_model)

    out = [""] * len(lines)
    # similar lengths in a batch = little padding = less memory and time
    todo = sorted((i for i, l in enumerate(lines) if l),
                  key=lambda i: len(lines[i]))
    batch = 16
    for b in range(0, len(todo), batch):
        run_safe(todo[b:b + batch])
        log(f"translated {min(b + batch, len(todo))}/{len(todo)}")
    return out


# ---------------------------------------------------------------- LLM

class LLM:
    def __init__(self, path):
        # load torch's (newer) CUDA runtime first: if llama.cpp's bundled
        # libcudart is loaded first, the NLLB fallback cannot import torch
        try:
            import torch  # noqa: F401
        except ImportError:
            pass
        from llama_cpp import Llama
        global MODEL_NAME
        MODEL_NAME = os.path.basename(path)
        gpu = os.environ.get("DUBBER_DEVICE") != "cpu"
        log(f"LLM {os.path.basename(path)} on {'cuda' if gpu else 'cpu'}")
        # 4096 context and 256 batch fit a 4 GB GPU next to the 4B weights;
        # a bigger model gets as many layers on the GPU as fit, rest on CPU
        tries = [-1, 40, 32, 26, 20, 14, 8, 0] if gpu else [0]
        if os.environ.get("DUBBER_LLM_GPU_LAYERS"):
            tries = [int(os.environ["DUBBER_LLM_GPU_LAYERS"])]
        # LoRA distilled from the 12B (tools/train_lora.py), for the 4B only
        lora = lora_for(path)
        if lora:
            log(f"LoRA adapter {os.path.basename(lora)}")
        for n in tries:
            try:
                self.m = Llama(path, n_gpu_layers=n, n_ctx=4096, lora_path=lora,
                               n_batch=256, n_ubatch=256, flash_attn=gpu,
                               n_threads=max(1, (os.cpu_count() or 4) // 2),
                               verbose=False)
                break
            except ValueError:          # does not fit in GPU memory
                if n == tries[-1]:
                    raise
        if n not in (-1, 0):
            log(f"({n} layers on the GPU, the rest on the CPU)")

    def tokens(self, text):
        return len(self.m.tokenize(text.encode(), add_bos=False))

    def ask(self, prompt, max_tokens=1500):
        # free text, not a JSON grammar: grammar-constrained sampling runs
        # on the CPU over the whole 262k vocabulary and is ~10x slower
        r = self.m.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, repeat_penalty=1.05,
            max_tokens=max_tokens)
        return r["choices"][0]["message"]["content"]

    def numbered(self, prompt, count, max_tokens, ok):
        """Ask for COUNT answers as "N. text" lines; missing ones are "".
        Small models sometimes repeat the source after the number and put
        the translation on the next line, so every line of an item's block
        is a candidate and the first one OK() accepts wins."""
        blocks, cur = {}, None
        for l in self.ask(prompt, max_tokens).splitlines():
            m = NUMBERED.match(l)
            if m and 1 <= int(m.group(1)) <= count:
                cur = blocks.setdefault(int(m.group(1)), [])
                l = m.group(2)
            if cur is not None:
                t = clean(LIMIT_ECHO.sub("", l).strip(QUOTES))
                if t:
                    cur.append(t)
        return [next((t for t in blocks.get(n, []) if ok(t)), "")
                for n in range(1, count + 1)]


def lora_for(path):
    """The adapter to load on top of model PATH, or None."""
    lora = os.path.expanduser(os.environ.get("DUBBER_LORA", DEFAULT_LORA))
    if lora == "none" or not os.path.exists(lora) or \
            os.path.basename(path) != os.path.basename(DEFAULT_LLM):
        return None
    return lora


def sample(llm, lines, budget):
    """Evenly spaced lines of the transcript, up to BUDGET tokens."""
    text = [l for l in lines if l]
    if not text:
        return ""
    step = 1
    while True:
        pick = text[::step]
        joined = "\n".join(pick)
        if llm.tokens(joined) <= budget or step >= len(text):
            return joined
        step += 1


def study(llm, lines, src, tgt, hint):
    """Subject of the video and a glossary of its technical terms."""
    s, t = NAMES.get(src, src), NAMES.get(tgt, tgt)
    prompt = (
        f"Below are lines sampled from the {s} transcript of a video (speech "
        f"recognition output, may contain errors).\n"
        + (f"The uploader describes it as: {hint}\n" if hint else "") +
        f"1. In English, describe the subject of the video in one or two "
        f"sentences (domain, what is being made or explained).\n"
        f"2. List up to 30 technical or domain terms that appear in the "
        f"transcript, as written there, with the correct {t} term a "
        f"professional in that field would use. Speakers often use English "
        f"loanwords written in {s} script; map them back to the real "
        f"technical term (e.g. a word that sounds like 'bearing' -> the "
        f"{t} word for bearing). Skip ordinary words.\n\n"
        f"Answer in exactly this format, nothing else:\n"
        f"SUBJECT: <one or two sentences>\n"
        f"TERM: <term as in transcript> = <{t} term>\n"
        f"TERM: ...\n\n"
        f"Transcript sample:\n{sample(llm, lines, 1800)}")
    subject, gl = hint or "", []
    for l in llm.ask(prompt, 1200).splitlines():
        l = l.strip(" *-")
        if l.upper().startswith("SUBJECT:"):
            subject = l[8:].strip()
        elif l.upper().startswith("TERM:") and "=" in l:
            a, b = (x.strip(" *\"'") for x in l[5:].split("=", 1))
            b = re.sub(r"\s*\(.*$", "", b)   # drop "(no direct equivalent...)"
            if a and b and a != b and len(b) < 40:
                gl.append((a, b))
    if subject:
        record(prompt, "\n".join([f"SUBJECT: {subject}"] +
                                 [f"TERM: {a} = {b}" for a, b in gl[:30]]))
    return subject, gl[:30]


def script_share(text, scripts):
    """Fraction of the letters of TEXT written in one of SCRIPTS."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    ok = sum(unicodedata.name(c, "?").startswith(scripts) for c in letters)
    return ok / len(letters)


def plausible(text, src, tgt):
    """Reject empty output, source text copied back, and transliteration
    instead of translation."""
    if not text or "=>" in text:
        return False
    if script_share(text, SCRIPTS.get(tgt, "LATIN")) < 0.8:
        return False
    if src in ("ur", "hi", "pa") and tgt not in ("ur", "hi", "pa"):
        words = re.findall(r"[a-z]+", text.lower())
        if sum(w in ROMAN for w in words) >= max(2, len(words) // 5):
            return False
    return True


def record(prompt, answer):
    """DUBBER_RECORD=FILE.jsonl: keep each request and its checked answer,
    e.g. from the 12B model, as training data for a LoRA on the small one."""
    path = os.environ.get("DUBBER_RECORD")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"model": MODEL_NAME, "prompt": prompt,
                                "answer": answer},
                               ensure_ascii=False) + "\n")


def budget_of(slot, cps):
    return max(8, int(slot * cps)) if slot else None


def translate_window(llm, lines, idx, done, ctx, src, tgt, budgets,
                     history=True):
    """Translate lines[idx] in one call, with previous lines as context."""
    s, t = NAMES.get(src, src), NAMES.get(tgt, tgt)
    subject, glossary = ctx
    window = " ".join(lines[i] for i in idx)
    terms = [f"  {a} = {b}" for a, b in glossary if a in window]
    hist = [i for i in range(idx[0] - 1, -1, -1)
            if lines[i] and done[i]][:HISTORY] if history else []
    prompt = [f"You are dubbing a video from {s} into {t}. Subject: {subject}",
              f"Translate each numbered {s} line into natural, spoken {t}. "
              f"The lines are consecutive pieces of speech cut by timing, so a "
              f"line may be half a sentence: translate exactly the words of "
              f"that line into that line, even if it ends up incomplete. Never "
              f"merge lines, never move words to another line, never add "
              f"content. Each translation is spoken over the time of its own "
              f"line. Keep technical terms exact; numbers and units as spoken; "
              f"be concise and stay within each line's character limit, "
              f"dropping filler words but never meaning."]
    if terms:
        prompt.append("Glossary:\n" + "\n".join(terms))
    if hist:   # only the translation: source pairs get copied into answers
        prompt.append(f"What was said just before (already translated):\n" +
                      " ".join(done[i] for i in reversed(hist)))
    if any(budgets[i] for i in idx):
        prompt.append("Character limits: " + ", ".join(
            f"line {n}: {budgets[i]}" for n, i in enumerate(idx, 1) if budgets[i]))
    prompt.append("Lines to translate:\n" + "\n".join(
        f"{n}. {lines[i]}" for n, i in enumerate(idx, 1)))
    prompt.append(f"Answer with only the {t} translations, one per line, "
                  f"numbered like the input: \"1. ...\" to \"{len(idx)}. ...\". "
                  f"Do not repeat the {s} text.")
    prompt = "\n\n".join(prompt)
    res = llm.numbered(prompt, len(idx), 100 + 100 * len(idx),
                       lambda x: plausible(x, src, tgt))
    if all(res):
        record(prompt, "\n".join(f"{n}. {r}" for n, r in enumerate(res, 1)))
    return res


def shorten(llm, lines, out, idx, ctx, src, tgt, budgets):
    """Ask for shorter wordings of translations that do not fit their slot."""
    s, t = NAMES.get(src, src), NAMES.get(tgt, tgt)
    body = [f"{n}. source: {lines[i]}\n   current: {out[i]}\n"
            f"   limit: {budgets[i]} characters (now {len(out[i])})"
            for n, i in enumerate(idx, 1)]
    prompt = (f"These {t} dubbing lines (video subject: {ctx[0]}) are too long "
              f"to be spoken in the time of the original {s} speech. Rewrite "
              f"each one in {t} within its character limit: keep the meaning "
              f"and technical terms, drop filler, use shorter words.\n\n" +
              "\n".join(body) +
              f"\n\nAnswer with only the new {t} lines, numbered \"1. ...\" "
              f"to \"{len(idx)}. ...\".")
    return llm.numbered(prompt, len(idx), 100 + 80 * len(idx),
                        lambda x: plausible(x, src, tgt))


def strip_foreign(text, tgt):
    """Drop words left in another writing system ("... InshaAllah انشاءاللہ")."""
    want = SCRIPTS.get(tgt, "LATIN")
    return " ".join(w for w in text.split() if script_share(w, want) >= 0.5)


def translate_llm(lines, src, tgt, budgets, path):
    llm = LLM(path)
    hint = os.environ.get("DUBBER_PROMPT", "").strip()

    t0 = time.time()
    ctx = study(llm, lines, src, tgt, hint)
    if hint:
        ctx = (f"{ctx[0]} ({hint})", ctx[1])
    log(f"subject: {ctx[0]}")
    log("glossary: " + "; ".join(f"{a} = {b}" for a, b in ctx[1]))

    out = [""] * len(lines)
    todo = [i for i, l in enumerate(lines) if l]
    for b in range(0, len(todo), WINDOW):
        idx = todo[b:b + WINDOW]
        res = translate_window(llm, lines, idx, out, ctx, src, tgt, budgets)
        for i, r in zip(idx, res):
            out[i] = r
        for i in idx:       # retry missing lines alone, without history
            if not out[i]:
                out[i] = translate_window(llm, lines, [i], out, ctx, src, tgt,
                                          budgets, history=False)[0]
        log(f"translated {min(b + WINDOW, len(todo))}/{len(todo)}")

    # isochrony: rewrite what is clearly too long to be spoken in its slot
    long_ = [i for i in todo if budgets[i] and out[i] and
             len(out[i]) > budgets[i] * SLACK]
    if long_:
        log(f"shortening {len(long_)} lines that exceed their time slot")
        fixed = 0
        for b in range(0, len(long_), WINDOW):
            idx = long_[b:b + WINDOW]
            res = shorten(llm, lines, out, idx, ctx, src, tgt, budgets)
            for i, r in zip(idx, res):
                if r and len(r) < len(out[i]):
                    out[i] = r
                    fixed += 1
        log(f"shortened {fixed}/{len(long_)}")
    log(f"LLM translation {src} -> {tgt} took {time.time() - t0:.0f} s")
    return [strip_foreign(o, tgt) for o in out]


def translate_stage(lines, src, tgt, budgets, path):
    """LLM translation; lines it could not do go to NLLB (after the LLM
    is freed: both do not fit a 4 GB GPU together)."""
    out = translate_llm(lines, src, tgt, budgets, path)
    miss = [i for i, l in enumerate(lines) if l and not out[i]]
    if miss:
        log(f"{len(miss)} lines left empty by the LLM, using NLLB for them")
        fill = translate_nllb([lines[i] for i in miss], src, tgt)
        for i, t in zip(miss, fill):
            out[i] = t
    return out


def english_pivot(lines, src, budgets, path, outfile):
    """SRC -> English, cached next to OUTFILE and shared by every target
    language. Small LLMs understand Urdu/Hindi far worse than they write
    other languages from English, and all dubs then use the same terms."""
    import hashlib
    key = hashlib.sha256(json.dumps(
        [lines, budgets, src, path, os.environ.get("DUBBER_PROMPT", ""),
         PIVOT_VERSION, lora_for(path),
         lora_for(path) and os.path.getmtime(lora_for(path))]).encode()).hexdigest()
    cache = os.path.join(os.path.dirname(os.path.abspath(outfile)), "pivot_en.txt")
    try:
        with open(cache, encoding="utf-8") as f:
            head, *rest = f.read().split("\n")
        if head == key and len(rest) >= len(lines):
            log(f"reusing {cache}")
            return rest[:len(lines)]
    except (OSError, ValueError):
        pass
    en = translate_stage(lines, src, "en", budgets, path)
    with open(cache, "w", encoding="utf-8") as f:
        f.write(key + "\n" + "\n".join(" ".join(l.split()) for l in en) + "\n")
    return en


# ----------------------------------------------------------------

def main():
    if len(sys.argv) not in (5, 6):
        sys.exit(__doc__)
    src, tgt, infile, outfile = sys.argv[1:5]
    with open(infile, encoding="utf-8") as f:
        lines = [clean(l.rstrip("\n")) for l in f]
    slots = None
    if len(sys.argv) == 6:
        with open(sys.argv[5]) as f:
            slots = [float(x) for x in f.read().split()]
        if len(slots) != len(lines):
            sys.exit("translate.py: SLOTS and INFILE differ in length")

    engine = os.environ.get("DUBBER_TRANSLATOR", "auto")
    path = os.path.expanduser(os.environ.get("DUBBER_LLM", DEFAULT_LLM))
    if engine == "auto":
        engine = "llm" if os.path.exists(path) else "nllb"
    if engine == "llm" and not os.path.exists(path):
        sys.exit(f"translate.py: LLM model {path} not found; run fetch_models.sh")

    cps = float(os.environ.get("DUBBER_CPS", "17"))
    budgets = [budget_of(slots[i] if slots else 0, cps) for i in range(len(lines))]
    pivot = os.environ.get("DUBBER_PIVOT", "1") != "0"

    if src == tgt:
        out = lines
    elif engine == "llm" and pivot and src != "en":
        # the hard step (understanding the source) may use a bigger model
        big = os.path.expanduser(os.environ.get("DUBBER_LLM_PIVOT", "") or path)
        out = english_pivot(lines, src, budgets, big, outfile)
        if tgt != "en":
            out = translate_stage(out, "en", tgt, budgets, path)
    elif engine == "llm":
        out = translate_stage(lines, src, tgt, budgets, path)
    else:
        out = translate_nllb(lines, src, tgt)

    with open(outfile, "w", encoding="utf-8") as f:
        for l in out:
            f.write(" ".join(l.split()) + "\n")


if __name__ == "__main__":
    main()
