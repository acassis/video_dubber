#!/usr/bin/env python3
"""Distil the big translator into the small one with a LoRA adapter.

usage: train_lora.py DATA.jsonl [DATA.jsonl ...]

DATA is what translate.py records with DUBBER_RECORD=FILE while running
with the big model (dubber --best): each line holds a request and Gemma 3
12B's checked answer. Only records made by a 12B model are used.

The 4B model (4-bit, QLoRA) learns to give the 12B's answers. The result
is converted to a llama.cpp adapter, ~/.local/share/dubber/lora/
gemma3-4b-dubber.gguf, which translate.py loads automatically on top of
the 4B model. The last 10% of the records are held out and used to score
4B, 4B + LoRA and the 12B reference (chrF, 0..100).

Needs a 4 GB GPU: the 4-bit weights sit on the GPU, while the 262k-token
embedding / output matrix (1.3 GB) stays on the CPU and the output layer
is computed only for the answer tokens.

Environment: DUBBER_LORA_EPOCHS (3), DUBBER_LORA_RANK (8).
"""
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter

# a 4 GB GPU is full: avoid fragmentation of the CUDA caching allocator
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DATA = os.path.expanduser("~/.local/share/dubber")
BASE = os.path.join(DATA, "lora", "base")          # unsloth/gemma-3-4b-it-bnb-4bit
CONVERTER = os.path.join(DATA, "lora", "llama.cpp", "convert_lora_to_gguf.py")
SMALL = os.path.join(DATA, "llm", "google_gemma-3-4b-it-Q4_K_M.gguf")
OUT = os.path.join(DATA, "lora", "gemma3-4b-dubber.gguf")
ADAPTER = os.path.join(DATA, "lora", "adapter")


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def load(paths):
    recs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            recs += [r for r in map(json.loads, f) if "12b" in r.get("model", "")]
    if len(recs) < 20:
        sys.exit(f"train_lora.py: only {len(recs)} 12B records; need more data")
    return recs


# ------------------------------------------------------------- training

def train(train_recs):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE, device_map={"": 0}, dtype=torch.bfloat16)
    inner = model.model
    # text only: drop the image encoder; keep the huge embedding on the CPU
    inner.vision_tower = None
    inner.multi_modal_projector = None
    lm = inner.language_model
    emb = lm.embed_tokens.to("cpu")
    out_w = emb.weight.detach().float()           # tied output matrix, CPU fp32
    torch.cuda.empty_cache()
    for p in model.parameters():
        p.requires_grad_(False)
    lm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})

    rank = int(os.environ.get("DUBBER_LORA_RANK", "8"))
    cfg = LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.05, bias="none",
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|"
                       r"gate_proj|up_proj|down_proj)")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    def encode(r):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}], tokenize=False,
            add_generation_prompt=True)
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        a = tok(r["answer"] + "<end_of_turn>\n", add_special_tokens=False)["input_ids"]
        return p, a

    data = [encode(r) for r in train_recs]
    data = [d for d in data if len(d[0]) + len(d[1]) <= 2048]
    params = [p for p in model.parameters() if p.requires_grad]
    # foreach=False: no batched temporaries, they do not fit next to the model
    opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.0, foreach=False)
    epochs = int(os.environ.get("DUBBER_LORA_EPOCHS", "3"))
    accum = 4
    total = epochs * math.ceil(len(data) / accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 5) * max(0.05, 1 - s / total))

    model.train()
    step, t0 = 0, time.time()
    for ep in range(epochs):
        random.Random(ep).shuffle(data)
        run = 0.0
        for k, (p, a) in enumerate(data):
            ids = torch.tensor([p + a])
            with torch.no_grad():
                x = emb(ids).to(0, torch.bfloat16)       # scaled embeddings
            x.requires_grad_(True)
            h = lm(inputs_embeds=x).last_hidden_state[0]
            # position t predicts token t+1: only the answer tokens count
            hs = h[len(p) - 1:len(p) + len(a) - 1]
            hc = hs.detach().float().cpu().requires_grad_(True)
            logits = hc @ out_w.T
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor(a))
            (loss / accum).backward()
            hs.backward(hc.grad.to(0, hs.dtype))
            run += loss.item()
            if (k + 1) % accum == 0 or k + 1 == len(data):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            if (k + 1) % 10 == 0:
                log(f"  epoch {ep + 1}/{epochs}  {k + 1}/{len(data)}  "
                    f"loss {run / 10:.3f}  "
                    f"{time.time() - t0:.0f} s  "
                    f"GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
                run = 0.0
    model.save_pretrained(ADAPTER)
    log(f"adapter saved in {ADAPTER} ({time.time() - t0:.0f} s)")


def convert():
    # the converter reads only the base config, and refuses a bitsandbytes
    # one: give it a copy without the quantization section
    cfgdir = os.path.join(DATA, "lora", "base_cfg")
    os.makedirs(cfgdir, exist_ok=True)
    with open(os.path.join(BASE, "config.json")) as f:
        cfg = json.load(f)
    cfg.pop("quantization_config", None)
    with open(os.path.join(cfgdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    for name in os.listdir(BASE):
        dst = os.path.join(cfgdir, name)
        if name.startswith(("tokenizer", "special_tokens", "added_tokens")) \
                and not os.path.exists(dst):
            os.symlink(os.path.join(BASE, name), dst)
    py = sys.executable
    env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(CONVERTER), "gguf-py"))
    subprocess.run([py, CONVERTER, "--base", cfgdir, "--outtype", "f16",
                    "--outfile", OUT, ADAPTER], check=True, env=env)
    log(f"llama.cpp adapter: {OUT}")


# ------------------------------------------------------------- scoring

def chrf(hyp, ref, n=6, beta=2.0):
    """Character n-gram F-score (Popovic 2015), 0..100."""
    def grams(s, k):
        s = s.replace(" ", "")
        return Counter(s[i:i + k] for i in range(len(s) - k + 1))
    p = r = 0.0
    m = 0
    for k in range(1, n + 1):
        h, g = grams(hyp, k), grams(ref, k)
        if not h or not g:
            continue
        hit = sum((h & g).values())
        p += hit / sum(h.values())
        r += hit / sum(g.values())
        m += 1
    if not m or not (p or r):
        return 0.0
    p, r = p / m, r / m
    return 100 * (1 + beta ** 2) * p * r / (beta ** 2 * p + r) if p + r else 0.0


def evaluate(recs):
    from llama_cpp import Llama
    res = {}
    for name, lora in (("4B", None), ("4B + LoRA", OUT)):
        m = Llama(SMALL, n_gpu_layers=-1, n_ctx=4096, n_batch=256, n_ubatch=256,
                  flash_attn=True, lora_path=lora, verbose=False)
        scores, outs = [], []
        for r in recs:
            a = m.create_chat_completion(
                messages=[{"role": "user", "content": r["prompt"]}],
                temperature=0.0, max_tokens=900)["choices"][0]["message"]["content"]
            scores.append(chrf(a, r["answer"]))
            outs.append(a)
        res[name] = (sum(scores) / len(scores), outs)
        del m
        log(f"  {name}: chrF vs 12B = {res[name][0]:.1f}")
    for i, r in enumerate(recs[:3]):
        log(f"\n--- held-out example {i + 1}\n12B:\n{r['answer']}\n"
            f"4B:\n{res['4B'][1][i]}\n4B + LoRA:\n{res['4B + LoRA'][1][i]}")
    return res


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    recs = load(sys.argv[1:])
    cut = int(len(recs) * 0.9)
    log(f"{len(recs)} 12B records: training on {cut}, scoring on {len(recs) - cut}")
    if os.environ.get("DUBBER_LORA_SKIP_TRAIN") != "1":
        train(recs[:cut])
    if not os.path.exists(OUT) or \
            os.path.getmtime(OUT) < os.path.getmtime(ADAPTER):
        convert()
    evaluate(recs[cut:])


if __name__ == "__main__":
    main()
