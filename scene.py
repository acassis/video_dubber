#!/usr/bin/env python3
"""Who speaks each phrase, whether their lips are on screen, and how they
say it.

usage: scene.py WORKDIR

Reads WORKDIR/orig.json (phrases with word times, from asr.py),
WORKDIR/vocals.wav and WORKDIR/lips.npz (from lips.py). Writes
WORKDIR/scene.srt (the phrases, split where the speaker changes) and
WORKDIR/scene.json, one entry per phrase:

  speaker, gender     speaker cluster (ECAPA embeddings) and its sex
                      (audeering wav2vec2 age/gender model; pitch is a poor
                      cue: an animated man can speak higher than a woman)
  words               [[start, end, word], ...]
  f0                  pitch every HOP s from 0.2 s before the first word,
                      in semitones above 100 Hz (null = unvoiced)
  db                  loudness every HOP s, same grid
  face                the face track whose mouth moves with the voice, or
                      null when the speaker's lips are not visible
  lip_weight          0..1: how much lip sync matters here (face size,
                      frontal pose, share of the phrase on screen)
  mouth               lip opening 0..1 every frame of the phrase (null where
                      the face is not seen), at the video frame rate
"""
import json
import os
import sys

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")

HOP = 0.01          # pitch / loudness grid (s)
PAD = 0.2           # analysis starts this long before the first word
WIN, STEP = 2.0, 1.0   # speaker-embedding windows inside long phrases
SAME_SPK = 0.75     # average-linkage cosine distance joining one speaker
MIN_CLUSTER = 8     # smaller clusters (coughs, crosstalk) join a big one
MIN_TURN = 0.8      # a speaker turn shorter than this is not split off
FEMALE_HZ = 165     # fallback without the gender model: median pitch
GENDER_MODEL = "audeering/wav2vec2-large-robust-6-ft-age-gender"
MIN_FACE = 0.035    # faces narrower than this (frame fraction) are too small
FULL_FACE = 0.15    # faces this wide count as a full close-up
MAX_YAW = 0.55      # more turned than this: lips not readable
LIP_LAG = 3         # frames the lips may lead the sound (~100 ms)
STILL = 0.012       # lifetime lip motion below this: a photo or still image


def load_mono(path, rate):
    import soundfile as sf
    from scipy.signal import resample_poly
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != rate:
        g = np.gcd(sr, rate)
        x = resample_poly(x, rate // g, sr // g).astype(np.float32)
    return x


# ------------------------------------------------------------- speakers

def embed(chunks, pcm, rate):
    """ECAPA speaker embedding of each (start, end) chunk of pcm."""
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    dev = os.environ.get("DUBBER_DEVICE", "cpu")
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        dev = "cuda:0"          # speechbrain wants an explicit index
    enc = EncoderClassifier.from_hparams(
        "speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": dev},
        savedir=os.path.expanduser("~/.local/share/dubber/ecapa"))
    out = []
    n = int(WIN * rate)
    for k in range(0, len(chunks), 32):
        batch = np.zeros((len(chunks[k:k + 32]), n), np.float32)
        lens = []
        for j, (a, b) in enumerate(chunks[k:k + 32]):
            x = pcm[int(a * rate):int(b * rate)][:n]
            batch[j, :len(x)] = x
            lens.append(max(len(x), 1) / n)
        with torch.no_grad():
            e = enc.encode_batch(torch.from_numpy(batch).to(dev),
                                 torch.tensor(lens, device=dev))
        out.append(e.squeeze(1).cpu().numpy())
    e = np.concatenate(out)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def speaker_turns(segs, pcm, rate):
    """Label every word with a speaker cluster; returns labels per segment
    and the number of clusters."""
    from scipy.cluster.hierarchy import fcluster, linkage
    chunks, owner = [], []
    for i, s in enumerate(segs):
        a, b = s["words"][0][0], s["words"][-1][1]
        if b - a <= WIN + STEP / 2:
            chunks.append((a, b))
            owner.append(i)
            continue
        t = a
        while t < b - STEP / 2:
            chunks.append((t, min(t + WIN, b)))
            owner.append(i)
            t += STEP
    emb = embed(chunks, pcm, rate)
    lab = fcluster(linkage(emb, "average", metric="cosine"), SAME_SPK,
                   criterion="distance")
    # tiny clusters (a cough, crosstalk) join their nearest big cluster
    ids, cnt = np.unique(lab, return_counts=True)
    big = [c for c, n in zip(ids, cnt) if n >= MIN_CLUSTER] or list(ids)
    cent = {c: emb[lab == c].mean(0) for c in big}
    for j, c in enumerate(lab):
        if c not in cent:
            lab[j] = max(big, key=lambda b: emb[j] @ cent[b])
    remap = {c: k for k, c in enumerate(sorted(set(lab),
             key=lambda c: -np.sum(lab == c)))}   # 0 = most talkative
    words = []
    for i, s in enumerate(segs):
        mine = [(c, lab[j]) for j, c in enumerate(chunks) if owner[j] == i]
        wl = []
        for w0, w1, _ in s["words"]:
            m = (w0 + w1) / 2
            # the window whose centre is closest to the word
            c = min(mine, key=lambda cl: abs((cl[0][0] + cl[0][1]) / 2 - m))
            wl.append(remap[c[1]])
        words.append(wl)
    return words, len(remap)


def split_turns(segs, labels):
    """Split phrases where the speaker changes for at least MIN_TURN s."""
    out = []
    for s, lab in zip(segs, labels):
        runs = []                # [first word, last word, speaker]
        for k, c in enumerate(lab):
            if runs and runs[-1][2] == c:
                runs[-1][1] = k
            else:
                runs.append([k, k, c])
        w = s["words"]
        merged = []
        for r in runs:           # short runs join the previous turn
            if merged and w[r[1]][1] - w[r[0]][0] < MIN_TURN:
                merged[-1][1] = r[1]
            else:
                merged.append(r)
        if len(merged) > 1 and \
                w[merged[0][1]][1] - w[merged[0][0]][0] < MIN_TURN:
            merged[1][0] = 0
            merged.pop(0)
        for a, b, _ in merged:
            ws = w[a:b + 1]
            spk = np.bincount(lab[a:b + 1]).argmax()
            out.append(dict(start=ws[0][0], end=ws[-1][1], words=ws,
                            text=" ".join(x[2] for x in ws), speaker=int(spk)))
    return out


def female_share(phrases, pcm, rate):
    """Per speaker: mean probability that the voice is female, from up to
    20 phrases of 1.5 s or more. None if the model is not installed."""
    import torch
    import torch.nn as nn
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2Model, Wav2Vec2PreTrainedModel)

    class Head(nn.Module):
        def __init__(self, c, n):
            super().__init__()
            self.dense = nn.Linear(c.hidden_size, c.hidden_size)
            self.dropout = nn.Dropout(c.final_dropout)
            self.out_proj = nn.Linear(c.hidden_size, n)

        def forward(self, x):
            x = torch.tanh(self.dense(self.dropout(x)))
            return self.out_proj(self.dropout(x))

    class AgeGender(Wav2Vec2PreTrainedModel):   # as in the model card
        def __init__(self, c):
            super().__init__(c)
            self.wav2vec2 = Wav2Vec2Model(c)
            self.age = Head(c, 1)
            self.gender = Head(c, 3)            # female, male, child
            self.post_init()

        def forward(self, x):
            h = self.wav2vec2(x)[0].mean(1)
            return torch.softmax(self.gender(h), 1)

    try:
        dev = "cuda" if torch.cuda.is_available() and \
            os.environ.get("DUBBER_DEVICE") != "cpu" else "cpu"
        m = AgeGender.from_pretrained(GENDER_MODEL).to(dev).eval()
    except OSError:
        return None
    share = {}
    for k in sorted({p["speaker"] for p in phrases}):
        mine = [p for p in phrases if p["speaker"] == k]
        mine = sorted(mine, key=lambda p: p["start"] - p["end"])[:20]
        probs = []
        for p in mine:
            x = pcm[int(p["start"] * rate):int(p["end"] * rate)]
            x = (x - x.mean()) / (x.std() + 1e-7)
            with torch.no_grad():
                g = m(torch.from_numpy(x)[None].to(dev))[0].cpu().numpy()
            probs.append(g[0] / max(g[0] + g[1], 1e-6))
        share[k] = float(np.mean(probs))
    del m
    return share


# --------------------------------------------------------------- prosody

def prosody(pcm, rate, a, b):
    """Pitch (semitones above 100 Hz, NaN unvoiced) and loudness (dB) on
    the HOP grid from a - PAD to b."""
    import parselmouth
    t0 = max(0.0, a - PAD)
    x = pcm[int(t0 * rate):int((b + 0.05) * rate)]
    n = int(round((b - t0) / HOP))
    if len(x) < rate // 10:
        return np.full(n, np.nan), np.full(n, -60.0)
    snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=rate)
    p = snd.to_pitch_ac(time_step=HOP, pitch_floor=70, pitch_ceiling=500)
    grid = t0 + HOP * np.arange(n)
    hz = np.array([p.get_value_at_time(t - t0) for t in grid], dtype=float)
    hz[(hz <= 0)] = np.nan
    st = 12 * np.log2(hz / 100)
    hop = int(HOP * rate)
    db = np.array([10 * np.log10(np.mean(x[k * hop:(k + 1) * hop] ** 2) + 1e-10)
                   for k in range(n)])
    return st, db


# ---------------------------------------------------------------- lips

class Lips:
    def __init__(self, path):
        d = np.load(path)
        self.fps = float(d["fps"])
        self.keys = list(d["keys"])
        r = d["rows"]
        self.tracks = {}
        for t in np.unique(r[:, 1]).astype(int):
            m = r[r[:, 1] == t]
            self.tracks[t] = {k: m[:, 2 + j] for j, k in enumerate(self.keys)}
            self.tracks[t]["frame"] = m[:, 0].astype(int)
        # lip opening scaled per track to 0 (closed) .. 1 (wide open)
        for tr in self.tracks.values():
            lo, hi = np.percentile(tr["open"], [5, 95])
            tr["norm"] = np.clip((tr["open"] - lo) / max(hi - lo, 1e-3), 0, 1)
        # faces in photos and graphics: landmarks jitter, lips never move
        for t, tr in self.tracks.items():
            f = tr["frame"]
            tr["still"] = motion(self.series(t, "open", f[0], f[-1] + 1)) < STILL

    def series(self, t, key, f0, f1):
        tr = self.tracks[t]
        x = np.full(f1 - f0, np.nan)
        m = (tr["frame"] >= f0) & (tr["frame"] < f1)
        x[tr["frame"][m] - f0] = tr[key][m]
        return x


def motion(x):
    """Syllable-rate movement (2-8 Hz) of a lip-opening series."""
    ok = ~np.isnan(x)
    if ok.sum() < 8 or len(x) < 12:
        return 0.0
    y = np.where(ok, x, np.nanmean(x))
    y = y - np.convolve(y, np.ones(9) / 9, "same")   # drop slow drift
    return float(np.sqrt(np.mean(y[ok] ** 2)))


def active_face(lips, seg, env):
    """The face that is speaking this phrase: its lips move at syllable
    rate and with the voice's loudness. Returns (track, score) or (None, 0).
    env: voice loudness per video frame."""
    fps = lips.fps
    f0, f1 = int(seg["start"] * fps), int(seg["end"] * fps) + 1
    best, bs = None, 0.0
    for t, tr in lips.tracks.items():
        if tr["frame"][0] >= f1 or tr["frame"][-1] < f0 or tr["still"]:
            continue
        x = lips.series(t, "open", f0, f1)
        ok = ~np.isnan(x)
        if ok.mean() < 0.5:
            continue
        mv = motion(x)
        e = env[f0:f1]
        corr = 0.0
        for lag in range(LIP_LAG + 1):      # lips lead the sound slightly
            xs, es = x[:len(x) - lag], e[lag:]
            o = ~np.isnan(xs)
            if o.sum() > 10 and np.std(xs[o]) > 0 and np.std(es[o]) > 0:
                corr = max(corr, float(np.corrcoef(xs[o], es[o])[0, 1]))
        size = float(np.nanmedian(lips.series(t, "size", f0, f1)))
        score = mv * (0.5 + corr) * ok.mean()
        if score > bs and size >= MIN_FACE:
            best, bs = t, score
    return best, bs


def lip_info(lips, seg, track):
    fps = lips.fps
    f0 = int((seg["start"] - PAD) * fps)
    f1 = int(seg["end"] * fps) + 2
    f0 = max(f0, 0)
    mouth = lips.series(track, "norm", f0, f1)
    size = lips.series(track, "size", f0, f1)
    yaw = lips.series(track, "yaw", f0, f1)
    seen = ~np.isnan(mouth)
    good = seen & (yaw < MAX_YAW)
    if not good.any():
        return None, 0.0, f0
    sz = float(np.nanmedian(size[good]))
    w = np.clip((sz - MIN_FACE) / (FULL_FACE - MIN_FACE), 0, 1) ** 0.7
    w *= good.mean() * float(np.clip(1.2 - np.nanmedian(yaw[good]), 0, 1))
    mouth[~good] = np.nan
    return mouth, float(w), f0


# ---------------------------------------------------------------- main

def srt_time(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}," \
           f"{ms % 1000:03d}"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    wd = sys.argv[1]
    with open(os.path.join(wd, "orig.json")) as f:
        segs = json.load(f)["segments"]
    rate = 16000
    pcm = load_mono(os.path.join(wd, "vocals.wav"), rate)

    labels, nspk = speaker_turns(segs, pcm, rate)
    phrases = split_turns(segs, labels)
    print(f"  {nspk} speakers; {len(segs)} phrases -> {len(phrases)} after "
          f"splitting at speaker turns", file=sys.stderr)

    for p in phrases:
        p["f0"], p["db"] = prosody(pcm, rate, p["start"], p["end"])
    share = female_share(phrases, pcm, rate)
    gender = {}
    for k in range(nspk):
        f = np.concatenate([p["f0"] for p in phrases if p["speaker"] == k])
        hz = float(100 * 2 ** (np.nanmedian(f) / 12)) if np.isfinite(f).any() \
            else 0.0
        secs = sum(p["end"] - p["start"] for p in phrases if p["speaker"] == k)
        if share:
            gender[k] = "f" if share[k] > 0.5 else "m"
            why = f"p(female) {share[k]:.2f}"
        else:
            gender[k] = "f" if hz > FEMALE_HZ else "m"
            why = "from pitch"
        print(f"  speaker {k}: {gender[k]} ({why}), median pitch {hz:.0f} Hz, "
              f"{secs:.0f} s of speech", file=sys.stderr)

    lp = os.path.join(wd, "lips.npz")
    lips = Lips(lp) if os.path.exists(lp) else None
    if lips:
        hop = rate / lips.fps
        nf = int(len(pcm) / hop)
        env = np.array([10 * np.log10(np.mean(pcm[int(i * hop):int((i + 1) * hop)]
                                              ** 2) + 1e-10) for i in range(nf)])
        env = np.clip(env, env.max() - 50, None)
    onscreen = 0
    for p in phrases:
        p["gender"] = gender[p["speaker"]]
        p["face"], p["lip_weight"], p["mouth"], p["mouth_frame0"] = \
            None, 0.0, None, 0
        if lips:
            t, score = active_face(lips, p, env)
            if t is not None:
                mouth, w, fr0 = lip_info(lips, p, t)
                if mouth is not None and w > 0.05:
                    p["face"], p["lip_weight"] = int(t), round(w, 3)
                    p["mouth"] = [None if np.isnan(v) else round(float(v), 3)
                                  for v in mouth]
                    p["mouth_frame0"] = fr0
                    p["face_score"] = score
    # a face belongs to one speaker: the one whose phrases chose it most
    # strongly. Others chose it because the listener's lips moved (nods,
    # smiles) while the real speaker was off screen.
    owner = {}
    for p in phrases:
        if p["face"] is not None:
            o = owner.setdefault(p["face"], {})
            o[p["speaker"]] = o.get(p["speaker"], 0) + p["face_score"]
    for p in phrases:
        if p["face"] is not None:
            o = owner[p["face"]]
            if max(o, key=o.get) != p["speaker"]:
                p["face"], p["lip_weight"], p["mouth"] = None, 0.0, None
        p.pop("face_score", None)
        onscreen += p["face"] is not None
        p["f0"] = [None if np.isnan(v) else round(float(v), 2) for v in p["f0"]]
        p["db"] = [round(float(v), 1) for v in p["db"]]
    print(f"  lips visible in {onscreen}/{len(phrases)} phrases", file=sys.stderr)

    with open(os.path.join(wd, "scene.json"), "w") as f:
        json.dump({"fps": lips.fps if lips else 0, "hop": HOP, "pad": PAD,
                   "speakers": {str(k): v for k, v in gender.items()},
                   "phrases": phrases}, f, ensure_ascii=False)
    with open(os.path.join(wd, "scene.srt"), "w", encoding="utf-8") as f:
        for n, p in enumerate(phrases, 1):
            f.write(f"{n}\n{srt_time(p['start'])} --> {srt_time(p['end'])}\n"
                    f"{p['text']}\n\n")


if __name__ == "__main__":
    main()
