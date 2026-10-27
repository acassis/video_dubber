#!/usr/bin/env python3
"""Lip-sync dubbing: fit the translation to the speaker's time and lips the
way a dubbing adapter does, then voice it with the speaker's intonation.

usage: lipdub.py WORKDIR SRC LANG

Reads WORKDIR/scene.json (scene.py), WORKDIR/src.txt (one phrase per line)
and WORKDIR/LANG.txt (first translation, translate.py). Writes

  tts_LANG/NNNNN.wav    the final clip of each phrase (24 kHz, trimmed)
  LANG_dub.txt          the wording chosen for each phrase
  LANG_dub.srt          subtitles at the times the lines are spoken
  place_LANG.txt        "<start seconds> <clip>" per phrase, for dubber.c
  lipsync_LANG.json     candidates, durations and scores of every phrase

The speech is never sped up or slowed down. Instead (Brannon et al. 2023:
human dubbers keep their speaking rate and adapt the words):

1. Every translation is spoken by Kokoro and measured: its real duration and
   its phoneme timing, hence when the lips open and close.
2. Lines whose spoken length misses the original's (isochrony), or whose
   speaker is visible, get alternative wordings from the LLM with a syllable
   target, plus where the visible lips close (bilabials p/b/m) and open.
   A reworded line must still be a correct translation. It keeps the
   numbers and names, and either it and the first translation entail each
   other (multilingual NLI), or it is a condensation: the first translation
   entails it (nothing added or changed), neither contradicts the other, and
   its LaBSE similarity to the source stays close to the first translation's
   (the main point is kept). Only redundancy may go: fillers, repetitions,
   hedges. A rewrite that drops words pays for it in the score, so a
   complete one wins whenever it fits.
3. Candidates are scored on duration fit, and where lips are visible on how
   well the visemes of the candidate follow the measured lip opening (weight:
   face size, frontal pose, time on screen). The best wins; lines that still
   do not fit go back to the LLM with the measured time (like an adapter
   re-reading a take).
4. The winner gets the speaker's intonation: the original pitch contour,
   relative to the speaker's own mean, is laid on the clip with PSOLA
   (Praat), keeping the voice's micro-prosody and the clip's duration; the
   phrase's loudness relative to the speaker is kept too.
"""
import json
import os
import re
import sys
import time

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import translate as T                               # noqa: E402

REPO = "hexgrad/Kokoro-82M"
RATE = 24000
UNIT = 600                  # samples per Kokoro duration unit (25 ms)
# Kokoro language code and voices by sex; a speaker gets the next unused
# voice of their sex ("a,b" blends two voice styles into a new timbre)
KOKORO = {
    "pt": ("p", {"f": ["pf_dora", "pf_dora,af_heart"],
                 "m": ["pm_alex", "pm_santa", "pm_alex,am_michael"]}),
    "en": ("a", {"f": ["af_heart", "af_bella"],
                 "m": ["am_michael", "am_fenrir"]}),
    "es": ("e", {"f": ["ef_dora"], "m": ["em_alex", "em_santa"]}),
    "fr": ("f", {"f": ["ff_siwis"], "m": ["ff_siwis"]}),
    "it": ("i", {"f": ["if_sara"], "m": ["im_nicola"]}),
    "hi": ("h", {"f": ["hf_alpha", "hf_beta"], "m": ["hm_omega", "hm_psi"]}),
}
GAP = 0.12          # pause before another speaker's line (turn-taking)
GAP_SAME = 0.06     # one speaker's lines keep their own pause, at least this
GAP_MAX = 0.25      # ... and at most this
TURN_GAP = 0.8      # one speaker's phrases closer than this form a turn
INNER_ON = 0.6      # inside a turn, on camera, a line may come this late
INNER_OFF = 2.0     # ... and off camera this late
CUT_INSIDE = 200.0  # cutting a line its own speaker continues: almost never
LOST_W = 0.6        # shortened ending: cost of losing a number/name/negation
EARLY = 0.35        # off-screen lines may start this much earlier
SHIFT = 0.10        # on-screen lines may move this much to meet the lips
SHIFT_ON = 0.15     # timeline: on-camera lines stay this close to the lips
EARLY_OFF = 0.8     # timeline: off-camera lines may start this much earlier
LATE_OFF = 1.5      # ... or this much later, into the following pause
CUT_COST = 20.0     # a second of speech cut costs as much as moving 20 s
TOL = 0.08          # duration error (fraction) that costs nothing
MIN_ERR = 0.25      # s; a line closer than this to its time is left alone
BLOCK_GAP = 0.8     # block mode: one speaker's phrases closer than this join
BLOCK_MAX = 15.0    # ... into utterances of at most this many seconds
ISO_N = 4           # isochronous translations asked per line
LIP_W = 1.2         # weight of lip agreement against duration fit
NEED_LIPS = 0.25    # lip weight from which lip-aware variants are asked for
VARIANTS = 6          # about half survive the meaning check
SILENCE = 0.01      # |sample| below this is silence (-40 dBFS)
PITCH_RANGE = 8.0   # semitones the imposed contour may leave the voice mean
PITCH_SMOOTH = 0.20  # s; the contour's slow part (intonation, not segmental)
STRENGTH = 0.85     # share of the original intonation laid on the voice
GAIN_RANGE = 6.0    # dB a phrase may be louder/softer than the voice mean
NLI = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
SAME = 0.8          # a rewrite and the first translation entail each other
CONTRA = 0.2        # condensation: no contradiction either way above this
COND_DROP = 0.15    # condensation: LaBSE may drop this much at most
LOSS_W = 0.3        # score cost of a condensation that drops everything
LABSE = "sentence-transformers/LaBSE"
LABSE_FILES = ["*.json", "*.txt", "model.safetensors", "1_Pooling/*",
               "2_Dense/*"]   # what fetch_models.sh downloads
SIM_DROP = 0.06     # a variant may score this much below the first translation
SIM_MIN = 0.70      # and never below this (LaBSE cosine, source vs variant)

# lip opening of each phoneme, 0 = lips closed, 1 = wide open
OPEN = {}
OPEN.update(dict.fromkeys("aɐAæ", 1.0))
OPEN.update(dict.fromkeys("ɛɔO", 0.8))
OPEN.update(dict.fromkeys("eoəɜʌEI", 0.6))
OPEN.update(dict.fromkeys("iɪyʏj", 0.35))
OPEN.update(dict.fromkeys("uʊwɯW", 0.3))
OPEN.update(dict.fromkeys("pbm", 0.0))
OPEN.update(dict.fromkeys("fv", 0.12))
REST = 0.08                     # pauses and punctuation: mouth at rest
INHERIT = set("ˈˌː̃")      # stress, length, nasal: like their neighbour
VOWELS = re.compile(r"[aeiouyáéíóúâêôãõàü]+", re.IGNORECASE)


def log(*a):
    print("  " + " ".join(str(x) for x in a), file=sys.stderr, flush=True)


def syllables(text):
    return max(1, len(VOWELS.findall(text)))


# ------------------------------------------------------------------ TTS

class Speaker:
    """Kokoro: speech plus the timing of every phoneme."""

    def __init__(self, lang, device):
        from kokoro import KModel, KPipeline
        self.code = KOKORO[lang][0]
        self.device = device
        self.model = KModel(repo_id=REPO).to(device).eval()
        self.pipe = KPipeline(lang_code=self.code, repo_id=REPO,
                              model=self.model)

    def free(self):
        import torch
        del self.pipe, self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def say(self, text, voice):
        """(audio, phonemes as [(symbol, start s, end s)]), leading and
        trailing silence trimmed."""
        audio, phon, t = [], [], 0.0
        for r in self.pipe(text, voice=voice, speed=1.0):
            if r.audio is None:
                continue
            a = r.audio.cpu().numpy().astype(np.float32)
            if r.pred_dur is not None and r.phonemes:
                d = r.pred_dur.cpu().numpy() * UNIT / RATE
                edges = t + np.concatenate([[0], np.cumsum(d)])
                # pred_dur: BOS, one per phoneme symbol Kokoro knows, EOS
                # (symbols missing from its vocabulary are dropped first)
                known = [c for c in r.phonemes if c in self.model.vocab]
                for k, ch in enumerate(known[:len(edges) - 2]):
                    phon.append((ch, edges[k + 1], edges[k + 2]))
            audio.append(a)
            t += len(a) / RATE
        if not audio:
            return np.zeros(0, np.float32), []
        x = np.concatenate(audio)
        loud = np.flatnonzero(np.abs(x) > SILENCE)
        if not len(loud):
            return np.zeros(0, np.float32), []
        pad = RATE // 200
        a, b = max(loud[0] - pad, 0), min(loud[-1] + pad, len(x))
        off = a / RATE
        phon = [(c, s - off, e - off) for c, s, e in phon if e > off]
        return x[a:b], phon


def visemes(phon, fps, n):
    """Lip opening of a clip's phonemes on the video frame grid (n frames
    from the clip's start), smoothed for coarticulation."""
    v = np.full(n, REST)
    vals = []
    for k, (c, s, e) in enumerate(phon):
        if c in INHERIT:
            continue
        vals.append((OPEN.get(c, REST if not c.isalpha() else 0.3), s, e))
    # spread each phoneme over the frames it covers
    for o, s, e in vals:
        f0, f1 = int(round(s * fps)), max(int(round(e * fps)), int(round(s * fps)) + 1)
        v[max(f0, 0):min(f1, n)] = o
    k = np.array([0.25, 0.5, 0.25])
    return np.convolve(v, k, "same")


# ----------------------------------------------------------- phrase data

class Phrase:
    def __init__(self, i, p, src, base, nxt_start, fps):
        self.i, self.src, self.base = i, src, base
        self.start, self.end = p["start"], p["end"]
        self.target = self.end - self.start
        self.next_start = nxt_start
        self.speaker, self.gender = p["speaker"], p["gender"]
        self.lip_weight = p["lip_weight"] if p["mouth"] else 0.0
        self.f0 = np.array([np.nan if v is None else v for v in p["f0"]])
        self.db = np.array(p["db"])
        self.mouth = None
        if p["mouth"]:
            self.mouth = np.array([np.nan if v is None else v for v in p["mouth"]])
            self.frame0 = p["mouth_frame0"]
        self.fps = fps
        self.cands = {}         # text -> measurement dict
        self.best = None

    def lip_events(self):
        """Where (fraction of the line) the visible lips close, and whether
        the line ends open: hints for the adapter."""
        m = self.mouth
        f_a = int(self.start * self.fps) - self.frame0
        f_b = int(self.end * self.fps) - self.frame0
        seg = m[max(f_a, 0):max(f_b, 1)]
        closes = []
        for k in range(1, len(seg) - 1):
            if np.isnan(seg[k]) or seg[k] > 0.12:
                continue
            if (np.isnan(seg[k - 1]) or seg[k - 1] > seg[k]) and \
                    (np.isnan(seg[k + 1]) or seg[k + 1] >= seg[k]):
                frac = k / max(len(seg) - 1, 1)
                if 0.05 < frac < 0.95 and \
                        (not closes or frac - closes[-1] > 0.12):
                    closes.append(frac)
        tail = seg[-max(3, len(seg) // 8):]
        ends_open = bool(np.nanmean(tail) > 0.5) if np.isfinite(tail).any() \
            else False
        return closes[:4], ends_open


def lip_score(ph, phon, dur, start):
    """Agreement between the candidate's visemes, spoken from START, and
    the measured lip opening: (correlation, mismatch share)."""
    fps = ph.fps
    a = int(np.floor((min(ph.start, start) - 0.1) * fps))
    b = int(np.ceil((max(ph.end, start + dur) + 0.1) * fps))
    m = np.full(b - a, np.nan)
    lo = max(a, ph.frame0)
    hi = min(b, ph.frame0 + len(ph.mouth))
    if hi > lo:
        m[lo - a:hi - a] = ph.mouth[lo - ph.frame0:hi - ph.frame0]
    v = np.full(b - a, REST)
    off = int(round((start - a / fps) * fps))
    n = int(np.ceil(dur * fps)) + 1
    vis = visemes(phon, fps, n)
    s0, s1 = max(off, 0), min(off + n, b - a)
    if s1 > s0:
        v[s0:s1] = vis[s0 - off:s1 - off]
    ok = ~np.isnan(m)
    if ok.sum() < 6 or np.std(m[ok]) < 1e-3 or np.std(v[ok]) < 1e-3:
        return 0.0, 0.0
    corr = float(np.corrcoef(m[ok], v[ok])[0, 1])
    miss = float(np.mean(np.abs(m[ok] - v[ok]) > 0.6))
    return corr, miss


def evaluate(ph, text, audio, phon, prev_end, rank):
    """Measure one candidate and place it; returns its record."""
    dur = len(audio) / RATE
    room_end = ph.next_start - GAP
    start = ph.start
    corr = miss = 0.0
    if ph.lip_weight > 0:
        best = None
        for off in np.arange(-SHIFT, SHIFT + 1e-6, 1 / ph.fps):
            s = ph.start + off
            if s < prev_end + GAP / 2:
                continue
            c, mi = lip_score(ph, phon, dur, s)
            if best is None or c - 0.5 * mi > best[0]:
                best = (c - 0.5 * mi, s, c, mi)
        if best:
            _, start, corr, miss = best
    if start + dur > room_end and ph.lip_weight < NEED_LIPS:
        # off screen: the voice may come in a little early, as long as the
        # previous line is over
        start = min(max(prev_end + GAP, ph.start - EARLY, room_end - dur),
                    ph.start)
    over = start + dur - room_end
    rel = (dur - ph.target) / ph.target
    if over > 0.02:
        cost = 5.0 + over / ph.target
    elif rel > 0:
        cost = 2.0 * max(0.0, rel - TOL)
    else:
        cost = 1.5 * max(0.0, -rel - 1.5 * TOL)
    lip_cost = (1 - corr) / 2 + 0.5 * miss
    cost += ph.lip_weight * LIP_W * lip_cost + 0.04 * rank + \
        LOSS_W * (1 - getattr(ph, "full", {}).get(text, 1.0))
    return dict(text=text, dur=round(dur, 3), start=round(float(start), 3),
                over=round(float(max(over, 0)), 3), corr=round(corr, 3),
                miss=round(miss, 3), cost=round(float(cost), 4), rank=rank,
                syl=syllables(text))


# ------------------------------------------------------------ adaptation

def keeps_facts(cand, base):
    """A variant must keep the base's numbers and names."""
    num = set(re.findall(r"\d+", base))
    if num - set(re.findall(r"\d+", cand)):
        return False
    names = set(w for w in re.findall(r"(?<=[^.!?]\s)([A-Z][\w-]+)", base))
    low = cand.lower()
    return all(n.lower() in low for n in names)


# what a dubbing adapter cuts first: discourse markers and fillers (whole
# comma-delimited units, or opening the sentence) and, in pro-drop
# languages, subject pronouns opening a clause. None of these carries a
# fact, so the cut cannot make the translation wrong.
MARKERS = {
    "pt": ["eu quero dizer", "quero dizer", "quer dizer", "você sabe",
           "sabe", "na verdade", "de fato", "tipo", "então", "bem", "bom",
           "olha", "veja", "sim", "ah", "oh", "né", "ok"],
}
OPENERS = {"pt": ["então", "bem", "olha", "ah", "oh", "bom"]}
LOOSE = {"pt": ["literalmente"]}            # removable anywhere
SUBJECTS = {"pt": ["eu", "nós", "eles", "elas"]}


def trims(text, lang):
    """Every version of TEXT with some of its fillers removed."""
    markers = MARKERS.get(lang)
    if not markers:
        return []
    ops = []        # (start, end) character spans that may go
    for m in re.finditer(r"(?:^|(?<=[,.;:!?]))\s*([^,.;:!?]+?)\s*[,.;:!?]", text):
        if m.group(1).strip().lower() in markers:
            ops.append((m.start(1), m.end()))
    words = OPENERS[lang]
    m = re.match(r"\s*(" + "|".join(words) + r")\b\s+(?=\w)", text, re.I)
    if m and not any(a <= m.start(1) < b for a, b in ops):
        ops.append((m.start(1), m.end()))
    for w in LOOSE[lang]:
        for m in re.finditer(r"\b" + w + r"\b\s*", text, re.I):
            ops.append((m.start(), m.end()))
    subj = "|".join(SUBJECTS[lang])
    for m in re.finditer(r"(?:^|[,.;!?]\s*|\b(?:e|mas|que|porque|se)\s+)"
                         r"(" + subj + r")\s+(?=\w)", text, re.I):
        ops.append((m.start(1), m.end()))
    ops = sorted(set(ops))[:5]
    out = []
    for mask in range(1, 1 << len(ops)):
        pick = [ops[k] for k in range(len(ops)) if mask >> k & 1]
        if any(pick[k][1] > pick[k + 1][0] for k in range(len(pick) - 1)):
            continue            # overlapping cuts ("eu" in "eu quero dizer")
        t = text
        for k in reversed(range(len(ops))):
            if mask >> k & 1:
                a, b = ops[k]
                t = t[:a] + t[b:]
        t = re.sub(r"\s+([,.;:!?])", r"\1", t)
        t = re.sub(r"^[\s,.;:]+|(?<=[,.;:!?])[,;:]+", "", t)
        t = re.sub(r"\s{2,}", " ", t).strip()
        if re.search(r"\w", t):
            out.append(t[0].upper() + t[1:])
    return list(dict.fromkeys(o for o in out if o != text))


# words a rewrite may add although the first translation does not have
# them: pure grammar. Even short words such as "sim", "bem", "isso", "já"
# carry meaning; a 4B adds them freely ("Are there still?" -> "Sim,
# ainda?"), so they are not on the list.
GRAMMAR = {
    "pt": set("a o as os um uma uns umas de do da dos das em no na nos nas "
              "num numa por pelo pela pelos pelas para pra com e ou que se "
              "ao à aos às".split()),
}


def reuses_words(cand, base, lang):
    """The rewrite only deletes, reorders or re-inflects the first
    translation's words (plus grammar words). Measured on this pipeline's
    own accepted rewrites: with new words allowed, about 1 in 4 changed the
    meaning in ways neither NLI nor LaBSE noticed ("Tallarico plus three",
    a poll lead, -> "Tallarico somou três")."""
    free = GRAMMAR.get(lang, set())
    words = re.findall(r"\w+", base.lower())

    def known(w):
        return w in free or any(
            w == b or (len(w) >= 5 and len(b) >= 5 and w[:5] == b[:5])
            for b in words)
    return all(known(w) for w in re.findall(r"\w+", cand.lower()))


def ask_variants(get_llm, ph, phrases, done, src, tgt, cache, feedback=None):
    s, t = T.NAMES.get(src, src), T.NAMES.get(tgt, tgt)
    best = ph.best
    want = max(2, round(best["syl"] * ph.target / max(best["dur"], 0.1)))
    prev = [done[k] for k in range(max(0, ph.i - 2), ph.i) if done[k]]
    nxt = phrases[ph.i + 1].src if ph.i + 1 < len(phrases) else ""
    hint = os.environ.get("DUBBER_PROMPT", "").strip()
    lines = [
        f"You are the adapter of a lip-sync {t} dub of a {s} video"
        + (f" ({hint})" if hint else "") + ".",
        "Previous dubbed lines: " + " / ".join(prev) if prev else "",
        f"{s} line: {ph.src}",
        f"Next {s} line (context only): {nxt}" if nxt else "",
        f"Current {t} line: {best['text']}",
    ]
    if feedback:
        lines.append(
            f"Spoken, the current line lasts {best['dur']:.1f} s but the "
            f"speaker talks for only {ph.target:.1f} s"
            if best["dur"] > ph.target else
            f"Spoken, the current line lasts {best['dur']:.1f} s but the "
            f"speaker talks for {ph.target:.1f} s")
        lines.append(
            f"Rewrite it with about {want} syllables (it has {best['syl']}). "
            + ("Condense it the way a dubbing adapter does: drop fillers, "
               "repetitions, hedges and discourse markers (\"I mean\", "
               "\"as you know\", \"you know\"), use shorter synonyms and "
               "tighter syntax. Never change or add a fact; keep every "
               "claim, name and number." if want < best["syl"] else
               "Use fuller, natural phrasing; do not add new facts."))
    else:
        lines.append(
            f"Write {VARIANTS} alternative {t} versions with about {want} "
            f"syllables each (the current one has {best['syl']}).")
        if want < best["syl"]:
            lines.append(
                "You may drop fillers, repetitions and hedges to get there, "
                "but never change or add a fact.")
    if ph.lip_weight >= NEED_LIPS:
        closes, ends_open = ph.lip_events()
        if closes:
            pts = ", ".join(f"{100 * c:.0f}%" for c in closes)
            lines.append(
                f"The speaker is on camera. Their lips close completely at "
                f"about {pts} of the line; where it is natural, put a word "
                f"with p, b or m near those points.")
        lines.append(
            "The line ends with the mouth "
            + ("open: prefer ending on an open vowel (a, e, o)." if ends_open
               else "closed or nearly closed."))
    n = 3 if feedback else VARIANTS
    lines.append(
        f"Use only the words of the current line: you may delete words, "
        f"reorder them and change their form (tense, number), but do not "
        f"add new words. Keep the meaning, names and numbers; natural spoken "
        f"{t}; no explanations. Answer with {n} versions numbered 1. to "
        f"{n}.")
    prompt = "\n".join(l for l in lines if l)
    if prompt not in cache:
        ok = lambda x: T.plausible(x, src, tgt) and keeps_facts(x, ph.base)
        out = get_llm().numbered(prompt, n, 120 * n, ok)
        cache[prompt] = [T.strip_foreign(T.clean(x), tgt) for x in out if x]
    return cache[prompt]


def merge(P, idx, pad, hop, fps):
    """One scene entry for the consecutive phrases P[idx] of one speaker:
    their pitch, loudness and mouth curves on one grid."""
    ps = [P[i] for i in idx]
    a, b = ps[0]["start"], ps[-1]["end"]
    g0 = max(0.0, a - pad)
    n = int(round((b - g0) / hop))
    f0, db = [None] * n, [-60.0] * n
    for p in ps:
        k0 = int(round((max(0.0, p["start"] - pad) - g0) / hop))
        for k, (f, d) in enumerate(zip(p["f0"], p["db"])):
            if 0 <= k0 + k < n:
                f0[k0 + k], db[k0 + k] = f, d
    m0 = int((a - pad) * fps) if any(p["mouth"] for p in ps) else 0
    m0 = max(m0, 0)
    mouth = None
    seen = [p for p in ps if p["mouth"]]
    if seen:
        mouth = [None] * (int(b * fps) + 2 - m0)
        for p in seen:
            for k, v in enumerate(p["mouth"]):
                if 0 <= p["mouth_frame0"] + k - m0 < len(mouth) and v is not None:
                    mouth[p["mouth_frame0"] + k - m0] = v
    dur = sum(p["end"] - p["start"] for p in ps)
    lw = sum(p["lip_weight"] * (p["end"] - p["start"]) for p in ps) / dur
    return dict(start=a, end=b, speaker=ps[0]["speaker"],
                gender=ps[0]["gender"], f0=f0, db=db, mouth=mouth,
                mouth_frame0=m0, lip_weight=lw if seen else 0.0)


def units_of(P, mode):
    """Index groups: one phrase each, or (block mode) one speaker's
    utterance, so that Whisper's arbitrary phrase cuts do not split a
    sentence and its inner pauses can be shared."""
    if mode != "block":
        return [[i] for i in range(len(P))]
    out = []
    for i, p in enumerate(P):
        if out:
            q = P[out[-1][-1]]
            if p["speaker"] == q["speaker"] and \
                    p["start"] - q["end"] < BLOCK_GAP and \
                    p["end"] - P[out[-1][0]]["start"] <= BLOCK_MAX:
                out[-1].append(i)
                continue
        out.append([i])
    return out


def ask_iso(get_llm, ph, units, src, tgt, cache, rate):
    """Isometric translation (Lakew et al. 2022): ISO_N translations of the
    source written to be spoken in the speaker's own time."""
    s, t = T.NAMES.get(src, src), T.NAMES.get(tgt, tgt)
    want = max(2, round(ph.target * rate * 0.95))
    prev = units[ph.k - 1].src if ph.k > 0 else ""
    nxt = units[ph.k + 1].src if ph.k + 1 < len(units) else ""
    hint = os.environ.get("DUBBER_PROMPT", "").strip()
    prompt = "\n".join(l for l in [
        f"You translate a {s} TV programme"
        + (f" ({hint})" if hint else "") + f" into {t} for dubbing.",
        f"Before (context only): {prev}" if prev else "",
        f"Line to translate: {ph.src}",
        f"After (context only): {nxt}" if nxt else "",
        f"A reference translation: {ph.base}",
        f"The speaker says this line in {ph.target:.1f} s, so the {t} must be "
        f"speakable in that time: about {want} syllables "
        f"(the reference has {syllables(ph.base)}).",
        "Keep every fact, claim, name and number. You may drop fillers "
        "(\"you know\", \"I mean\"), repetitions and hesitations, and use "
        f"concise, natural spoken {t}.",
        f"Write {ISO_N} different versions, numbered 1. to {ISO_N}, nothing "
        "else."] if l)
    if prompt not in cache:
        ok = lambda x: T.plausible(x, src, tgt) and keeps_facts(x, ph.base)
        out = get_llm().numbered(prompt, ISO_N, 140 * ISO_N, ok)
        cache[prompt] = [T.strip_foreign(T.clean(x), tgt) for x in out if x]
    return cache[prompt]


class Meaning:
    """LaBSE (Feng et al. 2022): cross-lingual sentence embeddings made for
    scoring translation pairs. Runs on the CPU, next to the LLM."""

    def __init__(self):
        import torch
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer
        d = snapshot_download(LABSE, allow_patterns=LABSE_FILES)
        self.tok = AutoTokenizer.from_pretrained(d)
        self.m = AutoModel.from_pretrained(d).eval()
        w = load_file(os.path.join(d, "2_Dense", "model.safetensors"))
        self.w, self.b = w["linear.weight"], w["linear.bias"]
        self.torch = torch
        self.cache = {}

    def emb(self, texts):
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for k in range(0, len(todo), 64):
            b = self.tok(todo[k:k + 64], padding=True, truncation=True,
                         return_tensors="pt")
            with self.torch.no_grad():
                h = self.m(**b).last_hidden_state[:, 0]   # CLS pooling
                h = self.torch.tanh(h @ self.w.T + self.b)
                h = self.torch.nn.functional.normalize(h, dim=-1)
            for t, v in zip(todo[k:k + 64], h.numpy()):
                self.cache[t] = v
        return np.array([self.cache[t] for t in texts])

    def sim(self, src, texts):
        return self.emb(texts) @ self.emb([src])[0]


class Entail:
    """Multilingual NLI (mDeBERTa-v3, XNLI + 2.7M pairs). Two sentences mean
    the same when each entails the other. Trained on contradictions and
    negations, it rejects the subtly wrong rewrites a 4B LLM makes and does
    not notice itself as a judge (tested: 47% of deliberately falsified
    lines passed the LLM, ~2% pass this)."""

    def __init__(self, dev):
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
        self.torch, self.dev = torch, dev
        self.tok = AutoTokenizer.from_pretrained(NLI)
        self.m = AutoModelForSequenceClassification.from_pretrained(NLI) \
            .to(dev).eval()

    def probs(self, prem, hyp):
        """[P(entailment), P(neutral), P(contradiction)] per pair."""
        out = []
        for k in range(0, len(prem), 32):
            b = self.tok(prem[k:k + 32], hyp[k:k + 32], return_tensors="pt",
                         padding=True, truncation=True).to(self.dev)
            with self.torch.no_grad():
                out.append(self.torch.softmax(self.m(**b).logits, -1).cpu())
        return np.concatenate([o.numpy() for o in out])

    def free(self):
        del self.m
        if self.dev == "cuda":
            self.torch.cuda.empty_cache()


def synth_all(spk, todo, voice_of):
    """todo: [(phrase, text)] -> {(i, text): (audio, phon)}"""
    res = {}
    t0 = time.time()
    for k, (ph, text) in enumerate(todo):
        res[(ph.i, text)] = spk.say(text, voice_of[ph.speaker])
        if (k + 1) % 50 == 0:
            log(f"spoke {k + 1}/{len(todo)} ({time.time() - t0:.0f} s)")
    return res


def choose(phrases, audio, prev_end_of):
    """Re-measure every phrase's candidates in order (placement depends on
    where the previous line ended) and keep the cheapest."""
    prev_end = 0.0
    for ph in phrases:
        if not ph.cands:
            continue
        recs = []
        for text, rank in ph.cands.items():
            a, phon = audio[(ph.i, text)]
            if len(a) == 0:
                continue
            recs.append(evaluate(ph, text, a, phon, prev_end, rank))
        if not recs:
            continue
        ph.best = min(recs, key=lambda r: r["cost"])
        ph.recs = recs
        prev_end = ph.best["start"] + ph.best["dur"]
        prev_end_of[ph.i] = prev_end


# -------------------------------------------------------------- timeline

def schedule(phrases, total):
    """Place every chosen clip, as spoken, on the timeline: a linear program
    (HiGHS). Clips never overlap, and a natural pause separates them.

    Whisper often splits one person's continuous speech into several
    phrases; cutting such a phrase short and starting the same voice again
    sounds like a broken recording. So phrases of one speaker closer than
    TURN_GAP form a turn: only the turn's first line is held to the lips
    (on camera within SHIFT_ON); later lines follow the speech already
    under way and may come later (INNER_ON / INNER_OFF), since the speaker's
    mouth is moving all through the turn anyway. Cutting is cheapest at the
    end of a turn and nearly forbidden inside one. Off camera, lines may also
    slide into pauses. Sets ph.at and ph.cut (seconds that do not fit)."""
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix
    ps = [ph for ph in phrases if ph.best]
    n = len(ps)
    if not n:
        return
    # variables: x (start), u, v (|x - s| = u + v), o (cut)
    X, U, V, O = 0, n, 2 * n, 3 * n
    c = np.zeros(4 * n)
    bounds = []
    cont = [k + 1 < n and ps[k + 1].speaker == ph.speaker and
            ps[k + 1].start - ph.end < TURN_GAP for k, ph in enumerate(ps)]
    for k, ph in enumerate(ps):
        lw = ph.lip_weight
        inner = k > 0 and cont[k - 1]
        if lw >= NEED_LIPS:
            lo, hi = -SHIFT_ON, INNER_ON if inner else SHIFT_ON
        elif lw > 0:
            lo, hi = -0.4, 1.0 if inner else 0.6
        else:
            lo, hi = -EARLY_OFF, INNER_OFF if inner else LATE_OFF
        s = ph.best["start"]
        bounds.append((max(0.0, s + lo), max(0.0, s + hi)))
        # inside a turn the timing is the voice's own: weigh it lightly
        c[U + k] = c[V + k] = (0.3 if inner else 1) * (1 + 5 * lw)
        c[O + k] = CUT_INSIDE if cont[k] else CUT_COST
    bounds += [(0, None)] * (2 * n)
    bounds += [(0, ph.best["dur"]) for ph in ps]
    A = lil_matrix((n - 1 + 2 * n + 1, 4 * n))
    b = np.zeros(A.shape[0])
    r = 0
    for k in range(n - 1):      # x_k + d_k - o_k + pause <= x_k+1
        A[r, X + k], A[r, O + k], A[r, X + k + 1] = 1, -1, -1
        pause = min(max(GAP_SAME, ps[k + 1].start - ps[k].end), GAP_MAX) \
            if cont[k] else GAP
        b[r] = -ps[k].best["dur"] - pause
        r += 1
    for k, ph in enumerate(ps):  # x - s = u - v, as two inequalities
        s = ph.best["start"]
        A[r, X + k], A[r, U + k], A[r, V + k] = 1, -1, 1
        b[r] = s
        r += 1
        A[r, X + k], A[r, U + k], A[r, V + k] = -1, 1, -1
        b[r] = -s
        r += 1
    A[r, X + n - 1], A[r, O + n - 1] = 1, -1   # the last ends in the video
    b[r] = total - ps[-1].best["dur"]
    res = linprog(c, A_ub=A.tocsr(), b_ub=b, bounds=bounds, method="highs")
    if res.status != 0:
        log(f"timeline: {res.message}; keeping the first placement")
        for ph in ps:
            ph.at, ph.cut = ph.best["start"], 0.0
        return
    for k, ph in enumerate(ps):
        ph.at = float(res.x[X + k])
        ph.cut = float(res.x[O + k]) if res.x[O + k] > 0.005 else 0.0
    moved = [ph.at - ph.start for ph in ps]
    cut = [ph.cut for ph in ps if ph.cut]
    inside = sum(1 for k, ph in enumerate(ps) if ph.cut and cont[k])
    log(f"timeline: {sum(cont)} lines continue their speaker's turn; lines "
        f"moved by {np.median(np.abs(moved)):.2f} s median (max "
        f"{max(moved):+.2f} / {min(moved):+.2f} s); {len(cut)} lines too "
        f"long, {sum(cut):.1f} s in all ({inside} inside a turn)")


NEGATION = re.compile(r"\b(não|nunca|nem|nenhum|nenhuma|nada|jamais)\b",
                      re.IGNORECASE)


def facts(text):
    """Numbers, names and negations: what a shortened line must not lose."""
    return set(re.findall(r"\d+", text)) | \
        set(re.findall(r"(?<=\s)[A-Z][\w-]+", " " + text)) | \
        set(w.lower() for w in NEGATION.findall(text))


def fit_endings(spk, phrases, voice_of, audio):
    """A line the timeline cannot fit is not chopped mid-word: it is spoken
    again, shorter, ending at a word boundary (at a clause boundary when
    possible) with a full stop, so the voice falls and finishes as a
    speaker does. Among the shortened versions of every accepted wording,
    the one keeping the most, and losing no number, name or negation where
    possible, wins."""
    fixed = 0
    for ph in phrases:
        if not ph.best or ph.cut <= 0.02:
            continue
        room = ph.best["dur"] - ph.cut
        keep = facts(ph.base)
        tries = []
        for text in ph.cands:
            words = text.split()
            dur = ph.best["dur"] if text == ph.best["text"] else \
                len(audio[(ph.i, text)][0]) / RATE
            syl = syllables(text)
            for j in range(len(words) - 1, 0, -1):
                pre = " ".join(words[:j]).rstrip(",;:—-–… ")
                est = dur * syllables(pre) / syl
                if est <= room * 1.15 and re.search(r"\w", pre):
                    clause = words[j - 1][-1] in ",;:.!?"
                    tries.append((pre.rstrip(".!?") + ".", clause))
        best = None
        tries = sorted(set(tries), key=lambda t: -syllables(t[0]))[:8]
        for pre, clause in tries:
            a, phon = spk.say(pre, voice_of[ph.speaker])
            d = len(a) / RATE
            if not len(a) or d > room + 0.03:
                continue
            lost = len(keep - facts(pre))
            score = syllables(pre) / syllables(ph.base) - LOST_W * lost + \
                (0.1 if clause else 0)
            if best is None or score > best[0]:
                best = (score, pre, a, phon, d)
        if best is None:
            # not even a few words fit: leave the line to the fade-out
            continue
        _, pre, a, phon, d = best
        audio[(ph.i, pre)] = (a, phon)
        ph.shortened = dict(full=ph.best["text"], kept=pre)
        ph.best = dict(ph.best, text=pre, dur=round(d, 3),
                       syl=syllables(pre), over=0.0)
        ph.cut = 0.0
        fixed += 1
    left = sum(1 for ph in phrases if ph.best and ph.cut > 0.02)
    log(f"{fixed} lines that did not fit end on a word, spoken again "
        f"shorter; {left} still fade out")


# ------------------------------------------------------------ intonation

def contour(st, step, width):
    """Slow part of a pitch contour (semitones, NaN unvoiced): unvoiced gaps
    interpolated, then a moving average WIDTH s wide."""
    ok = np.isfinite(st)
    if ok.sum() < 3:
        return None
    x = np.arange(len(st))
    y = np.interp(x, x[ok], st[ok])
    k = max(1, int(width / step))
    return np.convolve(np.pad(y, k, mode="edge"), np.ones(k) / k, "same")[k:-k]


def tts_pitch(audio):
    import parselmouth
    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=RATE)
    p = snd.to_pitch_ac(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
    hz = p.selected_array["frequency"]
    st = np.where(hz > 0, 12 * np.log2(np.maximum(hz, 1) / 100), np.nan)
    return p.xs(), st


def intonate(audio, ph, spk_mean, voice_mean, hop, pad):
    """Lay the speaker's intonation on the clip (PSOLA, same duration)."""
    import parselmouth
    from parselmouth.praat import call
    times, st = tts_pitch(audio)
    voiced = np.isfinite(st)
    if voiced.sum() < 5:
        return audio, None
    # original contour over the spoken part of the phrase
    k0 = int(round(pad / hop))
    orig = ph.f0[k0:]
    oslow = contour(orig, hop, PITCH_SMOOTH)
    if oslow is None:
        return audio, None
    tslow = contour(st, 0.01, PITCH_SMOOTH)
    # time-proportional map: the clip's speech onto the original's
    rel = (times - times[0]) / max(times[-1] - times[0], 1e-3)
    o_at = np.interp(rel, np.linspace(0, 1, len(oslow)), oslow)
    target = voice_mean + np.clip(o_at - spk_mean, -PITCH_RANGE, PITCH_RANGE)
    new = st + STRENGTH * (target - tslow)
    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=RATE)
    manip = call(snd, "To Manipulation", 0.01, 75, 600)
    tier = call("Create PitchTier", "dub", 0, snd.duration)
    for t, v in zip(times[voiced], new[voiced]):
        call(tier, "Add point", float(t), float(100 * 2 ** (v / 12)))
    call([tier, manip], "Replace pitch tier")
    out = call(manip, "Get resynthesis (overlap-add)")
    y = out.values[0].astype(np.float32)[:len(audio)]
    if len(y) < len(audio):
        y = np.pad(y, (0, len(audio) - len(y)))
    # how well the clip now follows the original's melody
    t2, st2 = tts_pitch(y)
    s2 = contour(st2, 0.01, PITCH_SMOOTH)
    fit = None
    if s2 is not None and np.std(o_at) > 0:
        fit = (float(np.corrcoef(tslow, o_at)[0, 1]),
               float(np.corrcoef(np.interp(times, t2, s2), o_at)[0, 1]))
    return y, fit


def loudness(ph, pad, hop):
    db = ph.db[int(round(pad / hop)):]
    db = db[db > db.max() - 30] if len(db) else db
    return float(np.mean(db)) if len(db) else None


# ------------------------------------------------------------------ main

def srt_time(t):
    ms = int(round(max(t, 0) * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}," \
           f"{ms % 1000:03d}"


def write_wav(path, x):
    import soundfile as sf
    sf.write(path, np.clip(x, -1, 1), RATE, subtype="PCM_16")


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    wd, src, tgt = sys.argv[1:4]
    if tgt not in KOKORO:
        sys.exit(f"lipdub.py: no Kokoro voice for '{tgt}'")
    with open(os.path.join(wd, "scene.json")) as f:
        scene = json.load(f)
    with open(os.path.join(wd, "src.txt"), encoding="utf-8") as f:
        srcl = [l.rstrip("\n") for l in f]
    with open(os.path.join(wd, f"{tgt}.txt"), encoding="utf-8") as f:
        base = [l.rstrip("\n") for l in f]
    P = scene["phrases"]
    if not (len(P) == len(srcl) == len(base)):
        sys.exit("lipdub.py: scene.json, src.txt and the translation differ "
                 "in length")
    import soundfile as sf
    total = sf.info(os.path.join(wd, "vocals.wav")).duration
    fps, hop, pad = scene["fps"] or 25.0, scene["hop"], scene["pad"]
    adapt = os.environ.get("DUBBER_ADAPT", "trim")
    groups = units_of(P, adapt)
    phrases = []    # translation units: phrases, or utterances (block mode)
    for k, idx in enumerate(groups):
        nxt = P[groups[k + 1][0]]["start"] if k + 1 < len(groups) else total
        p = P[idx[0]] if len(idx) == 1 else merge(P, idx, scene["pad"],
                                                  scene["hop"], scene["fps"])
        ph = Phrase(k, p, " ".join(srcl[i] for i in idx).strip(),
                    " ".join(base[i] for i in idx if base[i].strip()).strip(),
                    nxt, fps)
        ph.k, ph.members = k, idx
        phrases.append(ph)
    if adapt == "block":
        log(f"{len(P)} phrases joined into {len(phrases)} utterances")

    # one voice per speaker, by sex, most talkative first
    talk = {}
    for ph in phrases:
        talk[ph.speaker] = talk.get(ph.speaker, 0) + ph.target
    voices = KOKORO[tgt][1]
    used = {"f": 0, "m": 0}
    voice_of = {}
    for k in sorted(talk, key=lambda k: -talk[k]):
        g = scene["speakers"][str(k)]
        voice_of[k] = voices[g][used[g] % len(voices[g])]
        used[g] += 1
        log(f"speaker {k} ({g}, {talk[k]:.0f} s) -> voice {voice_of[k]}")

    import torch
    dev = os.environ.get("DUBBER_DEVICE", "auto")
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    path = os.path.expanduser(os.environ.get("DUBBER_LLM", T.DEFAULT_LLM))
    use_llm = os.path.exists(path) and \
        os.environ.get("DUBBER_TRANSLATOR", "auto") != "nllb"

    live = [ph for ph in phrases if ph.base.strip() and
            re.search(r"\w", ph.base)]
    meaning = Meaning() if use_llm else None
    for ph in live:
        ph.cands[ph.base] = 0
        ph.wrong = {}
        ph.base_sim = float(meaning.sim(ph.src, [ph.base])[0]) if meaning \
            else 1.0
        ph.sim = {ph.base: ph.base_sim}
        ph.full = {ph.base: 1.0}    # share of the first translation's meaning
    # the LLM's rewrites, kept between runs (same prompt, same answer)
    cache_path = os.path.join(wd, f"lipdub_{tgt}_llm.json")
    try:
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    except (OSError, ValueError):
        cache = {}
    # the adapter writes target-language rewrites: the LoRA distilled for
    # source -> English translation is not meant for that
    os.environ["DUBBER_LORA"] = "none"

    t0 = time.time()
    spk = Speaker(tgt, dev)
    audio = synth_all(spk, [(ph, ph.base) for ph in live], voice_of)
    ends = {}
    choose(live, audio, ends)
    base_rec = {ph.i: dict(ph.best) for ph in live if ph.best}
    log(f"first translation spoken in {time.time() - t0:.0f} s")

    def needy(ph, strict):
        b = ph.best
        if b is None:
            return False
        if b["over"] > 0.02:
            return True
        err = abs(b["dur"] - ph.target)
        rel = err / ph.target
        return (rel > (0.25 if strict else 0.12) and err > MIN_ERR) or \
            (not strict and ph.lip_weight >= NEED_LIPS)

    # default: only cut what cannot carry a fact (fillers, subject
    # pronouns). LLM rewriting is opt-in: checked on this pipeline's own
    # output, about 1 in 5 of its accepted rewrites was still wrong
    # (e.g. deleting "first ... since" from "first senator since 1988").
    todo = [ph for ph in live if needy(ph, False)]
    offers = [(ph, k + 1, v) for ph in todo
              for k, v in enumerate(trims(ph.base, tgt))]
    if offers:
        nli = Entail(dev)
        fwd = nli.probs([ph.base for ph, _, _ in offers],
                        [v for _, _, v in offers])
        back = nli.probs([v for _, _, v in offers],
                         [ph.base for ph, _, _ in offers])
        nli.free()
        new = []
        for (ph, k, v), f, b in zip(offers, fwd, back):
            if f[0] >= SAME and max(f[2], b[2]) < CONTRA:
                ph.cands[v] = k
                ph.sim[v] = ph.base_sim
                new.append((ph, v))
            else:
                ph.wrong[v] = dict(entails=round(float(f[0]), 3),
                                   contra=round(float(max(f[2], b[2])), 3))
        log(f"trimming fillers: {len(new)} shorter versions of "
            f"{len({ph.i for ph, _ in new})} lines "
            f"({len(offers) - len(new)} rejected by the NLI guard)")
        audio.update(synth_all(spk, new, voice_of))
        choose(live, audio, ends)

    if use_llm and adapt in ("iso", "block"):
        # the voice's own pace, measured on what it has spoken so far
        rate = {}
        for v in set(voice_of.values()):
            mine = [ph.best for ph in live
                    if ph.best and voice_of[ph.speaker] == v]
            rate[v] = sum(b["syl"] for b in mine) / \
                max(sum(b["dur"] for b in mine), 1e-3) if mine else 5.0
        todo = [ph for ph in live if needy(ph, False)]
        log(f"isochronous translation of {len(todo)} "
            f"{'utterances' if adapt == 'block' else 'lines'}")
        spk.free()
        box = []

        def get_llm():
            if not box:
                box.append(T.LLM(path))
            return box[0]
        offers, t1 = [], time.time()
        for k, ph in enumerate(todo):
            vs = ask_iso(get_llm, ph, phrases, src, tgt, cache,
                         rate[voice_of[ph.speaker]])
            offers += [(ph, j + 1, v) for j, v in enumerate(vs)
                       if v and v not in ph.cands]
            if (k + 1) % 20 == 0:
                log(f"asked {k + 1}/{len(todo)} ({time.time() - t1:.0f} s)")
        if box:
            box[0].m.close()
            box.clear()
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        new = []
        if offers:
            nli = Entail(dev)
            bases = [ph.base for ph, _, _ in offers]
            vs = [v for _, _, v in offers]
            fwd, back = nli.probs(bases, vs), nli.probs(vs, bases)
            nli.free()
            for (ph, j, v), f, b in zip(offers, fwd, back):
                sim = float(meaning.sim(ph.src, [v])[0])
                same = min(f[0], b[0]) >= SAME
                condensed = f[0] >= SAME and max(f[2], b[2]) < CONTRA and \
                    sim >= ph.base_sim - COND_DROP
                if same or condensed:
                    ph.cands[v] = j
                    ph.sim[v] = sim
                    ph.full[v] = 1.0 if same else float(b[0])
                    new.append((ph, v))
                else:
                    ph.wrong[v] = dict(entails=round(float(f[0]), 3),
                                       entailed=round(float(b[0]), 3),
                                       contra=round(float(max(f[2], b[2])), 3),
                                       sim=round(sim, 3))
        log(f"{len(offers)} isochronous translations, {len(offers) - len(new)} "
            f"rejected by the meaning checks")
        spk = Speaker(tgt, dev)
        audio.update(synth_all(spk, new, voice_of))
        choose(live, audio, ends)

    for rnd in (1, 2) if use_llm and adapt == "llm" else ():
        todo = [ph for ph in live if needy(ph, rnd == 2)]
        if not todo:
            break
        log(f"round {rnd}: adapting {len(todo)} lines "
            f"({sum(ph.best['over'] > 0.02 for ph in todo)} too long for "
            f"their time, {sum(ph.lip_weight >= NEED_LIPS for ph in todo)} "
            f"on camera)")
        spk.free()
        done = {ph.i: ph.best["text"] if ph.best else "" for ph in phrases}
        offers = []
        t1 = time.time()
        box = []                # the LLM, loaded once a prompt is new

        def get_llm():
            if not box:
                box.append(T.LLM(path))
            return box[0]
        for k, ph in enumerate(todo):
            vs = ask_variants(get_llm, ph, phrases, done, src, tgt, cache,
                              feedback=rnd == 2)
            offers += [(ph, j, v) for j, v in enumerate(vs)
                       if v and v not in ph.cands and v not in ph.wrong]
            if (k + 1) % 20 == 0:
                log(f"asked {k + 1}/{len(todo)} ({time.time() - t1:.0f} s)")
        if box:
            box[0].m.close()    # llama.cpp keeps its VRAM until closed
            box.clear()
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        # meaning check, after the LLM is gone (both do not fit 4 GB)
        new, rejected = [], 0
        foreign = [(ph, j, v) for ph, j, v in offers
                   if not reuses_words(v, ph.base, tgt)]
        for ph, _, v in foreign:
            ph.wrong[v] = dict(new_words=True)
        offers = [o for o in offers if reuses_words(o[2], o[0].base, tgt)]
        if offers:
            nli = Entail(dev)
            bases = [ph.base for ph, _, _ in offers]
            vs = [v for _, _, v in offers]
            fwd = nli.probs(bases, vs)      # first translation => rewrite
            back = nli.probs(vs, bases)     # rewrite => first translation
            nli.free()
            for (ph, j, v), f, b in zip(offers, fwd, back):
                sim = float(meaning.sim(ph.src, [v])[0])
                same = min(f[0], b[0]) >= SAME and sim >= min(
                    ph.base_sim, max(SIM_MIN, ph.base_sim - SIM_DROP))
                condensed = f[0] >= SAME and max(f[2], b[2]) < CONTRA and \
                    sim >= ph.base_sim - COND_DROP
                if same or condensed:
                    ph.cands[v] = j + 1
                    ph.sim[v] = sim
                    ph.full[v] = 1.0 if same else float(b[0])
                    new.append((ph, v))
                else:
                    ph.wrong[v] = dict(entails=round(float(f[0]), 3),
                                       entailed=round(float(b[0]), 3),
                                       contra=round(float(max(f[2], b[2])), 3),
                                       sim=round(sim, 3))
                    rejected += 1
        log(f"{len(foreign)} rewrites rejected for adding words; "
            f"{len(offers)} others, {rejected} rejected: not the same "
            f"meaning as the first translation; "
            f"{sum(ph.full.get(v, 1) < 1 for ph, v in new)} accepted as "
            f"condensations")
        spk = Speaker(tgt, dev)
        audio.update(synth_all(spk, new, voice_of))
        choose(live, audio, ends)

    schedule(phrases, total)
    if not hasattr(spk, "pipe"):
        spk = Speaker(tgt, dev)
    fit_endings(spk, phrases, voice_of, audio)

    # intonation: the speaker's melody on the voice, PSOLA keeps duration
    spk_mean = {}
    for k in talk:
        f = np.concatenate([ph.f0 for ph in phrases if ph.speaker == k])
        spk_mean[k] = float(np.nanmedian(f)) if np.isfinite(f).any() else 0.0
    spk_db = {k: np.median([v for v in (loudness(ph, pad, hop)
                                        for ph in phrases if ph.speaker == k)
                            if v is not None]) for k in talk}
    voice_mean = {}
    for v in set(voice_of.values()):
        meds = []
        for ph in live:
            if ph.best and voice_of[ph.speaker] == v:
                _, st = tts_pitch(audio[(ph.i, ph.best["text"])][0])
                if np.isfinite(st).any():
                    meds.append(np.nanmedian(st))
        voice_mean[v] = float(np.median(meds)) if meds else 0.0
    out_dir = os.path.join(wd, f"tts_{tgt}")
    os.makedirs(out_dir, exist_ok=True)
    fits, place, report = [], [], []
    for ph in phrases:
        b = ph.best
        if not b:
            continue
        a = audio[(ph.i, b["text"])][0]
        a, fit = intonate(a, ph, spk_mean[ph.speaker],
                          voice_mean[voice_of[ph.speaker]], hop, pad)
        if fit:
            fits.append(fit)
        lv = loudness(ph, pad, hop)
        if lv is not None:
            g = np.clip(lv - spk_db[ph.speaker], -GAIN_RANGE, GAIN_RANGE)
            a = a * 10 ** (g / 20)
        a = a / max(1.0, float(np.max(np.abs(a))) / 0.98)
        wav = os.path.join(out_dir, f"{ph.i:05d}.wav")
        write_wav(wav, a)
        place.append(f"{ph.at:.3f} {wav}")
        report.append(dict(i=ph.i, phrases=ph.members, start=ph.start,
                           end=ph.end,
                           speaker=ph.speaker, voice=voice_of[ph.speaker],
                           lip_weight=ph.lip_weight, src=ph.src,
                           base=base_rec.get(ph.i), chosen=b,
                           at=round(ph.at, 3), cut=round(ph.cut, 3),
                           meaning=round(ph.sim.get(b["text"], 0), 3),
                           complete=round(ph.full.get(b["text"], 1.0), 3),
                           base_meaning=round(ph.base_sim, 3),
                           candidates=getattr(ph, "recs", []),
                           rejected=ph.wrong,
                           shortened=getattr(ph, "shortened", None),
                           intonation=fit))

    lines = [""] * len(P)
    for ph in phrases:
        if ph.best:
            lines[ph.members[0]] = ph.best["text"]
    with open(os.path.join(wd, f"{tgt}_dub.txt"), "w", encoding="utf-8") as f:
        f.write("".join(l + "\n" for l in lines))
    with open(os.path.join(wd, f"{tgt}_dub.srt"), "w", encoding="utf-8") as f:
        n = 0
        for ph in phrases:
            if ph.best:
                n += 1
                end = ph.at + ph.best["dur"] - ph.cut
                f.write(f"{n}\n{srt_time(ph.at)} --> {srt_time(end)}\n"
                        f"{ph.best['text']}\n\n")
    early = sum(ph.best and ph.at < ph.start - 0.02 for ph in phrases)
    late = sum(ph.best and ph.at > ph.start + 0.02 for ph in phrases)
    log(f"{early} lines start early, {late} late (off camera, into pauses); "
        f"{sum(bool(ph.best and ph.cut) for ph in phrases)} cut short")
    with open(os.path.join(wd, f"place_{tgt}.txt"), "w") as f:
        f.write("\n".join(place) + "\n")
    with open(os.path.join(wd, f"lipsync_{tgt}.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    # what changed, in numbers
    def stats(get):
        r = [(x, get(x)) for x in report if get(x)]
        iso = np.median([abs(c["dur"] / (x["end"] - x["start"]) - 1)
                         for x, c in r]) if r else 0
        over = sum(c["over"] > 0.02 for _, c in r)
        lip = [c["corr"] for x, c in r if x["lip_weight"] >= NEED_LIPS]
        return iso, over, np.mean(lip) if lip else 0.0
    for name, get in (("first translation", lambda x: x["base"]),
                      ("adapted", lambda x: x["chosen"])):
        iso, over, lip = stats(get)
        log(f"{name:17s}: median duration error {100 * iso:4.1f}%, "
            f"{over} lines longer than their time, mean lip correlation "
            f"{lip:+.2f} (on-camera lines)")
    if fits:
        f = np.array(fits)
        log(f"intonation: correlation with the speaker's melody "
            f"{np.nanmean(f[:, 0]):+.2f} before, {np.nanmean(f[:, 1]):+.2f} "
            f"after ({len(f)} clips)")
    log(f"lip-sync dubbing took {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
