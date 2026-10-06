"""Reference-free ("general") mode. No champion recording is needed. Expectations come from three sources:
  a) SELF-NORMALISATION: local features (rate, level, F0 range, 2-8 kHz energy) against the speaker's own clip statistics;
  b) TRANSCRIPT RULES: pause norms by boundary type (linguistics.py), statement-final rise, ASR-vs-transcript for fillers / repeats / skips / swaps;
  c) POPULATION NORMS: per boundary type and per feature, the spread (median, MAD, quantiles) across clean speakers (engine/norms.json,
     built from the TRAIN speakers' clean takes only by `python -m engine.free --build-norms`).
The detectors return the same Cand objects as detect.py, so arbitration, severity, scoring and explanations are shared."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import asr, audio, features, reference
from .align import needleman_wunsch, sim
from .detect import Cand, _runs, silence_runs
from .linguistics import annotate
from .render_facts import syllables

NORMS_PATH = Path(__file__).parent / "norms.json"
POP_SPEAKERS = ["B01", "B02", "B04", "B06", "B07"]                  # TRAIN speakers only (never dev B03, test B05/B08, extra B09/B10)
FTH = {"pause_bad": 0.8, "pause_loose_mult": 1.6, "lost_frac": 0.25, "pace_k_fast": 2.0, "pace_k_slow": 2.0, "pace_min_s": 1.2, "pace_win": 2,
       "mono_ratio": 0.62, "mono_min_s": 2.4, "uptalk_st": 2.0, "fade_db": 3.5, "shout_db": 4.0, "slur_db": 4.0,
       "swap_sim": 0.8, "ins_p": 0.5, "filler_min_s": 0.1, "repeat_sim": 0.6, "skip_min_words": 1}
FILLERS = asr.FILLERS
_NORMS: dict | None = None


def norms() -> dict:
    global _NORMS
    if _NORMS is None:
        _NORMS = json.loads(NORMS_PATH.read_text()) if NORMS_PATH.exists() else {}
    return _NORMS


@dataclass
class FA:
    """Everything the detectors need about one participant clip."""
    bid: str
    words: list[dict]                 # transcript words with participant times ps/pe and alignment status
    bt: list[str]                     # boundary type after each word
    zipf_next: list
    fr: features.Frames
    sil: list[tuple[float, float]]
    hyp: list[dict]
    verb: list[dict]
    ins: list[dict] = field(default_factory=list)   # hypothesis words with no transcript partner
    dur: float = 0.0

    @property
    def n(self):
        return len(self.words)


def _zipf(ref_w, ling):
    from wordfreq import zipf_frequency
    n = len(ref_w)
    return [zipf_frequency(ref_w[min(k + 1, n - 1)]["clean"], "en") if ling["w"][min(k + 1, n - 1)]["content"] and len(ref_w[min(k + 1, n - 1)]["clean"]) > 4 else None
            for k in range(n)]


def analyse(x: np.ndarray, bid: str) -> FA:
    """ASR twice (unprompted + disfluency-prompted, cached), align the transcript to what was heard, extract frames and silences."""
    ref_w = reference.words_of(bid)
    ling = annotate([{"w": r["w"], "sent_end": r["sent_end"], "clause_end": r["clause_end"]} for r in ref_w])
    hyp = asr.words(x, False)
    verb = asr.words(x, True)
    refc = [w["clean"] for w in ref_w]
    path = needleman_wunsch(refc, [h["w"] for h in hyp])
    words = [{"i": i, "w": w["w"], "clean": w["clean"], "ps": None, "pe": None, "status": "skip", "hyp": None} for i, w in enumerate(ref_w)]
    ins = []
    for i, j in path:
        if i is not None and j is not None:
            words[i].update(ps=hyp[j]["start"], pe=hyp[j]["end"], hyp=j, status="ok", sim=sim(refc[i], hyp[j]["w"]))
        elif j is not None:
            ins.append({**hyp[j], "j": j, "after": None})
        # i with no j stays "skip"
    # times for skipped words: interpolate between matched neighbours
    last = 0.0
    for k, w in enumerate(words):
        if w["ps"] is None:
            nxt = next((words[m]["ps"] for m in range(k + 1, len(words)) if words[m]["ps"] is not None), last)
            w["ps"], w["pe"] = last, max(last, min(nxt, last + 0.01))
        else:
            last = w["pe"]
    # where each inserted hypothesis word sits: index of the transcript word before it
    for e in ins:
        prev = [w["i"] for w in words if w["status"] != "skip" and w["pe"] <= e["start"] + 0.02]
        e["after"] = prev[-1] if prev else -1
    fr = features.compute(x)
    bt = ling["b"][: len(words) - 1] + ["sentence"]
    return FA(bid=bid, words=words, bt=bt, zipf_next=_zipf(ref_w, ling), fr=fr, sil=silence_runs(fr, 0.1), hyp=hyp, verb=verb, ins=ins, dur=len(x) / audio.SR)


# ----------------------------------------------------------------------------- shared measurements
def _iqr(fr, a, b):
    v = fr.f0_st[fr.sl(a, b)]
    v = v[~np.isnan(v)]
    return float(np.percentile(v, 75) - np.percentile(v, 25)) if len(v) > 10 else np.nan


def pause_at(fa: FA, k: int) -> tuple[float, float, float]:
    """(length, start, end) of the longest silence at the boundary after word k (assigned to its nearest boundary)."""
    w0, w1 = fa.words[k], fa.words[min(k + 1, fa.n - 1)]
    a, b = w0["pe"], max(w1["ps"], w0["pe"])
    mid = (a + b) / 2
    hit = [r for r in fa.sil if r[1] >= a - 0.25 and r[0] <= b + 0.25 and abs((r[0] + r[1]) / 2 - mid) <= 0.6 + (b - a) / 2]
    if not hit:
        return 0.0, a, a
    r = max(hit, key=lambda r: r[1] - r[0])
    return r[1] - r[0], r[0], r[1]


def pauses(fa: FA) -> list[tuple[float, float, float]]:
    """pause_at for every boundary, with each silence given to ONE boundary only (the nearest by centre)."""
    mids = np.array([(fa.words[k]["pe"] + fa.words[min(k + 1, fa.n - 1)]["ps"]) / 2 for k in range(fa.n - 1)])
    best: dict[int, tuple[float, float, float]] = {}
    for r in fa.sil:
        if len(mids) == 0:
            break
        k = int(np.argmin(np.abs(mids - (r[0] + r[1]) / 2)))
        if abs(mids[k] - (r[0] + r[1]) / 2) <= 0.7:
            L = r[1] - r[0]
            if k not in best or L > best[k][0]:
                best[k] = (L, r[0], r[1])
    return [best.get(k, (0.0, mids[k], mids[k])) for k in range(fa.n - 1)]


def local_rates(fa: FA, half: int):
    """syllables per second over a window of 2*half+1 transcript words that were all heard (NaN otherwise)."""
    n = fa.n
    r = np.full(n, np.nan)
    for i in range(half, n - half):
        ws = fa.words[i - half: i + half + 1]
        if any(w["status"] == "skip" for w in ws):
            continue
        span = ws[-1]["pe"] - ws[0]["ps"]
        if span > 0.3:
            r[i] = sum(syllables(w["clean"]) for w in ws) / span
    return r


# ----------------------------------------------------------------------------- detectors
def d_pauses(fa: FA, th) -> list[Cand]:
    out = []
    P = pauses(fa)
    for k in range(fa.n - 1):
        L, a, b = P[k]
        if a < 0.4 or b > fa.dur - 0.3:
            continue
        t = fa.bt[k]
        if t in ("phrase_tight", "phrase_loose"):
            lim = th["pause_bad"] * (1.0 if t == "phrase_tight" else th["pause_loose_mult"])
            if L >= lim:
                rare = fa.zipf_next[k] is not None and fa.zipf_next[k] < 3.5
                out.append(Cand("PAUSE_BAD", a, b, L, k, k + 1, {"boundary": t, "before_rare_word": bool(rare)}))
        elif t == "sentence" and k < fa.n - 2:
            med = norms().get("pause", {}).get("sentence", {}).get("median", 0.5)
            if L < th["lost_frac"] * med:
                mid = (fa.words[k]["pe"] + fa.words[k + 1]["ps"]) / 2
                out.append(Cand("PAUSE_LOST", mid - 0.05, mid + 0.05, max(med - L, 0.0), k, k + 1, {"boundary": t, "typical_s": med}))
    return out


FUNC_WORDS = {"the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or", "but", "is", "are", "was", "were", "be", "it", "its", "that", "this", "with", "as",
              "by", "from", "we", "he", "she", "they", "i", "you", "his", "her", "their", "our", "my"}


def _dur_feats(words, bt):
    return np.array([[syllables(w["clean"]), len(w["clean"]), w["clean"] in FUNC_WORDS, bt[i] in ("sentence", "clause_punct"), bt[i] == "sentence", 1.0]
                     for i, w in enumerate(words)], float)


def dur_resid(fa: FA) -> np.ndarray:
    """log(word duration) minus what the word's length and position predict (model fitted on clean speakers), centred on this clip's median.
    NaN for words that were not heard. Negative = spoken faster than the word calls for."""
    m = norms().get("dur_model")
    d = np.array([max(w["pe"] - w["ps"], 0.03) if w["status"] != "skip" else np.nan for w in fa.words])
    pred = _dur_feats(fa.words, fa.bt) @ np.array(m["coef"]) if m else 0.0
    r = np.log(d) - pred
    return r - np.nanmedian(r)


def d_pace(fa: FA, th) -> list[Cand]:
    half = int(th["pace_win"])
    r = dur_resid(fa)
    n = fa.n
    W = np.full(n, np.nan)
    for i in range(half, n - half):
        seg = r[i - half: i + half + 1]
        if not np.isnan(seg).any():
            W[i] = seg.mean()
    ok = ~np.isnan(W)
    if ok.sum() < 10:
        return []
    sd = max(1.4826 * float(np.median(np.abs(W[ok] - np.median(W[ok])))), 0.05)
    z = np.where(ok, W / sd, 0.0)
    out = []
    for fast in (True, False):
        flag = (z <= -th["pace_k_fast"]) if fast else (z >= th["pace_k_slow"])
        for i, j in _runs(flag):
            a, b = fa.words[max(0, i - half)]["ps"], fa.words[min(n - 1, j + half)]["pe"]
            if b - a >= th["pace_min_s"]:
                eff = float(np.exp(-np.median(W[i:j + 1]) if fast else np.median(W[i:j + 1])))
                out.append(Cand("PACE_FAST" if fast else "PACE_SLOW", a, b, eff, max(0, i - half), min(n - 1, j + half),
                                {"syll_per_s": round(sum(syllables(w["clean"]) for w in fa.words[max(0, i - half): min(n - 1, j + half) + 1]) / max(b - a, 1e-3), 2),
                                 "clip_median": round(sum(syllables(w["clean"]) for w in fa.words if w["status"] != "skip") / max(sum(w["pe"] - w["ps"] for w in fa.words if w["status"] != "skip"), 1e-3), 2)}))
    return out


def d_mono(fa: FA, th, win: int = 12) -> list[Cand]:
    ip = _iqr(fa.fr, 0, 1e9)
    if not ip > 0.5:
        return []
    ratio = np.full(fa.n, np.nan)
    flags = np.zeros(fa.n, bool)
    for i in range(0, max(1, fa.n - win + 1), 3):
        j = min(fa.n - 1, i + win - 1)
        w = _iqr(fa.fr, fa.words[i]["ps"], fa.words[j]["pe"])
        if not np.isnan(w):
            r = w / ip
            ratio[i:j + 1] = np.where(np.isnan(ratio[i:j + 1]), r, np.minimum(ratio[i:j + 1], r))
            if r <= th["mono_ratio"]:
                flags[i:j + 1] = True
    out = []
    for i, j in _runs(flags):
        a, b = fa.words[i]["ps"], fa.words[j]["pe"]
        if b - a >= th["mono_min_s"]:
            w = _iqr(fa.fr, a, b)
            out.append(Cand("MONOTONE", a, b, float(ip - w) if not np.isnan(w) else 0.0, i, j, {"iqr_here": round(w, 2), "iqr_clip": round(ip, 2)}))
    return out


def _sentences(fa: FA):
    out, s = [], 0
    for k in range(fa.n - 1):
        if fa.bt[k] == "sentence":
            out.append((s, k))
            s = k + 1
    if s < fa.n:
        out.append((s, fa.n - 1))
    return out


def final_rise(fa: FA, e: int) -> float:
    w = fa.words[e]
    sl = fa.fr.sl(w["ps"], w["pe"])
    f0, I = fa.fr.f0_st[sl], fa.fr.inten[sl]
    v = ~np.isnan(f0)
    if v.sum() < 6:
        return np.nan
    idx = np.where(v)[0]
    nuc = idx[np.argmax(I[idx])]
    return float(np.nanmedian(f0[idx[-5:]]) - f0[nuc])


def d_uptalk(fa: FA, th) -> list[Cand]:
    out = []
    pop = norms().get("final_rise", {}).get("median", -3.0)
    for s, e in _sentences(fa):
        if e - s < 4 or fa.words[e]["status"] == "skip":
            continue
        rise = final_rise(fa, e)
        if not np.isnan(rise) and rise - pop >= th["uptalk_st"]:
            i0 = e
            while i0 > s + 1 and (fa.words[e]["pe"] - fa.words[i0 - 1]["ps"]) < 1.0:
                i0 -= 1
            out.append(Cand("UPTALK", fa.words[i0]["ps"], fa.words[e]["pe"], float(rise - pop), i0, e, {"rise_st": round(rise, 2), "typical_st": round(pop, 2)}))
    return out


def d_fade(fa: FA, th) -> list[Cand]:
    out = []
    pop = norms().get("fade_db", {}).get("median", -1.0)
    for s, e in _sentences(fa):
        if e - s < 4 or fa.words[e]["status"] == "skip":
            continue
        i0 = e
        while i0 > s + 2 and (fa.words[e]["pe"] - fa.words[i0 - 1]["ps"]) < 1.4:
            i0 -= 1
        I = fa.fr.inten
        end = I[fa.fr.sl(fa.words[i0]["ps"], fa.words[e]["pe"])]
        rest = I[fa.fr.sl(fa.words[s]["ps"], fa.words[i0]["ps"])]
        if end.size < 5 or rest.size < 5:
            continue
        d = float(np.mean(end) - np.mean(rest)) - pop
        if d <= -th["fade_db"]:
            out.append(Cand("FADE", fa.words[i0]["ps"], fa.words[e]["pe"], float(-d), i0, e, {"typical_db": round(pop, 2)}))
    return out


def _win_series(fa: FA, fn, win: int):
    v = np.full(fa.n, np.nan)
    for i in range(fa.n):
        j = min(fa.n - 1, i + win - 1)
        v[i] = fn(fa.words[i]["ps"], fa.words[j]["pe"])
    return v


def d_shout(fa: FA, th, win: int = 6) -> list[Cand]:
    def lvl(a, b):
        I = fa.fr.inten[fa.fr.sl(a, b)]
        return float(np.percentile(I, 80)) if I.size > 5 else np.nan
    v = _win_series(fa, lvl, win)
    ok = ~np.isnan(v)
    if ok.sum() < 10:
        return []
    gain = np.where(ok, v - np.median(v[ok]), 0.0)
    out = []
    for i, j in _runs(gain >= th["shout_db"]):
        j = min(fa.n - 1, j + win - 1)
        a, b = fa.words[i]["ps"], fa.words[j]["pe"]
        if b - a >= 0.6:
            out.append(Cand("SHOUT", a, b, float(np.max(gain[i:j + 1])), i, j, {"re_clip_median_db": True}))
    return out


def d_slur(fa: FA, th, win: int = 7) -> list[Cand]:
    def top(a, b):
        h = fa.fr.hf[fa.fr.sl(a, b)]
        return float(np.mean(np.sort(h)[-max(1, int(0.3 * len(h))):])) if h.size > 8 else np.nan
    v = _win_series(fa, top, win)
    ok = ~np.isnan(v)
    if ok.sum() < 10:
        return []
    drop = np.where(ok, np.median(v[ok]) - v, 0.0)
    out = []
    for i, j in _runs(drop >= th["slur_db"]):
        j = min(fa.n - 1, j + win - 1)
        a, b = fa.words[i]["ps"], fa.words[j]["pe"]
        if b - a >= 1.0:
            out.append(Cand("SLUR", a, b, float(np.max(drop[i:j + 1])), i, j, {}))
    return out


def d_text(fa: FA, th) -> list[Cand]:
    """ASR against the transcript: skipped words, misread words, fillers, repeats."""
    out = []
    # skipped runs (a ref word with no hypothesis partner)
    k = 0
    while k < fa.n:
        if fa.words[k]["status"] == "skip":
            j = k
            while j + 1 < fa.n and fa.words[j + 1]["status"] == "skip":
                j += 1
            if 0 < k and j < fa.n - 1 and j - k + 1 >= th["skip_min_words"]:
                t = (fa.words[k - 1]["pe"] + fa.words[j + 1]["ps"]) / 2
                out.append(Cand("WORD_SKIP", t - 0.025, t + 0.025, float(j - k + 1), k, j, {"n_words": j - k + 1, "missing": " ".join(w["clean"] for w in fa.words[k:j + 1])}))
            k = j + 1
        else:
            k += 1
    # misread words: aligned but dissimilar
    for w in fa.words:
        if w["status"] == "ok" and w.get("sim", 1.0) < th["swap_sim"]:
            out.append(Cand("WORD_SWAP", w["ps"], max(w["pe"], w["ps"] + 0.1), 1.0, w["i"], w["i"], {"heard": fa.hyp[w["hyp"]]["w"] if w["hyp"] is not None else ""}))
    # insertions heard in the disfluency-prompted pass: fillers and repeated words
    refc = [w["clean"] for w in fa.words]
    seen = set()
    for e in fa.verb:
        if e["w"] in FILLERS and e["end"] - e["start"] >= th["filler_min_s"] and round(e["start"], 1) not in seen:
            seen.add(round(e["start"], 1))
            out.append(Cand("FILLER", e["start"], e["end"], e["end"] - e["start"], 0, 0, {"heard": e["w"]}))
    for e in fa.ins:
        if e["w"] in FILLERS:
            continue
        a = max(0, e["after"] - 3)
        near = refc[a: e["after"] + 5]
        if any(sim(e["w"], r) >= th["repeat_sim"] for r in near):
            out.append(Cand("REPEAT", e["start"], e["end"], e["end"] - e["start"], max(e["after"], 0), max(e["after"], 0), {"heard": e["w"]}))
    return out


INS_CLF: dict | None = None
INS_FEATS = ["len", "level", "f0sd", "voiced", "d_edge", "ctxL", "ctxR", "flux", "hf"]


def ins_candidates(fa: FA, minlen: float = 0.08):
    """Voiced stretches that no heard word explains (a hum, a drawn-out schwa, a restart), with the features that tell a filler from leftovers."""
    fr = fa.fr
    n = len(fr.inten)
    cov = np.zeros(n, bool)
    edges = []
    for h in list(fa.hyp) + [w for w in fa.verb if w["w"] not in FILLERS]:
        cov[int(max(h["start"] + 0.03, 0) * 100): int(max(h["end"] - 0.03, 0) * 100)] = True
        edges += [h["start"], h["end"]]
    edges = np.array(sorted(edges)) if edges else np.array([0.0])
    I = fr.inten
    thr = np.percentile(I, 5) * 0.65
    out = []
    for i, j in _runs((I > thr) & (~cov) & fr.voiced):
        L = (j - i + 1) * 0.01
        if L < minlen:
            continue
        a, b = i * 0.01, (j + 1) * 0.01
        sl = slice(i, j + 1)
        f0 = fr.f0_st[sl]
        f0 = f0[~np.isnan(f0)]
        cl, cr = I[max(0, i - 30): i], I[j + 1: j + 31]
        out.append((a, b, [L, float(np.mean(I[sl])), float(np.std(f0)) if len(f0) > 3 else 9.0, float(np.mean(fr.voiced[sl])),
                           float(min(np.abs(edges - a).min(), np.abs(edges - b).min())), float(cl.min()) if len(cl) else 0.0, float(cr.min()) if len(cr) else 0.0,
                           float(np.abs(np.diff(fr.hf[sl])).mean()) if L > 0.05 else 0.0, float(np.mean(fr.hf[sl]))]))
    return out


def ins_prob(f) -> float:
    if not INS_CLF:
        return 0.5
    z = (np.array(f) - np.array(INS_CLF["mean"])) / np.array(INS_CLF["std"])
    return float(1 / (1 + np.exp(-(z @ np.array(INS_CLF["w"]) + INS_CLF["b"]))))


def d_insert(fa: FA, th) -> list[Cand]:
    out = []
    for a, b, f in ins_candidates(fa):
        p = ins_prob(f)
        if p >= th["ins_p"] and a > 0.3 and b < fa.dur - 0.3:
            k = max([w["i"] for w in fa.words if w["status"] != "skip" and w["pe"] <= a + 0.05] or [0])
            out.append(Cand("FILLER", a, b, float(b - a), k, min(k + 1, fa.n - 1), {"heard": "voiced hum / drawl", "p": round(p, 2)}))
    return out


PRODUCERS_FREE = {"PAUSE_BAD": d_pauses, "PAUSE_LOST": d_pauses, "PACE_FAST": d_pace, "PACE_SLOW": d_pace, "MONOTONE": d_mono, "UPTALK": d_uptalk,
                  "FADE": d_fade, "SHOUT": d_shout, "SLUR": d_slur, "FILLER": lambda fa, th: d_text(fa, th) + d_insert(fa, th), "REPEAT": d_text, "WORD_SKIP": d_text, "WORD_SWAP": d_text}


def run(fa: FA, th: dict, flaw: str | None = None) -> list[Cand]:
    t = {**FTH, **{k: v for k, v in th.items() if k in FTH}}
    if flaw:
        fns = {PRODUCERS_FREE[flaw]} if flaw in PRODUCERS_FREE else set()
    else:
        fns = set(PRODUCERS_FREE.values())
    out = []
    for f in fns:
        out += f(fa, t)
    seen, uniq = set(), []
    for c in out:                                              # one function can serve several flaws (d_text): never report a detection twice
        key = (c.flaw, round(c.start, 3), round(c.end, 3))
        if (flaw is None or c.flaw == flaw) and key not in seen:
            seen.add(key)
            uniq.append(c)
    return sorted(uniq, key=lambda c: c.start)


# ----------------------------------------------------------------------------- population norms
def build_norms() -> dict:
    """Spread of each measurement across the clean TRAIN speakers (true word times from the takes' alignments)."""
    pause = {}
    rises, fades, rate_logs = [], [], []
    for b in POP_SPEAKERS:
        x = audio.normalise(audio.load(reference.audio_of(b)))[0]
        ref_w = reference.words_of(b)
        ling = annotate([{"w": r["w"], "sent_end": r["sent_end"], "clause_end": r["clause_end"]} for r in ref_w])
        words = [{"i": i, "w": w["w"], "clean": w["clean"], "ps": w["start_s"], "pe": w["end_s"], "status": "ok", "hyp": None} for i, w in enumerate(ref_w)]
        fr = features.compute(x)
        fa = FA(bid=b, words=words, bt=ling["b"][: len(words) - 1] + ["sentence"], zipf_next=[], fr=fr, sil=silence_runs(fr, 0.1), hyp=[], verb=[], dur=len(x) / audio.SR)
        for k, (L, _, _) in enumerate(pauses(fa)):
            pause.setdefault(fa.bt[k], []).append(L)
        for s, e in _sentences(fa):
            if e - s < 4:
                continue
            r = final_rise(fa, e)
            if not np.isnan(r):
                rises.append(r)
            i0 = e
            while i0 > s + 2 and (words[e]["pe"] - words[i0 - 1]["ps"]) < 1.4:
                i0 -= 1
            end = fr.inten[fr.sl(words[i0]["ps"], words[e]["pe"])]
            rest = fr.inten[fr.sl(words[s]["ps"], words[i0]["ps"])]
            if end.size > 4 and rest.size > 4:
                fades.append(float(np.mean(end) - np.mean(rest)))
        r = local_rates(fa, 2)
        ok = r[~np.isnan(r)]
        rate_logs += list(np.log(ok / np.median(ok)))
    def summ(v):
        v = np.asarray(v, float)
        return {"median": float(np.median(v)), "mad": float(1.4826 * np.median(np.abs(v - np.median(v)))), "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)), "n": int(len(v))}
    from sklearn.linear_model import Ridge
    Xs, ys = [], []
    for b in POP_SPEAKERS:
        W = reference.words_of(b)
        ling = annotate([{"w": r["w"], "sent_end": r["sent_end"], "clause_end": r["clause_end"]} for r in W])
        btb = ling["b"][: len(W) - 1] + ["sentence"]
        Xs.append(_dur_feats([{"clean": w["clean"]} for w in W], btb))
        y_ = np.log(np.array([max(w["end_s"] - w["start_s"], 0.03) for w in W]))
        ys.append(y_ - np.median(y_))
    mdl = Ridge(1.0, fit_intercept=False).fit(np.vstack(Xs), np.concatenate(ys))
    return {"speakers": POP_SPEAKERS, "dur_model": {"coef": [float(c) for c in mdl.coef_], "features": ["syllables", "letters", "function_word", "before_pause", "sentence_final", "const"]}, "pause": {k: summ(v) for k, v in pause.items()}, "final_rise": summ(rises), "fade_db": summ(fades),
            "rate_log_sigma": float(1.4826 * np.median(np.abs(np.array(rate_logs) - np.median(rate_logs))))}


if __name__ == "__main__":
    import sys
    if "--build-norms" in sys.argv:
        n = build_norms()
        NORMS_PATH.write_text(json.dumps(n, indent=1))
        print(json.dumps(n, indent=1))
