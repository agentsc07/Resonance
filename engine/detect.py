"""One detector per flaw: a threshold on a speaker-normalised feature delta (spec v1.1 "Detect"). Detectors only see the Cmp (participant
vs reference features) and the reference text's linguistics. They return Candidates in PARTICIPANT time with the measured deviation d in the
flaw's own unit; severity is mapped from d later (isotonic calibration)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .compare import Cmp, FR
from .features import HOP

CATEGORY = {"PACE_FAST": "Pacing", "PACE_SLOW": "Pacing", "PAUSE_BAD": "Pausing", "PAUSE_LOST": "Pausing", "MONOTONE": "Intonation",
            "UPTALK": "Intonation", "EMPH_FLAT": "Intonation", "FADE": "Volume", "SHOUT": "Volume", "FILLER": "Fluency", "REPEAT": "Fluency",
            "RARE_HESIT": "Fluency", "SLUR": "Clarity", "WORD_SKIP": "Text fidelity", "WORD_SWAP": "Text fidelity"}

# thresholds (tuned on train/dev only; see calibrate.py)
TH = {"pause_ins_min": 0.20, "pace_fast": 1.12, "pace_slow": 0.88, "pace_min_s": 1.2, "mono_ratio": 0.62, "mono_min_s": 2.4,
      "cost_ref": 0.03, "g_max": 5.0, "repeat_min": 0.14, "emph_min_words": 2, "swap_contrast": 0.12, "expand_emph": 1, "expand_slur": 0, "abs_pause": 0, "abs_lost": 0, "pause_abs": 0.7, "lost_frac": 0.35, "rare_min": 0.30, "uptalk_st": 1.5, "fade_db": 3.5, "shout_db": 3.0, "slur_db": 3.0, "emph_drop": 0.9}


import json as _json
from pathlib import Path as _P

MODEL_PATH = _P(__file__).parent / "model.json"
MODELS = _json.loads(MODEL_PATH.read_text()) if MODEL_PATH.exists() else {}      # {"same": {...}, "cross": {...}} fitted on TRAIN only
_DEFAULT_TH = dict(TH)
DISABLED: set = set()                          # flaws this mode cannot detect reliably (fitted on TRAIN): never reported
REPEAT_CLF = None                              # logistic on block structure: {"w": [...], "b": x, "mean": [...], "std": [...]}
MODE = "same"


G = 1.0
SCALED = ("swap_contrast", "emph_drop", "uptalk_st", "slur_db", "shout_db", "fade_db")


def set_match(C) -> float:
    """Adaptive sensitivity. The DTW path cost measures how well participant and reference match: ~0 for a clean same-speaker clip, higher
    under noise/phone/reverb and for a different speaker. Sensitive thresholds scale with how much worse the match is than the calibrated
    norm for this mode, so recording conditions and speaker differences cannot masquerade as delivery flaws."""
    global G
    ref = max(float(TH.get("cost_ref", 0.03)), 1e-3)
    G = float(np.clip(C.path_cost / ref, 1.0, float(TH.get("g_max", 5.0))))
    return G


def th(key: str) -> float:
    return TH[key] * (G if key in SCALED else 1.0)


def use(mode: str):
    """Select the calibrated parameter set for a reference mode (same / cross / free)."""
    global REPEAT_CLF, MODE, DISABLED
    MODE = mode
    m = MODELS.get(mode) or MODELS.get("same") or {}
    TH.clear()
    TH.update(_DEFAULT_TH)
    TH.update(m.get("thresholds", {}))
    REPEAT_CLF = m.get("repeat_clf")
    DISABLED = set(m.get("disabled", []))


use("same")
OPENER = ("sentence", "clause_punct", "before_coord", "before_subord", "before_relative", "after_discourse")
CLAUSE_END = ("sentence", "clause_punct", "before_coord", "before_subord", "before_relative")


@dataclass
class Cand:
    flaw: str
    start: float                      # participant time
    end: float
    d: float                          # deviation in the flaw's own unit
    w0: int                           # reference word indices
    w1: int
    facts: dict = field(default_factory=dict)

    @property
    def category(self) -> str:
        return CATEGORY[self.flaw]


def _span(C: Cmp, i0: int, i1: int) -> tuple[float, float]:
    return C.w[i0]["ps"], C.w[i1]["pe"]


def clause_span(C: Cmp, bt: list[str], i0: int, i1: int) -> tuple[int, int]:
    """Expand a flagged stretch to the clause that contains it (flaws that act on a phrase are labelled over the whole phrase)."""
    a = i0
    while a > 0 and bt[a - 1] not in CLAUSE_END:
        a -= 1
    b = i1
    while b < C.n - 1 and bt[b] not in CLAUSE_END:
        b += 1
    return a, b


def block_features(C: Cmp, a: float, b: float) -> list[float]:
    """Structure of an inserted block: a steady vowel (filler) vs consonant-rich speech (a restarted word)."""
    fr = C.par_fr
    s = fr.sl(a, b)
    H, V = fr.hf[s], fr.voiced[s]
    if H.size < 6:
        return [0.0, 0.5]
    return [float(np.mean(np.abs(np.diff(H)))), float(np.mean(V))]


def repeat_prob(feat: list[float]) -> float:
    if not REPEAT_CLF:
        return 1.0 if feat[1] < 0.55 else 0.0
    z = (np.array(feat) - np.array(REPEAT_CLF["mean"])) / np.array(REPEAT_CLF["std"])
    return float(1 / (1 + np.exp(-(z @ np.array(REPEAT_CLF["w"]) + REPEAT_CLF["b"]))))


def _silent_frac(C: Cmp, a: float, b: float) -> tuple[float, float]:
    """(fraction of silent frames, fraction of voiced frames) in a participant interval."""
    fr = C.par_fr
    s = fr.sl(a, b)
    I, V = fr.inten[s], fr.voiced[s]
    if I.size == 0:
        return 1.0, 0.0
    floor = float(np.percentile(fr.inten, 5))
    thr = floor + 0.35 * (0.0 - floor)                 # 35% of the clip's dynamic range above the floor
    return float(np.mean(I < thr)), float(np.mean(V))


def pauses_and_events(C: Cmp, bt: list[str], zipf_next: list[float]) -> list[Cand]:
    """Insertion / deletion events from the DTW lag curve -> PAUSE_BAD, PAUSE_LOST, FILLER / REPEAT (voiced insertion), WORD_SKIP."""
    out = []
    absp, absl = TH.get("abs_pause", 0) >= 1, TH.get("abs_lost", 0) >= 1
    if absp or absl:
        out += [c for c in pauses_abs(C, bt) if (c.flaw == "PAUSE_BAD" and absp) or (c.flaw == "PAUSE_LOST" and absl)]
    for e in C.events:
        k = min(e.boundary, C.n - 2)
        if k < 0:
            continue
        if e.kind == "ins" and e.mag >= TH["repeat_min"]:
            sil, voi = _silent_frac(C, e.q0, e.q1)
            a, b = e.q0, e.q1
            rare = zipf_next[k] is not None and zipf_next[k] < 3.5
            if rare and e.mag >= TH["rare_min"]:                            # hesitation before a hard word: the label spans pause + word
                out.append(Cand("RARE_HESIT", a, max(C.w[k + 1]["pe"], b + 0.05), e.mag, k + 1, k + 1, {"boundary": bt[k]}))
            elif sil >= 0.6:                                                # silent insertion = a pause
                if absp:
                    pass
                elif e.mag >= TH["pause_ins_min"] and (bt[k] == "phrase_tight" or (bt[k] == "phrase_loose" and e.mag >= 0.45)):
                    out.append(Cand("PAUSE_BAD", a, b, e.mag, k, k + 1, {"boundary": bt[k]}))
            else:                                                           # speech-like insertion = filler or restart
                pr = repeat_prob(block_features(C, a, b)) if bt[k] in OPENER else 0.0
                if pr >= 0.5:
                    out.append(Cand("REPEAT", a, b, e.mag, k + 1, k + 1, {"boundary": bt[k], "p_repeat": round(pr, 2)}))
                elif e.mag >= TH["pause_ins_min"]:
                    out.append(Cand("FILLER", a, b, e.mag, k, k + 1, {"boundary": bt[k], "voiced": round(voi, 2)}))
        elif e.kind == "del":
            # words wholly inside the deleted reference span => skipped words, else a removed pause
            inside = [i for i in range(C.n) if C.w[i]["rs"] >= e.t0 - 0.05 and C.w[i]["re"] <= e.t1 + 0.05]
            if inside and e.mag >= 0.08 and sum(C.w[i]["dur_r"] for i in inside) > 0.6 * e.mag:
                a, b = C.w[inside[0]]["ps"], C.w[inside[-1]]["pe"]
                out.append(Cand("WORD_SKIP", a, max(b, a + 0.05), e.mag, inside[0], inside[-1], {"n_words": len(inside)}))
            elif not absl and bt[k] in ("clause_punct", "sentence") and C.w[k]["gap_r"] >= 0.15:
                a = C.w[k]["pe"]
                out.append(Cand("PAUSE_LOST", a, a + max(0.05, C.w[k]["gap_r"] - e.mag), e.mag, k, k + 1, {"boundary": bt[k]}))
    return out


def silence_runs(fr, min_s: float = 0.12) -> list[tuple[float, float]]:
    """Silent stretches (frames below 35% of the clip's dynamic range above its floor) of at least min_s seconds."""
    I = fr.inten
    thr = float(np.percentile(I, 5)) * 0.65
    out = []
    for i, j in _runs(I < thr):
        if (j - i + 1) * 0.01 >= min_s:
            out.append((i * 0.01, (j + 1) * 0.01))
    return out


def pauses_abs(C: Cmp, bt: list[str]) -> list[Cand]:
    """Reference-free pause norms. Natural readers stop at sentence ends and clause punctuation and nowhere inside a phrase, in every voice, so
    a long silence at a phrase boundary is PAUSE_BAD and a missing silence at a sentence end is PAUSE_LOST, whoever the reference speaker is.
    The reference only says whether a stop is expected at a clause boundary."""
    out = []
    dur = C.w[-1]["pe"] if C.w else 0.0
    mids_p = np.array([(C.w[k]["pe"] + C.w[k + 1]["ps"]) / 2 for k in range(C.n - 1)])
    mids_r = np.array([(C.w[k]["re"] + C.w[k + 1]["rs"]) / 2 for k in range(C.n - 1)])

    def assign(runs, mids):
        """each silent stretch belongs to exactly one boundary: the nearest by its centre (a sentence-end pause must not leak into its neighbours)"""
        best = {}
        for r in runs:
            if len(mids) == 0:
                break
            k = int(np.argmin(np.abs(mids - (r[0] + r[1]) / 2)))
            if abs(mids[k] - (r[0] + r[1]) / 2) <= 0.6 and (k not in best or r[1] - r[0] > best[k][1] - best[k][0]):
                best[k] = r
        return best
    P, R = assign(silence_runs(C.par_fr), mids_p), assign(silence_runs(C.ref_fr), mids_r)
    for k in range(C.n - 1):
        if mids_p[k] < 0.4 or mids_p[k] > dur - 0.3:
            continue
        r = P.get(k)
        L = r[1] - r[0] if r else 0.0
        if bt[k] in ("phrase_tight", "phrase_loose"):
            lim = TH["pause_abs"] if bt[k] == "phrase_tight" else TH["pause_abs"] * 1.6
            rr_ = R.get(k)
            Lr = (rr_[1] - rr_[0]) if rr_ else 0.0                         # a stop the reference speaker also makes here is natural for this text
            if r and L - Lr >= lim:
                out.append(Cand("PAUSE_BAD", r[0], r[1], L - Lr, k, k + 1, {"boundary": bt[k]}))
        elif bt[k] in ("sentence", "clause_punct"):
            rr_ = R.get(k)
            Lr = rr_[1] - rr_[0] if rr_ else 0.0
            need = 0.45 if bt[k] == "sentence" else 0.3
            if Lr >= need and L < TH["lost_frac"] * Lr:
                a = C.w[k]["pe"]
                out.append(Cand("PAUSE_LOST", a, a + max(0.05, Lr - L), float(Lr - L), k, k + 1, {"boundary": bt[k]}))
    return out


def swaps(C: Cmp) -> list[Cand]:
    """A misread word matches the reference poorly along the DTW path: per-word cost well above its neighbours' (local contrast, so a whole
    time-warped stretch does not count)."""
    cost = np.array([w["cost"] for w in C.w], float)
    out = []
    for i, w in enumerate(C.w):
        if w["dur_r"] < 0.1 or not np.isfinite(cost[i]):
            continue
        nb = [cost[j] for j in range(max(0, i - 4), min(C.n, i + 5)) if j != i and np.isfinite(cost[j])]
        c = cost[i] - (np.median(nb) if nb else 0.0)
        if c >= th("swap_contrast"):
            out.append(Cand("WORD_SWAP", w["ps"], max(w["pe"], w["ps"] + 0.1), float(c), i, i, {"cost": round(float(cost[i]), 3)}))
    return out


def pace(C: Cmp) -> list[Cand]:
    rho = np.array([w["rho"] for w in C.w], float)
    valid = np.array([w["dur_r"] > 0.06 for w in C.w])
    sm = rho.copy()
    for i in range(len(rho)):
        s = slice(max(0, i - 1), min(len(rho), i + 2))
        v = rho[s][valid[s]]
        sm[i] = np.median(v) if len(v) else 1.0
    out = []
    for fast in (True, False):
        flag = (sm >= TH["pace_fast"]) if fast else (sm <= TH["pace_slow"])
        i = 0
        while i < len(flag):
            if flag[i]:
                j = i
                while j + 1 < len(flag) and flag[j + 1]:
                    j += 1
                a, b = _span(C, i, j)
                dr = sum(C.w[k]["dur_r"] for k in range(i, j + 1))
                if dr >= TH["pace_min_s"] * (0.8 if not fast else 1.0):
                    med = float(np.median(sm[i:j + 1]))
                    out.append(Cand("PACE_FAST" if fast else "PACE_SLOW", a, b, med if fast else 1.0 / max(med, 1e-3), i, j,
                                    {"ratio": round(med, 2)}))
                i = j + 1
            else:
                i += 1
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def _iqr(fr, a: float, b: float) -> float:
    v = fr.f0_st[fr.sl(a, b)]
    v = v[~np.isnan(v)]
    return float(np.percentile(v, 75) - np.percentile(v, 25)) if len(v) > 10 else np.nan


def monotone(C: Cmp, win_words: int = 12) -> list[Cand]:
    """F0 IQR of a stretch relative to the speaker's own clip IQR, compared with the same ratio in the reference (speaker-normalised)."""
    out = []
    ip, ir = _iqr(C.par_fr, 0, 1e9), _iqr(C.ref_fr, 0, 1e9)
    if not (ip > 0.5 and ir > 0.5):
        return out
    flags = np.zeros(C.n, bool)
    ratio = np.full(C.n, np.nan)
    for i in range(0, max(1, C.n - win_words + 1), 3):
        j = min(C.n - 1, i + win_words - 1)
        a, b = _span(C, i, j)
        ra, rb = C.w[i]["rs"], C.w[j]["re"]
        wp, wr = _iqr(C.par_fr, a, b), _iqr(C.ref_fr, ra, rb)
        if np.isnan(wp) or np.isnan(wr) or wr < 1.2:
            continue
        r = (wp / ip) / (wr / ir)
        ratio[i:j + 1] = np.where(np.isnan(ratio[i:j + 1]), r, np.minimum(ratio[i:j + 1], r))
        if r <= TH["mono_ratio"]:
            flags[i:j + 1] = True
    for i, j in _runs(flags):
        a, b = _span(C, i, j)
        if b - a >= TH["mono_min_s"]:
            wp = _iqr(C.par_fr, a, b)
            wr = _iqr(C.ref_fr, C.w[i]["rs"], C.w[j]["re"])
            out.append(Cand("MONOTONE", a, b, float(wr * ip / ir - wp) if ip else 0.0, i, j,
                            {"iqr_part": round(wp, 2), "iqr_ref": round(wr, 2)}))
    return out


def _sent_ends(C: Cmp, bt: list[str]) -> list[tuple[int, int]]:
    """(first word, last word) of each sentence inside the comparable span."""
    out, s = [], 0
    for k in range(C.n - 1):
        if bt[k] == "sentence":
            out.append((s, k))
            s = k + 1
    if s < C.n:
        out.append((s, C.n - 1))
    return out


def uptalk(C: Cmp, bt: list[str]) -> list[Cand]:
    out = []
    for s, e in _sent_ends(C, bt):
        if e - s < 4:
            continue
        rise = []
        for fr, w in ((C.par_fr, lambda i: (C.w[i]["ps"], C.w[i]["pe"])), (C.ref_fr, lambda i: (C.w[i]["rs"], C.w[i]["re"]))):
            a, b = w(e)
            sl = fr.sl(a, b)
            f0, I = fr.f0_st[sl], fr.inten[sl]
            v = ~np.isnan(f0)
            if v.sum() < 6:
                rise.append(np.nan)
                continue
            idx = np.where(v)[0]
            nuc = idx[np.argmax(I[idx])]
            rise.append(float(np.nanmedian(f0[idx[-5:]]) - f0[nuc]))
        if not np.isnan(rise).any() and rise[0] - rise[1] >= th("uptalk_st"):
            i0 = e
            while i0 > s + 1 and (C.w[e]["re"] - C.w[i0 - 1]["rs"]) < 1.0:
                i0 -= 1
            a, b = _span(C, i0, e)
            out.append(Cand("UPTALK", a, b, float(rise[0] - rise[1]), i0, e, {"rise_part": round(rise[0], 2), "rise_ref": round(rise[1], 2)}))
    return out


def fade(C: Cmp, bt: list[str]) -> list[Cand]:
    out = []
    for s, e in _sent_ends(C, bt):
        if e - s < 4:
            continue
        i0 = e
        while i0 > s + 2 and (C.w[e]["re"] - C.w[i0 - 1]["rs"]) < 1.4:
            i0 -= 1
        def delta(fr, f, g):
            a, b = f(i0), f(e)
            I_end = np.mean(fr.inten[fr.sl(a[0], b[1])])
            I_rest = np.mean(fr.inten[fr.sl(g(s)[0], g(i0)[0])]) if i0 > s else I_end
            return I_end - I_rest
        pw = lambda i: (C.w[i]["ps"], C.w[i]["pe"])
        rw = lambda i: (C.w[i]["rs"], C.w[i]["re"])
        d = delta(C.par_fr, pw, pw) - delta(C.ref_fr, rw, rw)
        if d <= -th("fade_db"):
            a, b = _span(C, i0, e)
            out.append(Cand("FADE", a, b, float(-d), i0, e, {}))
    return out


def shout(C: Cmp, win: int = 6) -> list[Cand]:
    """Speech level of a stretch (dB re the speaker's own median) vs the reference's same stretch."""
    out = []
    gain = np.zeros(C.n)
    for i in range(0, C.n):
        j = min(C.n - 1, i + win - 1)
        a, b = _span(C, i, j)
        ra, rb = C.w[i]["rs"], C.w[j]["re"]
        def lvl(fr, x, y):
            I = fr.inten[fr.sl(x, y)]
            return float(np.percentile(I, 80)) if I.size > 5 else np.nan
        gain[i] = lvl(C.par_fr, a, b) - lvl(C.ref_fr, ra, rb)
    flag = np.nan_to_num(gain, nan=0.0) >= th("shout_db")
    for i, j in _runs(flag):
        j = min(C.n - 1, j + win - 1)
        a, b = _span(C, i, j)
        if b - a >= 0.6:
            out.append(Cand("SHOUT", a, b, float(np.nanmax(gain[i:j + 1])), i, j, {}))
    return out


def slur(C: Cmp, win: int = 7) -> list[Cand]:
    """Consonant clarity: mean of the top-30% 2-7.5 kHz frames (dB re total), participant vs reference."""
    out = []
    drop = np.zeros(C.n)
    for i in range(0, C.n):
        j = min(C.n - 1, i + win - 1)
        a, b = _span(C, i, j)
        ra, rb = C.w[i]["rs"], C.w[j]["re"]
        def top(fr, x, y):
            h = fr.hf[fr.sl(x, y)]
            return float(np.mean(np.sort(h)[-max(1, int(0.3 * len(h))):])) if h.size > 8 else np.nan
        drop[i] = top(C.ref_fr, ra, rb) - top(C.par_fr, a, b)
    flag = np.nan_to_num(drop, nan=0.0) >= th("slur_db")
    for i, j in _runs(flag):
        j = min(C.n - 1, j + win - 1)
        a, b = _span(C, i, j)
        if b - a >= 1.0:
            out.append(Cand("SLUR", a, b, float(np.nanmax(drop[i:j + 1])), i, j, {}))
    return out


def emph(C: Cmp, win_s: float = 4.0) -> list[Cand]:
    """Prominence z (F0 peak + energy + duration) of words that were prominent in the reference but are not in the participant."""
    d = np.array([w["rf"]["prom"] - w["pf"]["prom"] if w["rf"]["prom"] > 0.8 else 0.0 for w in C.w])
    d = np.nan_to_num(d)
    flag = d >= th("emph_drop")
    out, i = [], 0
    idx = np.where(flag)[0]
    used = set()
    for k in idx:
        if k in used:
            continue
        grp = [m for m in idx if C.w[m]["rs"] >= C.w[k]["rs"] and C.w[m]["re"] <= C.w[k]["rs"] + win_s and m not in used]
        if len(grp) >= TH["emph_min_words"]:
            used.update(grp)
            a, b = _span(C, grp[0], grp[-1])
            out.append(Cand("EMPH_FLAT", a, b, float(np.mean(d[grp])), grp[0], grp[-1], {"words": len(grp)}))
    return out


# which detector function(s) produce each flaw (used by calibration to run only what a trial needs)
PRODUCERS = {"PACE_FAST": ("pace",), "PACE_SLOW": ("pace",), "PAUSE_BAD": ("events",), "PAUSE_LOST": ("events",), "RARE_HESIT": ("events",),
             "REPEAT": ("events",), "FILLER": ("events",), "WORD_SKIP": ("events",), "MONOTONE": ("monotone",), "UPTALK": ("uptalk",),
             "EMPH_FLAT": ("emph",), "FADE": ("fade",), "SHOUT": ("shout",), "SLUR": ("slur",), "WORD_SWAP": ("swaps",)}


def run_flaw(flaw: str, C: Cmp, bt: list[str], zipf_next: list[float]) -> list[Cand]:
    set_match(C)
    """Only the detector that can emit `flaw` (no conflict resolution): the fast path for threshold calibration."""
    f = PRODUCERS[flaw][0]
    out = {"pace": lambda: pace(C), "events": lambda: pauses_and_events(C, bt, zipf_next), "monotone": lambda: monotone(C),
           "uptalk": lambda: uptalk(C, bt), "emph": lambda: emph(C), "fade": lambda: fade(C, bt), "shout": lambda: shout(C),
           "slur": lambda: slur(C), "swaps": lambda: swaps(C)}[f]()
    return [c for c in _expand(C, bt, out) if c.flaw == flaw]


def _overlaps(a0, a1, b0, b1) -> bool:
    return min(a1, b1) > max(a0, b0)


def _expand(C: Cmp, bt: list[str], cands: list[Cand]) -> list[Cand]:
    for c in cands:
        if (c.flaw == "EMPH_FLAT" and TH.get("expand_emph", 1)) or (c.flaw == "SLUR" and TH.get("expand_slur", 0)):
            a, b = clause_span(C, bt, c.w0, c.w1)
            c.start, c.end, c.w0, c.w1 = C.w[a]["ps"], C.w[b]["pe"], a, b
    return cands


def run_all(C: Cmp, bt: list[str], zipf_next: list[float]) -> list[Cand]:
    set_match(C)
    out = []
    pc = pace(C)
    ev = pauses_and_events(C, bt, zipf_next)
    dur = C.w[-1]["pe"] if C.w else 0.0
    keep = []
    for c in ev:
        if c.start < 0.4 or c.start > dur - 0.3:                                       # alignment edge effects at the clip ends
            continue
        if c.d < 0.5 and any(p.flaw == "PACE_SLOW" and _overlaps(c.start, c.end, p.start - 0.2, p.end + 0.2) for p in pc):
            continue                                                                   # small lengthenings inside a slow stretch ARE the slowness
        keep.append(c)
    out += keep
    out += pc
    sw = swaps(C)
    others = keep + pc + monotone(C) + slur(C) + shout(C) + emph(C)
    out += [c for c in sw if not any(_overlaps(c.start, c.end, o.start - 0.1, o.end + 0.1) for o in others if o.flaw != "WORD_SWAP")]
    out += monotone(C)
    out += uptalk(C, bt)
    out += fade(C, bt)
    out += shout(C)
    out += slur(C)
    out += emph(C)
    return sorted(_expand(C, bt, out), key=lambda c: c.start)
