"""Baseline-vs-participant comparison. DTW maps reference time to participant time. The LAG curve (participant minus reference time) is read
two ways: sharp jumps are EVENTS (an insertion = pause / filler / repeat when lag rises, a deletion = skipped words / removed pause when it
falls) and the sustained slope is TEMPO. Events are removed from the timing map before per-word durations are measured, so an inserted pause
never masquerades as slow words next to it."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import dtw, features
from .audio import SR

FR = 100.0   # frames per second of the DTW path


@dataclass
class Event:
    kind: str            # "ins" | "del"
    t0: float            # reference time span
    t1: float
    mag: float           # seconds (positive)
    p0: float            # participant time span
    p1: float
    boundary: int        # index k of the nearest reference word boundary (between k and k+1)
    q0: float = 0.0      # refined participant span of the inserted block (DTW local-cost peak of length mag)
    q1: float = 0.0


@dataclass
class Cmp:
    n: int
    ref_words: list[dict]
    ref_fr: features.Frames
    par_fr: features.Frames
    kappa: float
    w: list[dict] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    lag: np.ndarray | None = None
    path_cost: float = 0.0
    path: tuple | None = None          # (reference frames, participant frames) of the warping path (for the time-warped overlay)


def cancel_jitter(evs: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    """Alignment wobble (esp. under babble noise): an insertion and a deletion of about the same size within ~1 s cancel to zero net timing
    change, so neither is a real event. Real flaws do not arrive with an equal-and-opposite partner."""
    evs = sorted(evs)
    drop = set()
    for i in range(len(evs) - 1):
        a, b = evs[i], evs[i + 1]
        if i in drop or i + 1 in drop or np.sign(a[2]) == np.sign(b[2]):
            continue
        if (b[0] - a[1]) / FR <= 1.0 and abs(abs(a[2]) - abs(b[2])) / FR <= 0.08 + 0.4 * max(abs(a[2]), abs(b[2])) / FR:
            drop.update((i, i + 1))
    return [e for k, e in enumerate(evs) if k not in drop]


def _events(lag: np.ndarray, min_mag: float = 0.12) -> list[tuple[int, int, float]]:
    """Runs where |d lag / d t| is large over a short stretch. Returns (r0, r1, signed magnitude in frames)."""
    h = 10
    pad = np.pad(lag, h, mode="edge")
    S = (pad[2 * h:] - pad[: len(lag)]) / (2.0 * h)               # slope of lag (frames per frame)
    out, i, n = [], 0, len(lag)
    while i < n:
        if S[i] > 0.6 or S[i] < -0.45:
            sign = 1 if S[i] > 0 else -1
            j = i
            while j + 1 < n and ((S[j + 1] > 0.35) if sign > 0 else (S[j + 1] < -0.3)):
                j += 1
            r0, r1 = max(0, i - h // 2), min(n - 1, j + h // 2)
            mag = lag[r1] - lag[r0]
            if abs(mag) / FR >= min_mag and (r1 - r0) / FR <= 1.2 and np.sign(mag) == sign:
                out.append((r0, r1, float(mag)))
            i = j + 1
        else:
            i += 1
    return out


def build(ref_x: np.ndarray, ref_words: list[dict], par_x: np.ndarray, n_words: int, ref_fr=None, par_fr=None) -> Cmp:
    n = min(n_words, len(ref_words))
    ref_x = ref_x[: int((ref_words[n - 1]["end_s"] + 0.25) * SR)]
    rr, pp, cost, pcost = dtw.align(ref_x, par_x)
    nref = int(rr.max()) + 1
    p_lo = np.full(nref, np.inf)
    p_hi = np.full(nref, -np.inf)
    np.minimum.at(p_lo, rr, pp)
    np.maximum.at(p_hi, rr, pp)
    ok = np.isfinite(p_lo)
    p_mid = np.interp(np.arange(nref), np.arange(nref)[ok], ((p_lo + p_hi) / 2)[ok])
    lag = p_mid - np.arange(nref)
    # robust local tempo: median of the lag slope over ~1.5 s (a single insertion jump cannot move a median; a genuinely slow/fast stretch does)
    h = 10
    padl = np.pad(lag, h, mode="edge")
    S = (padl[2 * h:] - padl[: len(lag)]) / (2.0 * h)
    wlen = 75
    tempo = np.array([np.median(S[max(0, r - wlen): r + wlen + 1]) for r in range(len(S))])
    cost_f = np.zeros(nref)
    cnt = np.zeros(nref)
    np.add.at(cost_f, rr, pcost)
    np.add.at(cnt, rr, 1)
    cost_f = np.where(cnt > 0, cost_f / np.maximum(cnt, 1), np.nan)
    evs_raw = cancel_jitter(_events(lag))
    # corrected map: remove each event's jump (smooth) so tempo is measured on the rest of the speech
    cp = np.zeros(int(pp.max()) + 1)                          # smoothed DTW local cost per PARTICIPANT frame: inserted audio matches nothing
    cn = np.zeros_like(cp)
    np.add.at(cp, pp, pcost)
    np.add.at(cn, pp, 1)
    cp = np.where(cn > 0, cp / np.maximum(cn, 1), 0.0)
    cp = np.convolve(cp, np.ones(5) / 5, mode="same")
    corr = p_mid.copy()
    for r0, r1, mag in evs_raw:
        ramp = np.clip((np.arange(nref) - r0) / max(r1 - r0, 1), 0, 1)
        corr = corr - mag * (ramp * ramp * (3 - 2 * ramp))
    corr = np.maximum.accumulate(corr)
    fc = lambda t: float(corr[min(nref - 1, max(0, int(round(t * FR))))]) / FR        # participant time, events removed
    fm = lambda t: float(p_mid[min(nref - 1, max(0, int(round(t * FR))))]) / FR       # participant time, as is
    starts = np.array([w["start_s"] for w in ref_words[:n]])
    ends = np.array([w["end_s"] for w in ref_words[:n]])
    events = []
    for r0, r1, mag in evs_raw:
        tc = (r0 + r1) / 2 / FR
        # nearest word boundary (between word k end and word k+1 start) to the event centre
        mids = (ends[:-1] + starts[1:]) / 2 if n > 1 else np.array([tc])
        k = int(np.argmin(np.abs(mids - tc)))
        ev = Event("ins" if mag > 0 else "del", r0 / FR, r1 / FR, abs(mag) / FR, float(p_mid[r0]) / FR, float(p_mid[r1]) / FR, k)
        ev.q0, ev.q1 = ev.p0, ev.p1
        if mag > 0:                                                  # locate the inserted block: the window of length mag with the highest local cost
            M = max(2, int(round(mag)))
            lo, hi = max(0, int(p_mid[r0]) - 15), min(len(cp) - 1, int(p_mid[r1]) + 15)
            if hi - lo > M:
                scores = np.convolve(cp[lo:hi + 1], np.ones(M) / M, mode="valid")
                b0 = lo + int(np.argmax(scores))
                ev.q0, ev.q1 = b0 / FR, (b0 + M) / FR
        events.append(ev)
    rf = ref_fr or features.compute(ref_x)
    pf = par_fr or features.compute(par_x)
    recs = []
    for i in range(n):
        rs, re_ = float(starts[i]), float(ends[i])
        recs.append({"i": i, "rs": rs, "re": re_, "ps": fm(rs), "pe": max(fm(re_), fm(rs)),
                     "dur_r": re_ - rs, "dur_p": max(fc(re_) - fc(rs), 0.0)})
    ratios = [o["dur_r"] / max(o["dur_p"], 1e-3) for o in recs if o["dur_r"] > 0.08 and o["dur_p"] > 0.02]
    kappa = float(np.median(ratios)) if ratios else 1.0
    tempo_med = float(np.median(tempo))
    kap = max(kappa, 1e-3)
    ins_at, del_at = {}, {}
    for e in events:
        (ins_at if e.kind == "ins" else del_at)[e.boundary] = (ins_at if e.kind == "ins" else del_at).get(e.boundary, 0.0) + e.mag
    rw = features.word_feats(rf, [{"start": o["rs"], "end": o["re"]} for o in recs])
    pw = features.word_feats(pf, [{"start": o["ps"], "end": max(o["pe"], o["ps"] + 0.01)} for o in recs])
    for i, (o, a, b) in enumerate(zip(recs, rw, pw)):
        nxt = i + 1 < n
        g_r = float(starts[i + 1] - ends[i]) if nxt else 0.0
        o["gap_r"] = g_r
        o["dgap"] = ins_at.get(i, 0.0) - del_at.get(i, 0.0)                         # seconds added (+) / removed (-) at the boundary after word i
        o["gap_p"] = g_r / kap + o["dgap"]
        o["rf"], o["pf"] = a, b
        r_mid = int(round((o["rs"] + o["re"]) / 2 * FR))
        o["tempo"] = float(tempo[min(nref - 1, r_mid)])                            # lag slope: >0 participant slower than the reference
        o["rho"] = (1.0 + tempo_med) / max(1.0 + o["tempo"], 0.2)                           # >1 = faster than the clip's typical tempo (speaker-normalised)
        cs = cost_f[int(round(o["rs"] * FR)): int(round(o["re"] * FR)) + 1]
        o["cost"] = float(np.nanmean(cs)) if np.isfinite(cs).any() else np.nan     # acoustic mismatch of this word (a swapped word matches badly)
    return Cmp(n=n, ref_words=ref_words[:n], ref_fr=rf, par_fr=pf, kappa=kappa, w=recs, events=events, lag=lag, path_cost=cost, path=(rr, pp))
