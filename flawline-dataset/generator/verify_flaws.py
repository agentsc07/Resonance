"""Objective dose-response check of the flaw factory (no listening required).

For each flaw and level we measure the labelled region in the variant against the SAME region in the
baseline, using a feature that matches the flaw's claimed signature. The measured effect should rise
monotonically with level. Output: pilot/objective_check.csv

Usage: python verify_flaws.py B01-CHAMP [more takes]
"""
from __future__ import annotations

import csv
import sys

import numpy as np
import pyworld as pw
from scipy.signal import stft
from scipy.stats import spearmanr

from common import PILOT, SR, db, rms, s2n
from flaws import FLAW_CODES, Ctx, make_flawed, seed_for

LEVELS = (1, 2, 3, 4, 5)


def f0_st(x):
    x64 = x.astype(np.float64)
    f0, t = pw.harvest(x64, SR, f0_floor=71.0, f0_ceil=500.0, frame_period=5.0)
    v = f0 > 0
    return 12 * np.log2(f0[v]) if v.sum() > 5 else np.array([np.nan])


def band_db(x, lo=2000, hi=8000):
    f, _, Z = stft(x, SR, nperseg=512)
    m = (f >= lo) & (f <= hi)
    return 10 * np.log10(np.mean(np.abs(Z[m]) ** 2) + 1e-18)


def seg(x, a_s, b_s):
    return x[s2n(a_s): s2n(b_s)]


def emph_loss(ctx, r, B, V):
    """F0 peak of the words the injector flattened, above the region mean, baseline minus variant."""
    ws = [w for w in ctx.words[r["word_start"]: r["word_end"] + 1] if w["clean"] in r["params"]["stressed_words"]]
    a0 = ctx.words[r["word_start"]]["start"]
    out = []
    for X in (B, V):
        mean = np.nanmean(f0_st(X))
        pk = []
        for w in ws:
            f = f0_st(X[w["start"] - a0: w["end"] - a0])
            pk.append(np.nanmax(f) - mean)
        out.append(np.nanmean(pk))
    return float(out[0] - out[1])


def measure(ctx, code, xb, xv, r):
    ev = (r.get('params') or {}).get('evidence') or {}
    for key, unit in (('achieved_rate', 'achieved speed ratio'), ('rise_after_nucleus_st', 'final rise from nucleus (st)'),
                      ('restart_words', 'restart words (frac-weighted)'), ('prominence_drop', 'focal prominence z lost'),
                      ('level_gain_db', 'speech level gain (dB)')):
        if ev.get(key) is not None and code in ('PACE_FAST', 'PACE_SLOW', 'UPTALK', 'REPEAT', 'EMPH_FLAT', 'SHOUT'):
            return float(ev[key]), unit
    bs, be, cs, ce = r["baseline_start_s"], r["baseline_end_s"], r["start_s"], r["end_s"]
    B, V = seg(xb, bs, be), seg(xv, cs, ce)
    if code in ("PACE_FAST", "PACE_SLOW"):
        return (be - bs) / max(ce - cs, 1e-6), "speed ratio"
    if code in ("PAUSE_BAD", "FILLER", "REPEAT"):
        return ce - cs, "inserted s"
    if code == "PAUSE_LOST":
        return (be - bs) - (ce - cs), "pause removed s"
    if code == "RARE_HESIT":
        return (ce - cs) - (be - bs), "added s"
    if code == "WORD_SKIP":
        return be - bs, "removed s"
    if code == "WORD_SWAP":
        return 1.0, "words swapped"
    if code == "MONOTONE":
        a, b = f0_st(B), f0_st(V)
        return (np.nanpercentile(a, 75) - np.nanpercentile(a, 25)) - (np.nanpercentile(b, 75) - np.nanpercentile(b, 25)), "F0 IQR lost (st)"
    if code == "UPTALK":
        a, b = f0_st(B), f0_st(V)                     # voiced frames only; compare the final 10 (the rise lives there)
        return float(np.nanmedian(b[-10:]) - np.nanmedian(a[-10:])), "final rise added (st)"
    if code == "EMPH_FLAT":
        return emph_loss(ctx, r, B, V), "stressed-word F0 peak lost (st)"
    if code == "FADE":
        n = s2n(0.3)
        return (db(rms(B[-n:])) - db(rms(V[-n:]))), "end level drop (dB)"
    if code == "SHOUT":
        return db(rms(V)) - db(rms(B)), "level gain (dB)"
    if code == "SLUR":
        return band_db(B) - band_db(V), "2-8 kHz loss (dB)"
    return np.nan, ""


def run(take_id: str):
    ctx = Ctx(take_id)
    rows = []
    for code in FLAW_CODES:
        series, unit = [], ""
        for lvl in LEVELS:
            y, regs = make_flawed(ctx, [(code, lvl)], seed_for(take_id, code))
            vals = [measure(ctx, code, ctx.x, y, r) for r in regs]
            unit = vals[0][1]
            if code in ("FILLER", "WORD_SKIP", "WORD_SWAP"):
                v = float(np.nansum([m[0] for m in vals]))
                unit = {"FILLER": "total inserted s", "WORD_SKIP": "total removed s"}.get(code, unit)
            else:
                v = float(np.nanmean([m[0] for m in vals]))
            series.append(v)
        rho = spearmanr(LEVELS, series).statistic if np.ptp(series) > 0 else float("nan")
        rows.append({"take_id": take_id, "flaw": code, "unit": unit, **{f"L{l}": round(v, 3) for l, v in zip(LEVELS, series)},
                     "spearman": round(float(rho), 3), "monotonic": bool(all(np.diff(series) >= -1e-6) or all(np.diff(series) <= 1e-6))})
    return rows


if __name__ == "__main__":
    takes = sys.argv[1:] or ["B01-CHAMP"]
    rows = [r for t in takes for r in run(t)]
    PILOT.mkdir(exist_ok=True)
    with open(PILOT / "objective_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['take_id']:10s} {r['flaw']:11s} {r['unit']:24s} " + " ".join(f"{r[f'L{l}']:8.2f}" for l in LEVELS)
              + f"  rho={r['spearman']:5.2f} {'OK' if r['monotonic'] else 'NOT MONOTONIC'}")
