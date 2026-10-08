"""Fit the reference-free mode on the TRAIN split only: per-flaw thresholds (coordinate descent on that flaw's event F1@0.5), isotonic severity
maps, per-flaw reliability and the list of detectors too weak to report. Writes the "free" section of engine/model.json.
   python -m engine.calibrate_free [--rounds 2]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import calibrate as CAL
from . import detect
from . import tune as T

GRID_FREE = {
    "PAUSE_LOST": {"lost_frac": [0.1, 0.2, 0.3, 0.4, 0.55]},
    "PACE_FAST": {"pace_k_fast": [1.2, 1.6, 2.0, 2.5, 3.0], "pace_win": [1, 2, 3], "pace_min_s": [0.8, 1.2, 1.6]},
    "PACE_SLOW": {"pace_k_slow": [1.2, 1.6, 2.0, 2.5, 3.0]},
    "MONOTONE": {"mono_ratio": [0.4, 0.5, 0.62, 0.75, 0.85]},
    "UPTALK": {"uptalk_st": [0.8, 1.2, 1.6, 2.0, 3.0, 4.0]},
    "FADE": {"fade_db": [2.0, 3.0, 4.0, 5.0, 6.5]},
    "SHOUT": {"shout_db": [2.0, 3.0, 4.0, 5.0, 6.5]},
    "SLUR": {"slur_db": [2.0, 3.0, 4.0, 5.0, 6.5]},
    "FILLER": {"ins_p": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], "filler_min_s": [0.05, 0.1, 0.2, 0.3]},
    "REPEAT": {"repeat_sim": [0.5, 0.6, 0.7, 0.8]},
    "WORD_SKIP": {"skip_min_words": [1, 2]},
    "WORD_SWAP": {"swap_sim": [0.5, 0.6, 0.7, 0.8, 0.9]},
}


# Pinned, not fitted: the F1-optimal pause limit on injected pauses (0.8 s) misses real hesitations such as a 0.55 s stop inside a noun phrase. 0.5 s with a doubled limit
# at looser phrase boundaries is the lowest setting that keeps false pause flags on the clean baseline readings near one per minute (see docs/RESULTS.md).
# The held-vowel detector is pinned the same way: the fitted minimum length (0.4 s) would miss a real 0.27 s "uhh" in the regression set (tests/real); its range, steadiness and
# word-allowance settings come from the grid search described in docs/RESULTS.md.
PINNED = {"pause_bad": 0.5, "pause_loose_mult": 2.0, "flat_range": 2.0, "flat_min_s": 0.25, "flat_unexpl": 0.2, "flat_mult": 0.8, "flat_istd": 4.0}


def fit_ins_clf(labs):
    """Logistic classifier for 'voiced stretch nobody explains' -> filler (label: IoU >= 0.4 with an injected FILLER), fitted on TRAIN clips."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from . import free
    X, y = [], []
    for lab in labs:
        fa = T.free_analysis(lab)
        gts = [g for g in lab["what"] if g["flaw"] == "FILLER"]
        for a, b, f in free.ins_candidates(fa):
            hit = 0
            for g in gts:
                gw, cw = (g["start_s"], max(g["end_s"], g["start_s"] + 0.3)), (a, max(b, a + 0.3))
                iou = max(0, min(gw[1], cw[1]) - max(gw[0], cw[0])) / (max(gw[1], cw[1]) - min(gw[0], cw[0]))
                hit = hit or int(iou >= 0.4)
            X.append(f)
            y.append(hit)
    X, y = np.array(X), np.array(y)
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=500).fit((X - mu) / sd, y)
    print(f"insertion classifier: {len(y)} candidates, {int(y.sum())} fillers, train accuracy {clf.score((X - mu) / sd, y):.2f}")
    return {"w": clf.coef_[0].tolist(), "b": float(clf.intercept_[0]), "mean": mu.tolist(), "std": sd.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    labs = T.clips("train")
    print(f"calibrating reference-free mode on {len(labs)} TRAIN clips")
    detect.use("free")
    from . import free
    free.INS_CLF = fit_ins_clf(labs)
    th = {**dict(detect.TH), **PINNED}
    for rnd in range(a.rounds):
        for flaw, params in GRID_FREE.items():
            for key, vals in params.items():
                best, bv = -1.0, th[key]
                for v in vals:
                    detect.TH.update(th)
                    detect.TH[key] = v
                    f = CAL.flaw_f1(labs, "free", flaw)
                    if f > best + 1e-9:
                        best, bv = f, v
                th[key] = bv
                print(f"  round {rnd} {flaw:10s} {key:16s} -> {bv}   (F1 {best:.3f})", flush=True)
    detect.TH.update(th)
    model = {"thresholds": th, "repeat_clf": None, "ins_clf": free.INS_CLF, "isotonic": CAL.fit_isotonic(labs, "free")}
    detect.DISABLED = set()
    g, _ = T.evaluate(labs, "free", verbose=False)
    model["disabled"] = sorted(f for f in detect.PRODUCERS if g["per"].get(f, {"f1": 0.0})["f1"] < 0.10)
    detect.DISABLED = set(model["disabled"])
    print("free mode disables:", model["disabled"])
    model["reliability"] = CAL.fit_reliability(labs, "free")
    mp = Path(__file__).parent / "model.json"
    allm = json.loads(mp.read_text())
    allm["free"] = model
    mp.write_text(json.dumps(allm, indent=1))
    print("wrote engine/model.json (free)")
    T.evaluate(labs, "free")


if __name__ == "__main__":
    main()
