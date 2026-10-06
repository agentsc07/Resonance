"""Fit every learned piece of the engine on the TRAIN split only, then write engine/model.json:
   1. per-flaw thresholds (coordinate descent on that flaw's own event-F1@0.5),
   2. REPEAT-vs-FILLER block classifier (logistic),
   3. per-flaw isotonic severity map d -> injected level (monotone, so a bigger deviation can never score as less severe).
   python -m engine.calibrate --mode same"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import metrics as M  # noqa: E402

from . import detect  # noqa: E402
from . import tune as T  # noqa: E402

GRID = {
    "PACE_FAST": {"pace_fast": [1.07, 1.10, 1.14, 1.18]},
    "PACE_SLOW": {"pace_slow": [0.80, 0.85, 0.88, 0.92]},
    "PAUSE_BAD": {"abs_pause": [0, 1], "pause_abs": [0.5, 0.6, 0.7, 0.8, 1.0], "pause_ins_min": [0.18, 0.25, 0.32]},
    "PAUSE_LOST": {"abs_lost": [0, 1], "lost_frac": [0.2, 0.35, 0.5]},
    "REPEAT": {"repeat_min": [0.10, 0.14, 0.18, 0.24]},
    "RARE_HESIT": {"rare_min": [0.2, 0.3, 0.4, 0.5]},
    "MONOTONE": {"mono_ratio": [0.45, 0.55, 0.62, 0.72, 0.82]},
    "UPTALK": {"uptalk_st": [0.8, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0]},
    "EMPH_FLAT": {"emph_drop": [0.15, 0.25, 0.4, 0.6, 0.9, 1.3], "emph_min_words": [1, 2], "expand_emph": [0, 1]},
    "FADE": {"fade_db": [3.5, 4.5, 5.5, 7.0]},
    "SHOUT": {"shout_db": [3.0, 4.0, 5.0, 6.5, 8.0]},
    "SLUR": {"slur_db": [1.5, 2.0, 2.5, 3.0, 4.0, 5.0], "expand_slur": [0, 1]},
    "WORD_SWAP": {"swap_contrast": [0.2, 0.3, 0.4, 0.55, 0.75, 1.0]},
}


def flaw_f1(labs, mode, flaw):
    g, _ = T.evaluate(labs, mode, verbose=False, flaw=flaw)
    v = g["per"].get(flaw)
    return v["f1"] if v else 0.0


def fit_repeat(labs, mode):
    from sklearn.linear_model import LogisticRegression
    X, y = [], []
    for lab in labs:
        if not lab["what"]:
            continue
        C, q, ref, n = T.analysed(lab, mode)
        for r in lab["what"]:
            if r["flaw"] in ("REPEAT", "FILLER") and r["kind"] == "insert":
                ev = [e for e in C.events if e.kind == "ins" and abs((e.q0 + e.q1) / 2 - (r["start_s"] + r["end_s"]) / 2) < 0.5]
                if ev:
                    X.append(detect.block_features(C, ev[0].q0, ev[0].q1))
                    y.append(int(r["flaw"] == "REPEAT"))
    X, y = np.array(X), np.array(y)
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    clf = LogisticRegression(class_weight="balanced", C=1.0).fit((X - mu) / sd, y)
    print(f"REPEAT classifier: n={len(y)} ({int(y.sum())} repeats), train accuracy {clf.score((X - mu) / sd, y):.2f}")
    return {"w": clf.coef_[0].tolist(), "b": float(clf.intercept_[0]), "mean": mu.tolist(), "std": sd.tolist()}


def fit_isotonic(labs, mode):
    from sklearn.isotonic import IsotonicRegression
    pts: dict[str, list] = {}
    for lab in labs:
        if not lab["what"]:
            continue
        C, q, ref, n = T.analysed(lab, mode)
        preds = [T.to_pred(c) for c in T.run_detectors(C, ref)]
        for i, j, _ in M.match(lab["what"], preds, 0.3, lambda r: r["flaw"]):
            if lab["what"][i].get("level") is not None:
                pts.setdefault(lab["what"][i]["flaw"], []).append((preds[j]["severity"], lab["what"][i]["level"]))
    out = {}
    for f, v in pts.items():
        v = np.array(v, float)
        if len(v) >= 6 and np.ptp(v[:, 0]) > 0:
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(v[:, 0], v[:, 1])
            xs = np.unique(v[:, 0])
            out[f] = {"x": [float(a) for a in xs], "y": [float(b) for b in iso.predict(xs)], "n": len(v)}
    return out


def fit_reliability(labs, mode, floor: float = 0.1):
    """Per-flaw reliability = precision of the full detector pass on TRAIN at the calibrated thresholds (used to weight penalties and to
    arbitrate overlapping detections of different causes)."""
    g, _ = T.evaluate(labs, mode, verbose=False)
    return {f: round(max(floor, v["precision"]), 3) for f, v in g["per"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="same")
    ap.add_argument("--only-reliability", action="store_true", help="refit just the per-flaw reliability from the current model.json")
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    labs = T.clips("train")
    if a.only_reliability:
        mp = Path(__file__).parent / "model.json"
        allm = json.loads(mp.read_text())
        detect.use(a.mode)
        allm[a.mode]["reliability"] = fit_reliability(labs, a.mode)
        mp.write_text(json.dumps(allm, indent=1))
        print(a.mode, allm[a.mode]["reliability"])
        return
    print(f"calibrating on {len(labs)} TRAIN clips, mode={a.mode}")
    detect.use(a.mode)
    model = {"mode": a.mode, "repeat_clf": fit_repeat(labs, a.mode)}
    detect.REPEAT_CLF = model["repeat_clf"]
    th = dict(detect._DEFAULT_TH)
    costs = [T.analysed(l, a.mode)[0].path_cost for l in labs]               # how well a NORMAL clip matches its reference in this mode
    th["cost_ref"] = float(max(0.02, np.percentile(costs, 75)))
    print(f"match norm cost_ref = {th['cost_ref']:.3f} (median path cost {np.median(costs):.3f})")
    for rnd in range(a.rounds):
        for flaw, params in GRID.items():
            for key, vals in params.items():
                best, bv = -1.0, th[key]
                for v in vals:
                    detect.TH.update(th)
                    detect.TH[key] = v
                    f = flaw_f1(labs, a.mode, flaw)
                    if f > best + 1e-9:
                        best, bv = f, v
                th[key] = bv
                print(f"  round {rnd} {flaw:10s} {key:14s} -> {bv}   (F1 {best:.3f})")
    detect.TH.update(th)
    model["thresholds"] = th
    model["isotonic"] = fit_isotonic(labs, a.mode)
    model["reliability"] = fit_reliability(labs, a.mode)
    mp = Path(__file__).parent / "model.json"
    allm = json.loads(mp.read_text()) if mp.exists() else {}
    allm[a.mode] = {k: v for k, v in model.items() if k != "mode"}
    mp.write_text(json.dumps(allm, indent=1))
    print("wrote engine/model.json; isotonic fitted for:", sorted(model["isotonic"]))
    T.evaluate(labs, a.mode)


if __name__ == "__main__":
    main()
