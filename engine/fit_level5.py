"""A2: one weight per headline flaw so a single level-5 flaw costs about the same overall score for every flaw type (target ~60).
Bisection on TRAIN single-flaw clips (cached by `python -m engine.fit_scoring --collect`); weights land in engine/rubric.yaml `flaw_weight`.
   python -m engine.fit_level5 [--write]"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import score as S  # noqa: E402
from engine.fit_scoring import PKL  # noqa: E402

TARGET = 60.0
HEADLINE = ["FADE", "SHOUT", "PACE_SLOW", "PAUSE_BAD", "FILLER", "PACE_FAST", "WORD_SKIP", "SLUR", "PAUSE_LOST", "WORD_SWAP", "MONOTONE"]
START = {"FADE": 3.2, "SLUR": 2.7, "PAUSE_LOST": 2.6, "PAUSE_BAD": 2.4, "PACE_FAST": 2.2, "WORD_SWAP": 1.3, "MONOTONE": 1.1}


CAP = 8.0           # upper bound of the search


def mean_at(D, rub, f, lv, detected_only=True):
    """Mean overall score of level-lv clips with flaw f. detected_only: only clips where the engine flagged that flaw (a miss costs nothing whatever the weight)."""
    v = [S.score(d["what"], d["dur"], d["genre"], rub, False, d.get("badge"))["overall"] for d in D
         if d["flaw"] == f and d["level"] == lv and (not detected_only or any(r["flaw"] == f for r in d["what"]))]
    return float(np.mean(v)) if v else float("nan"), len(v)


def table(D, rub, flaws):
    return {f: [round(mean_at(D, rub, f, lv)[0], 1) for lv in range(1, 6)] for f in flaws}


def fit(write=False):
    D = pickle.loads(PKL.read_bytes())
    rub = S.load_rubric()
    rub["flaw_weight"] = {}
    print("BEFORE (weight 1.0), mean overall at L1..L5")
    for f, v in table(D, rub, HEADLINE).items():
        print(f"  {f:11s} {v}")
    w = {}
    for f in HEADLINE:
        lo, hi = 0.2, CAP
        m, n = mean_at(D, {**rub, "flaw_weight": {f: hi}}, f, 5)
        if n == 0 or m > TARGET:               # too weak a detector to reach the target even at the cap
            w[f] = hi if n else 1.0
            continue
        for _ in range(30):
            mid = (lo + hi) / 2
            m, _ = mean_at(D, {**rub, "flaw_weight": {f: mid}}, f, 5)
            lo, hi = (mid, hi) if m > TARGET else (lo, mid)
        w[f] = round((lo + hi) / 2, 2)
    rub["flaw_weight"] = w
    print("weights:", w)
    print("AFTER, mean overall at L1..L5")
    for f, v in table(D, rub, HEADLINE).items():
        print(f"  {f:11s} {v}")
    if write:
        p = ROOT / "engine" / "rubric.yaml"
        s = re.sub(r"^flaw_weight:.*$", "flaw_weight: " + "{" + ", ".join(f"{k}: {v}" for k, v in w.items()) + "}   # fitted on TRAIN single-flaw clips by engine/fit_level5.py so a level-5 flaw scores ~60", p.read_text(), flags=re.M)
        p.write_text(s)
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    fit(ap.parse_args().write)
