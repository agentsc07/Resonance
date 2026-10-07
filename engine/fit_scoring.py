"""Fit per-flaw weights so score spread is comparable across flaws (spec: a single L5 flaw puts its category near 55, an L3 near 77, an L4 near 66).
Uses TRAIN single-flaw clips only. For each flaw, w_f minimises sum over levels 3-5 of (w_f * P_obs - P_target)^2 where P = -tau ln(S/100) is the
penalty per clip-minute behind the own-category score. Weights are clipped to [0.4, 2] so no weak detector is blown up.
   python -m engine.fit_scoring --collect      # run the engine on train single-flaw clips (slow), cache /tmp/tau_data.pkl
   python -m engine.fit_scoring --fit [--write]  # fit weights and optionally patch engine/rubric.yaml"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import predict as P, score as S, tune as T  # noqa: E402

CAT = {"PACE_FAST": "Pacing", "PACE_SLOW": "Pacing", "PAUSE_BAD": "Pausing", "PAUSE_LOST": "Pausing", "MONOTONE": "Intonation", "UPTALK": "Intonation",
       "EMPH_FLAT": "Intonation", "FADE": "Volume", "SHOUT": "Volume", "FILLER": "Fluency", "REPEAT": "Fluency", "RARE_HESIT": "Fluency",
       "SLUR": "Clarity", "WORD_SKIP": "Text fidelity", "WORD_SWAP": "Text fidelity"}
TARGET = {3: 77.5, 4: 66.0, 5: 55.0}
PKL = Path("/tmp/tau_data.pkl")


def collect(mode="same"):
    labs = [l for l in T.clips("train") if l["what"] and len({w["flaw"] for w in l["what"]}) == 1 and not l.get("multi_set")]
    out = []
    for l in labs:
        lv = l["what"][0].get("level")
        if lv not in (1, 2, 3, 4, 5):
            continue
        meta = {k: v for k, v in l.items() if k not in ("what", "seed", "join_check", "multi_set")}
        p = P.predict(str(T.DATA / "variants" / f"{l['clip_id']}.flac"), meta, mode)
        out.append({"clip": l["clip_id"], "flaw": l["what"][0]["flaw"], "level": lv, "what": p["what"], "dur": l["duration_s"], "genre": l.get("genre"), "badge": p["quality"]["badge"]})
    PKL.write_bytes(pickle.dumps(out))
    print("collected", len(out))


def fit(write=False):
    D = pickle.loads(PKL.read_bytes())
    rub = S.load_rubric()
    rub["flaw_weight"] = {}
    tau = rub["tau"]
    w = {}
    for f, cat in CAT.items():
        num = den = 0.0
        for lv, tgt in TARGET.items():
            obs = [-tau * np.log(max(S.score(d["what"], d["dur"], d["genre"], rub)["categories"][cat], 1e-3) / 100.0) for d in D if d["flaw"] == f and d["level"] == lv]
            if obs:
                p_t = -tau * np.log(tgt / 100.0)
                po = float(np.mean(obs))
                num += po * p_t
                den += po * po
        w[f] = round(float(np.clip(num / den, 0.4, 2.0)), 2) if den > 1e-6 else 1.0
    print("flaw_weight:", w)
    if write:
        p = ROOT / "engine" / "rubric.yaml"
        s = p.read_text()
        s = re.sub(r"^flaw_weight:.*$", "flaw_weight: " + "{" + ", ".join(f"{k}: {v}" for k, v in w.items()) + "}   # fitted on TRAIN by engine/fit_scoring.py", s, flags=re.M)
        p.write_text(s)
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.collect:
        collect()
    if a.fit:
        fit(a.write)
