"""Spec v1.1 "What the score must prove": dose-response, invariance, locality (reproducibility is checked separately).
   python eval/acceptance.py --mode same        -> results/acceptance.json (+ prints a table)
Predictions are cached in /tmp/engine_pred_cache so reruns are fast."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import predict as P, score as SC  # noqa: E402

DATA = ROOT / "flawline-dataset"
CACHE = Path("/tmp/engine_pred_cache")
CAT_OF = {"PACE_FAST": "Pacing", "PACE_SLOW": "Pacing", "PAUSE_BAD": "Pausing", "PAUSE_LOST": "Pausing", "MONOTONE": "Intonation", "UPTALK": "Intonation",
          "EMPH_FLAT": "Intonation", "FADE": "Volume", "SHOUT": "Volume", "FILLER": "Fluency", "REPEAT": "Fluency", "RARE_HESIT": "Fluency",
          "SLUR": "Clarity", "WORD_SKIP": "Text fidelity", "WORD_SWAP": "Text fidelity"}


def label_of(clip_id: str) -> dict:
    p = DATA / "variants" / f"{clip_id}.json"
    if p.exists():
        return json.loads(p.read_text())
    r = next(r for r in csv.DictReader(open(DATA / "manifest.csv")) if r["clip_id"] == clip_id)
    return {"clip_id": clip_id, "take_id": r["take_id"], "baseline_id": r["baseline_id"], "genre": r["genre"], "who": {"speaker_id": r["speaker_id"]},
            "where": {"base_condition": "C0", "added_condition": None}, "what": [], "duration_s": float(r["duration_s"])}


def _rescore(out: dict) -> dict:
    """Scores are recomputed from the cached regions with the CURRENT rubric, so scoring changes need no new engine run."""
    q = out["quality"]
    sc = SC.score(out["what"], out["duration_s"], out.get("genre"), None, False, q.get("badge"), bool(q.get("matched")))
    out["scores"] = {**out["scores"], "overall": sc["overall"], "band": sc["band"], "worst_area_score": round(sc["worst"], 1), "categories": sc["categories"]}
    return out


def pred(clip_id: str, mode: str) -> dict:
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{clip_id}__{mode}.json"
    if f.exists():
        return _rescore(json.loads(f.read_text()))
    lab = label_of(clip_id)
    meta = {k: v for k, v in lab.items() if k not in ("what", "seed", "join_check", "multi_set", "scenario")}
    audio = DATA / "variants" / f"{clip_id}.flac"
    if not audio.exists():
        audio = DATA / "takes" / f"{clip_id}.flac"
    out = P.predict(str(audio), meta, mode)
    f.write_text(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="same")
    ap.add_argument("--max-locality", type=int, default=60)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(DATA / "manifest.csv")))
    res = {"mode": a.mode}

    # ---- 1. dose-response: B01 has the full 5-level pilot grid (a TRAIN speaker, so this is a sanity bound, not a held-out result)
    clean = pred("B01-CHAMP_C0", a.mode)["scores"]["overall"]
    dr, bad = {}, []
    for f in CAT_OF:
        sc = [clean]
        for lv in range(1, 6):
            cid = next((r["clip_id"] for r in rows if r["clip_id"].startswith(f"B01-CHAMP_C0__{f}_L{lv}_s")), None)
            if cid:
                sc.append(pred(cid, a.mode)["scores"]["overall"])
        if len(sc) == 6:
            rho = float(spearmanr(range(6), sc).statistic)
            dr[f] = {"scores_L0_to_L5": sc, "spearman": round(rho, 3)}
            if rho > -0.9:
                bad.append(f)
    res["dose_response"] = {"min_ok": -0.9, "per_flaw": dr, "failing": bad, "median_spearman": round(float(np.median([v["spearman"] for v in dr.values()])), 3)}

    # ---- 2. invariance: clean take under each Where condition: |score shift| < 3 and false flags per minute <= 0.5
    shifts, ff, mins = [], 0, 0.0
    for r in rows:
        if r["flaw_codes"] == "" and r["added_condition"]:
            p = pred(r["clip_id"], a.mode)
            base = pred(f"{r['take_id']}_C0", a.mode)["scores"]["overall"]
            shifts.append((r["added_condition"], abs(p["scores"]["overall"] - base)))
            ff += len(p["what"])
            mins += float(r["duration_s"]) / 60
    by = {}
    for c, s in shifts:
        by.setdefault(c, []).append(s)
    res["invariance"] = {"false_flags_per_min": round(ff / max(mins, 1e-9), 3), "max_score_shift": round(max(s for _, s in shifts), 2) if shifts else None,
                         "mean_shift_by_condition": {c: round(float(np.mean(v)), 2) for c, v in by.items()}, "n": len(shifts), "pass_under_10_points": bool(shifts and max(s for _, s in shifts) < 10),
                         "pass": bool(shifts and max(s for _, s in shifts) < 3 and ff / max(mins, 1e-9) <= 0.5)}

    # ---- 3. locality: single-flaw L3 clips: points lost in categories OTHER than the flaw's own
    leak = []
    clips = [r["clip_id"] for r in rows if "_L3_" in r["clip_id"] and not r["added_condition"] and r["multi_set"] == "" and r["flaw_codes"] in CAT_OF
             and "PURE_" not in r["clip_id"]][: a.max_locality]
    for cid in clips:
        lab = label_of(cid)
        own = CAT_OF[lab["what"][0]["flaw"]]
        cats = pred(cid, a.mode)["scores"]["categories"]
        leak.append(sum(100 - v for k, v in cats.items() if k != own))
    res["locality"] = {"mean_other_category_loss": round(float(np.mean(leak)), 2) if leak else None, "n": len(leak), "pass": bool(leak and np.mean(leak) < 2)}

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / f"acceptance_{a.mode}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: (v if k != "dose_response" else {kk: vv for kk, vv in v.items() if kk != "per_flaw"}) for k, v in res.items()}, indent=1))
    print("dose-response per flaw (score L0..L5):")
    for f, v in dr.items():
        print(f"  {f:11s} {v['scores_L0_to_L5']}  rho {v['spearman']}")


if __name__ == "__main__":
    main()
