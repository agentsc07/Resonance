"""Scoring (spec v1.1): every point lost traces to a timestamped region with a measured cause. Deterministic, rubric-driven.
  step 1  severity s in [0,5] = per-flaw ISOTONIC map of the measured deviation d (monotone: a bigger deviation never scores milder)
  step 2  penalty p = w_f * c * s^1.5 * m        (c = confidence, m = 1 for events, region seconds/2 capped at 3 for spans)
  step 3  P_dim = sum p / clip minutes ; S_dim = 100 exp(-P_dim/tau) ; overall = genre-weighted mean ; bands."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).parent


def load_rubric(path: str | None = None) -> dict:
    return yaml.safe_load(open(path or HERE / "rubric.yaml"))


def load_iso(mode: str = "same") -> dict:
    mp = HERE / "model.json"
    if not mp.exists():
        return {}
    allm = json.loads(mp.read_text())
    return (allm.get(mode) or allm.get("same") or {}).get("isotonic", {})


def load_reliability(mode: str = "same") -> dict:
    """Per-flaw reliability = the detector's precision on the TRAIN split at its calibrated thresholds (floored so no detector is silenced)."""
    mp = HERE / "model.json"
    if not mp.exists():
        return {}
    allm = json.loads(mp.read_text())
    return (allm.get(mode) or allm.get("same") or {}).get("reliability", {})


def severity(flaw: str, d: float, iso: dict | None = None, mode: str = "same") -> float:
    iso = iso if iso is not None else load_iso(mode)
    m = iso.get(flaw)
    if not m:
        return float(np.clip(d, 0, 5))
    return float(np.clip(np.interp(d, m["x"], m["y"]), 0.0, 5.0))


def band(score: float, rub: dict, worst: float | None = None) -> str:
    """Band of the overall score, capped by the worst area so one badly hurt category cannot hide behind an average:
    an area below 50 rules out polished and strong, an area below 30 means needs work."""
    b = rub["bands"]
    out = "polished" if score >= b["polished"] else "strong" if score >= b["strong"] else "noticeable" if score >= b["noticeable"] else "needs work"
    if worst is not None:
        if worst < 30:
            out = "needs work"
        elif worst < 50 and out in ("polished", "strong"):
            out = "noticeable"
    return out


def score(regions: list[dict], duration_s: float, genre: str | None, rub: dict | None = None, dont_score_fluency: bool = False, badge: str | None = None, rough: bool = False) -> dict:
    """regions: dicts with flaw, category, start_s, end_s, severity (0-5), confidence."""
    rub = rub or load_rubric()
    mins = max(duration_s / 60.0, 0.25)
    W = dict(rub["genre_weights"].get((genre or "").lower(), rub["genre_weights"]["interpretive reading"]))
    P = {c: 0.0 for c in rub["categories"]}
    rough = rough or badge in ("fair", "poor")             # a rough recording: quality gate fair/poor, or its conditions differ from the clean reference
    n_counted = {c: 0 for c in rub["categories"]}
    for r in regions:
        n_counted[r["category"]] += int(float(r.get("confidence", 1.0)) >= rub["min_confidence"] and not (dont_score_fluency and r["category"] == "Fluency"))
    lone = {c for c, n in n_counted.items() if n == 1 and rough}      # a lone flag in a rough recording is shown and counts half, but cannot be the weakest area
    rows = []
    for r in regions:
        conf = float(r.get("confidence", 1.0))
        s = float(r["severity"])
        m = 1.0 if r["flaw"] in rub["event_flaws"] else min(rub["span_cap"], max(0.0, r["end_s"] - r["start_s"]) / 2.0)
        w = float(rub["flaw_weight"].get(r["flaw"], 1.0))
        counted = conf >= rub["min_confidence"] and not (dont_score_fluency and r["category"] == "Fluency")
        p = w * conf * float(r.get("reliability", 1.0)) * (s ** rub["severity_exponent"]) * m if counted else 0.0
        if r["category"] in lone:
            p *= rub.get("lone_discount", 0.5)
        P[r["category"]] += p
        rows.append({**{k: r[k] for k in ("flaw", "category", "start_s", "end_s")}, "severity": round(s, 2), "points_lost": 0.0,
                     "penalty": round(p, 3), "counted": counted, "explanation": r.get("explanation", "")})
    if dont_score_fluency and W.get("Fluency"):            # share the fluency weight out across the other categories
        f = W["Fluency"]
        rest = sum(v for k, v in W.items() if k != "Fluency")
        W = {k: (0.0 if k == "Fluency" else v + f * v / rest) for k, v in W.items()}
    cat = {c: round(100.0 * float(np.exp(-(P[c] / mins) / rub["tau"])), 1) for c in rub["categories"]}
    mean = sum(W[c] * cat[c] for c in rub["categories"]) / max(sum(W.values()), 1e-9)
    live = [cat[c] for c in rub["categories"] if W[c] > 0]                      # areas that count under this genre / rubric
    live = [cat[c] for c in rub["categories"] if W[c] > 0 and c not in lone] or live
    worst = min(live) if live else 100.0
    overall = round(rub.get("blend_mean", 0.6) * mean + (1 - rub.get("blend_mean", 0.6)) * worst, 1)   # one bad area drags the score, an average cannot hide it
    # points lost by a region = its share of its category's loss, expressed in overall points
    for row in rows:
        c = row["category"]
        loss_c = 100.0 - cat[c]
        row["points_lost"] = round(W[c] * loss_c * (row["penalty"] / P[c]) / max(sum(W.values()), 1e-9), 2) if P[c] > 0 else 0.0
    return {"overall": overall, "band": band(overall, rub, worst), "worst": worst, "mean": round(mean, 1), "categories": cat, "weights": W, "regions": rows, "minutes": round(mins, 2)}
