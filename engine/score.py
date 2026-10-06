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


def severity(flaw: str, d: float, iso: dict | None = None, mode: str = "same") -> float:
    iso = iso if iso is not None else load_iso(mode)
    m = iso.get(flaw)
    if not m:
        return float(np.clip(d, 0, 5))
    return float(np.clip(np.interp(d, m["x"], m["y"]), 0.0, 5.0))


def band(score: float, rub: dict) -> str:
    b = rub["bands"]
    return "polished" if score >= b["polished"] else "strong" if score >= b["strong"] else "noticeable" if score >= b["noticeable"] else "needs work"


def score(regions: list[dict], duration_s: float, genre: str | None, rub: dict | None = None, dont_score_fluency: bool = False) -> dict:
    """regions: dicts with flaw, category, start_s, end_s, severity (0-5), confidence."""
    rub = rub or load_rubric()
    mins = max(duration_s / 60.0, 0.25)
    W = dict(rub["genre_weights"].get((genre or "").lower(), rub["genre_weights"]["interpretive reading"]))
    P = {c: 0.0 for c in rub["categories"]}
    rows = []
    for r in regions:
        conf = float(r.get("confidence", 1.0))
        s = float(r["severity"])
        m = 1.0 if r["flaw"] in rub["event_flaws"] else min(rub["span_cap"], max(0.0, r["end_s"] - r["start_s"]) / 2.0)
        w = float(rub["flaw_weight"].get(r["flaw"], 1.0))
        counted = conf >= rub["min_confidence"] and not (dont_score_fluency and r["category"] == "Fluency")
        p = w * conf * (s ** rub["severity_exponent"]) * m if counted else 0.0
        P[r["category"]] += p
        rows.append({**{k: r[k] for k in ("flaw", "category", "start_s", "end_s")}, "severity": round(s, 2), "points_lost": 0.0,
                     "penalty": round(p, 3), "counted": counted, "explanation": r.get("explanation", "")})
    if dont_score_fluency and W.get("Fluency"):            # share the fluency weight out across the other categories
        f = W["Fluency"]
        rest = sum(v for k, v in W.items() if k != "Fluency")
        W = {k: (0.0 if k == "Fluency" else v + f * v / rest) for k, v in W.items()}
    cat = {c: round(100.0 * float(np.exp(-(P[c] / mins) / rub["tau"])), 1) for c in rub["categories"]}
    overall = round(sum(W[c] * cat[c] for c in rub["categories"]) / max(sum(W.values()), 1e-9), 1)
    # points lost by a region = its share of its category's loss, expressed in overall points
    for row in rows:
        c = row["category"]
        loss_c = 100.0 - cat[c]
        row["points_lost"] = round(W[c] * loss_c * (row["penalty"] / P[c]) / max(sum(W.values()), 1e-9), 2) if P[c] > 0 else 0.0
    return {"overall": overall, "band": band(overall, rub), "categories": cat, "weights": W, "regions": rows, "minutes": round(mins, 2)}
