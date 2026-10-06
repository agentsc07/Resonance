"""Evaluation metrics for Flawline (spec v1.1 "Evaluation harness"). Pure functions over label/prediction dicts in schema 1.1.0.

Time base: CLIP time (start_s/end_s). Insert/delete flaws can have a (near) zero-length region, so every region is padded to a minimum
width before IoU, identically for ground truth and prediction.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

MIN_W = 0.30        # seconds: point-like events (deletions, short inserts) are widened to this before IoU


def _iv(r: dict) -> tuple[float, float]:
    a, b = float(r["start_s"]), float(r["end_s"])
    if b - a < MIN_W:
        c = (a + b) / 2
        a, b = c - MIN_W / 2, c + MIN_W / 2
    return a, b


def iou(r1: dict, r2: dict) -> float:
    (a1, b1), (a2, b2) = _iv(r1), _iv(r2)
    inter = max(0.0, min(b1, b2) - max(a1, a2))
    union = (b1 - a1) + (b2 - a2) - inter
    return inter / union if union > 0 else 0.0


def match(gt: list[dict], pred: list[dict], thr: float, key=None):
    """Greedy one-to-one matching by descending IoU. key(r) restricts matches to equal keys (e.g. flaw code or category).
    Returns list of (gt_idx, pred_idx, iou)."""
    cand = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            if key is not None and key(g) != key(p):
                continue
            v = iou(g, p)
            if v >= thr:
                cand.append((v, i, j))
    cand.sort(reverse=True)
    used_g, used_p, out = set(), set(), []
    for v, i, j in cand:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        out.append((i, j, v))
    return out


def prf(tp: int, n_gt: int, n_pred: int) -> tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def grounding(pairs: list[tuple[dict, dict]], thr: float, level: str = "flaw") -> dict:
    """Event precision/recall/F1 at IoU>=thr over all clips; level = 'flaw' (code must match) or 'category'."""
    key = (lambda r: r["flaw"]) if level == "flaw" else (lambda r: r["category"])
    per: dict[str, list[int]] = {}
    tot = [0, 0, 0]
    for gt, pr in pairs:
        g, p = gt["what"], pr["what"]
        m = match(g, p, thr, key)
        tp = len(m)
        tot[0] += tp
        tot[1] += len(g)
        tot[2] += len(p)
        for lab in {key(r) for r in g} | {key(r) for r in p}:
            gl = sum(key(r) == lab for r in g)
            pl = sum(key(r) == lab for r in p)
            tpl = sum(key(g[i]) == lab for i, _, _ in m)
            a = per.setdefault(lab, [0, 0, 0])
            a[0] += tpl
            a[1] += gl
            a[2] += pl
    out = {"overall": dict(zip(("precision", "recall", "f1"), prf(*tot)), tp=tot[0], n_gt=tot[1], n_pred=tot[2])}
    out["per"] = {k: dict(zip(("precision", "recall", "f1"), prf(*v)), tp=v[0], n_gt=v[1], n_pred=v[2]) for k, v in sorted(per.items())}
    return out


def timing(pairs, thr: float = 0.3) -> dict:
    on, off = [], []
    for gt, pr in pairs:
        for i, j, _ in match(gt["what"], pr["what"], thr, lambda r: r["flaw"]):
            g, p = gt["what"][i], pr["what"][j]
            on.append(abs(g["start_s"] - p["start_s"]) * 1000)
            off.append(abs(g["end_s"] - p["end_s"]) * 1000)
    return {"n": len(on), "median_onset_ms": float(np.median(on)) if on else float("nan"),
            "median_offset_ms": float(np.median(off)) if off else float("nan")}


def category_confusion(pairs, cats: list[str], thr: float = 0.3):
    """Confusion matrix over regions matched on time alone (any flaw), then category accuracy."""
    idx = {c: k for k, c in enumerate(cats)}
    M = np.zeros((len(cats), len(cats)), int)
    for gt, pr in pairs:
        for i, j, _ in match(gt["what"], pr["what"], thr, None):
            M[idx[gt["what"][i]["category"]], idx[pr["what"][j]["category"]]] += 1
    acc = float(np.trace(M) / M.sum()) if M.sum() else float("nan")
    return M, acc


def dose_response(pairs, thr: float = 0.3) -> dict:
    """Per flaw: Spearman between injected level and predicted severity over matched regions."""
    by: dict[str, tuple[list, list]] = {}
    for gt, pr in pairs:
        for i, j, _ in match(gt["what"], pr["what"], thr, lambda r: r["flaw"]):
            g, p = gt["what"][i], pr["what"][j]
            if g.get("level") is None or p.get("severity") is None:
                continue
            xs, ys = by.setdefault(g["flaw"], ([], []))
            xs.append(g["level"])
            ys.append(p["severity"])
    out = {}
    for f, (x, y) in sorted(by.items()):
        out[f] = {"n": len(x), "spearman": float(spearmanr(x, y).statistic) if len(set(x)) > 1 and len(set(y)) > 1 else float("nan")}
    return out


def false_flags_per_min(pairs) -> dict:
    """On CLEAN clips (ground truth has no regions): predicted regions per minute."""
    n, mins = 0, 0.0
    for gt, pr in pairs:
        if not gt["what"]:
            n += len(pr["what"])
            mins += gt["duration_s"] / 60.0
    return {"clean_clips_minutes": mins, "false_flags": n, "per_minute": n / mins if mins else float("nan")}
