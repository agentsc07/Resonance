"""Leakage audit (spec v1.1, cross-cutting #7): can a classifier find our flaws from EDITING ARTIFACTS alone?

Features never look at delivery or content: click energy at the join, noise-floor step across it, spectral-flux spike and sample-exact
repetition (peak normalised cross-correlation). PAIRED control: a positive is a window centred on a region edge of a flawed clip; its
negative is the window at the SAME place in the clean baseline (baseline time of that edge). Content, position and word boundaries are
therefore identical, and only the edit differs. (An unpaired design with random negatives scores 0.69 but only because positives sit at word
boundaries: level features then separate boundary from mid-vowel, which is content, not an artifact.) Train on the train split, report AUC
on the held-out test split. Near 0.5 means the dataset tests delivery, not editing; the pass mark is AUC <= 0.6.

   python eval/leakage_audit.py [--max-clips N]        -> results/leakage.csv, prints AUC overall and per flaw
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "flawline-dataset" / "generator"))
from common import SR, TAKES, click_db, db, frame_rms_db, read_audio, rms, s2n  # noqa: E402

DATA = ROOT / "flawline-dataset"
FEATS = ["click_db", "floor_step_db", "flux_peak", "repeat_corr"]


def window_features(x: np.ndarray, t: float) -> list[float]:
    c = s2n(t)
    h = s2n(0.30)
    seg = x[max(0, c - h): c + h]
    if len(seg) < s2n(0.4):
        return [np.nan] * len(FEATS)
    ck = click_db(x, t)
    fl = frame_rms_db(x[max(0, c - s2n(0.5)): c])
    fr = frame_rms_db(x[c: c + s2n(0.5)])
    floor_step = abs(np.percentile(fl, 10) - np.percentile(fr, 10)) if len(fl) > 3 and len(fr) > 3 else 0.0
    # spectral flux spike near the centre vs the window's median flux
    n_fft, hop = 512, 128
    win = np.hanning(n_fft)
    frames = [np.abs(np.fft.rfft(seg[i: i + n_fft] * win)) for i in range(0, len(seg) - n_fft, hop)]
    F = np.log(np.array(frames) + 1e-6)
    flux = np.sqrt(np.sum(np.diff(F, axis=0) ** 2, axis=1))
    mid = len(flux) // 2
    peak = float(flux[max(0, mid - 6): mid + 6].max() / (np.median(flux) + 1e-6))
    # sample-exact repetition: the 150 ms block before the join vs everything in the next 1 s
    blk = x[max(0, c - s2n(0.15)): c].astype(np.float64)
    nxt = x[c: c + s2n(1.0)].astype(np.float64)
    best = 0.0
    if len(blk) > 100 and len(nxt) > len(blk):
        from scipy.signal import fftconvolve
        num = fftconvolve(nxt, blk[::-1], mode="valid")
        e2 = np.sqrt(np.convolve(nxt ** 2, np.ones(len(blk)), mode="valid") * np.sum(blk ** 2)) + 1e-9
        best = float(np.max(num[s2n(0.12):] / e2[s2n(0.12):])) if len(num) > s2n(0.12) else 0.0
    return [ck, floor_step, peak, best]


def build(split: str, max_clips: int | None, rng):
    """Paired windows. Returns X, y, flaw-of-edit."""
    rows = list(csv.DictReader(open(DATA / "manifest.csv")))
    flawed = [r for r in rows if r["split"] == split and not r["added_condition"] and r["flaw_codes"]]
    if max_clips:
        flawed = flawed[:max_clips]
    X, y, meta = [], [], []
    base_cache: dict = {}
    for r in flawed:
        lab = json.loads((DATA / "variants" / f"{r['clip_id']}.json").read_text())
        x, _ = read_audio(DATA / "variants" / f"{r['clip_id']}.flac")
        if lab["take_id"] not in base_cache:
            base_cache[lab["take_id"]] = read_audio(TAKES / f"{lab['take_id']}_C0.flac")[0]
        xb = base_cache[lab["take_id"]]
        for reg in lab["what"]:
            pts = [("start_s", "baseline_start_s")] + ([("end_s", "baseline_end_s")] if reg["end_s"] - reg["start_s"] > 0.02 else [])
            for kv, kb in pts:
                if not (0.4 < reg[kv] < len(x) / SR - 0.4 and 0.4 < reg[kb] < len(xb) / SR - 0.4):
                    continue
                X.append(window_features(x, reg[kv])); y.append(1); meta.append(reg["flaw"])
                X.append(window_features(xb, reg[kb])); y.append(0); meta.append(reg["flaw"])
    return np.nan_to_num(np.array(X, float), nan=0.0), np.array(y), np.array(meta)


def main():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-clips", type=int)
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    Xtr, ytr, _ = build("train", a.max_clips, rng)
    Xte, yte, mte = build("test", a.max_clips, rng)
    clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0).fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    rows = [{"scope": "ALL", "n_pos": int(yte.sum()), "n_neg": int((1 - yte).sum()), "auc": round(float(auc), 3)}]
    for f in sorted(set(mte)):
        m = mte == f                                   # paired: both members of every pair carry the flaw label
        if (yte[m] == 1).sum() >= 4 and (yte[m] == 0).sum() >= 4:
            rows.append({"scope": f, "n_pos": int(((yte == 1) & m).sum()), "n_neg": int(((yte == 0) & m).sum()),
                         "auc": round(float(roc_auc_score(yte[m], p[m])), 3)})
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    with open(out / "leakage.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    imp = sorted(zip(FEATS, clf.feature_importances_), key=lambda kv: -kv[1])
    print(f"train windows {len(ytr)} ({int(ytr.sum())} pos) | test windows {len(yte)} ({int(yte.sum())} pos)")
    for r in rows:
        print(f"  {r['scope']:12s} AUC {r['auc']:.3f}  (pos {r['n_pos']}, neg {r['n_neg']})")
    print("feature importance:", ", ".join(f"{k} {v:.2f}" for k, v in imp))
    print("PASS" if auc <= 0.6 else "ABOVE the 0.6 pass mark: the dataset still leaks editing artifacts", f"(overall AUC {auc:.3f}, pass mark <= 0.60)")


if __name__ == "__main__":
    main()
