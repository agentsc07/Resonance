"""Leakage audit (spec v1.1, cross-cutting #7): can a classifier find our flaws from EDITING ARTIFACTS alone?

Features never look at delivery or content: click energy at the join, noise-floor step across it, spectral-flux spike and sample-exact
repetition (peak normalised cross-correlation). PAIRED control (audit v2: for flaws that ARE a silence, PAUSE_BAD and RARE_HESIT, the negative is a natural silence edge of the same baseline, because the position-matched negative differs by the pause itself, i.e. by the flaw, and not by an artifact): a positive is a window centred on a region edge of a flawed clip; its
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


def _floor_step(x: np.ndarray, c: int) -> float:
    """Noise-floor continuity across the join (audit v2). Compares the ROOM TONE on either side, using only frames within 6 dB of the clip's own
    floor. v1 compared the 10th percentile of each 0.5 s side, which at a pause edge compares speech with silence, i.e. it measured the pause
    (the flaw itself) and not an editing artifact. Zero when a side has no quiet frames, for positives and paired negatives alike."""
    allf = frame_rms_db(x)
    floor = np.percentile(allf, 5)
    sides = []
    for seg in (x[max(0, c - s2n(0.6)): c], x[c: c + s2n(0.6)]):
        f = frame_rms_db(seg)
        q = f[f <= floor + 6.0]
        sides.append(float(np.median(q)) if len(q) >= 3 else np.nan)
    return float(abs(sides[0] - sides[1])) if not np.isnan(sides).any() else 0.0


def window_features(x: np.ndarray, t: float) -> list[float]:
    c = s2n(t)
    h = s2n(0.30)
    seg = x[max(0, c - h): c + h]
    if len(seg) < s2n(0.4):
        return [np.nan] * len(FEATS)
    ck = click_db(x, t)
    floor_step = _floor_step(x, c)
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


SILENCE_FLAWS = {"PAUSE_BAD", "RARE_HESIT"}     # the edit IS a silence: the control is a NATURAL silence edge in the same baseline (audit v2)


def natural_silence_edges(x: np.ndarray, min_s: float = 0.25) -> tuple[list[float], list[float]]:
    """Start and end times (s) of natural silences >= min_s in a clean baseline (frames within 6 dB of its floor)."""
    f = frame_rms_db(x)
    quiet = f <= np.percentile(f, 5) + 6.0
    starts, ends, i, n = [], [], 0, len(quiet)
    while i < n:
        if quiet[i]:
            j = i
            while j + 1 < n and quiet[j + 1]:
                j += 1
            if (j - i + 1) * 0.010 >= min_s and i * 0.010 > 0.5 and j * 0.010 < len(x) / SR - 0.5:
                starts.append(i * 0.010)
                ends.append((j + 1) * 0.010)
            i = j + 1
        else:
            i += 1
    return starts, ends


def snap_to_silence_edge(x: np.ndarray, t: float, start: bool, win: float = 0.35) -> float:
    """Re-centre a window on the acoustic landmark both classes share: the nearest silence onset (start=True) or offset (start=False) within
    +-win of the labelled time. Natural negatives are centred on such landmarks by construction; without this, positives would be centred on
    the label convention and the classifier could separate them by the offset alone."""
    f = frame_rms_db(x)
    quiet = f <= np.percentile(f, 5) + 6.0
    lo, hi = max(1, int((t - win) / 0.010)), min(len(quiet) - 2, int((t + win) / 0.010))
    cand = [i for i in range(lo, hi) if (quiet[i] and not quiet[i - 1]) == start and (quiet[i - 1] and not quiet[i]) == (not start)] if False else \
           [i for i in range(lo, hi) if (quiet[i] and not quiet[i - 1] if start else quiet[i - 1] and not quiet[i])]
    return min((i * 0.010 for i in cand), key=lambda v: abs(v - t)) if cand else t


def build(split: str, max_clips: int | None, rng):
    """Paired windows. Returns X, y, flaw-of-edit."""
    rows = list(csv.DictReader(open(DATA / "manifest.csv")))
    flawed = [r for r in rows if r["split"] == split and not r["added_condition"] and r["flaw_codes"]]
    if max_clips:
        flawed = flawed[:max_clips]
    X, y, meta, spk = [], [], [], []
    base_cache: dict = {}
    sil_cache: dict = {}
    for r in flawed:
        lab = json.loads((DATA / "variants" / f"{r['clip_id']}.json").read_text())
        x, _ = read_audio(DATA / "variants" / f"{r['clip_id']}.flac")
        if lab["take_id"] not in base_cache:
            base_cache[lab["take_id"]] = read_audio(TAKES / f"{lab['take_id']}_C0.flac")[0]
        xb = base_cache[lab["take_id"]]
        for reg in lab["what"]:
            pts = [("start_s", "baseline_start_s")] + ([("end_s", "baseline_end_s")] if reg["end_s"] - reg["start_s"] > 0.02 else [])
            if reg["flaw"] in SILENCE_FLAWS and lab["take_id"] not in sil_cache:
                sil_cache[lab["take_id"]] = natural_silence_edges(xb)
            for kv, kb in pts:
                if not (0.4 < reg[kv] < len(x) / SR - 0.4 and 0.4 < reg[kb] < len(xb) / SR - 0.4):
                    continue
                if reg["flaw"] in SILENCE_FLAWS:
                    edges = sil_cache[lab["take_id"]][0 if kv == "start_s" else 1]
                    if not edges:
                        continue
                    t_neg = float(edges[int(rng.integers(len(edges)))])
                else:
                    t_neg = reg[kb]
                t_pos = snap_to_silence_edge(x, reg[kv], kv == "start_s") if reg["flaw"] in SILENCE_FLAWS else reg[kv]
                X.append(window_features(x, t_pos)); y.append(1); meta.append(reg["flaw"]); spk.append(lab["baseline_id"])
                X.append(window_features(xb, t_neg)); y.append(0); meta.append(reg["flaw"]); spk.append(lab["baseline_id"])
    return np.nan_to_num(np.array(X, float), nan=0.0), np.array(y), np.array(meta), np.array(spk)


def loso(max_clips=None):
    """Leave-one-speaker-out over the TRAIN+DEV speakers (never the test speakers): out-of-fold AUC overall and per flaw. About 5x more windows
    than the held-out test split alone, so per-flaw numbers stop being noise (8-24 windows per flaw on the test split)."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(7)
    parts = [build(sp, max_clips, rng) for sp in ("train", "dev")]
    X, y, m, spk = (np.concatenate([p[i] for p in parts]) for i in range(4))
    pred = np.zeros(len(y))
    for s_ in sorted(set(spk)):
        te = spk == s_
        clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0).fit(X[~te], y[~te])
        pred[te] = clf.predict_proba(X[te])[:, 1]
    rows = [{"scope": "ALL", "n_pos": int(y.sum()), "auc": round(float(roc_auc_score(y, pred)), 3)}]
    for f in sorted(set(m)):
        k = m == f
        if (y[k] == 1).sum() >= 6 and (y[k] == 0).sum() >= 6:
            rows.append({"scope": f, "n_pos": int((y[k] == 1).sum()), "auc": round(float(roc_auc_score(y[k], pred[k])), 3)})
    return rows


def main():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-clips", type=int)
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    Xtr, ytr, _, _ = build("train", a.max_clips, rng)
    Xte, yte, mte, _ = build("test", a.max_clips, rng)
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
    print("\nleave-one-speaker-out over train+dev speakers (more windows, no test speakers):")
    lo = loso(a.max_clips)
    for r in lo:
        print(f"  {r['scope']:12s} AUC {r['auc']:.3f}  (pos {r['n_pos']})")
    with open(out / "leakage_loso.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lo[0]))
        w.writeheader()
        w.writerows(lo)
    print("PASS" if auc <= 0.6 else "ABOVE the 0.6 pass mark: the dataset still leaks editing artifacts", f"(overall AUC {auc:.3f}, pass mark <= 0.60)")


if __name__ == "__main__":
    main()
