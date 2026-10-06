"""Objective splice/artifact meter for altered clips (a proxy for "does it sound awkward"; it does NOT replace listening).

For every region edge in clip time we measure, relative to what the SAME place in the baseline does naturally:
  click_db     high-band (>2 kHz) energy in a 10 ms window at the join vs the local median      (cuts landing mid-sound)
  step_db      level jump across the join (40 ms before vs after) minus the baseline's natural jump  (level/noise mismatch)
  timbre_db    for same-length edits (pitch/gain/clarity): mean log-mel distance region vs baseline region  (vocoder timbre shift)
  floor_db     for inserted silence: log-mel distance of the inserted tone vs the clip's own quiet frames  (fake room tone)
Lower is better for all four. Output: pilot/artifact_check.csv + per-flaw summary.   Usage: python artifact_check.py [clip-glob]
"""
from __future__ import annotations

import csv
import json
import sys

import librosa
import numpy as np
from scipy.signal import butter, sosfilt

from common import PILOT, SR, TAKES, VARIANTS, click_db, db, join_clicks, read_audio, rms, s2n

HP = butter(4, 2000, btype="high", fs=SR, output="sos")


def hp_energy_db(x):
    return db(rms(sosfilt(HP, x))) if len(x) else -120.0


def step_db(x, t):
    w = s2n(0.040)
    c = s2n(t)
    return abs(db(rms(x[c: c + w])) - db(rms(x[max(0, c - w): c])))


def logmel(x):
    if len(x) < 1024:
        x = np.pad(x, (0, 1024 - len(x)))
    m = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=1024, hop_length=256, n_mels=40, fmax=8000)
    return 10 * np.log10(m + 1e-10)


def mel_dist(a, b):
    return float(np.mean(np.abs(logmel(a).mean(axis=1) - logmel(b).mean(axis=1))))


def frame_db(x):
    h = 441
    return np.array([db(rms(x[i:i + h])) for i in range(0, len(x) - h, h)])


def analyse(label, xb, xv):
    rows = []
    fr = frame_db(xb)
    quiet_idx = np.argsort(fr)[: max(3, len(fr) // 20)]
    quiet = np.concatenate([xb[i * 441:(i + 1) * 441] for i in quiet_idx])
    for r in label["what"]:
        k = r["kind"]
        cs, ce, bs, be = r["start_s"], r["end_s"], r["baseline_start_s"], r["baseline_end_s"]
        edges_clip = [("start", cs), ("end", ce)] if ce - cs > 0.02 else [("point", cs)]
        edges_base = {"start": bs, "end": be, "point": bs}
        for name, t in edges_clip:
            if t < 0.05 or t > len(xv) / SR - 0.05:
                continue
            tb = min(max(edges_base[name], 0.05), len(xb) / SR - 0.05)
            exc = click_db(xv, t) - click_db(xb, tb)                      # excess over what the baseline does naturally here
            stp = "" if k == "insert" else round(max(0.0, step_db(xv, t) - step_db(xb, tb)), 2)   # a drop to silence is natural at an insert
            rows.append({"clip_id": label["clip_id"], "flaw": r["flaw"], "level": r["level"], "edge": name,
                         "click_db": round(max(0.0, exc), 2), "step_db": stp, "timbre_db": "", "floor_db": ""})
        if k == "modify" and abs((ce - cs) - (be - bs)) < 0.03:        # same-length edit: timbre shift
            rows[-1]["timbre_db"] = round(mel_dist(xb[s2n(bs): s2n(be)], xv[s2n(cs): s2n(ce)]), 2)
        if k == "insert" and r["flaw"] in ("PAUSE_BAD", "RARE_HESIT") and ce - cs > 0.15:
            rows[-1]["floor_db"] = round(mel_dist(quiet, xv[s2n(cs + 0.03): s2n(ce - 0.03)]), 2)
    return rows


def main(pattern="*"):
    out = []
    for jp in sorted(VARIANTS.glob(f"{pattern}.json")):
        lab = json.loads(jp.read_text())
        if lab["where"]["added_condition"] or not lab["what"]:
            continue
        xb, _ = read_audio(TAKES / f"{lab['take_id']}_C0.flac")
        xv, _ = read_audio(jp.with_suffix(".flac"))
        out += analyse(lab, xb, xv)
    PILOT.mkdir(exist_ok=True)
    with open(PILOT / "artifact_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    import pandas as pd
    d = pd.DataFrame(out)
    for c in ("timbre_db", "floor_db", "step_db"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    s = d.groupby("flaw").agg(n=("clip_id", "count"), click_db_p90=("click_db", lambda v: np.percentile(v, 90)),
                              step_db_p90=("step_db", lambda v: np.nanpercentile(v, 90) if v.notna().any() else np.nan),
                              timbre_db=("timbre_db", "median"), floor_db=("floor_db", "median")).round(1)
    print(s.to_string())
    return s


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "*")
