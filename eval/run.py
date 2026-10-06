"""One command: python eval/run.py --split test --mode cross [--predictor oracle|null|engine] [--limit N]

Reads the frozen label JSONs (schema 1.1.0) for a split, asks a predictor for a prediction JSON in the SAME schema, and writes
results/*.csv + results/summary.json. The predictor is pluggable:
  oracle  returns the ground truth with severity = level        (must score F1 = 1.0: proves the harness is right)
  null    returns no regions                                    (must score F1 = 0 and 0 false flags)
  engine  imports engine.predict.predict(audio_path, label_path, mode) when the engine package exists
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "schema"))
import metrics as M  # noqa: E402
from validate import errors  # noqa: E402

DATA = ROOT / "flawline-dataset"
CATS = ["Pacing", "Pausing", "Intonation", "Volume", "Fluency", "Clarity", "Text fidelity"]
PASS = {"f1@0.5": 0.7, "onset_ms": 150, "offset_ms": 150, "category_acc": 0.85, "spearman": 0.85, "false_flags_per_min": 0.5}


def oracle(label: dict, **_):
    p = copy.deepcopy(label)
    p["labels_from"] = "engine"
    for r in p["what"]:
        r["severity"], r["confidence"] = float(r.get("level", 3)), 1.0
    return p


def null(label: dict, **_):
    p = copy.deepcopy(label)
    p["labels_from"] = "engine"
    p["what"] = []
    return p


def load_predictor(name: str):
    if name == "oracle":
        return oracle
    if name == "null":
        return null
    sys.path.insert(0, str(ROOT))
    from engine.predict import predict  # noqa: WPS433  (the engine package must not import from the generator)
    def run(label, mode, **_):
        audio = DATA / "variants" / f"{label['clip_id']}.flac"
        if not audio.exists():
            audio = DATA / "takes" / f"{label['clip_id']}.flac"
        meta = {k: v for k, v in label.items() if k not in ("what", "seed", "join_check", "multi_set", "scenario")}   # NO ground truth to the engine
        return predict(str(audio), meta, {"same": "same", "cross": "cross", "free": "free"}[mode])
    return run


def load_split(split: str, limit: int | None):
    rows = list(csv.DictReader(open(DATA / "manifest.csv")))
    out = []
    for r in rows:
        if split != "all" and r["split"] != split:
            continue
        p = DATA / "variants" / f"{r['clip_id']}.json"
        if not p.exists():                                  # clean takes have no variant label: build the empty-region label
            lab = {"schema_version": "1.1.0", "clip_id": r["clip_id"], "take_id": r["take_id"], "baseline_id": r["baseline_id"],
                   "genre": r["genre"], "who": {"speaker_id": r["speaker_id"]}, "where": {"base_condition": "C0", "added_condition": None},
                   "what": [], "duration_s": float(r["duration_s"]), "labels_from": "generator"}
        else:
            lab = json.loads(p.read_text())
        out.append(lab)
    return out[:limit] if limit else out


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "dev", "test", "all"])
    ap.add_argument("--mode", default="cross", choices=["same", "cross", "free"])
    ap.add_argument("--predictor", default="oracle")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-conditions", action="store_true", help="skip clips with an added Where condition")
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()
    out = Path(a.out)
    labels = load_split(a.split, a.limit)
    if a.no_conditions:
        labels = [l for l in labels if not l["where"].get("added_condition")]
    predict = load_predictor(a.predictor)
    pairs, bad = [], 0
    for lab in labels:
        pr = predict(lab, mode=a.mode)
        e = errors({**pr, "schema_version": "1.1.0"})
        if e:
            bad += 1
            print("INVALID PREDICTION", lab["clip_id"], e[:2])
        pairs.append((lab, pr))
    summary = {"split": a.split, "mode": a.mode, "predictor": a.predictor, "n_clips": len(pairs), "invalid_predictions": bad}
    rows = []
    for thr in (0.3, 0.5):
        for level in ("flaw", "category"):
            g = M.grounding(pairs, thr, level)
            summary[f"{level}_f1@{thr}"] = round(g["overall"]["f1"], 4)
            for k, v in g["per"].items():
                rows.append({"iou": thr, "level": level, "label": k, **{kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}})
    write_csv(out / "grounding.csv", rows)
    t = M.timing(pairs)
    summary.update(median_onset_ms=round(t["median_onset_ms"], 1), median_offset_ms=round(t["median_offset_ms"], 1), n_timing_pairs=t["n"])
    write_csv(out / "timing.csv", [t])
    cm, acc = M.category_confusion(pairs, CATS)
    summary["category_acc"] = round(acc, 4) if acc == acc else None
    write_csv(out / "category_confusion.csv", [{"true": CATS[i], **{CATS[j]: int(cm[i, j]) for j in range(len(CATS))}} for i in range(len(CATS))])
    dr = M.dose_response(pairs)
    write_csv(out / "dose_response.csv", [{"flaw": k, **v} for k, v in dr.items()])
    sp = [v["spearman"] for v in dr.values() if v["spearman"] == v["spearman"]]
    summary["dose_response_min_spearman"] = round(min(sp), 3) if sp else None
    ff = M.false_flags_per_min(pairs)
    write_csv(out / "invariance.csv", [ff])
    summary["false_flags_per_min"] = round(ff["per_minute"], 3) if ff["per_minute"] == ff["per_minute"] else None
    blob = json.dumps(summary, sort_keys=True).encode()
    summary["sha256"] = hashlib.sha256(blob).hexdigest()[:16]
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
