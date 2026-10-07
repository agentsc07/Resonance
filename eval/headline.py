"""Headline vs experimental metrics. A flaw is HEADLINE when it is detectable (same-speaker train F1 >= 0.3), shows a dose-response and is not an
outlier in the leakage audit; the rest are reported but marked EXPERIMENTAL and excluded from the headline number (spec: say so in the README).
   python eval/headline.py --split dev --mode same        -> results/headline_<split>_<mode>.json"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
import metrics as M  # noqa: E402
from engine import detect, tune as T  # noqa: E402

EXPERIMENTAL = {"EMPH_FLAT": "VCTK speakers are not expressive enough for a flattening to scale with level; detector F1 0.09",
                "REPEAT": "detector F1 0.13; leakage AUC 0.64",
                "UPTALK": "detector F1 0.25 (no leakage)",
                "RARE_HESIT": "dose-response fails (non-monotone); leakage AUC 0.70"}
LEAK_FLAGGED = {"PAUSE_BAD": 0.68, "FILLER": 0.69}      # headline, but the leave-one-speaker-out leakage AUC is ~0.7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--mode", default="same")
    a = ap.parse_args()
    detect.use(a.mode)
    labs = T.clips(a.split)
    g, pairs = T.evaluate(labs, a.mode, verbose=False)
    head = {k: v for k, v in g["per"].items() if k not in EXPERIMENTAL and k not in detect.DISABLED}
    keep = set(head)
    pairs_h = [({**lab, "what": [r for r in lab["what"] if r["flaw"] in keep]}, {**p, "what": [r for r in p["what"] if r["flaw"] in keep]}) for lab, p in pairs]
    gh = M.grounding(pairs_h, 0.5, "flaw")
    gh3 = M.grounding(pairs_h, 0.3, "flaw")
    gc = M.onset_f1(pairs_h, 0.25)
    tm = M.timing(pairs_h, 0.3)
    cats = sorted({r["category"] for _, p in pairs_h for r in p["what"]} | {r["category"] for l, _ in pairs_h for r in l["what"]})
    _, cacc = M.category_confusion(pairs_h, cats, 0.3)

    def pr(x):
        o = x["overall"] if "overall" in x else x
        return {"f1": round(o["f1"], 3), "precision": round(o["precision"], 3), "recall": round(o["recall"], 3), "tp": o["tp"], "n_gt": o["n_gt"], "n_pred": o["n_pred"]}
    out = {"split": a.split, "mode": a.mode, "clips": len(labs), "all_flaws": {"f1": round(g["overall"]["f1"], 3), "precision": round(g["overall"]["precision"], 3), "recall": round(g["overall"]["recall"], 3)},
           "headline": {"flaws": sorted(keep), "f1": round(gh["overall"]["f1"], 3), "precision": round(gh["overall"]["precision"], 3), "recall": round(gh["overall"]["recall"], 3)},
           "metrics_headline": {
               "strict_iou50": {"label": "Strict: flag overlaps the real moment by at least half", **pr(gh)},
               "overlap_iou30": {"label": "Standard: flag overlaps the real moment by at least 30%", **pr(gh3)},
               "onset_250ms": {"label": "Onset: right flaw type, start within 250 ms of the real start", **pr(gc)},
               "median_onset_error_ms": {"label": "How far off the flag start is, median (milliseconds)", "value": round(tm["median_onset_ms"], 1), "n": tm["n"]},
               "category_accuracy": {"label": "Share of located flags that name the right area", "value": round(cacc, 3), "categories": cats}},
           "experimental": EXPERIMENTAL, "cross_disabled": sorted(detect.DISABLED), "leakage_flagged": LEAK_FLAGGED,
           "per_flaw_f1": {k: round(v["f1"], 3) for k, v in g["per"].items()}}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"headline_{a.split}_{a.mode}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("split", "mode", "clips", "all_flaws", "headline", "metrics_headline")}, indent=1))


if __name__ == "__main__":
    main()
