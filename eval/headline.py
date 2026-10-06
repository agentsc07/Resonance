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
    out = {"split": a.split, "mode": a.mode, "clips": len(labs), "all_flaws": {"f1": round(g["overall"]["f1"], 3), "precision": round(g["overall"]["precision"], 3), "recall": round(g["overall"]["recall"], 3)},
           "headline": {"flaws": sorted(keep), "f1": round(gh["overall"]["f1"], 3), "precision": round(gh["overall"]["precision"], 3), "recall": round(gh["overall"]["recall"], 3)},
           "experimental": EXPERIMENTAL, "cross_disabled": sorted(detect.DISABLED), "leakage_flagged": LEAK_FLAGGED,
           "per_flaw_f1": {k: round(v["f1"], 3) for k, v in g["per"].items()}}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"headline_{a.split}_{a.mode}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("split", "mode", "clips", "all_flaws", "headline")}, indent=1))


if __name__ == "__main__":
    main()
