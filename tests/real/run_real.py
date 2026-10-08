"""Regression check for upload (free) mode on real recordings with known slips.
   python tests/real/run_real.py          prints, per file, expected items against what upload mode found (hit / miss) and any extra flags.
Files and expected times live next to this script (expected.json; times +-0.15 s). The recordings are personal and are not committed (.wav is gitignored)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from engine import audio, predict  # noqa: E402

TOL = 0.15


def hit(exp, r):
    return r["flaw"] == exp["flaw"] and r["start_s"] <= exp["end_s"] + TOL and r["end_s"] >= exp["start_s"] - TOL


def main():
    spec = json.loads((HERE / "expected.json").read_text())
    n_exp = n_hit = n_extra = 0
    for name, d in spec.items():
        if name.startswith("_"):
            continue
        path = HERE / name
        if not path.exists():
            print(f"{name}: file missing, skipped")
            continue
        x = audio.load(str(path))
        meta = {"clip_id": "REAL-" + name, "take_id": "REAL", "baseline_id": "B01", "genre": "interpretive reading", "duration_s": len(x) / audio.SR,
                "who": {"speaker_id": "real"}, "where": {"base_condition": "C0", "added_condition": None}}
        if d.get("transcript"):
            meta["transcript"] = d["transcript"]
        else:
            meta["no_transcript"] = True
        out = predict.predict(str(path), meta, "free")
        found = out["what"]
        print(f"\n{name}  ({meta['duration_s']:.1f} s, score {out['scores']['overall']})")
        used = set()
        for exp in d["expected"] + [dict(e, possible=True) for e in d.get("possible", [])]:
            m = next((k for k, r in enumerate(found) if hit(exp, r)), None)
            tag = "possible" if exp.get("possible") else "expected"
            if m is not None:
                used.add(m)
                print(f"  HIT   {tag} {exp['flaw']} {exp['start_s']:.2f}-{exp['end_s']:.2f}  found {found[m]['flaw']} {found[m]['start_s']:.2f}-{found[m]['end_s']:.2f}")
            else:
                print(f"  MISS  {tag} {exp['flaw']} {exp['start_s']:.2f}-{exp['end_s']:.2f}")
            if not exp.get("possible"):
                n_exp += 1
                n_hit += m is not None
        for k, r in enumerate(found):
            if k not in used:
                n_extra += 1
                print(f"  EXTRA {r['flaw']} {r['start_s']:.2f}-{r['end_s']:.2f}  ({r['explanation'][:80]})")
    print(f"\nexpected items found: {n_hit} of {n_exp}; extra flags: {n_extra}")


if __name__ == "__main__":
    main()
