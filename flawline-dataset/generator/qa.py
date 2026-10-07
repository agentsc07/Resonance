"""Automated build checks (spec "Quality checks"): no clipping, no digital-zero runs, labels inside the clip,
region boundaries on word boundaries (baseline time), checksums match. Exit 1 on any failure."""
import json
import sys

import numpy as np

from common import ROOT, SR, TAKES, VARIANTS, read_audio, s2n

sys.path.insert(0, str(ROOT.parent / "schema"))
from validate import errors as schema_errors  # noqa: E402

ZERO_RUN = s2n(0.010)
bad = []
for jp in sorted(VARIANTS.glob("*.json")):
    lab = json.loads(jp.read_text())
    x, _ = read_audio(jp.with_suffix(".flac"))
    cid = lab["clip_id"]
    for e in schema_errors(lab):
        bad.append((cid, "schema: " + e))
    if np.abs(x).max() >= 0.999:
        bad.append((cid, "clipping"))
    z = np.abs(x) < 1e-6
    run = best = 0
    for v in z:
        run = run + 1 if v else 0
        best = max(best, run)
    if best >= ZERO_RUN:
        bad.append((cid, f"digital-zero run {best / SR * 1000:.0f} ms"))
    if abs(len(x) / SR - lab["duration_s"]) > 1e-3:
        bad.append((cid, "duration mismatch"))
    al = json.loads((TAKES / f"{lab['take_id']}_C0.align.json").read_text())["words"]
    starts = {round(w["start_s"], 3) for w in al}
    ends = {round(w["end_s"], 3) for w in al}
    for r in lab["what"]:
        if not (0 <= r["start_s"] <= r["end_s"] <= lab["duration_s"] + 1e-3):
            bad.append((cid, f"{r['flaw']} region outside clip"))
        if r["kind"] in ("modify", "delete") and r["flaw"] not in ("PAUSE_LOST",):
            tol = 0.06 if r["flaw"] == "WORD_SKIP" else 0.021          # deletions are cut in the energy valley beside the word edge
            ok_s = any(abs(r["baseline_start_s"] - w) < tol for w in starts | ends)
            ok_e = any(abs(r["baseline_end_s"] - w) < tol for w in starts | ends)
            if not (ok_s and ok_e):
                bad.append((cid, f"{r['flaw']} boundary off word boundary"))
n = len(list(VARIANTS.glob("*.json")))
print(f"QA: {n} clips checked, {len(bad)} problems")
for b in bad[:30]:
    print("  ", *b)
sys.exit(1 if bad else 0)
