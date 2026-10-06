"""Did the injectors follow the grammar rules? Reads every label JSON and tabulates, per flaw, the boundary type each edit landed on,
plus the measured evidence for MONOTONE / SLUR. Output: pilot/placement_audit.csv"""
import csv
import json
from collections import Counter, defaultdict

import numpy as np

from common import PILOT, VARIANTS

by, ev = defaultdict(Counter), defaultdict(list)
for jp in sorted(VARIANTS.glob("*.json")):
    lab = json.loads(jp.read_text())
    if lab["where"]["added_condition"]:
        continue
    for r in lab["what"]:
        by[r["flaw"]][r.get("boundary") or r["params"].get("token") or r["kind"]] += 1
        e = r["params"].get("evidence")
        if e:
            ev[r["flaw"]].append(e)
rows = []
for f, c in sorted(by.items()):
    tot = sum(c.values())
    for k, v in c.most_common():
        rows.append({"flaw": f, "placed_at": k, "n": v, "share": round(v / tot, 3)})
with open(PILOT / "placement_audit.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["flaw", "placed_at", "n", "share"])
    w.writeheader()
    w.writerows(rows)
for f in ("PAUSE_BAD", "FILLER", "REPEAT", "PAUSE_LOST", "RARE_HESIT"):
    tot = sum(by[f].values())
    print(f"{f:11s} n={tot:3d}  " + ", ".join(f"{k} {v / tot:.0%}" for k, v in by[f].most_common(6)))
m = ev["MONOTONE"]
print("\nMONOTONE evidence: n=%d  IQR before median %.1f st -> after %.1f st; audible (>=1 st lost) %.0f%%" % (
    len(m), np.median([e["f0_iqr_before_st"] for e in m]), np.median([e["f0_iqr_after_st"] for e in m]), 100 * np.mean([e["audible"] for e in m])))
s = ev["SLUR"]
print("SLUR evidence:     n=%d  consonant energy drop median %.1f dB (min %.1f)" % (len(s), np.median([e["drop_db"] for e in s]), min(e["drop_db"] for e in s)))
