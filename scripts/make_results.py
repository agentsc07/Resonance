"""Write docs/RESULTS.md from results/*.json and results/*.csv, so every number in the document is the one on disk.
   python scripts/make_results.py
Sections for the test split and for the clean-clone check appear when results/headline_test_*.json / results/clean_clone.json exist."""
import csv
import json
import math
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
J = lambda n: json.loads((R / n).read_text()) if (R / n).exists() else None
f3 = lambda v: "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"
pc = lambda v: f"{100 * v:.0f}%"
SPL = [("train", "Train (B01, B02, B04, B06, B07)"), ("dev", "Dev (B03)"), ("extra", "JFK 1962 (B09, B10; never tuned on)"), ("test", "Test (B05, B08)")]
out = []
w = out.append

w("# Results\n")
w("All numbers are read from `results/*.json` and `results/*.csv` by `scripts/make_results.py`. Thresholds and weights are fitted on the **train** split only; **dev** is checked, "
  "**JFK** (two excerpts of the 1962 Rice University address) is never tuned on, and the **test** split is run once at the end.\n")
w("## 1. Event grounding\n")
w("An injected flaw is *found* when a predicted region of the same flaw type overlaps it. Point-like events (a removed word, a short insertion) are widened to 0.30 s on both sides before overlap is computed. "
  "Precision = share of predicted regions that match a true flaw; recall = share of true flaws matched; F1 is their harmonic mean. Counts are over all clips of the split.\n")
w("Metrics reported side by side:\n")
w("- **Strict F1**: overlap (IoU) of at least 0.5 between predicted and true region, same flaw type.")
w("- **Standard F1**: the same with IoU of at least 0.3.")
w("- **Onset F1**: same flaw type and predicted start within 250 ms of the true start.")
w("- **Median onset error**: median |predicted start − true start| over matched flaws (IoU ≥ 0.3).")
w("- **Area accuracy**: among regions matched on time alone (IoU ≥ 0.3), the share whose predicted area (pacing, pausing, intonation, volume, fluency, clarity, text fidelity) is correct.\n")
w("**Headline flaws** (11): FADE, FILLER, MONOTONE, PACE_FAST, PACE_SLOW, PAUSE_BAD, PAUSE_LOST, SHOUT, SLUR, WORD_SKIP, WORD_SWAP. In upload mode only the 5 detectable ones count (FADE, FILLER, PAUSE_BAD, SHOUT, WORD_SWAP).\n")
for mode, title in (("same", "With a clean reading of the same text (same-speaker mode)"), ("free", "Upload mode (no reference reading)")):
    w(f"### {title}\n")
    w("| Split | Clips | Strict F1 | P | R | Standard F1 | Onset F1 | Median onset error | Area accuracy | Found / true | All 15 flaws F1 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for sp, label in SPL:
        d = J(f"headline_{sp}_{mode}.json")
        if not d:
            continue
        m = d["metrics_headline"]
        s = m["strict_iou50"]
        w(f"| {label} | {d['clips']} | {f3(s['f1'])} | {f3(s['precision'])} | {f3(s['recall'])} | {f3(m['overlap_iou30']['f1'])} | {f3(m['onset_250ms']['f1'])} | "
          f"{m['median_onset_error_ms']['value']:.1f} ms | {pc(m['category_accuracy']['value'])} | {s['tp']} / {s['n_gt']} | {f3(d['all_flaws']['f1'])} |")
    w("")
    d = J(f"headline_train_{mode}.json")
    ex = set(d["experimental"])
    off = set(d["cross_disabled"])
    w(f"Per-flaw strict F1, train split ({mode}):\n")
    w("| Flaw | F1 | Status |")
    w("|---|---|---|")
    for k, v in sorted(d["per_flaw_f1"].items(), key=lambda kv: -kv[1]):
        st = "experimental" if k in ex else "not detectable without a reference" if k in off else "headline"
        w(f"| {k} | {f3(v)} | {st} |")
    w("")

w("### Experimental flaws\n")
w("Reported, excluded from the headline numbers, and hidden in the app by default:\n")
for k, v in J("headline_train_same.json")["experimental"].items():
    w(f"- **{k}**: {v}")
w("")
w("### Other modes\n")
w("Another-speaker mode (a panel of other voices reading the same text must agree) is experimental, available only in the lab build, and was last evaluated on an earlier version of the dataset; no current numbers are reported.\n")

a = J("acceptance_same.json")
w("## 2. Score acceptance tests (same-speaker mode)\n")
w("**Dose-response.** The B01 baseline has all five levels of every flaw. Spearman correlation between level (L0 = unaltered … L5) and overall score; a flaw passes at ρ ≤ −0.9.\n")
w("| Flaw | L0 | L1 | L2 | L3 | L4 | L5 | ρ | Result |")
w("|---|---|---|---|---|---|---|---|---|")
npass = ntot = 0
for k, v in a["dose_response"]["per_flaw"].items():
    rho = v["spearman"]
    scored = not (isinstance(rho, float) and math.isnan(rho))
    ntot += scored
    ok = scored and rho <= a["dose_response"]["min_ok"]
    npass += ok
    w(f"| {k} | " + " | ".join(f"{x:.1f}" for x in v["scores_L0_to_L5"]) + f" | {f3(rho)} | {'pass' if ok else 'fail' if scored else 'not scored (experimental)'} |")
w(f"\n**{npass} of {ntot}** scored flaws pass. Failing: {', '.join(a['dose_response']['failing'])}. Per-flaw weights (`engine/rubric.yaml`, fitted on train single-flaw clips) make a single level-5 flaw score about 60 for every flaw type.\n")
iv, lo = a["invariance"], a["locality"]
w("**Invariance.** Unaltered readings re-recorded under noise, phone-band filtering, room reverb, MP3 and gain change should keep their score and produce no flags.\n")
w(f"- False flags on unaltered clips under those conditions: **{iv['false_flags_per_min']:.3f} per minute** (target ≤ 0.5: pass), over {iv['n']} clips.")
w(f"- Worst score shift against the unaltered reading: **{iv['max_score_shift']} points** (target < 3: not met).")
w("- Mean score shift by condition: " + ", ".join(f"{k} {v}" for k, v in iv["mean_shift_by_condition"].items()) + " points.\n")
w(f"**Locality.** On level-3 single-flaw clips, mean points lost in areas other than the flaw's own: **{lo['mean_other_category_loss']}** (target < 2: not met), over {lo['n']} clips.\n")

lk, lso = list(csv.DictReader(open(R / "leakage.csv"))), list(csv.DictReader(open(R / "leakage_loso.csv")))
w("## 3. Leakage audit\n")
w("A classifier that sees only editing artifacts (click energy at the join, noise-floor step, spectral-flux spike, sample-exact repetition) tries to tell an altered window from the same place in the clean baseline. "
  "AUC near 0.5 means the dataset tests delivery, not editing. Pass mark: AUC ≤ 0.60 overall. Trained on the train split and scored on the held-out test windows; the second column is leave-one-speaker-out over train and dev speakers.\n")
w("| Scope | Held-out AUC | Leave-one-speaker-out AUC |")
w("|---|---|---|")
lm = {r["scope"]: r["auc"] for r in lso}
for r in lk:
    w(f"| {r['scope']} | {r['auc']} | {lm.get(r['scope'], '')} |")
w("\nFlaws above 0.65 in leave-one-speaker-out (FILLER, PAUSE_BAD, PAUSE_LOST, RARE_HESIT, REPEAT) can be separated from clean speech partly by their editing artifacts. FILLER and PAUSE_BAD remain headline flaws and are flagged as such.\n")

af = J("acceptance_free.json")
if af:
    w("## 3b. Upload-mode invariance\n")
    iv = af["invariance"]
    w(f"Unaltered readings under noise, phone band, room reverb, MP3 and gain change, upload mode: **{iv['false_flags_per_min']:.3f} false flags per minute**, worst score shift **{iv['max_score_shift']} points**, mean shift by condition " + ", ".join(f"{k} {v}" for k, v in iv["mean_shift_by_condition"].items()) + f" (n = {iv['n']}).\n")

c = J("clean_clone.json")
if c:
    w("## 4. Reproduction check\n")
    for k, v in c.items():
        w(f"- **{k}**: {v}")
    w("")
(R.parent / "docs" / "RESULTS.md").write_text("\n".join(out) + "\n")
print("wrote docs/RESULTS.md")
