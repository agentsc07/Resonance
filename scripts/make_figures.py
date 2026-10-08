"""Figures for docs/technical_report.md, drawn from results/*.json: pipeline diagram, dose-response plot, per-flaw F1 bars, timeline crop."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
R, OUT = ROOT / "results", ROOT / "docs" / "figures"
OUT.mkdir(exist_ok=True)
INK, GOLD, CORAL, GREY, BLUE = "#1d1d22", "#c98a14", "#d6483c", "#8a8a93", "#2f6fb5"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

# 1. pipeline
fig, ax = plt.subplots(figsize=(7.2, 1.55))
ax.axis("off"); ax.set_xlim(0, 104); ax.set_ylim(0, 22)
steps = ["Ingest", "Quality\ngate", "Alignment", "Detectors\n(one per flaw)", "Arbitration", "Severity", "Rubric\nscore", "Explanation\n+ tip"]
w, gap = 10.6, 2.2
for i, s in enumerate(steps):
    x = 1 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, 6), w, 10, boxstyle="round,pad=0.2,rounding_size=1.2", fc="#f6f1e4", ec=GOLD, lw=1.1))
    ax.text(x + w / 2, 11, s, ha="center", va="center", fontsize=7.6, color=INK)
    if i < len(steps) - 1:
        ax.add_patch(FancyArrowPatch((x + w + 0.3, 11), (x + w + gap - 0.3, 11), arrowstyle="-|>", mutation_scale=7, color=GREY, lw=1))
ax.text(1, 19.5, "audio (+ text)", fontsize=7.2, color=GREY); ax.text(103, 19.5, "flagged moments, score, report", fontsize=7.2, color=GREY, ha="right")
fig.savefig(OUT / "pipeline.png", dpi=220, bbox_inches="tight"); plt.close(fig)

# 2. dose-response
a = json.loads((R / "acceptance_same.json").read_text().replace("NaN", "null"))["dose_response"]["per_flaw"]
fig, ax = plt.subplots(figsize=(7.2, 2.7))
ends = []
for k, v in a.items():
    sc = v["scores_L0_to_L5"]
    if v["spearman"] is None:
        continue
    bad = v["spearman"] > -0.9
    ax.plot(range(6), sc, color=CORAL if bad else GREY, lw=1.9 if bad else 1.0, alpha=1 if bad else .8)
    ends.append([sc[-1], k, bad])
ends.sort()
for i in range(1, len(ends)):                      # spread the end labels so none overlap
    ends[i][0] = max(ends[i][0], ends[i - 1][0] + 2.3)
for y, k, bad in ends:
    ax.text(5.08, y, k, fontsize=6.2, va="center", color=CORAL if bad else INK)
ax.set_xlim(0, 6.4); ax.set_ylim(50, 104); ax.set_xticks(range(6)); ax.set_xticklabels(["L0\nunaltered", "L1", "L2", "L3", "L4", "L5"])
ax.set_ylabel("overall score"); ax.set_title("Score against flaw level, baseline B01 (red: fails ρ ≤ −0.9)", fontsize=8.5, loc="left")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT / "dose_response.png", dpi=220, bbox_inches="tight"); plt.close(fig)

# 3. per-flaw F1
same = json.loads((R / "headline_train_same.json").read_text())
free = json.loads((R / "headline_train_free.json").read_text())
exp, off = set(same["experimental"]), set(free["cross_disabled"])
order = sorted(same["per_flaw_f1"], key=lambda k: -same["per_flaw_f1"][k])
fig, ax = plt.subplots(figsize=(7.2, 2.7))
xs = range(len(order))
ax.bar([x - .2 for x in xs], [same["per_flaw_f1"][k] for k in order], .4, color=[GREY if k in exp else GOLD for k in order], label="with a clean reading (grey: experimental)")
ax.bar([x + .2 for x in xs], [free["per_flaw_f1"][k] for k in order], .4, color=BLUE, label="upload mode")
ax.set_xticks(list(xs)); ax.set_xticklabels(order, rotation=40, ha="right", fontsize=7)
ax.set_ylabel("F1 (IoU ≥ 0.5)"); ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=7, loc="upper right")
ax.set_title("Per-flaw event F1, train split", fontsize=8.5, loc="left"); ax.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT / "per_flaw_f1.png", dpi=220, bbox_inches="tight"); plt.close(fig)

# 4. timeline crop from the app screenshot
im = Image.open(ROOT / "docs" / "screenshots" / "analyse_shout_1440x900.png")
im.crop((72, 170, 1368, 770)).save(OUT / "timeline.png")
print("figures written")
