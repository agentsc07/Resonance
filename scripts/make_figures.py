"""Figures for docs/technical_report.md. One style: white background, dark text, one gold accent, coral for flaws, sans-serif.
Diagrams (overview, benchmark_build, engine_modes) are drawn on a fixed 7.2 in wide canvas so they print at 1:1; plots are drawn from results/*.json.
   python scripts/make_figures.py"""
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
R, OUT = ROOT / "results", ROOT / "docs" / "figures"
OUT.mkdir(exist_ok=True)
INK, GOLD, CORAL, GREY, FILL, LINE = "#1d1d22", "#C8901F", "#E0674F", "#6b6b73", "#F6F5F1", "#B9B8B0"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7.5,
                     "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
FS = 7.0                                                       # smallest text in any diagram


def canvas(h_in, h_units):
    """7.2 in wide canvas whose x axis runs 0..72, so one unit = 0.1 in."""
    fig = plt.figure(figsize=(7.2, h_in))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(-1.3, 73.3); ax.set_ylim(-0.6, h_units)
    return fig, ax


def box(ax, x, y, w, h, title, body="", accent=False, flaw=False, fs=FS):
    ec = GOLD if accent else (CORAL if flaw else LINE)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9", fc="#FFF9EA" if accent else FILL, ec=ec, lw=1.3 if accent or flaw else 0.9, zorder=2))
    cpl = max(8, int(w * 1.95 * 7.0 / fs))                     # characters per line that fit in the box at this size
    lines = textwrap.wrap(body, cpl) if body else []
    tl = textwrap.wrap(title, cpl)
    n = len(tl) + len(lines)
    lh = fs * 0.0145 * 10 * 0.98                               # line height in units (pt -> in -> units)
    top = y + h / 2 + (n * lh) / 2 - lh / 2
    for i, t in enumerate(tl):
        ax.text(x + w / 2, top - i * lh, t, ha="center", va="center", fontsize=fs, fontweight="bold", color=INK, zorder=3)
    for i, t in enumerate(lines):
        ax.text(x + w / 2, top - (len(tl) + i) * lh, t, ha="center", va="center", fontsize=fs, color=GREY if not accent else INK, zorder=3)


def route(ax, pts, color=GREY, ls="-", lw=1.0, head=True, z=1):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] if head else xs, ys[:-1] if head else ys, color=color, ls=ls, lw=lw, zorder=z, solid_capstyle="butt")
    if head:
        ax.annotate("", xy=pts[-1], xytext=(pts[-2][0] * 0.4 + pts[-1][0] * 0.6 if pts[-2][0] != pts[-1][0] else pts[-1][0],
                                            pts[-2][1] * 0.4 + pts[-1][1] * 0.6 if pts[-2][1] != pts[-1][1] else pts[-1][1]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=7, shrinkA=0, shrinkB=0), zorder=z)
        ax.plot([pts[-2][0], pts[-1][0]], [pts[-2][1], pts[-1][1]], color=color, ls=ls, lw=lw, zorder=z, solid_capstyle="butt")


def badge(ax, x, y, n):
    ax.add_patch(Circle((x, y), 1.15, fc=GOLD, ec="none", zorder=4))
    ax.text(x, y - 0.05, str(n), ha="center", va="center", fontsize=6.6, color="white", fontweight="bold", zorder=5)


# ---------------------------------------------------------------- Fig 1: overview
fig, ax = canvas(2.1, 21)
W, G = 11.2, 4.0
xs = [i * (W + G) for i in range(5)]
Y, H = 7.2, 8.6
box(ax, xs[0], Y, W, H, "Generator", "clean takes become flawed clips with exact labels", accent=True)
box(ax, xs[1], Y, W, H, "Benchmark", "563 clips, 15 flaws, 5 levels, 6 conditions", accent=True)
box(ax, xs[2], Y, W, H, "Engine", "audio and optional text become labels, blind", accent=True)
box(ax, xs[3], Y, W, H, "Harness", "compares engine labels with true labels: metrics")
box(ax, xs[4], Y, W, H, "App", "score, timeline, flaw cards, replay")
mid = Y + H / 2
route(ax, [(xs[0] + W, mid), (xs[1], mid)])
route(ax, [(xs[1] + W, mid), (xs[2], mid)])
ax.text((xs[1] + W + xs[2]) / 2, mid + 1.0, "audio", fontsize=FS, ha="center", color=GREY)
route(ax, [(xs[2] + W, mid), (xs[3], mid)])
ax.text((xs[2] + W + xs[3]) / 2, mid + 1.0, "labels", fontsize=FS, ha="center", color=GREY)
route(ax, [(xs[1] + W * 0.5, Y + H), (xs[1] + W * 0.5, 18.3), (xs[3] + W * 0.5, 18.3), (xs[3] + W * 0.5, Y + H)])
ax.text((xs[2] + W * 0.5 + xs[3] + W * 0.5) / 2, 19.5, "true labels", fontsize=FS, ha="center", color=GREY)
route(ax, [(xs[2] + W * 0.75, Y), (xs[2] + W * 0.75, 3.6), (xs[4] + W * 0.5, 3.6), (xs[4] + W * 0.5, Y)])
ax.text(xs[3] + W * 0.5, 4.5, "predicted labels, scored", fontsize=FS, ha="center", color=GREY)
route(ax, [(xs[0] + W * 0.5, Y), (xs[0] + W * 0.5, 1.6), (xs[2] + W * 0.25, 1.6), (xs[2] + W * 0.25, Y)], color=CORAL, ls=(0, (3, 2)), head=False)
ax.text((xs[0] + xs[2] + W * 0.75) / 2, 0.45, "×  never import each other", fontsize=FS + 0.3, ha="center", color=CORAL, fontweight="bold")
fig.savefig(OUT / "overview.png", dpi=250); plt.close(fig)

# ---------------------------------------------------------------- Fig 2: benchmark build
fig, ax = canvas(2.25, 22.5)
W, G, Hh = 15.0, 4.0, 8.4
steps = [("Clean take + text", "real speaker reading a known text"),
         ("Word times", "recogniser times matched to the text, snapped to energy valleys"),
         ("Boundary types", "spaCy: sentence end, comma, tight phrase, before a clause"),
         ("Placement rules", "each flaw has grammar rules for where it may go"),
         ("Render in own voice", "PSOLA edits, own-voice fillers, speech-only gain"),
         ("Quality gates", "objective checks, ASR removal gate, listening pass"),
         ("Recording conditions", "6: babble, pink noise, reverb, phone, MP3, gain"),
         ("Clip + label", "audio with exact start, end, type, level and reason")]
row1, row2 = list(range(0, 4)), list(range(7, 3, -1))          # snake: left to right, then right to left
YT, YB = 12.2, 1.4
pos = {}
for c, i in enumerate(row1):
    pos[i] = (c * (W + G), YT)
for c, i in enumerate(row2):
    pos[i] = (c * (W + G), YB)
for i in range(8):
    x, y = pos[i]
    box(ax, x, y, W, Hh, steps[i][0], steps[i][1], accent=(i in (0, 7)), flaw=False)
    badge(ax, x + 0.1, y + Hh + 0.1, i + 1)
for i in range(7):
    (x0, y0), (x1, y1) = pos[i], pos[i + 1]
    if y0 == y1:
        if x1 > x0:
            route(ax, [(x0 + W, y0 + Hh / 2), (x1, y1 + Hh / 2)])
        else:
            route(ax, [(x0, y0 + Hh / 2), (x1 + W, y1 + Hh / 2)])
    else:
        route(ax, [(x0 + W / 2, y0), (x1 + W / 2, y1 + Hh)])
fig.savefig(OUT / "benchmark_build.png", dpi=250); plt.close(fig)

# ---------------------------------------------------------------- Fig 3: engine modes
fig, ax = canvas(2.35, 23.5)
bw, bg = 12.2, 1.9
lanes = [("A  With a clean reading", 12.2, [("Condition matching", "reference degraded to the recording's noise, band and reverb"),
                                           ("DTW alignment", "reference and recording warped onto each other"),
                                           ("Lag events and tempo", "insertions, deletions, local speed")]),
         ("B  Upload mode", 1.4, [("ASR words, snapped", "word times moved to the audio's silences"),
                                 ("Audio measures", "silences, held vowels, features vs the clip's own norms"),
                                 ("Transcript rules", "pause norms by boundary type, text checks")])]
LH = 8.4
for label, y, items in lanes:
    ax.text(0, y + LH + 0.9, label, fontsize=FS + 0.5, fontweight="bold", va="center", color=INK)
    for k, (t, b) in enumerate(items):
        x = k * (bw + bg)
        box(ax, x, y, bw, LH, t, b, accent=False)
        if k < 2:
            route(ax, [(x + bw, y + LH / 2), (x + bw + bg, y + LH / 2)])
sx, sw = 46.5, 25.5
ax.add_patch(FancyBboxPatch((sx - 1.0, 0.3), sw + 2.0, 20.4, boxstyle="round,pad=0,rounding_size=0.9", fc="white", ec=GOLD, lw=1.3, zorder=1))
shared = ["Detectors, one per flaw", "Arbitration", "Severity (0 to 5)", "Score: areas and overall", "Explanation and tip"]
sh = 3.2
for i, t in enumerate(shared):
    yy = 16.4 - i * 4.0
    box(ax, sx, yy, sw, sh, t, "", fs=FS + 0.2)
    if i < 4:
        route(ax, [(sx + sw / 2, yy), (sx + sw / 2, yy - 0.95)])
for y in (12.2 + LH / 2, 1.4 + LH / 2):
    route(ax, [(2 * (bw + bg) + bw, y), (sx - 1.0, y)], color=GOLD)
ax.text(sx - 1.0, 22.0, "Shared by both modes", fontsize=FS + 0.5, fontweight="bold", va="center", color=INK)
fig.savefig(OUT / "engine_modes.png", dpi=250); plt.close(fig)

# ---------------------------------------------------------------- Fig 5 plots (half width each)
a = json.loads((R / "acceptance_same.json").read_text().replace("NaN", "null"))["dose_response"]["per_flaw"]
fig, ax = plt.subplots(figsize=(3.55, 3.0))
ends = []
for k, v in a.items():
    sc = v["scores_L0_to_L5"]
    if v["spearman"] is None:
        continue
    bad = v["spearman"] > -0.9
    ax.plot(range(6), sc, color=CORAL if bad else GOLD, lw=1.8 if bad else 0.9, alpha=1 if bad else .75)
    ends.append([sc[-1], k, bad])
ends.sort()
for i in range(1, len(ends)):                                  # spread the end labels so none overlap
    ends[i][0] = max(ends[i][0], ends[i - 1][0] + 2.7)
for y, k, bad in ends:
    ax.text(5.1, y, k, fontsize=6.3, va="center", color=CORAL if bad else INK)
ax.set_xlim(0, 7.3); ax.set_ylim(50, 104); ax.set_xticks(range(6)); ax.set_xticklabels(["L0", "L1", "L2", "L3", "L4", "L5"], fontsize=7)
ax.tick_params(axis="y", labelsize=7)
ax.set_xlabel("flaw level (L0 = unaltered)", fontsize=7); ax.set_ylabel("overall score", fontsize=7)
ax.set_title("Score against flaw level (coral: ρ > −0.9)", fontsize=7.5, loc="left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.4); fig.savefig(OUT / "dose_response.png", dpi=250); plt.close(fig)

same = json.loads((R / "headline_train_same.json").read_text())
free = json.loads((R / "headline_train_free.json").read_text())
exp = set(same["experimental"])
order = sorted(same["per_flaw_f1"], key=lambda k: same["per_flaw_f1"][k])
fig, ax = plt.subplots(figsize=(3.55, 3.0))
ys = range(len(order))
ax.barh([y + .2 for y in ys], [same["per_flaw_f1"][k] for k in order], .38, color=[LINE if k in exp else GOLD for k in order])
ax.barh([y - .2 for y in ys], [free["per_flaw_f1"][k] for k in order], .38, color=INK)
ax.set_yticks(list(ys)); ax.set_yticklabels(order, fontsize=6.5); ax.tick_params(axis="x", labelsize=7)
ax.set_xlabel("event F1 (IoU ≥ 0.5)", fontsize=7); ax.set_xlim(0, 1)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GOLD), plt.Rectangle((0, 0), 1, 1, color=LINE), plt.Rectangle((0, 0), 1, 1, color=INK)],
          labels=["clean reading", "(experimental)", "upload mode"], frameon=False, fontsize=6.3, loc="lower right")
ax.set_title("Per-flaw F1, train split", fontsize=7.5, loc="left"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.4); fig.savefig(OUT / "per_flaw_f1.png", dpi=250); plt.close(fig)

# ---------------------------------------------------------------- Fig 4: analysis view, cropped from a screenshot of the running app
Image.open(ROOT / "docs" / "screenshots" / "analyse_shout_1440x900.png").crop((72, 170, 1368, 770)).save(OUT / "timeline.png")
print("figures written")
