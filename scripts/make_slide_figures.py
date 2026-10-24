"""Phase 8 (T8.4): the two slide figures not produced by the phase drivers.

    python scripts/make_slide_figures.py

Reads saved tables only. Writes outputs/figures/09_group_ranking.png and
09_scenario_bracket.png, sized for half / two-thirds of a 16:9 slide, and
09_pipeline.png, the method diagram used by the report and the deck.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mirageai import config as c
from mirageai import plotting as P

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def style_axes(ax, grid_axis="x") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=11, length=3, width=0.6)


# ---------------------------------------------------------------------------
# Grouped SHAP ranking (T6.5 groups, D-038)
# ---------------------------------------------------------------------------
g = pd.read_csv(c.table("phase6_group_importance")).set_index("group")
labels = {"spectral": "Spectral (NDVI + dryness + moisture)", "dry cover": "Dry cover (grass, crop, bare, shrub)",
          "tree cover": "Tree cover", "terrain": "Terrain (elevation, slope)",
          "water proximity": "Water proximity", "built": "Built-up fraction"}
order = list(g.sort_values("lgbm_all").index)
fig, ax = plt.subplots(figsize=(8.6, 4.9))
y = np.arange(len(order))
h = 0.38
ax.barh(y + h / 2, g.loc[order, "lgbm_all"], height=h, color=BLUE, label="All land cells", edgecolor="white")
ax.barh(y - h / 2, g.loc[order, "lgbm_urban"], height=h, color=ORANGE, label="Urban cells", edgecolor="white")
for yi, k in zip(y, order):
    ax.text(g.loc[k, "lgbm_all"] + 0.02, yi + h / 2, f"{g.loc[k, 'lgbm_all']:.2f}", va="center", fontsize=11, color=INK2)
    ax.text(g.loc[k, "lgbm_urban"] + 0.02, yi - h / 2, f"{g.loc[k, 'lgbm_urban']:.2f}", va="center", fontsize=11, color=INK2)
ax.set_yticks(y, [labels[k] for k in order], fontsize=11)
ax.set_xlabel("Mean |SHAP| (°C of LST anomaly)", fontsize=11, color=INK2)
ax.legend(frameon=False, fontsize=11, loc="lower right")
style_axes(ax)
fig.tight_layout()
P.save_figure(fig, c.figure(9, "group_ranking"))
plt.close(fig)

# ---------------------------------------------------------------------------
# Scenario bracket: lower to realistic, first point excluded (D-040)
# ---------------------------------------------------------------------------
s = pd.read_csv(c.table("phase7_summary"))
e = s[s.model == "ensemble"].set_index(["scenario", "bound"])
rows = [("C", "To the 10.2% knot\n(avg +9.2 points)"), ("A", "+10 points"), ("B", "+20 points")]
fig, ax = plt.subplots(figsize=(6.6, 4.4))
for i, (sc, lab) in enumerate(rows):
    lo, hi = e.loc[(sc, "lower")], e.loc[(sc, "realistic")]
    ax.plot([lo.dLST_excl_first_point, hi.dLST_excl_first_point], [i, i], color=BLUE, linewidth=8,
            solid_capstyle="round", alpha=0.35)
    for b, mk in ((lo, "o"), (hi, "D")):
        ax.errorbar(b.dLST_excl_first_point, i, xerr=[[b.dLST_excl_first_point - b.dLST_excl_min],
                                                     [b.dLST_excl_max - b.dLST_excl_first_point]],
                    fmt=mk, color=BLUE, markersize=9, capsize=3, elinewidth=1.2, markeredgecolor="white")
    ax.text(lo.dLST_excl_first_point, i + 0.22, f"{lo.dLST_excl_first_point:+.2f}", fontsize=11,
            color=INK, ha="center")
    ax.text(hi.dLST_excl_first_point, i + 0.22, f"{hi.dLST_excl_first_point:+.2f}", fontsize=11,
            color=INK, ha="center")
ax.set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=11)
ax.set_ylim(-0.6, len(rows) - 0.3)
ax.axvline(0, color="#c3c2b7", linewidth=0.8)
ax.set_xlim(-1.9, 0.1)
ax.set_xlabel("Mean ΔLST over target cells (°C)  ← cooler", fontsize=11, color=INK2)
ax.text(-1.88, -0.55, "● lower bound (indices fixed)   ◆ realistic bound (indices respond)\n"
        "whiskers: range across LightGBM, XGBoost, RF", fontsize=9, color=MUTED, va="bottom")
style_axes(ax)
fig.tight_layout()
P.save_figure(fig, c.figure(9, "scenario_bracket"))
plt.close(fig)

# ---------------------------------------------------------------------------
# Pipeline diagram (report Fig. 1, slide "ML pipeline"); text only, no numbers
# ---------------------------------------------------------------------------
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

TINT, NAVY = "#eef4fc", "#14325c"
STAGES = [
    ("Extract", "Earth Engine composites:\nLandsat LST, Sentinel-2,\nWorldCover, SRTM", False),
    ("Preprocess", "100 m grid, 3 years,\nLST anomaly target,\nwater cells removed", False),
    ("Validate", "residual variogram\n-> 4 km blocks,\n5 random layouts", True),
    ("Model", "baseline, linear, RF,\nXGBoost, LightGBM;\ntuned inside folds", False),
    ("Evaluate", "random vs spatial vs\nspatio-temporal CV;\nerror diagnostics", True),
    ("Explain", "interventional SHAP,\ndriver groups,\nthresholds", True),
    ("Scenarios", "+tree cover in the\nhottest urban cells,\nlower / realistic bound", False),
    ("Communicate", "report, slides,\ndashboard; numbers\nchecked vs tables", False),
]
fig, ax = plt.subplots(figsize=(12, 4.6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 4.6)
ax.axis("off")
bw, bh, gx = 2.55, 1.75, 0.45
for i, (name, detail, key) in enumerate(STAGES):
    row, col = divmod(i, 4)
    x0, y0 = 0.25 + col * (bw + gx), 2.55 - row * 2.3
    fc, fg, sub = (BLUE, "white", "white") if key else (TINT, NAVY, INK2)
    ax.add_patch(FancyBboxPatch((x0, y0), bw, bh, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=fc, edgecolor="none"))
    ax.text(x0 + 0.18, y0 + bh - 0.32, f"{i + 1}", fontsize=15, fontweight="bold", color=fg, va="center")
    ax.text(x0 + 0.5, y0 + bh - 0.32, name, fontsize=14, fontweight="bold", color=fg, va="center")
    ax.text(x0 + 0.18, y0 + bh - 0.62, detail, fontsize=10.5, color=sub, va="top", linespacing=1.35)
    if col < 3:
        ax.add_patch(FancyArrowPatch((x0 + bw + 0.04, y0 + bh / 2), (x0 + bw + gx - 0.04, y0 + bh / 2),
                                     arrowstyle="-|>", mutation_scale=14, color=MUTED, linewidth=1.4))
ax.add_patch(FancyArrowPatch((0.25 + 3 * (bw + gx) + bw / 2, 2.55 - 0.04), (0.25 + bw / 2, 0.25 + bh + 0.04),
                             connectionstyle="arc3,rad=0.0", arrowstyle="-|>", mutation_scale=14,
                             color=MUTED, linewidth=1.4, linestyle=(0, (4, 3))))
ax.text(11.75, 0.02, "blue: the steps that decide whether the results can be trusted", fontsize=9.5,
        color=MUTED, ha="right", va="bottom")
fig.tight_layout(pad=0.2)
P.save_figure(fig, c.figure(9, "pipeline"))
plt.close(fig)
print("wrote 09_group_ranking.png, 09_scenario_bracket.png, 09_pipeline.png")
