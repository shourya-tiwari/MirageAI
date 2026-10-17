"""Phase 8: the three report figures not produced by the phase drivers.

    python scripts/make_report_figures.py

Reads only saved outputs (model table, outputs/tables/*) and fits nothing.
Writes outputs/tables/phase8_core_ring_land.csv and outputs/figures/
08_cool_island.png, 08_model_comparison.png and 08_tree_cover_dependence.png.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mirageai import config as c
from mirageai import plotting as P
from mirageai import preprocess as pp

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
YEAR_COL = dict(zip(c.YEARS, (BLUE, ORANGE, AQUA)))
YEAR_MARK = dict(zip(c.YEARS, ("o", "s", "^")))


def style_axes(ax, grid_axis="y") -> None:
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.6)


df = pp.load_model_table()

# ---------------------------------------------------------------------------
# Core-minus-ring LST on land cells (D-034), Phase 1 core/ring definition
# ---------------------------------------------------------------------------
full = pp.load_clean()
centre = ((full.x_utm.min() + full.x_utm.max()) / 2, (full.y_utm.min() + full.y_utm.max()) / 2)
p1 = pd.read_csv(c.table("phase1_cross_year_checks")).set_index("year")
assert np.allclose(pp.core_ring_contrast(full, centre).set_index("year").core_minus_ring,
                   p1.core_minus_ring, atol=5e-4), "core/ring definition no longer matches Phase 1"
pp.core_ring_contrast(df, centre).to_csv(c.table("phase8_core_ring_land"), index=False)

# ---------------------------------------------------------------------------
# Cool-island evidence: anomaly and dryness by built-up fraction, per year
# ---------------------------------------------------------------------------
bins = np.linspace(0, 1, 11)
df["built_bin"] = pd.cut(df.frac_built, bins, include_lowest=True)
g = df.groupby(["year", "built_bin"], observed=True).agg(
    anom=(c.TARGET_ANOM, "mean"), ndbi=("NDBI", "mean"), n=(c.TARGET_ANOM, "size")).reset_index()
g["mid"] = g.built_bin.apply(lambda iv: iv.mid).astype(float) * 100
g.to_csv(c.table("phase8_cool_island_bins"), index=False)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, col, ylab, title in (
        (axes[0], "anom", "Mean LST anomaly (°C)", "Built-up cells are cooler by day"),
        (axes[1], "ndbi", "Mean NDBI (dryness index)", "Dryness is highest in the least-built cells")):
    for y in c.YEARS:
        s = g[g.year == y]
        ax.plot(s.mid, s[col], color=YEAR_COL[y], marker=YEAR_MARK[y], markersize=5, linewidth=2,
                markeredgecolor="white", markeredgewidth=0.8, label=str(y))
    ax.set_xlabel("Built-up fraction of the 100 m cell (%, deciles)", fontsize=9, color=INK2)
    ax.set_ylabel(ylab, fontsize=9, color=INK2)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlim(0, 100)
    style_axes(ax, grid_axis="both")
axes[0].axhline(0, color="#c3c2b7", linewidth=0.8)
axes[0].legend(frameon=False, fontsize=8, loc="upper right", title="Year", title_fontsize=8)
fig.text(0.01, -0.03, "Land cells, pre-monsoon (Mar-May) composites, 2023-25. Anomaly = cell LST minus "
         "that year's AOI median. One line per year.", fontsize=7.5, color=MUTED)
fig.tight_layout()
P.save_figure(fig, c.figure(8, "cool_island"))
plt.close(fig)

# ---------------------------------------------------------------------------
# Model comparison: random CV, spatial 4 km, 2025 unseen and seen-place
# ---------------------------------------------------------------------------
comp = pd.read_csv(c.table("model_comparison")).set_index(["scheme", "model"])
th = pd.read_csv(c.table("phase5_temporal_holdout")).set_index(["scheme", "model"])
models = ["linear", "rf", "xgb", "lgbm"]
labels = {"linear": "Linear regression", "rf": "Random Forest", "xgb": "XGBoost", "lgbm": "LightGBM"}
series = [
    ("Random CV (cell-grouped)", BLUE, "o", True,
     lambda m: (comp.loc[("random", m), "RMSE_mean"], comp.loc[("random", m), "RMSE_std"])),
    ("Spatial CV, 4 km blocks", ORANGE, "s", True,
     lambda m: (comp.loc[("spatial_4km", m), "RMSE_mean"], comp.loc[("spatial_4km", m), "RMSE_std"])),
    ("2025, unseen place and year", AQUA, "^", True,
     lambda m: (th.loc[("spatio_temporal_4km", m), "RMSE_unseen_mean"],
                th.loc[("spatio_temporal_4km", m), "RMSE_unseen_std"])),
    ("2025, place seen in 2023-24", AQUA, "^", False,
     lambda m: (th.loc[("spatio_temporal_4km", m), "RMSE_seen_place_mean"],
                th.loc[("spatio_temporal_4km", m), "RMSE_seen_place_std"])),
]
fig, ax = plt.subplots(figsize=(9, 4.2))
offsets = np.linspace(-0.27, 0.27, len(series))
for (name, colr, mk, filled, f), off in zip(series, offsets):
    for i, m in enumerate(models):
        mean, sd = f(m)
        ax.errorbar(mean, i + off, xerr=sd, fmt=mk, color=colr, markersize=7, elinewidth=1,
                    capsize=0, markerfacecolor=colr if filled else "white", markeredgewidth=1.4,
                    label=name if i == 0 else None)
ax.set_yticks(range(len(models)), [labels[m] for m in models], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("RMSE of LST anomaly (°C); bars = ± 1 std over 5 seeds × 5 folds", fontsize=9, color=INK2)
ax.set_title("Accuracy depends on what the test hides", fontsize=10, color=INK, loc="left")
ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
style_axes(ax, grid_axis="x")
base = [comp.loc[("random", "baseline"), "RMSE_mean"], comp.loc[("spatial_4km", "baseline"), "RMSE_mean"],
        th.loc[("spatio_temporal_4km", "baseline"), "RMSE_unseen_mean"]]
fig.text(0.01, -0.03, f"Hollow vs filled triangles: the 2025 memorisation gap. Baseline (RMSE "
         f"{min(base):.2f}-{max(base):.2f} °C) omitted for scale.", fontsize=7.5, color=MUTED)
fig.tight_layout()
P.save_figure(fig, c.figure(8, "model_comparison"))
plt.close(fig)

# ---------------------------------------------------------------------------
# Tree-cover dependence with the fitted knot (Phase 6, D-037)
# ---------------------------------------------------------------------------
thr = pd.read_csv(c.table("phase6_thresholds")).set_index("feature").loc["frac_tree"]
fig, ax = plt.subplots(figsize=(8.5, 4.6))
for name, colr, ls in (("lgbm", BLUE, "-"), ("xgb", ORANGE, "--")):
    s = pd.read_parquet(c.TABLES / f"phase6_shap_{name}.parquet", columns=["row_id", "shap_frac_tree"])
    x = df.loc[s.row_id, "frac_tree"].to_numpy() * 100
    y = s.shap_frac_tree.to_numpy()
    if name == "lgbm":
        ax.scatter(x, y, s=4, alpha=0.25, color="#86b6ef", linewidths=0, rasterized=True,
                   label="cell-years (LightGBM)")
    pos = x > 0
    q = pd.qcut(x[pos], 20, duplicates="drop")
    bm = pd.DataFrame({"x": x[pos], "y": y[pos], "q": q}).groupby("q", observed=True).mean()
    xs = np.r_[0, bm.x.to_numpy()]
    ys = np.r_[y[~pos].mean(), bm.y.to_numpy()]
    ax.plot(xs, ys, color=colr, linestyle=ls, linewidth=2, marker="o", markersize=3,
            label=f"binned mean, {'LightGBM' if name == 'lgbm' else 'XGBoost'}")
k, lo, hi = thr.pos_knot * 100, thr.pos_knot_lo * 100, thr.pos_knot_hi * 100
ax.axvspan(lo, hi, color=ORANGE, alpha=0.15, linewidth=0)
ax.axvline(k, color=INK, linewidth=1)
ax.text(k + 0.6, ax.get_ylim()[1] * 0.9, f"knot {k:.1f}% (95% CI {lo:.1f}-{hi:.1f}%)\n"
        f"{thr.pos_slope_below / 10:+.2f} °C per 10 points below, "
        f"{thr.pos_slope_above / 10:+.2f} above", fontsize=8, color=INK, va="top")
ax.annotate(f"0 → any canopy: {thr.step_zero_to_low:+.2f} °C", xy=(0, y[~pos].mean()),
            xytext=(6, y[~pos].mean() + 0.35), fontsize=8, color=INK,
            arrowprops={"arrowstyle": "->", "color": INK2, "lw": 0.7})
ax.axhline(0, color="#c3c2b7", linewidth=0.6)
ax.set_xlim(-1, 60)
ax.set_xlabel("Tree cover fraction of the 100 m cell (%)", fontsize=9, color=INK2)
ax.set_ylabel("SHAP contribution to LST anomaly (°C)", fontsize=9, color=INK2)
ax.set_title("Tree cover: a step at first canopy, then cooling that flattens past ~10%",
             fontsize=10, color=INK, loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower right")
style_axes(ax, grid_axis="both")
fig.text(0.01, -0.03, "Interventional SHAP, 10,000 stratified cell-years. Knot and slopes: segmented "
         "fit on cells with tree cover > 0 (D-037). Associational, not causal.", fontsize=7.5, color=MUTED)
fig.tight_layout()
P.save_figure(fig, c.figure(8, "tree_cover_dependence"))
plt.close(fig)
print("wrote 08_cool_island.png, 08_model_comparison.png, 08_tree_cover_dependence.png")

# ---------------------------------------------------------------------------
# Source tables for numbers the report quotes that no phase table holds
# ---------------------------------------------------------------------------
from mirageai import evaluate as E  # noqa: E402

feats = list(c.FEATURES_TREE)
corr = df[feats].corrwith(df[c.TARGET_ANOM]).rename("r_with_LST_anom").to_frame()
corr.index.name = "feature"
corr.to_csv(c.table("phase8_feature_correlations"))
pairs = pd.DataFrame([{"a": a, "b": b, "r": float(df[a].corr(df[b]))}
                      for a, b in (("NDVI", "NDBI"), ("NDVI", "MNDWI"), ("NDBI", "MNDWI"))])
pairs.to_csv(c.table("phase8_index_correlations"), index=False)

# Residual range of the tuned LightGBM by the CV scheme that produced the
# residuals (D-028 note): seed 42 everywhere, plus seeds 43-46 at 4 km.
rows = []
for scheme in ("random", "spatial_1km", "spatial_2km", "spatial_4km", "spatial_8km"):
    for seed in c.CV_SEEDS if scheme == "spatial_4km" else (c.SEED,):
        o = E.load_oof(scheme, E.BEST_MODEL, seed=seed, columns=["row_id"])
        r = np.full(len(df), np.nan)
        r[o.row_id.to_numpy()] = o.resid.to_numpy()
        rr = E.residual_ranges(df, r)
        rows.append({"scheme": scheme, "seed": seed, **{f"range_95_{y}": v for y, v in
                                                         zip(rr.year, rr.range_95_m)},
                     "median_range_95_m": float(rr.range_95_m.median())})
pd.DataFrame(rows).to_csv(c.table("phase8_residual_range_by_scheme"), index=False)
print("wrote phase8_feature_correlations, phase8_index_correlations, phase8_residual_range_by_scheme")
