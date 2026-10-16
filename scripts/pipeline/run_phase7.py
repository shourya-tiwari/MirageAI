"""One-command driver for Phase 7 (T7.1-T7.5): greening scenarios for the hottest
urban cells, bracketed and run through three models.

    python scripts/pipeline/run_phase7.py

Targets: urban cells in the hottest decile of observed 3-year mean LST_anom
(D-039). Scenarios: A +10 points tree cover, B +20 points, C raise to the
fitted tree-cover knot from Phase 6 (D-040). Each runs under two bounds
(D-009): fractions only, indices held fixed (lower), and fractions plus
indices updated through land-cover -> index models (realistic, D-041). Each
runs through tuned LightGBM, XGBoost and RF refitted on all land rows
(D-042), per year, averaged over 2023-25. Scenario rows outside the training
support are flagged (D-043). Every number is an associational model response,
not a causal effect of planting trees.
"""

from __future__ import annotations

import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy import ndimage
from sklearn.metrics import r2_score

from mirageai import config as c
from mirageai import cv
from mirageai import explain as X
from mirageai import models as M
from mirageai import plotting as P
from mirageai import preprocess as pp
from mirageai import scenarios as SC

t_start = time.perf_counter()
c.ensure_dirs()
df = pp.load_model_table()
cv.require_land_only(df)
FEATS = M.features_for("lgbm")
PARAMS = {n: v["best_params"] for n, v in
          json.loads((c.MODELS / "best_params.json").read_text(encoding="utf-8")).items()}
KNOT = float(pd.read_csv(c.table("phase6_thresholds")).set_index("feature").loc["frac_tree", "pos_knot"])
SCEN = ("A", "B", "C")
SCEN_LABEL = {"A": "A: +10 points tree cover", "B": "B: +20 points tree cover",
              "C": f"C: raise tree cover to the {KNOT:.1%} knot"}
BOUNDS = ("lower", "realistic")
LST_MODELS = ("lgbm", "xgb", "rf")
MODEL_LABEL = {"lgbm": "LightGBM", "xgb": "XGBoost", "rf": "Random Forest"}
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
LAND, URBAN = "#eeede8", "#c3c2b7"


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


# ---------------------------------------------------------------------------
# T7.2 Target cells (D-039)
# ---------------------------------------------------------------------------
targets = SC.target_cells(df)
targets.to_csv(c.table("phase7_target_cells"))
cells_all = df.groupby("cell_id").agg(x_utm=("x_utm", "first"), y_utm=("y_utm", "first"),
                                      is_urban=("is_urban", "first"))
# Clusters of 8-connected target cells, for "where are they".
grid_t, extent = P.cells_to_grid(cells_all.x_utm.to_numpy(), cells_all.y_utm.to_numpy(),
                                 cells_all.index.isin(targets.index).astype(float))
lab, n_clusters = ndimage.label(np.nan_to_num(grid_t) > 0, structure=np.ones((3, 3)))
col = np.floor(targets.x_utm / c.SCALE).astype(int) - int(np.floor(cells_all.x_utm.min() / c.SCALE))
row = int(np.floor(cells_all.y_utm.max() / c.SCALE)) - np.floor(targets.y_utm / c.SCALE).astype(int)
targets["cluster"] = lab[row.to_numpy(), col.to_numpy()]
cl = (targets.groupby("cluster")
      .agg(cells=("mean_anom", "size"), lon=("lon", "mean"), lat=("lat", "mean"),
           x_km=("x_utm", lambda s: s.mean() / 1000), y_km=("y_utm", lambda s: s.mean() / 1000),
           mean_anom=("mean_anom", "mean"))
      .sort_values("cells", ascending=False))
cl.to_csv(c.table("phase7_target_clusters"))
cx, cy = cells_all.x_utm.mean(), cells_all.y_utm.mean()
dx, dy = targets.x_utm - cx, targets.y_utm - cy
quad = pd.Series(np.where(dy >= 0, "N", "S")) + pd.Series(np.where(dx >= 0, "E", "W"))
tdf = df[df.cell_id.isin(targets.index)]
print(f"T7.2 targets: {len(targets):,} of {targets.attrs['n_urban_cells']:,} urban cells "
      f"(observed 3-year mean LST_anom >= {targets.attrs['threshold']:+.2f} °C; "
      f"{len(tdf):,} cell-years)")
print(f"  mean anomaly {targets.mean_anom.mean():+.2f} °C; tree cover median "
      f"{tdf.frac_tree.median():.1%}, {(tdf.groupby('cell_id').frac_tree.first() < KNOT).mean():.0%} "
      f"below the {KNOT:.1%} knot; built median {tdf.frac_built.median():.0%}")
print(f"  {n_clusters} clusters of 8-connected cells; the 5 largest hold "
      f"{cl.cells.iloc[:5].sum() / len(targets):.0%}")
print(cl.head(8).round(3).to_string())
print("  quadrant shares (vs AOI centre):", (quad.value_counts(normalize=True).round(3)).to_dict(),
      f" median distance from centre {np.hypot(dx, dy).median() / 1000:.1f} km")

# ---------------------------------------------------------------------------
# D-041 Index models: spatial-CV validation, then per-year final fits
# ---------------------------------------------------------------------------
IDX_PARAMS = PARAMS["lgbm"]
cv_rows = []
t0 = time.perf_counter()
for year, part in df.groupby("year"):
    part = part.reset_index(drop=True)
    for seed in c.CV_SEEDS:
        for k, (tr, te) in enumerate(cv.spatial_splits(part, c.PRIMARY_BLOCK_SIZE_M, seed=seed)):
            for idx in SC.INDEX_TARGETS:
                m = M.make_model("lgbm", **IDX_PARAMS).fit(
                    part.loc[tr, list(SC.INDEX_FEATURES)], part.loc[tr, idx])
                pred = m.predict(part.loc[te, list(SC.INDEX_FEATURES)])
                y = part.loc[te, idx].to_numpy()
                cv_rows.append({"year": int(year), "seed": seed, "fold": k, "index": idx,
                                "R2": r2_score(y, pred), "RMSE": float(np.sqrt(np.mean((y - pred) ** 2))),
                                "target_sd": float(y.std())})
idx_cv = pd.DataFrame(cv_rows)
idx_cv.to_csv(c.table("phase7_index_model_cv"), index=False)
idx_sum = idx_cv.groupby("index").agg(R2_mean=("R2", "mean"), R2_std=("R2", "std"),
                                      RMSE_mean=("RMSE", "mean"), target_sd=("target_sd", "mean"))
idx_year = idx_cv.pivot_table(index="index", columns="year", values="R2", aggfunc="mean")
print(f"\nD-041 index models, spatial CV 4 km x {len(c.CV_SEEDS)} seeds, per year "
      f"[{(time.perf_counter() - t0) / 60:.1f} min]")
print(idx_sum.round(3).to_string())
print(idx_year.round(3).to_string())
index_models = SC.fit_index_models(df, IDX_PARAMS)

# ---------------------------------------------------------------------------
# D-042 LST models: LightGBM / XGBoost from Phase 6, RF refitted the same way
# ---------------------------------------------------------------------------
lst = {n: joblib.load(c.MODELS / f"{n}_all.joblib")["model"] for n in ("lgbm", "xgb")}
rf_path = c.MODELS / "rf_all.joblib"
with M.PeakMemory() as mem:
    t0 = time.perf_counter()
    lst["rf"] = X.fit_final(df, "rf", PARAMS["rf"])
    print(f"\nrf_all fit {time.perf_counter() - t0:.0f}s, peak {mem.peak_mb:.0f} MB")
joblib.dump({"model": lst["rf"], "features": FEATS, "params": PARAMS["rf"], "target": c.TARGET_ANOM,
             "fitted_on": "all_years_land", "years": list(c.YEARS), "water_excluded": True,
             "features_rounded": "float32"}, rf_path, compress=3)

# ---------------------------------------------------------------------------
# D-043 Support guard
# ---------------------------------------------------------------------------
guard = SC.SupportGuard(df, FEATS)
print(f"support guard: k = {guard.k}, threshold {guard.threshold:.3f} (99th pct of training "
      f"k-NN distance); observed target rows flagged {guard.flag(tdf).mean():.1%}")

# ---------------------------------------------------------------------------
# T7.3 Scenarios x bounds x models
# ---------------------------------------------------------------------------
base = tdf.reset_index(drop=True)
frac = list(c.FRACTION_FEATURES)
res = []
for s in SCEN:
    want = SC.scenario_delta(s, base.frac_tree.to_numpy(), KNOT)
    new_fr, short = SC.add_tree_cover(base[frac], want)
    lower = base.copy()
    lower[frac] = new_fr[frac]
    realistic = SC.updated_indices(base, lower, index_models)
    for bound, scen in (("lower", lower), ("realistic", realistic)):
        out = base[["cell_id", "year"]].copy()
        out["scenario"], out["bound"] = s, bound
        out["points_requested"] = want * 100
        out["points_added"] = (want - short) * 100
        out["shortfall"] = short > 1e-12
        out["flagged"] = guard.flag(scen)
        for idx in SC.INDEX_TARGETS:
            out[f"d_{idx}"] = (scen[idx] - base[idx]).to_numpy()
        for n in LST_MODELS:
            out[f"dLST_{n}"] = SC.delta_lst(lst[n], base, scen, FEATS)
        res.append(out)
res = pd.concat(res, ignore_index=True)  # saved after the sweep adds the first-point step


# ---------------------------------------------------------------------------
# Dose-response sweep: +0..30 points, for the diminishing-returns figure
# ---------------------------------------------------------------------------
STEPS = np.arange(0, 31)
sweep_rows, marg_rows = [], []
prev, step1 = {}, {}  # step1: each row's response to the first +1 point
for k in STEPS:
    new_fr, short = SC.add_tree_cover(base[frac], k / 100)
    lower = base.copy()
    lower[frac] = new_fr[frac]
    realistic = SC.updated_indices(base, lower, index_models)
    for bound, scen in (("lower", lower), ("realistic", realistic)):
        flagged = guard.flag(scen) if k else np.zeros(len(base), bool)
        for n in LST_MODELS:
            d = SC.delta_lst(lst[n], base, scen, FEATS)
            sweep_rows.append({"points": int(k), "bound": bound, "model": n,
                               "dLST": float(pd.Series(d).groupby(base.year).mean().mean()),
                               "pct_flagged": 100 * flagged.mean()})
            if k == 1:
                step1[(bound, n)] = d.copy()
            if k:
                step = d - prev[(bound, n)]
                reached = np.round((base.frac_tree.to_numpy() + k / 100 - short) * 100)
                marg_rows.append(pd.DataFrame({"bound": bound, "model": n, "reached_pct": reached,
                                               "marginal": step, "year": base.year.to_numpy(),
                                               "flagged": flagged}))
            prev[(bound, n)] = d
sweep = pd.DataFrame(sweep_rows)
sweep.to_csv(c.table("phase7_dose_response"), index=False)
marg = pd.concat(marg_rows, ignore_index=True)
marg = marg[marg.reached_pct <= 40]
marg_tab = (marg.groupby(["bound", "model", "reached_pct"])
            .agg(marginal=("marginal", "mean"), n=("marginal", "size")).reset_index())
marg_tab.to_csv(c.table("phase7_marginal_cooling"), index=False)
mt = marg_tab[marg_tab.n >= 200].pivot_table(index="reached_pct", columns=["bound"], values="marginal",
                                             aggfunc="mean")
knot_pct = KNOT * 100
marg_summary = []
for b in BOUNDS:
    first = mt.loc[1, b]
    below = mt.loc[(mt.index >= 2) & (mt.index <= knot_pct), b].mean()
    above = mt.loc[(mt.index > knot_pct + 2) & (mt.index <= 30), b].mean()
    marg_summary.append({"bound": b, "first_point_0_to_1pct": first, "per_point_2pct_to_knot": below,
                         "per_point_12_to_30pct": above, "ratio_above_to_below": above / below})
    print(f"marginal cooling per +1 point ({b}): first point (0 -> 1%) {first:+.3f} °C; "
          f"2% to the knot {below:+.4f}; 12-30% {above:+.4f} (ratio {above / below:.2f})")
pd.DataFrame(marg_summary).to_csv(c.table("phase7_marginal_summary"), index=False)

# Sensitivity: the first point from zero carries the presence step found in
# Phase 6 (D-037). Remove it from cells that start at 0% tree cover.
zero_start = (base.frac_tree.to_numpy() == 0)
print(f"target cell-years starting at 0% tree cover: {zero_start.mean():.1%}")


def year_avg(frame: pd.DataFrame, col: str) -> float:
    """Mean over target rows within each year, then the mean of the three years."""
    return float(frame.groupby("year")[col].mean().mean())


rows = []
for (s, bound), g in res.groupby(["scenario", "bound"]):
    kept = g[~g.flagged]
    per_model = {n: (year_avg(g, f"dLST_{n}"), year_avg(kept, f"dLST_{n}")) for n in LST_MODELS}
    # g keeps base's row order, so the per-row first-point step lines up.
    assert (g.year.to_numpy() == base.year.to_numpy()).all()
    excl = {n: year_avg(g.assign(x=g[f"dLST_{n}"].to_numpy() - step1[(bound, n)] * zero_start), "x")
            for n in LST_MODELS}
    common = {"scenario": s, "bound": bound, "cell_years": len(g),
              "pct_flagged": 100 * g.flagged.mean(),
              "cells_shortfall": int(g.loc[g.shortfall, "cell_id"].nunique()),
              "mean_points_added": g.points_added.mean(),
              "cells_changed": int(g.loc[g.points_added > 0, "cell_id"].nunique())}
    for n, (a, k) in per_model.items():
        rows.append({**common, "model": n, "dLST_mean": a, "dLST_mean_unflagged": k,
                     "dLST_excl_first_point": excl[n]})
    allm = np.array([v[0] for v in per_model.values()])
    kepm = np.array([v[1] for v in per_model.values()])
    exm = np.array(list(excl.values()))
    rows.append({**common, "model": "ensemble", "dLST_mean": allm.mean(), "dLST_min": allm.min(),
                 "dLST_max": allm.max(), "dLST_mean_unflagged": kepm.mean(),
                 "dLST_min_unflagged": kepm.min(), "dLST_max_unflagged": kepm.max(),
                 "dLST_excl_first_point": exm.mean(), "dLST_excl_min": exm.min(),
                 "dLST_excl_max": exm.max()})
summary = pd.DataFrame(rows)
summary["dLST_per_point"] = summary.dLST_mean / summary.mean_points_added
summary.to_csv(c.table("phase7_summary"), index=False)
# Per-row first-point step (zero for cells that start with some tree cover), so
# the headline, which excludes it, can be mapped (D-040).
for (s, bound), g in res.groupby(["scenario", "bound"]):
    for n in LST_MODELS:
        res.loc[g.index, f"first_point_{n}"] = step1[(bound, n)] * zero_start
res.to_parquet(c.TABLES / "phase7_cell_results.parquet", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print("\nT7.4 summary: mean ΔLST over target cells (°C; per-year means averaged over 2023-25)")
    print(summary.round(3).to_string(index=False))
    print("\nIndex changes under the realistic bound (mean over target cell-years)")
    print(res[res.bound == "realistic"].groupby("scenario")[[f"d_{i}" for i in SC.INDEX_TARGETS]]
          .mean().round(4).to_string())


# ===========================================================================
# Figures
# ===========================================================================
# Target cells
fig, ax = plt.subplots(figsize=(9, 7.4))
ug, _ = P.cells_to_grid(cells_all.x_utm.to_numpy(), cells_all.y_utm.to_numpy(),
                        cells_all.is_urban.astype(float).to_numpy())
layer = np.where(np.isnan(ug), np.nan, ug + np.nan_to_num(grid_t))
ax.imshow(layer, extent=extent, origin="upper", interpolation="nearest", vmin=-0.5, vmax=2.5,
          cmap=ListedColormap([LAND, URBAN, ORANGE]))
for i, (_, r) in enumerate(cl.head(5).iterrows(), 1):
    ax.annotate(f"{i}", (r.x_km, r.y_km), xytext=(8, 8), textcoords="offset points", fontsize=8,
                color=INK, arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.6})
ax.set_aspect("equal")
ax.set_xlabel("UTM 43N easting (km)", fontsize=9)
ax.set_ylabel("UTM 43N northing (km)", fontsize=9)
ax.tick_params(labelsize=8, length=3, width=0.6)
handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in (ORANGE, URBAN, LAND)]
ax.legend(handles, [f"Target: hottest urban decile ({len(targets):,} cells)",
                    f"Other urban cells (built ≥ 50%)", "Other land"], fontsize=8, frameon=True,
          framealpha=0.9, loc="upper left", bbox_to_anchor=(1.01, 1))
ax.set_title("Scenario targets: urban cells in the hottest decile of observed 2023-25 mean LST anomaly",
             fontsize=10, pad=8)
fig.text(0.01, 0.02, f"Numbers: the five largest clusters of 8-connected target cells. Threshold "
         f"{targets.attrs['threshold']:+.2f} °C. White: water-majority cells, excluded (D-034).",
         fontsize=7.5, color=MUTED)
P.save_figure(fig, c.figure(7, "target_cells"))
plt.close(fig)

# ΔLST maps: realistic bound with the first point excluded (the headline, D-040),
# ensemble mean over models and years
real = res[res.bound == "realistic"].copy()
real["dLST_ens"] = (real[[f"dLST_{n}" for n in LST_MODELS]].to_numpy()
                    - real[[f"first_point_{n}" for n in LST_MODELS]].to_numpy()).mean(axis=1)
per_cell = real.groupby(["scenario", "cell_id"]).dLST_ens.mean().unstack(0)
vmax = float(np.nanpercentile(np.abs(per_cell.to_numpy()), 99))
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
for ax, s in zip(axes, SCEN):
    v = per_cell[s].reindex(cells_all.index).to_numpy()
    g, ext = P.cells_to_grid(cells_all.x_utm.to_numpy(), cells_all.y_utm.to_numpy(), v)
    ax.imshow(np.where(np.isnan(ug), np.nan, ug), extent=ext, origin="upper",
              interpolation="nearest", cmap=ListedColormap([LAND, URBAN]), vmin=-0.5, vmax=1.5)
    im = ax.imshow(g, extent=ext, origin="upper", interpolation="nearest", cmap=c.DIVERGING_CMAP,
                   vmin=-vmax, vmax=vmax)
    ens = summary[(summary.scenario == s) & (summary.model == "ensemble")].set_index("bound")
    ax.set_title(f"{SCEN_LABEL[s]}\n{ens.loc['lower', 'dLST_excl_first_point']:+.2f} (lower) to "
                 f"{ens.loc['realistic', 'dLST_excl_first_point']:+.2f} °C (realistic)",
                 fontsize=9.5, loc="left")
    ax.set_aspect("equal")
    ax.set_xlabel("UTM 43N easting (km)", fontsize=8.5)
    ax.tick_params(labelsize=7.5, length=3, width=0.6)
axes[0].set_ylabel("UTM 43N northing (km)", fontsize=8.5)
cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
cb.set_label("ΔLST anomaly, scenario − observed (°C)\nblue = cooler", fontsize=8.5)
cb.ax.tick_params(labelsize=7.5)
fig.text(0.01, -0.02, "Map: realistic bound (fractions and indices updated, D-041), excluding the "
         "0 → 1% 'any canopy' step (D-040); mean of LightGBM, XGBoost and RF and of 2023-25. "
         "Titles: lower-realistic bracket. Associational model response, not a causal effect.",
         fontsize=7.5, color=MUTED)
P.save_figure(fig, c.figure(7, "delta_lst_maps"))
plt.close(fig)

# Cooling per +1 point of tree cover: cumulative and marginal
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
for b, colr, ls in (("realistic", BLUE, "-"), ("lower", ORANGE, "--")):
    w = sweep[sweep.bound == b].pivot_table(index="points", columns="model", values="dLST")
    ax.fill_between(w.index, w.min(axis=1), w.max(axis=1), color=colr, alpha=0.15, linewidth=0)
    ax.plot(w.index, w.mean(axis=1), color=colr, linestyle=ls, linewidth=2,
            label=f"{b} bound (mean of 3 models; band = model range)")
for s, pts in (("A", 10), ("B", 20)):
    ax.axvline(pts, color="#c3c2b7", linewidth=0.8)
    ax.text(pts, ax.get_ylim()[0], f" {s}", fontsize=8, color=INK2, va="bottom")
ax.axhline(0, color="#c3c2b7", linewidth=0.6)
ax.set_xlabel("Tree cover added (percentage points)", fontsize=9, color=INK2)
ax.set_ylabel("Mean ΔLST anomaly over target cells (°C)", fontsize=9, color=INK2)
ax.set_title("Cumulative cooling as tree cover is added", fontsize=10, color=INK, loc="left")
ax.legend(frameon=False, fontsize=7.5, loc="lower left")
style_axes(ax, grid_axis="both")
ax = axes[1]
for b, colr, ls in (("realistic", BLUE, "-"), ("lower", ORANGE, "--")):
    w = marg_tab[(marg_tab.bound == b) & (marg_tab.n >= 200)].pivot_table(
        index="reached_pct", columns="model", values="marginal")
    ax.fill_between(w.index, w.min(axis=1), w.max(axis=1), color=colr, alpha=0.15, linewidth=0)
    ax.plot(w.index, w.mean(axis=1), color=colr, linestyle=ls, linewidth=2, label=f"{b} bound")
# The first point from zero is several times larger than any other step; it is
# annotated off-scale so the rest of the curve stays readable.
ms = pd.DataFrame(marg_summary).set_index("bound")
ax.set_ylim(-0.45, 0.05)
ax.annotate(f"0 → 1%: {ms.loc['realistic', 'first_point_0_to_1pct']:+.2f} °C realistic,\n"
            f"{ms.loc['lower', 'first_point_0_to_1pct']:+.2f} °C lower (off scale; presence step)",
            xy=(1, -0.45), xytext=(3, -0.40), fontsize=7.5, color=INK,
            arrowprops={"arrowstyle": "->", "color": INK2, "lw": 0.7})
ax.axvline(knot_pct, color=INK, linewidth=1)
ax.text(knot_pct, 0.04, f" Phase 6 knot {knot_pct:.1f}% (scenario C target)",
        fontsize=7.5, color=INK, va="top")
ax.axhline(0, color="#c3c2b7", linewidth=0.6)
ax.set_xlabel("Tree cover reached after the step (%)", fontsize=9, color=INK2)
ax.set_ylabel("ΔLST per +1 point of tree cover (°C)", fontsize=9, color=INK2)
ax.set_title("Cooling per +1 point of tree cover", fontsize=10, color=INK, loc="left")
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
style_axes(ax, grid_axis="both")
fig.text(0.01, -0.03, "Target cells, +1 point steps from each cell's own cover, mean of 2023-25; "
         "bins with ≥ 200 cell-years. Associational model response, not a causal effect.",
         fontsize=7.5, color=MUTED)
fig.tight_layout()
P.save_figure(fig, c.figure(7, "cooling_per_point"))
plt.close(fig)

print(f"\nDONE in {(time.perf_counter() - t_start) / 60:.1f} min")
