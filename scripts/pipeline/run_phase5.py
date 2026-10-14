"""One-command driver for Phase 5 (T5.1-T5.8): compare models across CV schemes,
score the 2025 holdout, and diagnose where the errors are.

    python scripts/pipeline/run_phase5.py

Reads the Phase 4 outputs (fold metrics, out-of-fold predictions, final
2023-24 models); fits nothing new. Headline numbers are mean +/- std over every
seed x fold run of the shuffled block layouts; the 4 km strip layout is shown
beside them as a stress test (D-033). Water-majority cells are excluded
throughout (D-034). Tables go to outputs/tables/phase5_*.csv (plus
model_comparison.csv), figures to outputs/figures/05_*.png.
"""

from __future__ import annotations

import gc
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mirageai import config as c
from mirageai import cv
from mirageai import evaluate as E
from mirageai import models as M
from mirageai import plotting as P
from mirageai import preprocess as pp

t_start = time.perf_counter()
df = pp.load_model_table()
cv.require_land_only(df)
c.ensure_dirs()
fold_metrics = pd.read_csv(c.table("phase4_fold_metrics"))
BEST = E.BEST_MODEL
STRIP_4KM = E.PRIMARY_SCHEME + E.STRIP_SUFFIX
STRIP_ST = E.SPATIO_TEMPORAL_SCHEME + E.STRIP_SUFFIX

# Chart tokens: categorical slots in fixed order, validated for CVD separation.
# Two slots sit below 3:1 on white, so every series is also direct-labelled.
SERIES = {"linear": "#2a78d6", "rf": "#eb6834", "xgb": "#1baf7a", "lgbm": "#eda100"}
MARKERS = {"linear": "o", "rf": "s", "xgb": "^", "lgbm": "D"}
LABELS = {"baseline": "Baseline", "linear": "Linear", "rf": "Random Forest",
          "xgb": "XGBoost", "lgbm": "LightGBM"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def style_axes(ax) -> None:
    """Recessive hairline chrome."""
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.6)


def pm(mean: float, std: float, nd: int = 3) -> str:
    """Format mean +/- std."""
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


# ---------------------------------------------------------------------------
# T5.1 Model x scheme comparison (5 seeds x 5 folds)
# ---------------------------------------------------------------------------
comparison = E.model_comparison(fold_metrics)
comparison.to_csv(c.table("model_comparison"), index=False)
paired = E.paired_differences(fold_metrics, reference=BEST)
paired.to_csv(c.table("phase5_paired_vs_best"), index=False)

print("T5.1 model comparison (mean +/- std over seed x fold runs)")
for _, r in comparison.iterrows():
    print(f"  {r.scheme:<26} {r.model:<9} n={r.n_runs:<3} RMSE {pm(r.RMSE_mean, r.RMSE_std)} "
          f"(seed sd {r.RMSE_seed_std:.3f})  MAE {r.MAE_mean:.3f}  R2 {pm(r.R2_mean, r.R2_std)}")
print(f"\nPaired RMSE difference vs {BEST} (same seed x fold test sets)")
for _, r in paired[paired.model.isin(E.TREE_MODELS)].iterrows():
    print(f"  {r.scheme:<26} {r.model:<5} dRMSE {r.dRMSE_mean:+.4f} +/- {r.dRMSE_std:.4f}  "
          f"better on {r.runs_model_better}/{r.n_runs} runs")

# Strip stress test beside the headline (D-033).
stress_rows = []
for head, strip in ((E.PRIMARY_SCHEME, STRIP_4KM), (E.SPATIO_TEMPORAL_SCHEME, STRIP_ST)):
    for m in M.MODEL_NAMES:
        h = comparison[(comparison.scheme == head) & (comparison.model == m)].iloc[0]
        s = comparison[(comparison.scheme == strip) & (comparison.model == m)].iloc[0]
        stress_rows.append({
            "scheme": head, "model": m,
            "RMSE_shuffled_mean": h.RMSE_mean, "RMSE_shuffled_std": h.RMSE_std,
            "R2_shuffled_mean": h.R2_mean, "R2_shuffled_std": h.R2_std,
            "RMSE_strips_mean": s.RMSE_mean, "RMSE_strips_std": s.RMSE_std,
            "R2_strips_mean": s.R2_mean, "R2_strips_std": s.R2_std,
            "dRMSE_strips_minus_shuffled": s.RMSE_mean - h.RMSE_mean,
        })
stress = pd.DataFrame(stress_rows)
stress.to_csv(c.table("phase5_strip_stress_test"), index=False)
print("\nStrip stress test (4 km, unshuffled full-grid layout) beside the 5-seed headline")
for _, r in stress.iterrows():
    print(f"  {r.scheme:<22} {r.model:<9} shuffled {pm(r.RMSE_shuffled_mean, r.RMSE_shuffled_std)} "
          f"| strips {pm(r.RMSE_strips_mean, r.RMSE_strips_std)} "
          f"| d {r.dRMSE_strips_minus_shuffled:+.3f}")

# ---------------------------------------------------------------------------
# T5.2 Random vs spatial gap (RQ3 curve)
# ---------------------------------------------------------------------------
headline = comparison[~comparison.scheme.astype(str).str.endswith(E.STRIP_SUFFIX)]
gap = E.cv_gap(headline)
gap.to_csv(c.table("phase5_cv_gap"), index=False)

x_schemes = ["random"] + [f"spatial_{s // 1000}km" for s in c.BLOCK_SIZES_M]
x_labels = ["Random\n(cell-grouped)"] + [f"Spatial\n{s // 1000} km" for s in c.BLOCK_SIZES_M]
plot_models = ["linear", "rf", "xgb", "lgbm"]
offsets = dict(zip(plot_models, np.linspace(-0.15, 0.15, len(plot_models))))
i4 = x_schemes.index(E.PRIMARY_SCHEME)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, metric, unit in ((axes[0], "RMSE", "RMSE of LST anomaly (°C)"),
                         (axes[1], "R2", "R² (share of variance explained)")):
    cmp_idx = comparison.set_index(["scheme", "model"])
    for m in plot_models:
        part = cmp_idx.loc[[(s, m) for s in x_schemes]]
        xs = np.arange(len(x_schemes)) + offsets[m]
        ax.errorbar(xs, part[f"{metric}_mean"], yerr=part[f"{metric}_std"],
                    color=SERIES[m], marker=MARKERS[m], markersize=6, linewidth=2,
                    elinewidth=1, capsize=0, markeredgecolor="white",
                    markeredgewidth=1, label=LABELS[m])
        # Strip stress test: hollow marker beside the 4 km point, no line.
        sv = cmp_idx.loc[(STRIP_4KM, m), f"{metric}_mean"]
        ax.plot(i4 + 0.42 + offsets[m] * 0.6, sv, linestyle="none", marker=MARKERS[m],
                markersize=6, markerfacecolor="white", markeredgecolor=SERIES[m],
                markeredgewidth=1.4)
        # The trees tie at every spatial scheme, so they share one end label.
        if m in ("linear", "lgbm"):
            text = LABELS[m] if m == "linear" else "RF, XGBoost,\nLightGBM (tied)"
            ax.annotate(text, (xs[-1], part[f"{metric}_mean"].iloc[-1]),
                        xytext=(10, 0), textcoords="offset points", va="center",
                        fontsize=8, color=INK2)
    lo, hi = ax.get_ylim()
    ax.annotate("4 km strips\n(stress test)", (i4 + 0.42, hi), xytext=(0, -2),
                textcoords="offset points", ha="center", va="top", fontsize=7, color=MUTED)
    ax.axvline(0.5, color="#c3c2b7", linewidth=0.6)
    ax.set_xticks(range(len(x_schemes)), x_labels)
    ax.set_xlim(-0.4, len(x_schemes) - 0.3 + 0.9)
    ax.set_xlabel("Cross-validation scheme", fontsize=9, color=INK2)
    ax.set_ylabel(unit, fontsize=9, color=INK2)
    style_axes(ax)
axes[0].set_title("Error rises with block size once test areas are unseen",
                  fontsize=11, color=INK, loc="left")
axes[1].set_title("…and explained variance falls", fontsize=11, color=INK, loc="left")
axes[1].legend(frameon=False, fontsize=8, loc="lower left")
base_rmse = comparison.loc[(comparison.model == "baseline")
                           & ~comparison.scheme.astype(str).str.endswith(E.STRIP_SUFFIX), "RMSE_mean"]
fig.text(0.01, -0.05, f"Mean ± std over {len(c.CV_SEEDS)} seeds × {c.N_SPLITS} folds; "
         "water-majority cells excluded (D-034). Hollow markers: unshuffled 4 km layout, "
         f"every fold an east-west strip (D-033).\nBaseline (RMSE {base_rmse.min():.2f}-"
         f"{base_rmse.max():.2f} °C, R² ≈ 0) omitted for scale. Linear uses 12 features, "
         "trees 13 (D-025).", fontsize=8, color=MUTED)
fig.tight_layout()
P.save_figure(fig, c.figure(5, "cv_gap"))
plt.close(fig)
print("\nT5.2 RQ3 curve: RMSE by scheme, and gap vs random")
curve = headline.pivot(index="model", columns="scheme", values="RMSE_mean")
print(curve[[s for s in E.SCHEME_ORDER if s in curve.columns]].round(3).to_string())
for _, r in gap[gap.scheme == E.PRIMARY_SCHEME].iterrows():
    print(f"  {r.model:<9} random -> 4 km dRMSE {r.dRMSE:+.3f} ({r.dRMSE_pct:+.1f}%)  dR2 {r.dR2:+.3f}")

# ---------------------------------------------------------------------------
# T5.3 Temporal holdout (2025): unseen space+time vs seen place
# ---------------------------------------------------------------------------
y = df[c.TARGET_ANOM].to_numpy()
strip_layout = pp.load_clean()[[cv.block_column(c.PRIMARY_BLOCK_SIZE_M)]]
st_layouts = {(E.SPATIO_TEMPORAL_SCHEME, s): cv.spatio_temporal_splits(df, c.PRIMARY_BLOCK_SIZE_M, seed=s)
              for s in c.CV_SEEDS}
st_layouts[(STRIP_ST, E.STRIP_SEED)] = cv.strip_splits(df, strip_layout, temporal=True)
del strip_layout

hold_rows = []
for m in M.MODEL_NAMES:
    seen = E.seen_place_predictions(df, m)
    for (scheme, seed), splits in st_layouts.items():
        st = fold_metrics[(fold_metrics.scheme == scheme) & (fold_metrics.seed == seed)
                          & (fold_metrics.model == m)].set_index("fold")
        for fold, (_, te) in enumerate(splits):
            sp = E.error_metrics(y[te], seen[te])
            assert st.loc[fold, "n"] == sp["n"], "both evaluations must score the same rows"
            hold_rows.append({"scheme": scheme, "model": m, "seed": seed, "fold": fold,
                              "n": sp["n"],
                              "RMSE_unseen": st.loc[fold, "RMSE"], "RMSE_seen_place": sp["RMSE"],
                              "R2_unseen": st.loc[fold, "R2"], "R2_seen_place": sp["R2"],
                              "MAE_unseen": st.loc[fold, "MAE"], "MAE_seen_place": sp["MAE"]})
    del seen
    gc.collect()
hold_f = pd.DataFrame(hold_rows)
hold_f["memorisation_dRMSE"] = hold_f.RMSE_unseen - hold_f.RMSE_seen_place
hold_f.to_csv(c.table("phase5_temporal_holdout_folds"), index=False)
hold = (hold_f.groupby(["scheme", "model"])
        [["RMSE_unseen", "RMSE_seen_place", "R2_unseen", "R2_seen_place",
          "MAE_unseen", "MAE_seen_place", "memorisation_dRMSE"]]
        .agg(["mean", "std"]))
hold.columns = [f"{a}_{b}" for a, b in hold.columns]
hold["n_runs"] = hold_f.groupby(["scheme", "model"]).size()
hold = E._order(hold.reset_index())
hold.to_csv(c.table("phase5_temporal_holdout"), index=False)
print("\nT5.3 2025 holdout: unseen space+time (headline) vs seen place (contrast)")
for _, r in hold.iterrows():
    print(f"  {r.scheme:<26} {r.model:<9} unseen RMSE {pm(r.RMSE_unseen_mean, r.RMSE_unseen_std)} "
          f"R2 {r.R2_unseen_mean:.3f} | seen-place RMSE {pm(r.RMSE_seen_place_mean, r.RMSE_seen_place_std)} "
          f"R2 {r.R2_seen_place_mean:.3f} | memorisation "
          f"{pm(r.memorisation_dRMSE_mean, r.memorisation_dRMSE_std)}")

# ---------------------------------------------------------------------------
# T5.4 Predicted vs actual, best model (seed-42 layout shown)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
panels = ((E.PRIMARY_SCHEME, "Spatial CV, 4 km blocks (2023-25)"),
          (E.SPATIO_TEMPORAL_SCHEME, "Spatio-temporal, 4 km blocks (test year 2025)"))
lim = (-15, 15)
for ax, (scheme, title) in zip(axes, panels):
    oof = E.load_oof(scheme, BEST, seed=c.SEED, columns=["y_true", "y_pred"])
    met = E.error_metrics(oof.y_true.to_numpy(), oof.y_pred.to_numpy())
    hb = ax.hexbin(oof.y_true, oof.y_pred, gridsize=90, extent=lim + lim,
                   bins="log", cmap="Blues", mincnt=1, linewidths=0)
    ax.plot(lim, lim, color=INK2, linewidth=0.8)
    ax.text(lim[1] - 0.5, lim[1] - 1.2, "1:1", color=INK2, fontsize=8, ha="right")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Observed LST anomaly (°C)", fontsize=9, color=INK2)
    ax.set_ylabel("Predicted LST anomaly (°C)", fontsize=9, color=INK2)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.text(0.03, 0.97, f"{LABELS[BEST]}, out-of-fold\nlayout seed {c.SEED}\n"
            f"n = {met['n']:,}\nRMSE {met['RMSE']:.2f} °C\nR² {met['R2']:.3f}",
            transform=ax.transAxes, va="top", fontsize=8, color=INK)
    style_axes(ax)
    ax.grid(color=GRID, linewidth=0.6)
    cb = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Cells per hexagon (log scale)", fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.6)
    del oof
fig.tight_layout()
P.save_figure(fig, c.figure(5, "pred_vs_actual"))
plt.close(fig)

# ---------------------------------------------------------------------------
# T5.6 / T5.7 Subgroup errors (hottest decile; urban vs rest)
# ---------------------------------------------------------------------------
hot = E.hot_decile_flags(df)
df["hot_aoi"] = hot["hot_aoi"]
df["hot_urban"] = hot["hot_urban"]
df["urban_group"] = np.where(df["is_urban"], "urban", "rest")
sub_schemes = ("random", E.PRIMARY_SCHEME, E.SPATIO_TEMPORAL_SCHEME)
SUB_COLS = ["row_id", "year", "fold", "seed", "model", "scheme"]
urban_parts, hot_parts, overlap_parts = [], [], []
for scheme in sub_schemes:
    oof = E.attach_cells(E.load_oof(scheme, columns=SUB_COLS), df,
                         ["urban_group", "hot_aoi", "hot_urban", "is_urban"])
    urban_parts.append(E.subgroup_fold_metrics(oof, "urban_group"))
    # The two hot deciles overlap (an urban hot cell can be AOI-hot too), so each
    # is scored against its own complement rather than as one categorical column.
    for flag in ("hot_aoi", "hot_urban"):
        sel = oof if flag == "hot_aoi" else oof[oof.is_urban]
        fm_ = E.subgroup_fold_metrics(sel, flag).rename(columns={flag: "in_decile"})
        fm_.insert(4, "decile", "AOI-wide" if flag == "hot_aoi" else "within urban")
        hot_parts.append(fm_)
    overlap_parts += [E.top_decile_overlap(oof), E.top_decile_overlap(oof, urban_only=True)]
    del oof
    gc.collect()

urban = E.summarise_subgroups(pd.concat(urban_parts, ignore_index=True), ["urban_group"])
urban.to_csv(c.table("phase5_urban_vs_rest"), index=False)
hot_tab = E.summarise_subgroups(pd.concat(hot_parts, ignore_index=True), ["decile", "in_decile"])
hot_tab.to_csv(c.table("phase5_hot_decile"), index=False)

print("\nT5.6 hottest decile vs rest (resid = obs - pred; + means model too cool)")
for _, r in hot_tab[hot_tab.scheme == E.PRIMARY_SCHEME].iterrows():
    print(f"  {r.model:<9} {r.decile:<13} in={r.in_decile!s:<5} RMSE {r.RMSE_mean:.3f}  "
          f"MAE {r.MAE_mean:.3f}  mean resid {r.mean_resid_mean:+.3f}")
overlap = pd.concat(overlap_parts, ignore_index=True)
overlap.to_csv(c.table("phase5_hot_decile_overlap"), index=False)
ov = overlap.groupby(["scheme", "decile", "model"])["overlap"].agg(["mean", "std"])
print("\nT5.6 hottest-decile detection: share of observed top 10% also predicted top 10% "
      "(mean +/- std over seed x year)")
print(ov.loc[[E.PRIMARY_SCHEME, E.SPATIO_TEMPORAL_SCHEME]].round(3).to_string())

print("\nT5.7 urban vs rest")
for _, r in urban[urban.scheme != "random"].iterrows():
    print(f"  {r.scheme:<22} {r.model:<9} {r.urban_group:<6} RMSE {pm(r.RMSE_mean, r.RMSE_std)}  "
          f"MAE {r.MAE_mean:.3f}  R2 {pm(r.R2_mean, r.R2_std)}  mean resid {r.mean_resid_mean:+.3f}")

# ---------------------------------------------------------------------------
# T5.5 Residual map, residual range and Moran's I
# ---------------------------------------------------------------------------
oof = E.load_oof(E.PRIMARY_SCHEME, BEST, columns=["row_id", "cell_id", "seed"])
resid_cell = (
    E.attach_cells(oof, df, ["x_utm", "y_utm"])
    .groupby("cell_id")
    .agg(x_utm=("x_utm", "first"), y_utm=("y_utm", "first"), resid=("resid", "mean"))
)
grid, extent = P.cells_to_grid(resid_cell.x_utm.to_numpy(), resid_cell.y_utm.to_numpy(),
                               resid_cell.resid.to_numpy())
vmax = float(np.nanpercentile(np.abs(grid), 98))
fig, ax = plt.subplots(figsize=(9, 7.4))
ax.set_facecolor("#d9d9d9")  # excluded water cells show as flat grey
P.map_panel(ax, grid, extent, cmap=c.DIVERGING_CMAP, vmin=-vmax, vmax=vmax,
            cbar_label="Mean residual, observed − predicted (°C)",
            title=f"{LABELS[BEST]} out-of-fold residuals, spatial CV at 4 km "
                  f"(mean of 2023-25 × {len(c.CV_SEEDS)} seeds)")
bs = c.PRIMARY_BLOCK_SIZE_M / 1000
for xk in np.arange(np.ceil(extent[0] / bs) * bs, extent[1], bs):
    ax.axvline(xk, color="0.35", linewidth=0.4, alpha=0.6)
for yk in np.arange(np.ceil(extent[2] / bs) * bs, extent[3], bs):
    ax.axhline(yk, color="0.35", linewidth=0.4, alpha=0.6)
ax.text(0.0, -0.11, "Red: observed hotter than predicted (model too cool). Grey: water-majority "
        "cells, excluded (D-034). Lines: 4 km blocks.", transform=ax.transAxes,
        fontsize=8, color=MUTED)
P.save_figure(fig, c.figure(5, "residual_map"))
plt.close(fig)

# Residual range on the tuned model's residuals (seed-42 layout), against D-028.
r42 = oof[oof.seed == c.SEED]
resid42 = np.full(len(df), np.nan)
resid42[r42.row_id.to_numpy()] = r42.resid.to_numpy()
assert not np.isnan(resid42).any()
rr = E.residual_ranges(df, resid42)
rr.insert(0, "source", f"{BEST} tuned, {E.PRIMARY_SCHEME}, seed {c.SEED}")
rr.to_csv(c.table("phase5_residual_range"), index=False)
print(f"\nResidual range of the tuned {BEST} (4 km, seed {c.SEED}):\n{rr.round(3).to_string(index=False)}"
      f"\n  median {rr.range_95_m.median():,.0f} m -> rung {E.block_rung(rr.range_95_m.median())} m")
del oof, r42
gc.collect()

# Fold profile under the strip layout: what each latitude band has in common.
oof = E.load_oof(STRIP_4KM, BEST, columns=["row_id", "fold"])
prof_df = E.attach_cells(oof, df, ["is_urban", "frac_built", "frac_tree", "NDVI",
                                   "NDBI", "elevation", "slope", "y_utm"])
profile = prof_df.groupby("fold").agg(
    n=("y_true", "size"), target_std=("y_true", "std"),
    mean_northing_km=("y_utm", lambda s: s.mean() / 1000),
    urban_share=("is_urban", "mean"), frac_built=("frac_built", "mean"),
    frac_tree=("frac_tree", "mean"), NDVI=("NDVI", "mean"), NDBI=("NDBI", "mean"),
    elevation=("elevation", "mean"), slope=("slope", "mean"),
    mean_resid=("resid", "mean"),
)
fmb = fold_metrics[(fold_metrics.scheme == STRIP_4KM) & (fold_metrics.model == BEST)]
profile = profile.join(fmb.set_index("fold")[["RMSE", "R2"]])
profile.to_csv(c.table("phase5_fold_profile_strips"))
print(f"\nFold profile ({STRIP_4KM}, {BEST})\n{profile.round(3).to_string()}")
del oof, prof_df
gc.collect()

# Moran's I on residuals, per year and seed, against the raw target as reference.
# Land cells are identical across years, so one weights object serves every year.
yr0 = df[df.year == c.YEARS[0]]
w = E.queen_weights(yr0.x_utm.to_numpy(), yr0.y_utm.to_numpy())
if w.islands:
    raise RuntimeError(f"{len(w.islands)} land cells have no land neighbour; drop them first")
cell_order = yr0.cell_id.to_numpy()
moran_rows = []
for year in c.YEARS:
    yr = df[df.year == year].set_index("cell_id").loc[cell_order]
    moran_rows.append({"scheme": "none", "model": "raw_target", "seed": np.nan, "year": year,
                       **E.morans_i(yr[c.TARGET_ANOM].to_numpy(), w)})
moran_schemes = ("random", E.PRIMARY_SCHEME, E.SPATIO_TEMPORAL_SCHEME, STRIP_4KM, STRIP_ST)
for scheme in moran_schemes:
    oof = E.load_oof(scheme, columns=["cell_id", "year", "seed", "model"])
    for (m, seed, year), part in oof.groupby(["model", "seed", "year"]):
        vals = part.set_index("cell_id")["resid"].reindex(cell_order).to_numpy()
        assert not np.isnan(vals).any(), "every cell needs exactly one OOF residual"
        moran_rows.append({"scheme": scheme, "model": m, "seed": int(seed), "year": int(year),
                           **E.morans_i(vals, w)})
    del oof
    gc.collect()
moran = pd.DataFrame(moran_rows)
moran.to_csv(c.table("phase5_morans_i"), index=False)
print("\nT5.5 Moran's I (queen, 100 m), mean over seeds")
print(moran.pivot_table(index=["scheme", "model"], columns="year", values="morans_I",
                        aggfunc="mean").round(3).to_string())

print(f"\nDONE in {(time.perf_counter() - t_start) / 60:.1f} min")
