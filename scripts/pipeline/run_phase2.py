"""One-command driver for Phase 2 (T2.1-T2.7), plus the Phase 1 checks that need
only data/raw (T1.9-T1.10). Earth Engine is not touched.

    python scripts/pipeline/run_phase2.py

Reads data/raw/uhi_pune_{year}.csv. Writes data/processed/uhi_pune_clean.parquet,
outputs/tables/phase1_cross_year_checks.csv and phase2_lst_anom_correlations.csv,
and the 01_* and 02_* figures. Phase 1 and 2 describe the full grid, water cells
included; the water exclusion (D-034) starts at Phase 4.
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

YEAR_COL = dict(zip(c.YEARS, ("#b5503c", "#3b6ea5", "#6b8e3a")))
NOTE = f"{c.SCALE} m grid · {c.CRS}"
FEATS = list(c.FEATURES)

c.ensure_dirs()
raw = pp.load_all()

# ---------------------------------------------------------------------------
# Phase 1 checks from data/raw: cool island and driver signs per year (T1.9)
# ---------------------------------------------------------------------------
p1 = pp.core_ring_contrast(raw).drop(columns=["n_core", "n_ring"])
for col in ("NDVI", "NDBI", "frac_built", "frac_tree", "frac_grass", "MNDWI"):
    p1[f"corr_{col}"] = raw.groupby("year").apply(lambda g: g[c.TARGET].corr(g[col])).to_numpy()
p1.round(3).to_csv(c.table("phase1_cross_year_checks"), index=False)

# ---------------------------------------------------------------------------
# Phase 1 maps: LST per year on a shared range, NDVI 2024 (T1.10)
# ---------------------------------------------------------------------------
lo, hi = np.percentile(raw[c.TARGET], [1, 99])
for y in c.YEARS:
    g = raw[raw.year == y]
    grid, ext = P.cells_to_grid(g.x_utm.to_numpy(), g.y_utm.to_numpy(), g[c.TARGET].to_numpy())
    fig, ax = plt.subplots(figsize=(7, 5.4))
    P.map_panel(ax, grid, ext, cmap="inferno", cbar_label="Land surface temperature (°C)",
                title=f"Pune daytime LST, pre-monsoon {y} (Mar–May median)", vmin=lo, vmax=hi)
    fig.text(0.1, 0.01, f"{NOTE} · shared colour range across {c.YEARS[0]}–{c.YEARS[-1]} "
             f"[{lo:.1f}, {hi:.1f}] °C", fontsize=7.5, color="0.4")
    P.save_figure(fig, c.figure(1, f"lst_map_{y}"))
    plt.close(fig)

g = raw[raw.year == 2024]
lo, hi = np.percentile(g.NDVI, [1, 99])
grid, ext = P.cells_to_grid(g.x_utm.to_numpy(), g.y_utm.to_numpy(), g.NDVI.to_numpy())
fig, ax = plt.subplots(figsize=(7, 5.4))
P.map_panel(ax, grid, ext, cmap="YlGn", cbar_label="NDVI (unitless)",
            title="Pune NDVI, pre-monsoon 2024 (Mar–May median)", vmin=lo, vmax=hi)
fig.text(0.1, 0.01, f"{NOTE} · colour range = 1st–99th percentile [{lo:.2f}, {hi:.2f}]",
         fontsize=7.5, color="0.4")
P.save_figure(fig, c.figure(1, "ndvi_map_2024"))
plt.close(fig)

# ---------------------------------------------------------------------------
# T2.1-T2.7: the clean modelling table
# ---------------------------------------------------------------------------
df = pp.build_clean_table()
miss = pp.missing_report(df)
print(f"clean table {df.shape[0]:,} x {df.shape[1]}, nulls {int(df.isna().sum().sum())}")
print(miss.to_string())

# ---------------------------------------------------------------------------
# T2.3 Target distributions: absolute LST vs the within-year anomaly (D-021)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
for ax, col, xlab, title in ((axes[0], c.TARGET, "Absolute LST (°C)", "Absolute LST: years are offset"),
                             (axes[1], c.TARGET_ANOM, "Within-year LST anomaly (°C)",
                              "Anomaly: offset removed (D-021)")):
    for y in c.YEARS:
        v = df.loc[df.year == y, col]
        ax.hist(v, bins=150, histtype="step", linewidth=1.6, color=YEAR_COL[y],
                label=f"{y}  (median {v.median():+.2f})")
    ax.set_xlabel(xlab)
    ax.set_ylabel("cells")
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
P.save_figure(fig, c.figure(2, "target_distributions"))
plt.close(fig)

# ---------------------------------------------------------------------------
# T2.4 Feature distributions
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(4, 4, figsize=(13, 9.8))
for ax, f in zip(axes.flat, FEATS):
    ax.hist(df[f], bins=80, color="#4a6a8a")
    ax.set_title(f, fontsize=10)
    ax.set_ylabel("cells", fontsize=7)
    ax.tick_params(labelsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
for ax in axes.flat[len(FEATS):]:
    ax.set_visible(False)
fig.suptitle(f"Feature distributions, all years pooled (n = {len(df):,})", fontsize=12)
fig.tight_layout()
P.save_figure(fig, c.figure(2, "feature_distributions"))
plt.close(fig)

# ---------------------------------------------------------------------------
# T2.5 Correlation, VIF (D-023) and correlation with the target
# ---------------------------------------------------------------------------
cols = FEATS + [c.TARGET_ANOM]
corr = df[cols].corr()
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=9)
ax.set_yticks(range(len(cols)), cols, fontsize=9)
for i in range(len(cols)):
    for j in range(len(cols)):
        v = corr.iat[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(v) > 0.6 else "black")
ax.set_title("Feature correlation, all years pooled", fontsize=12, pad=10)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Pearson correlation")
fig.tight_layout()
P.save_figure(fig, c.figure(2, "correlation_heatmap"))
plt.close(fig)

for dropped in ("frac_built", "frac_bare"):
    vif = pp.vif_table(df, [f for f in FEATS if f != dropped])
    print(f"VIF without {dropped}: median {vif.VIF.median():.1f}")

urban = df[df.is_urban]
out = pd.DataFrame({"all_cells": df[FEATS].corrwith(df[c.TARGET_ANOM]),
                    "urban_only": urban[FEATS].corrwith(urban[c.TARGET_ANOM])})
out["difference"] = out.urban_only - out.all_cells
out = out.reindex(out.all_cells.abs().sort_values(ascending=False).index)
out.round(4).to_csv(c.table("phase2_lst_anom_correlations"))
print("wrote phase1_cross_year_checks, phase2_lst_anom_correlations, 01_* and 02_* figures")
