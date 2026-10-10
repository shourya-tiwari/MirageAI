"""One-command driver for Phase 3 (T3.1-T3.5): variogram, block layout, fold balance.

    python scripts/pipeline/run_phase3.py

Reads data/processed/uhi_pune_clean.parquet (scripts/pipeline/run_phase2.py). Writes
outputs/tables/phase3_fold_balance.csv and the 03_* figures.

This is the Phase 3 record as it was decided: the full grid, water included, with
unshuffled block folds and the 7.65 km variogram range that set 8 km as the first
primary block size (D-026). Phase 4 supersedes it with land cells only (D-034),
shuffled multi-seed folds (D-033) and 4 km blocks (D-028); the current fold
balance is phase4_fold_balance.csv. The splitters in cv.py refuse water rows, so
the historical layout is rebuilt here with GroupKFold directly.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import GroupKFold

from mirageai import config as c
from mirageai import cv
from mirageai import plotting as P
from mirageai import preprocess as pp

YEAR_COL = dict(zip(c.YEARS, ("#b5503c", "#3b6ea5", "#6b8e3a")))
FOLD_COL = ListedColormap(["#4a6a8a", "#b5503c", "#6b8e3a", "#8a6aa5", "#c98c3c"])
LAYOUT_BLOCK_M = 8000  # the D-026 primary block size this record documents

df = pp.load_clean()

# ---------------------------------------------------------------------------
# T3.1 Empirical variogram of the target, per year
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.7))
for y in c.YEARS:
    v = cv.empirical_variogram(df[df.year == y])
    r = cv.practical_range(v)
    ax.plot(v.lag_m / 1000, v.semivariance, color=YEAR_COL[y], linewidth=1.6,
            label=f"{y}  (sill {r['sill']:.1f}, range {r['range_m'] / 1000:.2f} km)")
    at = v.loc[v.lag_m == r["range_m"]].iloc[0]
    ax.plot(at.lag_m / 1000, at.semivariance, "o", color=YEAR_COL[y], markersize=6)
    print(f"variogram {y}: sill {r['sill']:.2f}, practical range {r['range_m']:.0f} m")
for b in c.BLOCK_SIZES_M:
    if b == LAYOUT_BLOCK_M:
        ax.axvline(b / 1000, color="#8a6a50", linewidth=2)
    elif b < LAYOUT_BLOCK_M // 2:
        ax.axvline(b / 1000, color="#8a6a50", linewidth=0.8, linestyle=":")
    else:
        continue
    ax.text(b / 1000 - 0.05, 16.9, f"{b // 1000} km block", rotation=90, ha="right", va="top",
            fontsize=7.5, color="#8a6a50")
ax.annotate("markers = 95% of that year's sill\n(practical range)", xy=(7.65, 10.9),
            xytext=(3.2, 4.4), fontsize=8.5, color="0.3",
            arrowprops=dict(arrowstyle="->", color="0.45", linewidth=0.8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 17)
ax.set_xlabel("Lag distance (km)")
ax.set_ylabel("Semivariance of LST anomaly (°C²)")
ax.set_title("Empirical variogram of LST anomaly, by year", fontsize=12)
ax.legend(frameon=False, loc="lower right", fontsize=9)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
P.save_figure(fig, c.figure(3, "variogram_lst_anom"))
plt.close(fig)

# ---------------------------------------------------------------------------
# T3.3-T3.4 Fold balance under every scheme (unshuffled blocks, full grid)
# ---------------------------------------------------------------------------
is_train_year = df.year.isin(c.TRAIN_YEARS).to_numpy()
is_test_year = (df.year == c.TEST_YEAR).to_numpy()
reports = [cv.fold_report(
    df, list(GroupKFold(c.N_SPLITS, shuffle=True, random_state=c.SEED).split(df, groups=df.cell_id)),
    "random")]
layouts = {}
for b in c.BLOCK_SIZES_M:
    spatial = list(GroupKFold(c.N_SPLITS).split(df, groups=df[cv.block_column(b)]))
    layouts[b] = spatial
    reports.append(cv.fold_report(df, spatial, "spatial", b))
    st = []
    for tr, te in spatial:
        tr_m, te_m = np.zeros(len(df), bool), np.zeros(len(df), bool)
        tr_m[tr], te_m[te] = True, True
        st.append((np.flatnonzero(tr_m & is_train_year), np.flatnonzero(te_m & is_test_year)))
    reports.append(cv.fold_report(df, st, "spatio_temporal", b))
fb = pd.concat(reports, ignore_index=True)
fb["urban_share"] = (fb.urban_share * 100).round(1)
fb["urban_dev_pts"] = fb.urban_dev_pts.round(1)
fb["flag"] = fb.urban_dev_pts.abs() > cv.URBAN_BALANCE_TOLERANCE_PTS
fb.to_csv(c.table("phase3_fold_balance"), index=False)

# ---------------------------------------------------------------------------
# T3.5 Block layout map and the urban cells it has to balance
# ---------------------------------------------------------------------------
one = df[df.year == c.TEST_YEAR].copy()
fold = np.empty(len(df), dtype="int64")
for k, (_, te) in enumerate(layouts[LAYOUT_BLOCK_M]):
    fold[te] = k
one["fold"] = fold[one.index]
col = cv.block_column(LAYOUT_BLOCK_M)
share = one.groupby(col).is_urban.mean() * 100

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
grid, ext = P.cells_to_grid(one.x_utm.to_numpy(), one.y_utm.to_numpy(), one.fold.to_numpy())
im = axes[0].imshow(grid, extent=ext, cmap=FOLD_COL, vmin=-0.5, vmax=c.N_SPLITS - 0.5,
                    interpolation="nearest")
edges_x = np.arange(np.floor(one.x_utm.min() / LAYOUT_BLOCK_M), np.ceil(one.x_utm.max() / LAYOUT_BLOCK_M) + 1)
edges_y = np.arange(np.floor(one.y_utm.min() / LAYOUT_BLOCK_M), np.ceil(one.y_utm.max() / LAYOUT_BLOCK_M) + 1)
for e in edges_x[1:-1]:
    axes[0].axvline(e * LAYOUT_BLOCK_M / 1000, color="white", linewidth=0.6)
for e in edges_y[1:-1]:
    axes[0].axhline(e * LAYOUT_BLOCK_M / 1000, color="white", linewidth=0.6)
axes[0].set_title(f"Spatial block layout, {LAYOUT_BLOCK_M // 1000} km blocks "
                  f"({one[col].nunique()} blocks, {c.N_SPLITS} folds)", fontsize=11)
cb = fig.colorbar(im, ax=axes[0], ticks=range(c.N_SPLITS), fraction=0.046, pad=0.03)
cb.set_label("CV fold (held out)")
grid, ext = P.cells_to_grid(one.x_utm.to_numpy(), one.y_utm.to_numpy(), one.is_urban.to_numpy(float))
im = axes[1].imshow(grid, extent=ext, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
axes[1].set_title("Urban cells: centrally concentrated with ribbon sprawl, so block\n"
                  f"urban share ranges {share.min():.1f}–{share.max():.1f}%", fontsize=11)
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03).set_label(
    f"urban (frac_built ≥ {c.URBAN_BUILT_THRESHOLD})")
for ax in axes:
    ax.set_xlabel("UTM 43N easting (km)")
    ax.set_ylabel("UTM 43N northing (km)")
    ax.set_aspect("equal")
fig.tight_layout()
P.save_figure(fig, c.figure(3, "block_layout"))
plt.close(fig)
print(f"wrote phase3_fold_balance ({len(fb)} rows, {int(fb.flag.sum())} flagged) and 03_* figures")
