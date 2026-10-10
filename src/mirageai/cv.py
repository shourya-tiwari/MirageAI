"""Phase 3: spatial block assignment and the random / spatial / spatio-temporal splitters."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from mirageai import config

# Fold urban-share deviation beyond this is flagged in the fold report. It is a
# reporting threshold, not a constraint: see D-026 for why it cannot be enforced
# at the primary block size.
URBAN_BALANCE_TOLERANCE_PTS: float = 10.0


# ---------------------------------------------------------------------------
# T3.1 Empirical variogram
# ---------------------------------------------------------------------------
def empirical_variogram(
    df: pd.DataFrame,
    value_col: str = config.TARGET_ANOM,
    n_sample: int = 20_000,
    n_pairs: int = 5_000_000,
    max_dist_m: float = 10_000.0,
    n_bins: int = 100,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Empirical semivariogram by random pair sampling.

    Exhaustive pairs over 20k cells would be 2e8 comparisons; a large random
    sample of pairs estimates the same binned means far more cheaply.
    """
    rng = np.random.default_rng(seed)
    sub = df.sample(n=min(n_sample, len(df)), random_state=seed)
    xy = sub[["x_utm", "y_utm"]].to_numpy()
    z = sub[value_col].to_numpy()

    i = rng.integers(0, len(sub), n_pairs)
    j = rng.integers(0, len(sub), n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]

    dist = np.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1])
    sq_diff = (z[i] - z[j]) ** 2

    within = dist <= max_dist_m
    dist, sq_diff = dist[within], sq_diff[within]

    edges = np.linspace(0.0, max_dist_m, n_bins + 1)
    idx = np.clip(np.digitize(dist, edges) - 1, 0, n_bins - 1)

    counts = np.bincount(idx, minlength=n_bins)
    sums = np.bincount(idx, weights=sq_diff, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = 0.5 * sums / counts

    return pd.DataFrame(
        {
            "lag_m": 0.5 * (edges[:-1] + edges[1:]),
            "semivariance": gamma,
            "n_pairs": counts,
        }
    ).dropna()


def practical_range(vario: pd.DataFrame, sill_frac: float = 0.95) -> dict[str, float]:
    """Lag at which the semivariogram first reaches sill_frac of its sill.

    The sill is taken as the plateau over the outer quarter of the lag range,
    which avoids assuming a parametric model.
    """
    tail = vario.iloc[int(len(vario) * 0.75) :]
    sill = float(tail["semivariance"].mean())
    nugget = float(vario["semivariance"].iloc[0])
    target = sill * sill_frac

    reached = vario.loc[vario["semivariance"] >= target, "lag_m"]
    rng_m = float(reached.iloc[0]) if len(reached) else float("nan")
    return {"sill": sill, "nugget": nugget, "target": target, "range_m": rng_m}


# ---------------------------------------------------------------------------
# T3.2 Blocks
# ---------------------------------------------------------------------------
def block_column(block_size_m: int) -> str:
    """Name of the block-ID column for a given block size."""
    return f"block_{block_size_m}m"


# ---------------------------------------------------------------------------
# T3.3 Splitters
# ---------------------------------------------------------------------------
def require_land_only(df: pd.DataFrame) -> None:
    """Refuse any table that still holds water-majority rows (D-034)."""
    n_water = int((df["frac_water"] >= config.WATER_EXCLUDE_THRESHOLD).sum())
    if n_water:
        raise ValueError(
            f"{n_water} water-majority rows reached a splitter; "
            "use preprocess.load_model_table() (D-034)"
        )


def random_splits(
    df: pd.DataFrame, n_splits: int = config.N_SPLITS, seed: int = config.SEED
) -> list:
    """Random CV, grouped on cell_id so a cell's three years stay together.

    This is the inflated baseline that D-003 reports for comparison. Grouping on
    cell_id is deliberate: without it the 2023, 2024 and 2025 rows of one cell
    land in different folds, and the measured inflation would be dominated by
    that near-duplicate leakage rather than by spatial autocorrelation. Holding
    the repeated measurements together isolates the spatial effect, which is the
    one RQ3 asks about.
    """
    require_land_only(df)
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(df, groups=df["cell_id"]))


def spatial_splits(
    df: pd.DataFrame,
    block_size_m: int = config.PRIMARY_BLOCK_SIZE_M,
    n_splits: int = config.N_SPLITS,
    seed: int | None = config.SEED,
) -> list:
    """Spatial block CV: whole blocks are held out, so test areas are unseen.

    Blocks are dealt to folds at random under `seed`. `seed=None` keeps
    GroupKFold's unshuffled tie-breaking, which deals equal-sized blocks in id
    order; at 4 km that makes every fold an east-west strip, and it is used only
    as the latitude-extrapolation stress test (D-033).
    """
    require_land_only(df)
    if seed is None:
        splitter = GroupKFold(n_splits=n_splits)
    else:
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(df, groups=df[block_column(block_size_m)]))


def spatio_temporal_splits(
    df: pd.DataFrame,
    block_size_m: int = config.PRIMARY_BLOCK_SIZE_M,
    n_splits: int = config.N_SPLITS,
    seed: int | None = config.SEED,
) -> list:
    """Spatial blocks crossed with the temporal holdout (amends D-010).

    For each spatial fold: train on the training blocks in 2023-24, test on the
    held-out blocks in 2025. A purely temporal split would train and test on the
    same cells, whose static features (land cover, terrain) are identical across
    years and whose indices barely move, so it measures memorisation of place
    rather than generalisation. Crossing the two makes the test set unseen in
    both space and time.
    """
    is_train_year = df["year"].isin(config.TRAIN_YEARS).to_numpy()
    is_test_year = (df["year"] == config.TEST_YEAR).to_numpy()

    splits = []
    for block_train, block_test in spatial_splits(df, block_size_m, n_splits, seed):
        train_mask = np.zeros(len(df), dtype=bool)
        test_mask = np.zeros(len(df), dtype=bool)
        train_mask[block_train] = True
        test_mask[block_test] = True
        splits.append(
            (
                np.flatnonzero(train_mask & is_train_year),
                np.flatnonzero(test_mask & is_test_year),
            )
        )
    return splits


def strip_splits(
    df: pd.DataFrame,
    layout_df: pd.DataFrame,
    block_size_m: int = config.PRIMARY_BLOCK_SIZE_M,
    n_splits: int = config.N_SPLITS,
    temporal: bool = False,
) -> list:
    """Stress test: the original unshuffled layout of the full grid, on land rows (D-033).

    Unshuffled GroupKFold deals equal-sized blocks in id order, which on the full
    100 m grid at 4 km makes every fold an east-west strip. Removing water (D-034)
    changes block sizes and would scramble that pattern, so the block-to-fold
    layout is computed on `layout_df` (the full grid, water included) and only
    then applied to the land-only `df`. `temporal=True` crosses it with the
    2023-24 / 2025 split, as spatio_temporal_splits does.
    """
    require_land_only(df)
    col = block_column(block_size_m)
    fold_of_row = np.empty(len(layout_df), dtype="int64")
    for k, (_, test) in enumerate(
        GroupKFold(n_splits=n_splits).split(layout_df, groups=layout_df[col])
    ):
        fold_of_row[test] = k
    fold_of_block = pd.Series(fold_of_row, index=layout_df[col].to_numpy())
    fold_of_block = fold_of_block[~fold_of_block.index.duplicated()]
    folds = df[col].map(fold_of_block).to_numpy()
    assert not np.isnan(folds.astype("float64")).any(), "land block missing from layout"

    train_ok = df["year"].isin(config.TRAIN_YEARS).to_numpy() if temporal else True
    test_ok = (df["year"] == config.TEST_YEAR).to_numpy() if temporal else True
    return [
        (np.flatnonzero((folds != k) & train_ok), np.flatnonzero((folds == k) & test_ok))
        for k in range(n_splits)
    ]


def block_folds(
    df: pd.DataFrame, block_size_m: int, seed: int | None = config.SEED
) -> pd.Series:
    """Fold of every block under spatial_splits, as a block-id -> fold mapping.

    Lets a subsample (e.g. the tuning sample) reuse the full table's layout
    exactly, instead of re-dealing its blocks under the same seed.
    """
    col = block_column(block_size_m)
    fold = np.empty(len(df), dtype="int64")
    for k, (_, test) in enumerate(spatial_splits(df, block_size_m, seed=seed)):
        fold[test] = k
    mapping = pd.Series(fold, index=df[col].to_numpy()).groupby(level=0).agg(["min", "max"])
    assert (mapping["min"] == mapping["max"]).all(), "a block spans two folds"
    return mapping["min"]


SCHEMES = {
    "random": random_splits,
    "spatial": spatial_splits,
    "spatio_temporal": spatio_temporal_splits,
}


# ---------------------------------------------------------------------------
# T3.4 Fold diagnostics
# ---------------------------------------------------------------------------
def fold_report(
    df: pd.DataFrame, splits: list, scheme: str, block_size_m: int | None = None
) -> pd.DataFrame:
    """Per-fold sizes, year span and urban share, for balance checking."""
    overall_urban = float(df["is_urban"].mean())
    rows = []
    for k, (tr, te) in enumerate(splits):
        test = df.iloc[te]
        urban = float(test["is_urban"].mean()) if len(test) else float("nan")
        rows.append(
            {
                "scheme": scheme,
                "block_m": block_size_m,
                "fold": k,
                "n_train": len(tr),
                "n_test": len(te),
                "test_years": ",".join(str(y) for y in sorted(test["year"].unique())),
                "urban_share": urban,
                "urban_dev_pts": (urban - overall_urban) * 100,
            }
        )
    return pd.DataFrame(rows)
