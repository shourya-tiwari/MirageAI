"""Phase 2: cleaning, missing-value handling and the modelling table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mirageai import config

# Integer packing base for cell and block IDs. Larger than any UTM index in the
# AOI (easting ~3.8e3 at 100 m, northing ~2.05e4), so the two never collide.
_ID_BASE = 1_000_000


def _require(path, hint: str) -> None:
    """Raise a FileNotFoundError that says how to create a missing input."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: {hint}")


def load_year(year: int) -> pd.DataFrame:
    """Load one year's raw extraction table."""
    path = config.raw_csv(year)
    _require(path, f"extract it with gee_extract.extract_year({year}) (README, 'Data')")
    return pd.read_csv(path)


def load_all(years: tuple[int, ...] = config.YEARS) -> pd.DataFrame:
    """Concatenate every year into one long table (T2.1)."""
    return pd.concat([load_year(y) for y in years], ignore_index=True)


def add_cell_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add a stable cell_id derived from UTM coordinates alone (D-022).

    Absolute, so the same ground cell gets the same id in every year and in any
    subset of the data.

    Uses floor, not round: cell centres sit at half-cell offsets (x = k*100 + 50),
    so x / SCALE lands exactly on .5 and round-half-to-even would merge adjacent
    columns onto one id.
    """
    ix = np.floor(df["x_utm"] / config.SCALE).astype("int64")
    iy = np.floor(df["y_utm"] / config.SCALE).astype("int64")
    df["cell_id"] = ix * _ID_BASE + iy
    return df


def block_id(df: pd.DataFrame, block_size_m: int) -> pd.Series:
    """Spatial block id for a given block size, from UTM coordinates alone (D-022)."""
    bx = np.floor(df["x_utm"] / block_size_m).astype("int64")
    by = np.floor(df["y_utm"] / block_size_m).astype("int64")
    return bx * _ID_BASE + by


def add_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Add one block-ID column per configured block size (D-022)."""
    for size in config.BLOCK_SIZES_M:
        df[f"block_{size}m"] = block_id(df, size)
    return df


def add_lst_anom(df: pd.DataFrame) -> pd.DataFrame:
    """Add LST_anom: LST minus that year's AOI-wide median LST (D-021).

    Removes the inter-annual level shift, which no feature can explain because
    nothing in the feature set encodes the year or the scene timing.
    """
    med = df.groupby("year")[config.TARGET].transform("median")
    df["LST_year_median"] = med
    df[config.TARGET_ANOM] = df[config.TARGET] - med
    return df


def add_urban_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag cells that are majority built-up (D-019)."""
    df["is_urban"] = df["frac_built"] >= config.URBAN_BUILT_THRESHOLD
    return df


def check_block_consistency(df: pd.DataFrame) -> dict[str, int]:
    """Assert each cell_id maps to exactly one block per size across years (D-022)."""
    report = {}
    for size in config.BLOCK_SIZES_M:
        col = f"block_{size}m"
        per_cell = df.groupby("cell_id")[col].nunique()
        offenders = int((per_cell > 1).sum())
        if offenders:
            raise AssertionError(
                f"{offenders} cell_id values map to more than one {col} across years"
            )
        report[col] = int(per_cell.size)
    return report


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-year row counts, nulls and cells failing the observation-count floor (T2.2)."""
    rows = []
    for year, part in df.groupby("year"):
        rows.append(
            {
                "year": int(year),
                "rows": len(part),
                "null_cells": int(part.isna().sum().sum()),
                "below_min_obs": int((part["LST_n_obs"] < config.MIN_LST_OBS).sum()),
                "min_n_obs": float(part["LST_n_obs"].min()),
                "median_n_obs": float(part["LST_n_obs"].median()),
            }
        )
    return pd.DataFrame(rows).set_index("year")


def vif_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Variance inflation factor per feature, with an intercept (T2.5)."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    mat = df[features].to_numpy(dtype="float64")
    mat = np.column_stack([np.ones(len(mat)), mat])  # intercept column
    vals = [variance_inflation_factor(mat, i + 1) for i in range(len(features))]
    return (
        pd.DataFrame({"feature": features, "VIF": vals})
        .set_index("feature")
        .sort_values("VIF", ascending=False)
    )


def build_clean_table(years: tuple[int, ...] = config.YEARS) -> pd.DataFrame:
    """Assemble the full modelling table and write it to data/processed (T2.7)."""
    df = load_all(years)
    df = add_cell_id(df)
    df = add_blocks(df)
    df = add_lst_anom(df)
    df = add_urban_flag(df)
    check_block_consistency(df)

    config.ensure_dirs()
    df.to_parquet(config.CLEAN_TABLE, index=False)
    return df


def load_clean() -> pd.DataFrame:
    """Read the processed modelling table."""
    _require(config.CLEAN_TABLE, "build it with scripts/pipeline/run_phase2.py (or scripts/pipeline/run_all.py)")
    return pd.read_parquet(config.CLEAN_TABLE)


def is_water(df: pd.DataFrame) -> pd.Series:
    """Water-majority rows, which never reach a model or a metric (D-034)."""
    return df["frac_water"] >= config.WATER_EXCLUDE_THRESHOLD


def core_ring_contrast(df: pd.DataFrame, centre: tuple[float, float] | None = None) -> pd.DataFrame:
    """Per-year mean LST and NDVI in the core and the outer ring, and their difference.

    `centre` is the (x, y) UTM midpoint of the grid extent; pass the full grid's
    centre when `df` is a subset, so the rings do not move with the subset.
    """
    if centre is None:
        centre = ((df.x_utm.min() + df.x_utm.max()) / 2, (df.y_utm.min() + df.y_utm.max()) / 2)
    d = np.hypot(df.x_utm - centre[0], df.y_utm - centre[1])
    core, ring = d < config.CORE_RADIUS_M, d > config.RING_RADIUS_M
    rows = []
    for year, g in df.groupby("year"):
        c, r = core[g.index], ring[g.index]
        rows.append({
            "year": year,
            "core_LST": g.loc[c, config.TARGET].mean(),
            "ring_LST": g.loc[r, config.TARGET].mean(),
            "core_NDVI": g.loc[c, "NDVI"].mean(),
            "ring_NDVI": g.loc[r, "NDVI"].mean(),
            "n_core": int(c.sum()),
            "n_ring": int(r.sum()),
        })
    out = pd.DataFrame(rows)
    out.insert(3, "core_minus_ring", out.core_LST - out.ring_LST)
    return out


def load_model_table() -> pd.DataFrame:
    """The clean table without water-majority cells, re-indexed from 0 (D-034).

    The index is what out-of-fold `row_id` refers to. LST_anom keeps its D-021
    definition (anomaly against the whole-AOI median, water included), so the
    target is unchanged for every land cell.
    """
    df = load_clean()
    return df.loc[~is_water(df)].reset_index(drop=True)
