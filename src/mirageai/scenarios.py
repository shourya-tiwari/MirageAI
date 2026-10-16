"""Phase 7: greening scenarios and associational cooling potential."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mirageai import config
from mirageai import explain as X

# New tree cover is taken from the dry/hot classes, in proportion to how much of
# each the cell has (D-040). Crop, shrub and water are left alone.
DONORS: tuple[str, ...] = ("frac_built", "frac_bare", "frac_grass")
INDEX_TARGETS: tuple[str, ...] = config.SPECTRAL_FEATURES
INDEX_FEATURES: tuple[str, ...] = config.FRACTION_FEATURES + config.TERRAIN_FEATURES
SUPPORT_K = 10          # k-th nearest training row for the support check (D-043)
SUPPORT_QUANTILE = 0.99
SUPPORT_REF_SAMPLE = 20_000


# ---------------------------------------------------------------------------
# T7.2 Target cells
# ---------------------------------------------------------------------------
def target_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Urban cells in the hottest decile of their OBSERVED 3-year mean LST_anom (D-039).

    Land cover is static (WorldCover 2021), so a cell is urban in every year or
    in none. The decile is taken within urban cells, as D-019 requires.
    """
    per_cell = df.groupby("cell_id").agg(
        mean_anom=(config.TARGET_ANOM, "mean"), n_years=("year", "nunique"),
        is_urban=("is_urban", "first"), urban_any=("is_urban", "max"),
        x_utm=("x_utm", "first"), y_utm=("y_utm", "first"),
        lon=("lon", "first"), lat=("lat", "first"),
    )
    assert (per_cell.is_urban == per_cell.urban_any).all(), "urban status varies by year"
    urban = per_cell[per_cell.is_urban]
    cut = urban.mean_anom.quantile(config.HOT_DECILE)
    out = urban[urban.mean_anom >= cut].drop(columns="urban_any")
    out.attrs["threshold"] = float(cut)
    out.attrs["n_urban_cells"] = len(urban)
    return out


# ---------------------------------------------------------------------------
# T7.3 Scenarios
# ---------------------------------------------------------------------------
def add_tree_cover(fr: pd.DataFrame, delta) -> tuple[pd.DataFrame, np.ndarray]:
    """Raise frac_tree by `delta`, taken from DONORS in proportion to their shares.

    Returns (new fractions, shortfall). Where the donors hold less than `delta`,
    all of them is converted and the unmet part is the shortfall, so fractions
    never go negative and their sum is unchanged.
    """
    delta = np.broadcast_to(np.asarray(delta, dtype="float64"), (len(fr),)).copy()
    delta = np.clip(delta, 0.0, None)
    pool = fr[list(DONORS)].sum(axis=1).to_numpy()
    applied = np.minimum(delta, pool)
    out = fr.copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        take = np.where(pool > 0, applied / pool, 0.0)
    for d in DONORS:
        out[d] = fr[d].to_numpy() * (1.0 - take)
    out["frac_tree"] = fr["frac_tree"].to_numpy() + applied
    frac_cols = list(config.FRACTION_FEATURES)
    assert np.allclose(out[frac_cols].sum(axis=1), fr[frac_cols].sum(axis=1)), "sum changed"
    assert (out[frac_cols].to_numpy() >= -1e-12).all(), "negative fraction"
    return out, delta - applied


def scenario_delta(name: str, tree: np.ndarray, knot: float) -> np.ndarray:
    """Points of tree cover (as a fraction) each scenario asks for."""
    if name == "A":
        return np.full(len(tree), 0.10)
    if name == "B":
        return np.full(len(tree), 0.20)
    if name == "C":
        return np.clip(knot - tree, 0.0, None)
    raise ValueError(f"unknown scenario {name!r}; expected 'A', 'B' or 'C'")


# ---------------------------------------------------------------------------
# D-009 realistic bound: index response to land cover
# ---------------------------------------------------------------------------
def fit_index_models(df: pd.DataFrame, params: dict) -> dict:
    """One LightGBM per (year, index), predicting the index from fractions + terrain."""
    from mirageai import models as M

    out = {}
    for year, part in df.groupby("year"):
        for idx in INDEX_TARGETS:
            m = M.make_model("lgbm", **params)
            m.fit(part[list(INDEX_FEATURES)], part[idx])
            out[(int(year), idx)] = m
    return out


def updated_indices(base: pd.DataFrame, new: pd.DataFrame, index_models: dict) -> pd.DataFrame:
    """Observed index + f(new fractions) - f(old fractions), per year (D-041).

    Using the delta keeps each cell's own baseline: a cell's residual from the
    index model (shade, irrigation, roof material) carries through unchanged.
    """
    out = new.copy()
    feats = list(INDEX_FEATURES)
    for year in np.unique(base["year"]):
        rows = (base["year"] == year).to_numpy()
        for idx in INDEX_TARGETS:
            m = index_models[(int(year), idx)]
            d = m.predict(new.loc[rows, feats]) - m.predict(base.loc[rows, feats])
            out.loc[rows, idx] = base.loc[rows, idx].to_numpy() + d
    return out


def delta_lst(model, base: pd.DataFrame, scen: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Predicted LST_anom under the scenario minus under observed features."""
    return model.predict(X.float32_exact(scen[features])) - model.predict(X.float32_exact(base[features]))


# ---------------------------------------------------------------------------
# D-043 Extrapolation guard
# ---------------------------------------------------------------------------
class SupportGuard:
    """Flags rows far from the training data in standardised feature space.

    The score is the distance to the k-th nearest training row. The threshold is
    the 99th percentile of the same score for training rows themselves (each
    queried against the others, its own row excluded). k = 10 is above the three
    near-duplicate rows every cell has (one per year), so a scenario row cannot
    look supported just because its own cell's observed rows are nearby.
    """

    def __init__(self, train: pd.DataFrame, features: list[str], k: int = SUPPORT_K,
                 quantile: float = SUPPORT_QUANTILE, ref_sample: int = SUPPORT_REF_SAMPLE,
                 seed: int = config.SEED):
        from sklearn.neighbors import NearestNeighbors

        self.features = list(features)
        self.mean = train[self.features].mean()
        self.sd = train[self.features].std()
        Z = self._z(train)
        self.k = k
        self.nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(Z)
        rng = np.random.default_rng(seed)
        ref = Z[rng.choice(len(Z), min(ref_sample, len(Z)), replace=False)]
        dist, _ = self.nn.kneighbors(ref, n_neighbors=k + 1)
        self.ref_scores = dist[:, k]  # column 0 is the row itself
        self.threshold = float(np.quantile(self.ref_scores, quantile))

    def _z(self, frame: pd.DataFrame) -> np.ndarray:
        return ((frame[self.features] - self.mean) / self.sd).to_numpy()

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        dist, _ = self.nn.kneighbors(self._z(frame), n_neighbors=self.k)
        return dist[:, self.k - 1]

    def flag(self, frame: pd.DataFrame) -> np.ndarray:
        return self.score(frame) > self.threshold
