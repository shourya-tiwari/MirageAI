"""Phase 6: SHAP explainability for the best tree model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mirageai import config
from mirageai import models as M

SPECTRAL_GROUP = "spectral (NDVI+NDBI+MNDWI)"
N_EXPLAIN = 10_000
N_BACKGROUND = 1_000
N_INTERACTION = 2_000
STRATA = ("year", "is_urban")


# ---------------------------------------------------------------------------
# Data and model
# ---------------------------------------------------------------------------
def float32_exact(X: pd.DataFrame) -> pd.DataFrame:
    """Round features to float32 precision, kept as float64.

    shap's interventional TreeSHAP evaluates trees on float32 inputs. LightGBM
    places thresholds between float64 values, so a float32 cast can move a cell
    across a split and the SHAP values stop summing to the prediction (up to
    0.4 °C on this data). A model fitted on float32-representable values has
    every threshold between float32 values, so the cast is lossless (D-036).
    """
    return X.astype(np.float32).astype(np.float64)


def fit_final(df: pd.DataFrame, name: str, params: dict):
    """The model SHAP explains: tuned params, every land row, all three years (D-035)."""
    feats = M.features_for(name)
    est = M.make_model(name, **params)
    est.fit(float32_exact(df[feats]), df[config.TARGET_ANOM])
    return est


def stratified_sample(
    df: pd.DataFrame, n: int, seed: int = config.SEED, exclude: pd.Index | None = None
) -> pd.Index:
    """Row labels of a sample with each year x urban/rest stratum at its population share."""
    pool = df if exclude is None else df.drop(index=exclude)
    sizes = pool.groupby(list(STRATA)).size()
    alloc = np.floor(sizes / sizes.sum() * n).astype(int)
    # Largest remainder, so the sample is exactly n.
    short = n - int(alloc.sum())
    remainder = (sizes / sizes.sum() * n) - alloc
    alloc[remainder.sort_values(ascending=False).index[:short]] += 1
    rng = np.random.default_rng(seed)
    picked = []
    for key, rows in pool.groupby(list(STRATA)).groups.items():
        picked.append(rng.choice(np.asarray(rows), int(alloc[key]), replace=False))
    return pd.Index(np.sort(np.concatenate(picked)))


# ---------------------------------------------------------------------------
# SHAP values
# ---------------------------------------------------------------------------
SHAP_N_JOBS = 8  # worker processes; ~250 MB each
SHAP_CHUNK = 2_000


def _interventional_chunk(model, X: pd.DataFrame, background: pd.DataFrame):
    import shap

    explainer = shap.TreeExplainer(
        model, data=background, feature_perturbation="interventional"
    )
    values = explainer.shap_values(X, check_additivity=True)
    return values, float(np.ravel(explainer.expected_value)[0])


def interventional_shap(
    model, X: pd.DataFrame, background: pd.DataFrame, n_jobs: int = 1, chunk: int = SHAP_CHUNK
):
    """Interventional TreeSHAP against a background sample; returns (values, base value).

    Interventional perturbation breaks the dependence between correlated
    features (the fractions sum to one; NDVI, NDBI and MNDWI move together), so
    a feature is credited only through the model's response to it, not through
    the features it happens to correlate with (D-036). Additivity is checked.
    shap runs this one row at a time on one core, so `n_jobs > 1` splits the
    rows into chunks across worker processes; the result is identical.
    """
    if n_jobs == 1 or len(X) <= chunk:
        return _interventional_chunk(model, X, background)
    from joblib import Parallel, delayed

    parts = Parallel(n_jobs=n_jobs)(
        delayed(_interventional_chunk)(model, X.iloc[i : i + chunk], background)
        for i in range(0, len(X), chunk)
    )
    bases = {round(b, 9) for _, b in parts}
    assert len(bases) == 1, "background base value differs between chunks"
    return np.vstack([v for v, _ in parts]), parts[0][1]


def path_dependent_shap(model, X: pd.DataFrame):
    """Exact path-dependent TreeSHAP; returns (values, base value).

    LightGBM's native `pred_contrib` is the same algorithm, multithreaded, which
    makes the full 243k-row grid take seconds rather than minutes.
    """
    if hasattr(model, "booster_"):
        contrib = model.predict(X, pred_contrib=True)
        values, base = contrib[:, :-1], float(contrib[0, -1])
    else:
        import shap

        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        values = explainer.shap_values(X)
        base = float(np.ravel(explainer.expected_value)[0])
    err = np.abs(values.sum(axis=1) + base - model.predict(X)).max()
    assert err < 1e-4, f"path-dependent SHAP not additive (max error {err:.2e})"
    return values, base


def interaction_values(model, X: pd.DataFrame):
    """SHAP interaction values (n, p, p) and base value. Path-dependent only: shap
    has no interventional interaction algorithm (D-036). Additivity is checked."""
    import shap

    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    inter = explainer.shap_interaction_values(X)
    base = float(np.ravel(explainer.expected_value)[0])
    err = np.abs(inter.sum(axis=(1, 2)) + base - model.predict(X)).max()
    assert err < 1e-4, f"interaction values not additive (max error {err:.2e})"
    return inter, base


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def importance(values: np.ndarray, features: list[str]) -> pd.Series:
    """Mean |SHAP| per feature, in °C of LST_anom, sorted descending."""
    return pd.Series(np.abs(values).mean(axis=0), index=features).sort_values(ascending=False)


def grouped(values: np.ndarray, features: list[str], members=config.SPECTRAL_FEATURES) -> np.ndarray:
    """Per-row SHAP of a feature group: the sum of its members' values (additive by construction)."""
    idx = [features.index(f) for f in members]
    return values[:, idx].sum(axis=1)


# Driver groups for the per-cell map (T6.5). The spectral members trade credit
# between themselves across models and SHAP methods (D-036), so a map at
# feature level flips arbitrarily between them; groups are stable.
DRIVER_GROUPS: dict[str, tuple[str, ...]] = {
    "spectral": config.SPECTRAL_FEATURES,
    "dry cover": ("frac_grass", "frac_crop", "frac_bare", "frac_shrub"),
    "tree cover": ("frac_tree",),
    "built": ("frac_built",),
    "water proximity": ("dist_water_m", "frac_water"),
    "terrain": ("elevation", "slope"),
}
assert sorted(f for g in DRIVER_GROUPS.values() for f in g) == sorted(config.FEATURES_TREE)


def group_values(values: np.ndarray, features: list[str], groups=DRIVER_GROUPS) -> np.ndarray:
    """Per-row SHAP summed within each group; columns in `groups` order."""
    return np.column_stack([grouped(values, features, members) for members in groups.values()])


def rank_agreement(a: pd.Series, b: pd.Series, top: int = 5) -> dict:
    """Spearman and Kendall rank correlation of two importance vectors, and top-k overlap."""
    from scipy.stats import kendalltau, spearmanr

    a = a.sort_values(ascending=False)
    b = b.reindex(a.index)
    top_a, top_b = list(a.index[:top]), list(b.sort_values(ascending=False).index[:top])
    return {
        "spearman": float(spearmanr(a, b).statistic),
        "kendall": float(kendalltau(a, b).statistic),
        f"top{top}_same_set": set(top_a) == set(top_b),
        f"top{top}_same_order": top_a == top_b,
        "only_in_a": ",".join(f for f in top_a if f not in top_b),
        "only_in_b": ",".join(f for f in top_b if f not in top_a),
    }


def top_interactions(inter: np.ndarray, features: list[str], k: int = 3) -> pd.DataFrame:
    """Feature pairs ranked by mean |phi_ij + phi_ji|, the pair's total interaction effect."""
    mean_abs = np.abs(inter + inter.transpose(0, 2, 1)).mean(axis=0)
    main = np.abs(np.einsum("nii->ni", inter)).mean(axis=0)
    rows = []
    p = len(features)
    for i in range(p):
        for j in range(i + 1, p):
            rows.append({"feature_a": features[i], "feature_b": features[j],
                         "mean_abs_interaction": float(mean_abs[i, j]),
                         "main_a": float(main[i]), "main_b": float(main[j])})
    out = pd.DataFrame(rows).sort_values("mean_abs_interaction", ascending=False)
    out["rank"] = np.arange(1, len(out) + 1)
    return out.reset_index(drop=True).head(k) if k else out.reset_index(drop=True)


def interaction_matrix(inter: np.ndarray, features: list[str]) -> pd.DataFrame:
    """Mean |SHAP interaction|: main effects on the diagonal, pair totals off it."""
    m = np.abs(inter + inter.transpose(0, 2, 1)).mean(axis=0)
    np.fill_diagonal(m, np.abs(np.einsum("nii->ni", inter)).mean(axis=0))
    return pd.DataFrame(m, index=features, columns=features)


# ---------------------------------------------------------------------------
# Thresholds and saturation (T6.3)
# ---------------------------------------------------------------------------
def _hinge_fit(x: np.ndarray, y: np.ndarray, knots: np.ndarray):
    """Best one-knot continuous piecewise-linear fit; returns (knot, coef, sse) arrays per knot."""
    X0 = np.column_stack([np.ones_like(x), x])
    coef0, *_ = np.linalg.lstsq(X0, y, rcond=None)
    sse0 = float(((y - X0 @ coef0) ** 2).sum())

    def search(candidates, best):
        for k in candidates:
            Xk = np.column_stack([np.ones_like(x), x, np.maximum(x - k, 0.0)])
            coef, *_ = np.linalg.lstsq(Xk, y, rcond=None)
            sse = float(((y - Xk @ coef) ** 2).sum())
            if sse < best[2]:
                best = (float(k), coef, sse)
        return best

    best = search(knots, (np.nan, None, np.inf))
    # Refine between the neighbouring coarse knots, so the estimate (and its
    # bootstrap spread) is not limited to the quantile grid.
    i = int(np.searchsorted(knots, best[0]))
    lo, hi = knots[max(i - 1, 0)], knots[min(i + 1, len(knots) - 1)]
    best = search(np.linspace(lo, hi, 41), best)
    return best, sse0


def breakpoint(
    x: np.ndarray, y: np.ndarray, n_knots: int = 60, n_boot: int = 200, seed: int = config.SEED
) -> dict:
    """One-breakpoint segmented regression of SHAP on feature value, with a bootstrap CI.

    Candidate knots are the 5th-95th percentiles of x, so each segment keeps at
    least 5% of the cells. Classification uses the slopes either side: a slope
    that falls below a quarter of the other side's is 'saturates' (after the
    knot) or 'onset' (before it); opposite signs are 'reverses'; otherwise
    'changes slope'. A segmented fit that adds under 2 points of R² over a
    straight line is 'near-linear' and its knot is not reported as a threshold.
    """
    knots = np.unique(np.quantile(x, np.linspace(0.05, 0.95, n_knots)))
    (k, coef, sse), sse0 = _hinge_fit(x, y, knots)
    sst = float(((y - y.mean()) ** 2).sum())
    r2_lin, r2_seg = 1 - sse0 / sst, 1 - sse / sst
    s1, s2 = float(coef[1]), float(coef[1] + coef[2])
    if r2_seg - r2_lin < 0.02:
        shape = "near-linear"
    elif np.sign(s1) != np.sign(s2) and min(abs(s1), abs(s2)) > 0.25 * max(abs(s1), abs(s2)):
        shape = "reverses"
    elif abs(s2) < 0.25 * abs(s1):
        shape = "saturates"
    elif abs(s1) < 0.25 * abs(s2):
        shape = "onset"
    else:
        shape = "changes slope"

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        boot[b] = _hinge_fit(x[i], y[i], knots)[0][0]
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"knot": k, "knot_lo": float(lo), "knot_hi": float(hi),
            "slope_below": s1, "slope_above": s2,
            "r2_linear": r2_lin, "r2_segmented": r2_seg, "shape": shape}
