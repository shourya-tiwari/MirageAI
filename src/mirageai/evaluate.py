"""Phase 5: metrics, CV-scheme comparison and residual diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from mirageai import config
from mirageai import models as M

OOF_PATH = config.TABLES / "oof_predictions.parquet"
# Columns read by default; block_size_m is implied by the scheme filter.
OOF_COLUMNS = ["row_id", "cell_id", "year", "fold", "seed", "model", "scheme", "y_true", "y_pred"]

PRIMARY_SCHEME = f"spatial_{config.PRIMARY_BLOCK_SIZE_M // 1000}km"
SPATIO_TEMPORAL_SCHEME = f"spatio_temporal_{config.PRIMARY_BLOCK_SIZE_M // 1000}km"
# Unshuffled GroupKFold at 4 km: every fold is an east-west strip. Reported as a
# latitude-extrapolation stress test, never as the headline (D-033).
STRIP_SUFFIX = "_strips"
STRIP_SEED = -1  # seed column value for the unshuffled layout
SCHEME_ORDER = (
    ("random",)
    + tuple(f"spatial_{s // 1000}km" for s in config.BLOCK_SIZES_M)
    + (SPATIO_TEMPORAL_SCHEME,)
    + (PRIMARY_SCHEME + STRIP_SUFFIX, SPATIO_TEMPORAL_SCHEME + STRIP_SUFFIX)
)
TREE_MODELS = ("rf", "xgb", "lgbm")

# Reference model for the per-cell diagnostics and for Phase 6. The tree models
# are tied under spatial CV, so the choice is made on the clean 2025 check and on
# SHAP cost, not on a meaningful accuracy gap (D-032).
BEST_MODEL = "lgbm"

METRICS = ("RMSE", "MAE", "R2")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_oof(
    scheme: str | None = None,
    model: str | None = None,
    seed: int | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Out-of-fold predictions from Phase 4, filtered and column-pruned on read.

    The full file holds every scheme x seed x model (~30M rows); it is never
    loaded whole. `columns` defaults to OOF_COLUMNS, and y_true / y_pred are
    always read because the residual is derived from them.
    """
    filters = []
    if scheme is not None:
        filters.append(("scheme", "=", scheme))
    if model is not None:
        filters.append(("model", "=", model))
    if seed is not None:
        filters.append(("seed", "=", seed))
    if not OOF_PATH.exists():
        raise FileNotFoundError(f"{OOF_PATH} not found: build it with scripts/pipeline/run_phase4.py (or scripts/pipeline/run_all.py)")
    cols = list(dict.fromkeys(list(columns or OOF_COLUMNS) + ["y_true", "y_pred"]))
    table = pq.read_table(OOF_PATH, columns=cols, filters=filters or None)
    oof = table.to_pandas()
    oof["resid"] = oof["y_true"] - oof["y_pred"]  # positive = model too cool
    return oof


def attach_cells(oof: pd.DataFrame, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Join model-table columns onto OOF rows via row_id (the model-table index)."""
    out = oof.copy()
    for col in cols:
        out[col] = df[col].to_numpy()[oof["row_id"].to_numpy()]
    return out


def _order(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort a results table by scheme, then model, in reporting order."""
    frame = frame.copy()
    frame["scheme"] = pd.Categorical(frame["scheme"], SCHEME_ORDER, ordered=True)
    frame["model"] = pd.Categorical(frame["model"], M.MODEL_NAMES, ordered=True)
    keys = ["scheme", "model"] + [c for c in ("decile", "in_decile") if c in frame]
    return frame.sort_values(keys).reset_index(drop=True)


# ---------------------------------------------------------------------------
# T5.1 Model x scheme comparison
# ---------------------------------------------------------------------------
def error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE, R-squared and mean residual (y_true - y_pred) for one set of rows."""
    y_true = np.asarray(y_true, dtype="float64")
    resid = y_true - np.asarray(y_pred, dtype="float64")
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "n": int(len(resid)),
        "RMSE": float(np.sqrt(np.mean(resid**2))),
        "MAE": float(np.mean(np.abs(resid))),
        "R2": float(1 - (resid**2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
        "mean_resid": float(resid.mean()),
    }


def model_comparison(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over every seed x fold run, per scheme x model (T5.1).

    `*_seed_std` is the std of the per-seed means: how much the headline moves
    with the block-to-fold layout alone.
    """
    keys = ["scheme", "model"]
    agg = fold_metrics.groupby(keys)[list(METRICS)].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    per_seed = fold_metrics.groupby(keys + ["seed"])[list(METRICS)].mean()
    seed_std = per_seed.groupby(level=[0, 1]).std()
    seed_std.columns = [f"{m}_seed_std" for m in seed_std.columns]
    agg = agg.join(seed_std).reset_index()
    agg["n_seeds"] = fold_metrics.groupby(keys)["seed"].nunique().to_numpy()
    agg["n_runs"] = fold_metrics.groupby(keys).size().to_numpy()
    agg["block_size_m"] = agg["scheme"].map(_block_size)
    # Linear and tree models see different feature counts by design (D-025).
    agg["n_features"] = agg["model"].map(lambda m: len(M.features_for(m)))
    agg = _order(agg)
    cols = ["scheme", "block_size_m", "model", "n_features", "n_seeds", "n_runs"]
    return agg[cols + [c for c in agg.columns if c not in cols]]


def paired_differences(
    fold_metrics: pd.DataFrame, reference: str = BEST_MODEL, metric: str = "RMSE"
) -> pd.DataFrame:
    """Per-run metric difference of each model against a reference, on identical folds.

    Every model is scored on the same seed x fold test sets, so the paired
    difference removes the between-fold variance that dominates the raw std.
    """
    rows = []
    for scheme, part in fold_metrics.groupby("scheme"):
        wide = part.pivot_table(index=["seed", "fold"], columns="model", values=metric)
        for model in wide.columns:
            if model == reference:
                continue
            diff = wide[model] - wide[reference]
            rows.append(
                {
                    "scheme": scheme,
                    "model": model,
                    "reference": reference,
                    f"d{metric}_mean": float(diff.mean()),
                    f"d{metric}_std": float(diff.std()),
                    "runs_model_better": int((diff < 0).sum()),
                    "n_runs": int(diff.size),
                }
            )
    return _order(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# T5.2 Random vs spatial gap
# ---------------------------------------------------------------------------
def cv_gap(comparison: pd.DataFrame) -> pd.DataFrame:
    """Random-CV optimism per model: scheme minus random RMSE and R-squared (T5.2)."""
    rows = []
    for model, part in comparison.groupby("model", observed=True):
        rnd = part[part.scheme == "random"].iloc[0]
        for _, r in part[part.scheme != "random"].iterrows():
            rows.append(
                {
                    "model": model,
                    "scheme": r.scheme,
                    "RMSE_random": rnd.RMSE_mean,
                    "RMSE_scheme": r.RMSE_mean,
                    "dRMSE": r.RMSE_mean - rnd.RMSE_mean,
                    "dRMSE_pct": 100 * (r.RMSE_mean - rnd.RMSE_mean) / rnd.RMSE_mean,
                    "R2_random": rnd.R2_mean,
                    "R2_scheme": r.R2_mean,
                    "dR2": r.R2_mean - rnd.R2_mean,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# T5.3 Temporal holdout
# ---------------------------------------------------------------------------
def seen_place_predictions(df: pd.DataFrame, model_name: str) -> np.ndarray:
    """Predictions of the final 2023-24 model for every row (NaN outside the test year).

    The final model was trained on every land cell in 2023-24, so on 2025 rows it
    has seen the place but not the year. Scored on a spatio-temporal fold's test
    rows, it pairs with the OOF result that has seen neither, and the difference
    isolates place memorisation (D-010 amended).
    """
    import joblib

    bundle = joblib.load(config.MODELS / f"{model_name}_trainyears.joblib")
    assert bundle["fitted_on"] == "train_years"
    test = (df["year"] == config.TEST_YEAR).to_numpy()
    pred = np.full(len(df), np.nan)
    pred[test] = bundle["model"].predict(df.loc[test, bundle["features"]])
    return pred


# ---------------------------------------------------------------------------
# T5.6 / T5.7 Subgroup errors
# ---------------------------------------------------------------------------
def hot_decile_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Per-year hottest-decile flags: AOI-wide and within urban cells (D-019).

    Thresholds are taken per year on the full model table, so they do not depend
    on which cells a fold holds out. Selecting on the target is fine here: it
    defines an evaluation subgroup, never a training input.
    """
    t = df[config.TARGET_ANOM]
    aoi_q = t.groupby(df["year"]).transform(lambda s: s.quantile(config.HOT_DECILE))
    urban_q = (
        t.where(df["is_urban"])
        .groupby(df["year"])
        .transform(lambda s: s.quantile(config.HOT_DECILE))
    )
    return pd.DataFrame(
        {"hot_aoi": t >= aoi_q, "hot_urban": df["is_urban"] & (t >= urban_q)},
        index=df.index,
    )


def top_decile_overlap(oof: pd.DataFrame, urban_only: bool = False) -> pd.DataFrame:
    """Share of the observed hottest decile that the predictions also place in their top decile.

    The predicted and observed sets are the same size, so this is both recall and
    precision. Unlike the mean residual inside the observed decile, it is not
    biased by selecting on the observed extreme, which pushes residuals positive
    for any imperfect model.
    """
    sel = oof[oof["is_urban"]] if urban_only else oof
    rows = []
    for (scheme, model, seed, year), part in sel.groupby(["scheme", "model", "seed", "year"]):
        obs = part.y_true >= part.y_true.quantile(config.HOT_DECILE)
        pred = part.y_pred >= part.y_pred.quantile(config.HOT_DECILE)
        rows.append(
            {
                "scheme": scheme,
                "model": model,
                "seed": int(seed),
                "year": int(year),
                "decile": "within urban" if urban_only else "AOI-wide",
                "n_hot": int(obs.sum()),
                "overlap": float((obs & pred).sum() / obs.sum()),
            }
        )
    return pd.DataFrame(rows)


def subgroup_fold_metrics(oof: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Per seed x fold metrics for every level of a subgroup column."""
    rows = []
    keys = ["scheme", "model", "seed", "fold", group_col]
    for (scheme, model, seed, fold, level), part in oof.groupby(keys, observed=True):
        rows.append(
            {
                "scheme": scheme,
                "model": model,
                "seed": int(seed),
                "fold": int(fold),
                group_col: level,
                **error_metrics(part.y_true.to_numpy(), part.y_pred.to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def summarise_subgroups(fm: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Mean and std over seed x fold runs of subgroup metrics."""
    keys = ["scheme", "model"] + group_cols
    agg = fm.groupby(keys)[["RMSE", "MAE", "R2", "mean_resid"]].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_runs"] = fm.groupby(keys).size()
    agg = agg.reset_index()
    return _order(agg).sort_values(
        ["scheme", "model"] + group_cols, kind="stable"
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# T5.5 Moran's I on residuals, and the residual autocorrelation range
# ---------------------------------------------------------------------------
def queen_weights(x_utm: np.ndarray, y_utm: np.ndarray):
    """Row-standardised queen contiguity on the 100 m grid, from cell centres.

    Built as a distance band of one diagonal step, so it is exactly the 8-cell
    queen neighbourhood and stays correct when water cells leave holes in the
    lattice (D-034). Land cells with no land neighbour are returned as islands.
    """
    from libpysal.weights import DistanceBand

    xy = np.column_stack([x_utm, y_utm])
    w = DistanceBand(
        xy, threshold=config.SCALE * 1.5, binary=True, silence_warnings=True
    )
    w.transform = "r"
    return w


def morans_i(values: np.ndarray, w, permutations: int = 999) -> dict[str, float]:
    """Global Moran's I with normal-approximation z and a permutation p-value."""
    from esda.moran import Moran

    np.random.seed(config.SEED)  # esda draws permutations from the global RNG
    mi = Moran(values, w, permutations=permutations)
    return {
        "morans_I": float(mi.I),
        "expected_I": float(mi.EI),
        "z_norm": float(mi.z_norm),
        "p_sim": float(mi.p_sim) if permutations else float("nan"),
    }


def residual_ranges(
    df: pd.DataFrame, resid: np.ndarray, sill_fracs: tuple[float, ...] = (0.90, 0.95, 0.99)
) -> pd.DataFrame:
    """Per-year residual variogram range with the T3.1 estimator (D-028 method)."""
    from mirageai import cv as cv_mod

    frame = df[["x_utm", "y_utm", "year"]].assign(resid=resid)
    rows = []
    for year, part in frame.groupby("year"):
        vario = cv_mod.empirical_variogram(part, value_col="resid")
        row = {"year": int(year), "resid_var": float(part.resid.var())}
        for frac in sill_fracs:
            pr = cv_mod.practical_range(vario, sill_frac=frac)
            row.update(nugget=pr["nugget"], sill=pr["sill"])
            row[f"range_{round(frac * 100)}_m"] = pr["range_m"]
        rows.append(row)
    return pd.DataFrame(rows)


def block_rung(range_m: float) -> int | None:
    """Smallest configured block size at or above a range (the D-028 rule)."""
    rungs = [s for s in config.BLOCK_SIZES_M if s >= range_m]
    return min(rungs) if rungs else None


def _block_size(scheme: str) -> int | None:
    """Block size in metres encoded in a scheme name, or None for random CV."""
    if scheme == "random":
        return None
    km = scheme.removesuffix(STRIP_SUFFIX).rsplit("_", 1)[1]
    return int(km.removesuffix("km")) * 1000
