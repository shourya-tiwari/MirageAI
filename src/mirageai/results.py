"""Read-only access to the pipeline's results, for the dashboard and the notebooks.

Everything here reads files the phase drivers already wrote to outputs/; nothing
is refitted. Each loader raises FileNotFoundError naming the driver that makes a
missing file, so a viewer knows what to run.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mirageai import config
from mirageai import evaluate as E
from mirageai import explain as X
from mirageai.plotting import cells_to_grid

# The table each driver writes first, so a missing file can name its producer.
_PRODUCER = {
    "phase2": "scripts/pipeline/run_phase2.py", "phase3": "scripts/pipeline/run_phase3.py", "phase4": "scripts/pipeline/run_phase4.py",
    "phase5": "scripts/pipeline/run_phase5.py", "model_comparison": "scripts/pipeline/run_phase5.py",
    "phase6": "scripts/pipeline/run_phase6.py", "phase7": "scripts/pipeline/run_phase7.py", "phase8": "scripts/make_report_figures.py",
}

MODEL_LABELS = {"baseline": "Mean baseline", "linear": "Linear regression",
                "rf": "Random forest", "xgb": "XGBoost", "lgbm": "LightGBM"}
SCENARIO_LABELS = {"A": "A: +10 points", "B": "B: +20 points", "C": "C: raise to the 10.2% knot"}


def _producer(name: str) -> str:
    """Driver script that writes a given output, for error messages."""
    for prefix, script in _PRODUCER.items():
        if name.startswith(prefix):
            return script
    return "scripts/pipeline/run_all.py"


def load_table(name: str) -> pd.DataFrame:
    """One CSV from outputs/tables, by its name without the extension."""
    path = config.table(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run {_producer(name)} (or scripts/pipeline/run_all.py)")
    return pd.read_csv(path)


def load_best_params() -> dict:
    """Tuned hyperparameters and tuning scores per model.

    Reads outputs/models/best_params.json. The submission ships no models
    directory, so without it the winners are rebuilt from the shipped search
    tables (rank 1 of phase4_tuning_<model>.csv), which hold the same search.
    """
    path = config.MODELS / "best_params.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for name in ("lgbm", "xgb", "rf"):
        if not config.table(f"phase4_tuning_{name}").exists():
            continue
        best = load_table(f"phase4_tuning_{name}").sort_values("rank_test_score").iloc[0]
        params = {k.removeprefix("param_"): v for k, v in best.items() if k.startswith("param_")}
        params = {k: int(v) if float(v).is_integer() and k in _INT_PARAMS else float(v)
                  for k, v in params.items()}
        out[name] = {"best_params": params, "best_cv_rmse": -float(best.mean_test_score),
                     "best_cv_rmse_std": float(best.std_test_score)}
    if not out:
        raise FileNotFoundError(f"{path} not found: run scripts/pipeline/run_phase4.py (or scripts/pipeline/run_all.py)")
    return out


_INT_PARAMS = {"n_estimators", "max_depth", "min_samples_leaf", "num_leaves",
               "min_child_samples", "min_child_weight"}


def load_model(name: str = "lgbm_all"):
    """A saved model bundle: a dict with 'model', 'features', 'params' and provenance."""
    import joblib

    path = config.MODELS / f"{name}.joblib"
    if not path.exists():
        hint = "scripts/pipeline/run_phase6.py" if name.endswith("_all") else "scripts/pipeline/run_phase4.py"
        raise FileNotFoundError(f"{path} not found: run {hint} (or scripts/pipeline/run_all.py)")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Headline numbers
# ---------------------------------------------------------------------------
def scheme_row(comparison: pd.DataFrame, scheme: str, model: str = E.BEST_MODEL) -> pd.Series:
    """The model_comparison row for one scheme and model."""
    row = comparison[(comparison.scheme == scheme) & (comparison.model == model)]
    if len(row) != 1:
        raise KeyError(f"no unique row for scheme={scheme!r}, model={model!r}")
    return row.iloc[0]


def headline(comparison: pd.DataFrame, core_ring: pd.DataFrame) -> dict[str, float]:
    """The handful of numbers the overview leads with, all read from the tables."""
    spatial = scheme_row(comparison, E.PRIMARY_SCHEME)
    temporal = scheme_row(comparison, E.SPATIO_TEMPORAL_SCHEME)
    rnd = scheme_row(comparison, "random")
    trees = comparison[(comparison.scheme == E.PRIMARY_SCHEME)
                       & comparison.model.isin(E.TREE_MODELS)]
    cool = -core_ring["core_minus_ring"]
    return {
        "spatial_rmse": float(spatial.RMSE_mean), "spatial_rmse_std": float(spatial.RMSE_std),
        "spatial_r2": float(spatial.R2_mean),
        "temporal_rmse": float(temporal.RMSE_mean), "temporal_r2": float(temporal.R2_mean),
        "random_rmse": float(rnd.RMSE_mean), "random_r2": float(rnd.R2_mean),
        "tree_rmse_min": float(trees.RMSE_mean.min()), "tree_rmse_max": float(trees.RMSE_mean.max()),
        "cool_island_min": float(cool.min()), "cool_island_max": float(cool.max()),
    }


# ---------------------------------------------------------------------------
# Evaluation views
# ---------------------------------------------------------------------------
def block_size_curve(comparison: pd.DataFrame) -> pd.DataFrame:
    """RMSE and R² against CV block size per model, with random CV at block size 0 (RQ3)."""
    keep = comparison.scheme.astype(str).str.fullmatch(r"random|spatial_\d+km")
    out = comparison.loc[keep, ["scheme", "model", "RMSE_mean", "RMSE_std", "R2_mean", "R2_std"]].copy()
    out["block_km"] = out.scheme.map(
        lambda s: 0 if s == "random" else int(s.removeprefix("spatial_").removesuffix("km")))
    out["model_label"] = out.model.map(MODEL_LABELS)
    return out.sort_values(["model", "block_km"]).reset_index(drop=True)


def hot_decile_detection(overlap: pd.DataFrame, scheme: str = E.PRIMARY_SCHEME) -> pd.DataFrame:
    """Mean share of the observed hottest decile also in the predicted one, per model."""
    part = overlap[overlap.scheme == scheme]
    out = part.groupby(["model", "decile"])["overlap"].agg(["mean", "std"]).reset_index()
    out["model_label"] = out.model.map(MODEL_LABELS)
    return out


def residual_morans_i(morans: pd.DataFrame) -> pd.DataFrame:
    """Mean Moran's I of residuals per scheme and model, with the raw target for reference."""
    out = morans.groupby(["scheme", "model"])["morans_I"].agg(["mean", "min", "max"]).reset_index()
    return out.sort_values("mean", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Scenario views
# ---------------------------------------------------------------------------
def scenario_bracket(summary: pd.DataFrame) -> pd.DataFrame:
    """Headline ΔLST per scenario and bound: three-model mean and range, first point excluded (D-040)."""
    ens = summary[summary.model == "ensemble"]
    out = ens[["scenario", "bound", "mean_points_added", "cells_changed", "pct_flagged",
               "dLST_excl_first_point", "dLST_excl_min", "dLST_excl_max", "dLST_mean"]].copy()
    out = out.rename(columns={"dLST_excl_first_point": "dLST", "dLST_excl_min": "dLST_lo",
                              "dLST_excl_max": "dLST_hi", "dLST_mean": "dLST_incl_first_point"})
    out["scenario_label"] = out.scenario.map(SCENARIO_LABELS)
    return out.reset_index(drop=True)


def dose_response(dose: pd.DataFrame) -> pd.DataFrame:
    """ΔLST against points of tree cover added, averaged over the three models, per bound."""
    out = dose.groupby(["bound", "points"])["dLST"].agg(["mean", "min", "max"]).reset_index()
    return out.sort_values(["bound", "points"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Maps from the per-cell SHAP table (shipped, so no data/ is needed)
# ---------------------------------------------------------------------------
def load_shap_cells(year: int | None = None) -> pd.DataFrame:
    """Per-row interventional SHAP of the final LightGBM for every land cell-year (Phase 6).

    The table holds no year column. `row_id` indexes the model table, which
    stacks the years in order with the same land cells in each (land cover is
    static), so `row_id // cells per year` is the year's position.
    """
    path = config.TABLES / "phase6_shap_cells_lgbm.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run scripts/pipeline/run_phase6.py (or scripts/pipeline/run_all.py)")
    cells = pd.read_parquet(path)
    per_year, rem = divmod(len(cells), len(config.YEARS))
    assert rem == 0, "SHAP cells table does not hold the same cells in every year"
    cells["year"] = np.asarray(config.YEARS)[cells.row_id.to_numpy() // per_year]
    return cells if year is None else cells[cells.year == year]


def shap_columns(features=config.FEATURES_TREE) -> list[str]:
    """Column names of the per-feature SHAP values in the cells table."""
    return [f"shap_{f}" for f in features]


def cell_mean_shap(cells: pd.DataFrame) -> pd.DataFrame:
    """SHAP per cell, averaged over the three years, as the Phase 6 driver map uses."""
    scols = shap_columns()
    return cells.groupby("cell_id").agg(
        {**{s: "mean" for s in scols}, "x_utm": "first", "y_utm": "first", "is_urban": "first"})


def driver_group_values(cells: pd.DataFrame) -> pd.DataFrame:
    """Grouped SHAP per row (D-038) and the dominant group by largest |SHAP|.

    Pass `cell_mean_shap(...)` to reproduce the Phase 6 per-cell map exactly.
    """
    feats = list(config.FEATURES_TREE)
    vals = cells[shap_columns(feats)].to_numpy()
    grouped = pd.DataFrame(X.group_values(vals, feats), columns=list(X.DRIVER_GROUPS),
                           index=cells.index)
    grouped["dominant"] = grouped.abs().idxmax(axis=1)
    return grouped


def to_grid(cells: pd.DataFrame, values) -> tuple[np.ndarray, tuple[float, ...]]:
    """Rasterise per-cell values onto the 100 m grid for a map; returns (grid, extent in km)."""
    return cells_to_grid(cells.x_utm.to_numpy(), cells.y_utm.to_numpy(), np.asarray(values))
