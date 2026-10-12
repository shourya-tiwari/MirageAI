"""Phase 4: model definitions and tuning inside spatial CV."""

from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mirageai import config

# Which feature set each model is entitled to. Trees keep every fraction so SHAP
# can attribute a built-up effect; the linear model drops one to break the
# sum-to-1 singularity (D-025).
FEATURE_SET = {
    "baseline": "linear",
    "linear": "linear",
    "rf": "tree",
    "xgb": "tree",
    "lgbm": "tree",
}


def features_for(model_name: str) -> list[str]:
    """Feature list a given model is fitted on."""
    kind = FEATURE_SET[model_name]
    return list(config.FEATURES_LINEAR if kind == "linear" else config.FEATURES_TREE)


def make_model(name: str, **params):
    """Construct an unfitted estimator by name, with the project seed applied."""
    if name == "baseline":
        return DummyRegressor(strategy="mean")
    if name == "linear":
        return Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression(**params))]
        )
    if name == "rf":
        return RandomForestRegressor(
            random_state=config.SEED, n_jobs=-1, **{"n_estimators": 300, **params}
        )
    if name == "xgb":
        from xgboost import XGBRegressor

        return XGBRegressor(
            random_state=config.SEED,
            n_jobs=-1,
            tree_method="hist",
            device="cpu",  # GPU histograms are not bitwise reproducible (D-031)
            verbosity=0,
            **params,
        )
    if name == "lgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            random_state=config.SEED,
            n_jobs=-1,
            device_type="cpu",
            verbose=-1,
            **params,
        )
    raise ValueError(f"unknown model: {name}")


MODEL_NAMES = ("baseline", "linear", "rf", "xgb", "lgbm")


# ---------------------------------------------------------------------------
# T4.6 Search spaces
# ---------------------------------------------------------------------------
SEARCH_SPACES: dict[str, dict] = {
    # Full space restored on the 16 GB machine (D-030 resolved). A 500-tree
    # forest with fully grown leaves is ~2.5-3 GB, so it fits as long as only
    # one forest is ever held at a time (D-031).
    "rf": {
        "n_estimators": [200, 300, 500],
        "max_depth": [8, 12, 16, 20],
        "min_samples_leaf": [1, 2, 5, 10, 20],
        "max_features": [0.3, 0.5, 0.7],
    },
    "xgb": {
        "n_estimators": [300, 500, 800, 1200],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 5, 20],
        "reg_lambda": [0.1, 1.0, 10.0],
    },
    "lgbm": {
        "n_estimators": [300, 500, 800, 1200],
        "num_leaves": [15, 31, 63, 127],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "min_child_samples": [10, 20, 50, 100],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0.1, 1.0, 10.0],
    },
}


# ---------------------------------------------------------------------------
# Out-of-fold prediction
# ---------------------------------------------------------------------------
def oof_predictions(
    df: pd.DataFrame,
    model_name: str,
    splits: list,
    params: dict | None = None,
    target: str = config.TARGET_ANOM,
) -> tuple[pd.DataFrame, float]:
    """Fit per fold and collect out-of-fold predictions. Returns (long frame, seconds)."""
    feats = features_for(model_name)
    X = df[feats]
    y = df[target].to_numpy()

    frames = []
    start = time.perf_counter()
    for fold, (train_idx, test_idx) in enumerate(splits):
        est = clone(make_model(model_name, **(params or {})))
        est.fit(X.iloc[train_idx], y[train_idx])
        pred = est.predict(X.iloc[test_idx])
        del est  # never hold two fitted forests at once (D-031)
        frames.append(
            pd.DataFrame(
                {
                    "row_id": df.index.to_numpy()[test_idx],
                    "cell_id": df["cell_id"].to_numpy()[test_idx],
                    "year": df["year"].to_numpy()[test_idx],
                    "fold": fold,
                    "model": model_name,
                    "y_true": y[test_idx],
                    "y_pred": pred,
                }
            )
        )
    elapsed = time.perf_counter() - start
    return pd.concat(frames, ignore_index=True), elapsed


def fold_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    """RMSE, MAE and R-squared per fold."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rows = []
    for fold, part in oof.groupby("fold"):
        rows.append(
            {
                "fold": int(fold),
                "n": len(part),
                "RMSE": float(np.sqrt(mean_squared_error(part.y_true, part.y_pred))),
                "MAE": float(mean_absolute_error(part.y_true, part.y_pred)),
                "R2": float(r2_score(part.y_true, part.y_pred)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Out-of-fold parts on disk
# ---------------------------------------------------------------------------
def prepare_parts_dir(parts_dir, force: bool = False) -> None:
    """Create an empty parts directory, refusing to discard saved parts unless forced.

    Saved parts are the output of a long CV run; a run that was interrupted
    after them is finished with --finish-from-parts, not by starting over.
    """
    import shutil

    existing = sorted(parts_dir.glob("*.parquet")) if parts_dir.exists() else []
    if existing and not force:
        raise FileExistsError(
            f"{len(existing)} saved OOF parts in {parts_dir}; finish them with "
            "--finish-from-parts, or pass --force to delete them and start over"
        )
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True)


def merge_parts(parts: list, out_path) -> dict[str, int]:
    """Stream parquet parts into one file, one part in memory at a time.

    Writes to a temporary name and renames only after the row count of the
    merged file matches the sum of the parts, so a crash never leaves a
    truncated file under the final name.
    """
    import pyarrow.parquet as pq

    tmp = out_path.with_name(out_path.name + ".tmp")
    expected = 0
    writer = None
    try:
        for part in parts:
            table = pq.read_table(part)
            expected += table.num_rows
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema)
            writer.write_table(table)
            del table
    finally:
        if writer is not None:
            writer.close()
    written = pq.ParquetFile(tmp).metadata.num_rows if writer is not None else 0
    if written != expected:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"merge wrote {written:,} rows, parts hold {expected:,}")
    tmp.replace(out_path)
    return {"n_parts": len(parts), "rows_in_parts": expected, "rows_written": written}

# ---------------------------------------------------------------------------
# T4.6 Tuning inside spatial CV
# ---------------------------------------------------------------------------
TUNE_N_ITER = 25
TUNE_SUBSAMPLE = 60_000


def tune(
    df: pd.DataFrame,
    model_name: str,
    block_size_m: int = config.PRIMARY_BLOCK_SIZE_M,
    n_iter: int = TUNE_N_ITER,
    subsample: int | None = TUNE_SUBSAMPLE,
    target: str = config.TARGET_ANOM,
):
    """RandomizedSearchCV inside spatial block CV, on training years only.

    The search never sees the temporal test year, so the spatio-temporal
    evaluation on 2025 stays clean (D-029). Folds follow the seed-42 shuffled
    block layout of the full training table (D-033).
    """
    from sklearn.model_selection import RandomizedSearchCV

    from mirageai import cv as cv_mod

    train = df[df["year"].isin(config.TRAIN_YEARS)]
    cv_mod.require_land_only(train)
    fold_of_block = cv_mod.block_folds(train, block_size_m, seed=config.SEED)
    if subsample and len(train) > subsample:
        train = train.sample(n=subsample, random_state=config.SEED)

    feats = features_for(model_name)
    folds = train[cv_mod.block_column(block_size_m)].map(fold_of_block).to_numpy()
    splits = [
        (np.flatnonzero(folds != k), np.flatnonzero(folds == k))
        for k in range(config.N_SPLITS)
    ]

    search = RandomizedSearchCV(
        estimator=make_model(model_name),
        param_distributions=SEARCH_SPACES[model_name],
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=splits,
        random_state=config.SEED,
        n_jobs=1,  # the estimators already use every core
        # Only best_params_ is used; refitting would keep a forest alive (D-031).
        refit=False,
        verbose=0,
    )
    search.fit(train[feats], train[target])
    return search


# ---------------------------------------------------------------------------
# Resource logging
# ---------------------------------------------------------------------------
class PeakMemory:
    """Context manager that samples this process's resident memory and keeps the peak.

    Windows can only report a lifetime peak working set, which cannot be reset
    between models, so a sampling thread is used instead. Spikes shorter than
    `interval` seconds can be missed; at 0.05 s that is far below a tree fit.
    """

    def __init__(self, interval: float = 0.05):
        import psutil

        self._proc = psutil.Process()
        self._interval = interval
        self._stop = threading.Event()
        self.start_mb = self.peak_mb = 0.0

    def _rss_mb(self) -> float:
        return self._proc.memory_info().rss / 2**20

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.peak_mb = max(self.peak_mb, self._rss_mb())

    def __enter__(self) -> "PeakMemory":
        self.start_mb = self.peak_mb = self._rss_mb()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_mb = max(self.peak_mb, self._rss_mb())
