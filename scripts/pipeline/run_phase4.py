"""One-command driver for Phase 4 (T4.1-T4.8): tune, evaluate every model under
every CV scheme, fit the final models, and log runtime and peak memory.

    python scripts/pipeline/run_phase4.py                     # full run
    python scripts/pipeline/run_phase4.py --range-check-only  # D-034 count + residual-range gate
    python scripts/pipeline/run_phase4.py --skip-tune         # reuse outputs/models/best_params.json
    python scripts/pipeline/run_phase4.py --finish-from-parts # after an interrupted run: merge the
                                                         # saved OOF parts and fit the final models
    python scripts/pipeline/run_phase4.py --force             # full run that discards saved OOF parts

A full run refuses to start while OOF parts from an earlier run are on disk,
so an interrupted run cannot be wiped by mistake.

Data: water-majority cells are excluded (D-034). Validation: spatial and
spatio-temporal CV deal blocks to folds at random, repeated over
config.CV_SEEDS; random CV is repeated over the same seeds. The unshuffled
4 km layout, in which every fold is an east-west strip, runs once as a stress
test (D-033). Before tuning, the residual autocorrelation range is re-measured
with the D-028 method and the run stops if it no longer supports the primary
block size.

Memory rule (D-031): parallelism lives inside each model (n_jobs=-1) and the
search runs one fit at a time, so at most one fitted forest exists at any
moment. Out-of-fold predictions are streamed to per-run parquet parts and
combined at the end rather than held in memory. Everything runs on the CPU.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import shutil
import sys
import time

import joblib
import numpy as np
import pandas as pd

from mirageai import config as c
from mirageai import cv, evaluate as E, models as M
from mirageai import preprocess as pp

TUNED = ("lgbm", "xgb", "rf")

ap = argparse.ArgumentParser()
ap.add_argument("--skip-tune", action="store_true")
ap.add_argument("--range-check-only", action="store_true")
ap.add_argument("--finish-from-parts", action="store_true")
ap.add_argument("--force", action="store_true", help="delete saved OOF parts and start over")
args = ap.parse_args()

parts_dir = c.TABLES / "_oof_parts"
if not (args.finish_from_parts or args.range_check_only):
    try:
        M.prepare_parts_dir(parts_dir, force=args.force)
    except FileExistsError as err:
        sys.exit(f"refusing to start: {err}")

# Keep Windows from sleeping mid-run; released automatically when the process exits.
if sys.platform == "win32":
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

t_start = time.perf_counter()
c.ensure_dirs()

# ---------------------------------------------------------------------------
# D-034 Water-majority cells leave the modelling table
# ---------------------------------------------------------------------------
full = pp.load_clean()
water = pp.is_water(full)
excl = pd.DataFrame({
    "year": list(c.YEARS) + ["all"],
    "rows_before": [int((full.year == y).sum()) for y in c.YEARS] + [len(full)],
    "rows_removed": [int((water & (full.year == y)).sum()) for y in c.YEARS] + [int(water.sum())],
})
excl["rows_after"] = excl.rows_before - excl.rows_removed
excl["cells_removed"] = full.loc[water, "cell_id"].nunique()
excl["urban_rows_removed"] = [int((water & full.is_urban & (full.year == y)).sum())
                              for y in c.YEARS] + [int((water & full.is_urban).sum())]
med_all = full.groupby("year")[c.TARGET].median()
med_land = full[~water].groupby("year")[c.TARGET].median()
excl["median_LST_shift_if_land_only"] = [float(med_land[y] - med_all[y]) for y in c.YEARS] + [np.nan]
excl.to_csv(c.table("phase4_water_exclusion"), index=False)
print(f"D-034: removed {int(water.sum()):,} of {len(full):,} rows "
      f"({full.loc[water, 'cell_id'].nunique():,} cells x {len(c.YEARS)} years)", flush=True)
print(excl.to_string(index=False), flush=True)
# The strip stress test takes its layout from the full grid, water included (D-033).
strip_layout = full[[cv.block_column(c.PRIMARY_BLOCK_SIZE_M)]].copy()
del full, water
gc.collect()

df = pp.load_model_table()
cv.require_land_only(df)

# ---------------------------------------------------------------------------
# D-028 re-check: residual range on the land-only table, before any tuning
# ---------------------------------------------------------------------------
# Same protocol as D-028: LightGBM defaults, out-of-fold under 2 km spatial CV,
# per-year residual variogram. Only the fold layout (seed-42 shuffled) and the
# table (D-034) differ.
rc_splits = cv.spatial_splits(df, 2000, seed=c.SEED)
rc_oof, _ = M.oof_predictions(df, "lgbm", rc_splits, params={})
rc_resid = np.empty(len(df))
rc_resid[rc_oof.row_id.to_numpy()] = (rc_oof.y_true - rc_oof.y_pred).to_numpy()
ranges = E.residual_ranges(df, rc_resid)
ranges["target_var"] = df.groupby("year")[c.TARGET_ANOM].var().to_numpy()
ranges.to_csv(c.table("phase4_residual_range"), index=False)
central = float(ranges["range_95_m"].median())
rung = E.block_rung(central)
print(f"\nResidual range (95% of sill), per year:\n{ranges.round(3).to_string(index=False)}")
print(f"median {central:,.0f} m, mean {ranges.range_95_m.mean():,.0f} m, "
      f"max {ranges.range_95_m.max():,.0f} m -> rung {rung} m "
      f"(max -> {E.block_rung(ranges.range_95_m.max())} m)", flush=True)
del rc_oof
if rung != c.PRIMARY_BLOCK_SIZE_M:
    print(f"STOP: the residual range no longer supports the {c.PRIMARY_BLOCK_SIZE_M} m "
          "primary block (D-028). Decide the block size before tuning.", flush=True)
    sys.exit(2)
if args.range_check_only:
    sys.exit(0)

resources = []  # one row per (stage, scheme, seed, model)
params_path = c.MODELS / "best_params.json"

# ---------------------------------------------------------------------------
# T4.6 Tuning inside spatial CV (D-029), seed-42 shuffled 4 km layout (D-033)
# ---------------------------------------------------------------------------
if not (args.skip_tune or args.finish_from_parts):
    best = {}
    for name in TUNED:
        with M.PeakMemory() as mem:
            t0 = time.perf_counter()
            s = M.tune(df, name)
            el = time.perf_counter() - t0
        best[name] = {
            "best_params": s.best_params_,
            "best_cv_rmse": float(-s.best_score_),
            "best_cv_rmse_std": float(s.cv_results_["std_test_score"][s.best_index_]),
            "tune_seconds": round(el, 1),
            "n_iter": M.TUNE_N_ITER,
            "subsample": M.TUNE_SUBSAMPLE,
            "block_size_m": c.PRIMARY_BLOCK_SIZE_M,
            "fold_layout_seed": c.SEED,
            "water_excluded": True,
            "search_space": M.SEARCH_SPACES[name],
        }
        pd.DataFrame(s.cv_results_).drop(columns="params").to_csv(
            c.table(f"phase4_tuning_{name}"), index=False
        )
        resources.append({"stage": "tune", "scheme": f"spatial_{c.PRIMARY_BLOCK_SIZE_M // 1000}km",
                          "seed": c.SEED, "model": name, "seconds": round(el, 1),
                          "start_rss_mb": round(mem.start_mb), "peak_rss_mb": round(mem.peak_mb)})
        params_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
        print(f"tune {name}: RMSE {-s.best_score_:.4f} in {el / 60:.1f} min, "
              f"peak {mem.peak_mb:.0f} MB -> {s.best_params_}", flush=True)
        del s
        gc.collect()

best = json.loads(params_path.read_text(encoding="utf-8"))
PARAMS = {name: best.get(name, {}).get("best_params", {}) for name in M.MODEL_NAMES}

# ---------------------------------------------------------------------------
# T4.1-T4.5 Out-of-fold evaluation under every scheme x seed
# ---------------------------------------------------------------------------
PRIMARY = c.PRIMARY_BLOCK_SIZE_M
RUNS: list[tuple[str, int, int, callable]] = []  # (scheme, seed, block_m, make_splits)
for seed in c.CV_SEEDS:
    RUNS.append(("random", seed, -1, lambda s=seed: cv.random_splits(df, seed=s)))
for size in c.BLOCK_SIZES_M:
    for seed in c.CV_SEEDS:
        RUNS.append((f"spatial_{size // 1000}km", seed, size,
                     lambda z=size, s=seed: cv.spatial_splits(df, z, seed=s)))
for seed in c.CV_SEEDS:
    RUNS.append((f"spatio_temporal_{PRIMARY // 1000}km", seed, PRIMARY,
                 lambda s=seed: cv.spatio_temporal_splits(df, PRIMARY, seed=s)))
# Stress test: the original unshuffled full-grid layout, each fold an
# east-west strip (D-033).
RUNS.append((E.PRIMARY_SCHEME + E.STRIP_SUFFIX, E.STRIP_SEED, PRIMARY,
             lambda: cv.strip_splits(df, strip_layout, PRIMARY)))
RUNS.append((E.SPATIO_TEMPORAL_SCHEME + E.STRIP_SUFFIX, E.STRIP_SEED, PRIMARY,
             lambda: cv.strip_splits(df, strip_layout, PRIMARY, temporal=True)))

expected = sorted(f"{s}__s{seed}__{n}.parquet" for s, seed, _, _ in RUNS for n in M.MODEL_NAMES)
if args.finish_from_parts:
    # Resume: every scheme x seed x model part must already be on disk.
    missing = set(expected) - {p.name for p in parts_dir.glob("*.parquet")}
    if missing:
        sys.exit(f"cannot finish: {len(missing)} OOF parts missing, e.g. {sorted(missing)[0]}")
    print(f"resuming from {len(expected)} saved OOF parts", flush=True)
    resources = pd.read_csv(c.table("phase4_runtimes")).to_dict("records")
    RUNS_TO_DO = []
else:
    RUNS_TO_DO = RUNS  # parts_dir was emptied at startup

metric_rows, balance_rows = [], []
t_oof = time.perf_counter()
for i, (scheme, seed, block_m, make_splits) in enumerate(RUNS_TO_DO, 1):
    splits = make_splits()
    br = cv.fold_report(df, splits, scheme, None if block_m < 0 else block_m)
    br.insert(1, "seed", seed)
    balance_rows.append(br)
    for name in M.MODEL_NAMES:
        with M.PeakMemory() as mem:
            oof, secs = M.oof_predictions(df, name, splits, PARAMS.get(name))
        oof["scheme"] = scheme
        oof["seed"] = seed
        oof["block_size_m"] = block_m

        # Narrow dtypes before anything touches the disk or the metrics pass.
        oof = oof.astype(
            {
                "row_id": "int32", "cell_id": "int64", "year": "int16",
                "fold": "int8", "seed": "int16", "block_size_m": "int32",
                "y_true": "float32", "y_pred": "float32",
            }
        )
        oof.to_parquet(parts_dir / f"{scheme}__s{seed}__{name}.parquet", index=False)

        fm = M.fold_metrics(oof)
        fm.insert(0, "scheme", scheme)
        fm.insert(1, "seed", seed)
        fm.insert(2, "block_size_m", None if block_m < 0 else block_m)
        fm.insert(3, "model", name)
        metric_rows.append(fm)
        resources.append({"stage": "oof", "scheme": scheme, "seed": seed, "model": name,
                          "seconds": round(secs, 1), "start_rss_mb": round(mem.start_mb),
                          "peak_rss_mb": round(mem.peak_mb)})
        print(
            f"[{i}/{len(RUNS)}] {scheme:<26} seed {seed:>3} {name:<9} "
            f"RMSE {fm.RMSE.mean():.4f} +/- {fm.RMSE.std():.4f}  "
            f"R2 {fm.R2.mean():.4f}  [{secs:.1f}s, peak {mem.peak_mb:.0f} MB]",
            flush=True,
        )
        del oof
        gc.collect()
    # Checkpoint so a crash mid-run still leaves every finished result on disk.
    pd.concat(metric_rows, ignore_index=True).to_csv(c.table("phase4_fold_metrics"), index=False)
    pd.concat(balance_rows, ignore_index=True).to_csv(c.table("phase4_fold_balance"), index=False)
    pd.DataFrame(resources).to_csv(c.table("phase4_runtimes"), index=False)
    done = time.perf_counter() - t_oof
    print(f"    elapsed {done / 60:.1f} min, eta {done / i * (len(RUNS) - i) / 60:.0f} min",
          flush=True)

# Stream the parts into one parquet, one part in memory at a time; the row
# count of the merged file is checked against the sum of the parts.
with M.PeakMemory() as mem:
    t0 = time.perf_counter()
    merged = M.merge_parts([parts_dir / n for n in expected], c.TABLES / "oof_predictions.parquet")
    el = time.perf_counter() - t0
resources.append({"stage": "merge", "scheme": "all", "seed": -1, "model": "all",
                  "seconds": round(el, 1), "start_rss_mb": round(mem.start_mb),
                  "peak_rss_mb": round(mem.peak_mb)})
pd.DataFrame(resources).to_csv(c.table("phase4_runtimes"), index=False)
print(f"wrote oof_predictions.parquet: {merged['n_parts']} parts, "
      f"{merged['rows_written']:,} rows (parts hold {merged['rows_in_parts']:,})  "
      f"[{el:.1f}s, start {mem.start_mb:.0f} MB, peak {mem.peak_mb:.0f} MB]", flush=True)

# ---------------------------------------------------------------------------
# T4.7 Final models, fitted on training years only so 2025 stays unseen
# ---------------------------------------------------------------------------
train = df[df.year.isin(c.TRAIN_YEARS)]
for name in M.MODEL_NAMES:
    feats = M.features_for(name)
    est = M.make_model(name, **PARAMS.get(name, {}))
    with M.PeakMemory() as mem:
        t0 = time.perf_counter()
        est.fit(train[feats], train[c.TARGET_ANOM])
        el = time.perf_counter() - t0
    joblib.dump(
        {"model": est, "features": feats, "params": PARAMS.get(name, {}),
         "target": c.TARGET_ANOM, "fitted_on": "train_years", "water_excluded": True},
        c.MODELS / f"{name}_trainyears.joblib",
        compress=3,
    )
    resources.append({"stage": "final_fit", "scheme": "train_years", "seed": c.SEED,
                      "model": name, "seconds": round(el, 1),
                      "start_rss_mb": round(mem.start_mb), "peak_rss_mb": round(mem.peak_mb)})
    print(f"saved {name}_trainyears.joblib  [{el:.1f}s, peak {mem.peak_mb:.0f} MB]", flush=True)
    del est
    gc.collect()

pd.DataFrame(resources).to_csv(c.table("phase4_runtimes"), index=False)
# Parts are removed only once the merged file and every final model exist.
shutil.rmtree(parts_dir)
print(f"DONE in {(time.perf_counter() - t_start) / 60:.1f} min")
