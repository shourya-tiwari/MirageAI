# Migration

How to move MirageAI to a new machine and prove the move changed nothing. Written
from the 2026-10-16 migration to an i7-13650HX (14 cores / 20 threads), 16 GB
RAM, RTX 5060, Windows 11.

## 1. What to copy

Copy the whole repo **including the gitignored data**. Without `data/raw/`
the clean table can only be rebuilt by re-extracting from Earth Engine, and
the scene catalogue may have changed since.

| Path | Needed for |
|---|---|
| `data/raw/uhi_pune_{2023,2024,2025}.csv` | rebuilding the clean table |
| `data/processed/uhi_pune_clean.parquet` | every Phase 3+ step |
| `outputs/` | the stored numbers the checks below compare against |

Do **not** reuse a copied `.venv/`. A virtual environment hard-codes absolute
paths, so a copied one may import but is not a verified environment. Move it out of the repo (not deleted until the checks pass) and
build a fresh one.

## 2. Environment

From the repo root:

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS / Linux: source .venv/bin/activate)
python -m pip install --upgrade pip
pip install -r requirements-lock.txt   # the exact verified environment
pip install -e .
earthengine authenticate               # only if Phase 1 will be re-run
```

- The lock pins every package; `requirements.txt` (direct dependencies only,
  D-055) is the file to edit, and the lock is regenerated from the environment
  it produces.
- After install, `pip check` must report no broken requirements, and
  `pip freeze --exclude-editable` must match `requirements-lock.txt`.

## 3. Verification (all must pass before any new result is trusted)

1. `python -m pytest -q` → every test passes (27 at the 2026-10-16
   migration; 114 as of the final polish, with `outputs/` and `data/` present).
2. Rebuild the clean table **in memory** and compare it with the stored
   parquet using `pd.testing.assert_frame_equal(..., check_exact=True)`.
   Chain `load_all → add_cell_id → add_blocks → add_lst_anom →
   add_urban_flag` yourself. Do not call `build_clean_table()`, which overwrites
   the stored file you are comparing against.
3. Recompute `phase2_lst_anom_correlations.csv` from the clean table: max
   absolute difference must be 0.0 at the stored 4 dp.
4. `preprocess.vif_table` on `FEATURES` minus `frac_built` → median 2.2;
   minus `frac_bare` → median 11.8 (D-023).
5. Refit LightGBM with a stored parameter set on one stored fold and compare
   the predictions with the stored out-of-fold predictions for that fold. On
   this migration that meant `best_params_provisional.json` against
   `outputs/tables/_oof_parts_provisional/random__lgbm.parquet`. For a future
   move, compare against `best_params.json` and `oof_predictions.parquet`.

### Result of the 2026-10-16 migration

| Check | Old machine | New machine |
|---|---|---|
| pytest | 27 passed | 27 passed |
| clean table (245,949 x 28) | stored | identical (exact) |
| correlation table | stored | max abs diff 0.0 |
| VIF median, drop `frac_built` / `frac_bare` | 2.2 / 11.8 | 2.2 / 11.8 |
| LightGBM, random CV fold 0, provisional params, RMSE | 1.614075 | 1.614075 |
| same, max abs prediction difference | | 0 (float32) |

Bit-identical tree predictions across different CPUs are expected here
because every model runs on the CPU with a fixed seed. The GPU is
deliberately left unused (see D-031): GPU histogram construction in
XGBoost/LightGBM is not bitwise reproducible against the CPU path.
