# MirageAI: Project Deep Dive

A complete technical guide to MirageAI, written so that someone who has never
seen the repository could understand every component and rebuild the project.
For each major component it answers five questions:

* **What** it is and what it produces
* **Why** it exists and why it is done this way (with the decision number from `docs/DECISIONS.md`)
* **How** it works, down to the functions involved
* **When** it runs in the pipeline
* **Where** it lives in the repository

Numbers quoted here come from `outputs/tables/`. Decision ids (D-0xx) point to
`docs/DECISIONS.md`, where each choice is argued in full.

---

## Contents

1. [The project in one page](#1-the-project-in-one-page)
2. [Domain background](#2-domain-background)
3. [Architecture](#3-architecture)
4. [Configuration: `config.py`](#4-configuration-configpy)
5. [Phase 1: Data extraction](#5-phase-1-data-extraction)
6. [Phase 2: Preprocessing and the modelling table](#6-phase-2-preprocessing-and-the-modelling-table)
7. [Phase 3: Validation design](#7-phase-3-validation-design)
8. [Phase 4: Modelling and tuning](#8-phase-4-modelling-and-tuning)
9. [Phase 5: Evaluation](#9-phase-5-evaluation)
10. [Phase 6: Explainability with SHAP](#10-phase-6-explainability-with-shap)
11. [Phase 7: Greening scenarios](#11-phase-7-greening-scenarios)
12. [Phase 8: Report, slides and number checks](#12-phase-8-report-slides-and-number-checks)
13. [Results layer and dashboard](#13-results-layer-and-dashboard)
14. [Testing strategy](#14-testing-strategy)
15. [Reproducibility engineering](#15-reproducibility-engineering)
16. [Design decisions and trade-offs](#16-design-decisions-and-trade-offs)
17. [Results](#17-results)
18. [Rebuilding the project from scratch](#18-rebuilding-the-project-from-scratch)
19. [Glossary](#19-glossary)

---

## 1. The project in one page

**Goal.** Explain what drives daytime land surface temperature (LST) across
Pune, and estimate how much cooler the hottest urban neighbourhoods would be
with more tree cover.

**Setting.** A ~32 × 26 km box over the Pune Municipal Corporation and
Pimpri-Chinchwad core, gridded at 100 m (81,983 cells), observed in the
pre-monsoon season (March–May) of 2023, 2024 and 2025.

**Inputs.** Thirteen features per cell and year:

* three Sentinel-2 spectral indices: NDVI (greenness), NDBI (dryness here), MNDWI (surface moisture);
* seven ESA WorldCover land-cover fractions: tree, shrub, grass, crop, built-up, bare, water;
* three terrain/water variables: elevation, slope, distance to water.

**Target.** `LST_anom`: the cell's median pre-monsoon LST minus that year's
city-wide median, in °C.

**Method.** Five regressors (mean baseline, linear regression, random forest,
XGBoost, LightGBM), tuned and evaluated under **spatial block cross-validation**
so that test areas are geographically separate from training areas; a
**spatio-temporal** test on unseen blocks in the unseen year 2025; **SHAP** to
attribute predictions to features; and **counterfactual scenarios** that add tree
cover to the hottest urban cells.

**Headline findings.**

1. Pune has a daytime surface urban *cool* island: the core is 3.4–4.0 °C cooler
   than the outer ring, because the dry, fallow periphery heats faster than the
   greener, irrigated city.
2. The tree models predict unseen places to ~1.75 °C RMSE (R² ≈ 0.73), and
   random CV would have overstated that by 0.12–0.15 °C.
3. Greenness, dryness and moisture dominate; the built-up fraction is a minor
   driver once they are accounted for.
4. +10 points of tree cover in the hottest urban cells is associated with
   0.37–1.02 °C lower modelled LST.

---

## 2. Domain background

### 2.1 Land surface temperature (LST)

**What.** The radiative "skin" temperature of the ground: roofs, roads, canopy,
soil. Landsat 8/9 estimate it from thermal infrared (band 10) at about 10:30 AM
local time.

**Why it is the target (D-047).** It is the only temperature measured
wall-to-wall at neighbourhood scale (~100 m). Air temperature would need weather
stations (Pune has only a handful) or reanalysis grids (~9 km). LST is the
standard measure of the *surface* urban heat island.

**Limit.** LST is not what people feel. Air temperature and comfort also depend
on humidity, wind and shade, and peak in the afternoon. Every result is about
daytime surface temperature.

### 2.2 Urban heat island, and why Pune's is inverted

A UHI is a city warmer than its surroundings. In temperate cities the daytime
surface UHI is usually positive. In semi-arid India before the monsoon, the
countryside is harvested, bare and dry, with almost no evapotranspiration, so it
heats faster than an irrigated, tree-shaded city. The result is a daytime
surface **cool** island. MirageAI found exactly this: the core (within 5 km of
the centre) is 3.4–4.0 °C cooler than the outer ring (beyond 12 km) in every
year (`phase8_core_ring_land.csv`). The project is named after it.

### 2.3 Spectral indices

All three are normalised differences of Sentinel-2 bands, ranging −1 to +1:

| Index | Formula | Usual meaning | Meaning in this project |
|---|---|---|---|
| NDVI | (NIR − Red) / (NIR + Red) = (B8 − B4)/(B8 + B4) | vegetation vigour | greenness |
| NDBI | (SWIR − NIR) / (SWIR + NIR) = (B11 − B8)/(B11 + B8) | built-up | **dryness / bare soil** (D-024) |
| MNDWI | (Green − SWIR) / (Green + SWIR) = (B3 − B11)/(B3 + B11) | open water | **surface moisture** (D-038) |

NDBI's reinterpretation matters: it correlates +0.63 with LST while the actual
built-up fraction correlates −0.19. Bare, dry soil has high SWIR reflectance, the
same signature NDBI was designed to detect in roofs.

### 2.4 Land-cover fractions

ESA WorldCover classifies every 10 m pixel. Averaging the binary class masks
onto the 100 m grid gives, per cell, the share of each class. The seven shares
sum to 1. WorldCover is a 2021 snapshot (D-006), so the fractions are identical
in all three years; only LST and the indices change between years.

---

## 3. Architecture

### 3.1 Layers

```
config.py                          constants, schema, paths, seeds (single source of truth)
   │
src/mirageai/*.py                  pure functions and small classes, one module per phase
   │
scripts/pipeline/run_phase{2..7}.py   drivers: load → call modules → write tables, figures, models
   │
outputs/{tables,figures,models}    every result, as CSV / parquet / PNG / joblib
   │
scripts/ → reports/                report, slides, viva notes; numbers checked against tables
results.py → app/, notebooks/      read-only views of the outputs
```

* **Modules** (`src/mirageai/`) contain the logic and are unit-tested.
* **Drivers** (`scripts/pipeline/run_phase*.py`) are scripts, top to bottom, that orchestrate a
  phase and write its outputs. They contain no reusable logic beyond plotting.
* **`scripts/pipeline/run_all.py`** runs the drivers and the report/slide builds in order, each in
  its own process, stopping at the first failure (D-048).
* **Views** (dashboard, notebooks) only read outputs.

### 3.2 Data flow

| Step | Reads | Writes |
|---|---|---|
| Phase 1 (`gee_extract.extract_year`) | Earth Engine | `data/raw/uhi_pune_{year}.csv`, `uhi_pune_stack_{year}.tif` |
| `scripts/pipeline/run_phase2.py` | `data/raw/*.csv` | `data/processed/uhi_pune_clean.parquet`, `phase1_*`, `phase2_*`, figures `01_*`, `02_*` |
| `scripts/pipeline/run_phase3.py` | clean table | `phase3_fold_balance.csv`, figures `03_*` |
| `scripts/pipeline/run_phase4.py` | clean table | `best_params.json`, `phase4_*`, `oof_predictions.parquet`, models `*_trainyears.joblib` |
| `scripts/pipeline/run_phase5.py` | Phase 4 outputs | `model_comparison.csv`, `phase5_*`, figures `05_*` |
| `scripts/pipeline/run_phase6.py` | clean table, `best_params.json` | `lgbm_all.joblib`, `xgb_all.joblib`, `phase6_*` (incl. per-cell SHAP), figures `06_*` |
| `scripts/pipeline/run_phase7.py` | clean table, Phase 6 models and thresholds | `rf_all.joblib`, `phase7_*`, figures `07_*` |
| `scripts/make_report_figures.py`, `make_slide_figures.py` | tables | `phase8_*`, figures `08_*`, `09_*` |
| `scripts/build_report.py`, `build_slides.py` | `reports/report.md`, tables | `reports/MirageAI_Report.*`, `MirageAI_Slides.*` |

### 3.3 Naming conventions

* Figures: `outputs/figures/<phase>_<name>.png`, 300 dpi (`config.figure`).
* Tables: `outputs/tables/<name>.csv` (`config.table`).
* Models: `outputs/models/<model>_<fitted_on>.joblib`, each a dict with `model`,
  `features`, `params`, `target` and provenance fields.
* Suffixes: `_strips` for the latitude-strip stress test, `_provisional` for
  results superseded but kept for comparison.

---

## 4. Configuration: `config.py`

**What.** One module of constants: study area, CRS, grid size, years, Earth
Engine collections and masking rules, feature lists, validation settings,
thresholds, seeds and paths.

**Why (D-014).** The project's central claim depends on rules such as
"coordinates are never features". Declaring the schema once makes those rules
machine-checkable: `config.py` asserts at import that `frac_built` stays in the
tree feature set (D-025), and the tests assert that no forbidden column is a
feature.

**How (key constants).**

| Constant | Value | Meaning |
|---|---|---|
| `AOI_BOUNDS` | (73.72, 18.42, 74.02, 18.65) | WGS84 bounding box |
| `CRS`, `SCALE` | EPSG:32643, 100 | UTM 43N, 100 m grid (D-001) |
| `YEARS`, `TRAIN_YEARS`, `TEST_YEAR` | 2023–25, 2023–24, 2025 | temporal design |
| `FEATURES` | 13 names | spectral + fractions + terrain |
| `FEATURES_LINEAR` | 12 names | drops `frac_built` (D-023, D-025) |
| `FORBIDDEN_FEATURES` | coordinates, QC, target, ids, block columns | never model inputs |
| `TARGET_ANOM` | `LST_anom` | modelling target (D-021) |
| `BLOCK_SIZES_M`, `PRIMARY_BLOCK_SIZE_M` | 1, 2, 4, 8 km; 4 km | CV blocks (D-028) |
| `CV_SEEDS` | 42–46 | five block layouts (D-033) |
| `URBAN_BUILT_THRESHOLD` | 0.5 | urban cell (D-019) |
| `WATER_EXCLUDE_THRESHOLD` | 0.5 | water-majority cells excluded (D-034) |
| `HOT_DECILE` | 0.9 | "hottest decile" quantile |
| `SEED` | 42 | global seed |
| `GEE_PROJECT` | env `MIRAGEAI_GEE_PROJECT` or default | Earth Engine project (D-013) |

Helper functions: `season_dates(year)`, `raw_csv(year)`, `raw_tif(year)`,
`figure(phase, name)`, `table(name)`, `ensure_dirs()`.

**When / where.** Imported by every module, driver, script and test.
`src/mirageai/config.py`.

---

## 5. Phase 1: Data extraction

**What.** For each year, a 19-band GeoTIFF on the 100 m grid (LST, observation
count, 3 indices, 7 fractions, distance to water, elevation, slope, 4 coordinate
bands) and the same data as a CSV with one row per cell.

**Why.** All sources are free, official and accessible through Google Earth
Engine, which does the heavy raster work server-side. The area is small enough
(~6 MB per year) to download synchronously, so there is no manual Google Drive
step (D-020).

**How** (`src/mirageai/gee_extract.py`):

| Function | Does |
|---|---|
| `init_ee()` | Initialise Earth Engine with `config.GEE_PROJECT`; a clear error if unset |
| `aoi()` | The study rectangle as an `ee.Geometry` |
| `to_grid(img, native_scale)` | **The key aggregation:** reproject to the source's native scale, `reduceResolution(mean)`, reproject to 100 m. Composites lose their projection, so the native scale must be imposed first (REFERENCE §2) |
| `_mask_landsat`, `landsat_collection`, `landsat_lst` | Merge Landsat 8 and 9, filter `CLOUD_COVER < 30`, mask cloud/shadow/cirrus bits of `QA_PIXEL`, scale `ST_B10` to °C (`DN × 0.00341802 + 149 − 273.15`), take the per-pixel **median** (D-005) and the count of clear observations |
| `_mask_s2`, `s2_collection`, `s2_indices` | Sentinel-2 SR, `SCL` cloud/shadow/snow mask, median composite, NDVI at 10 m, NDBI and MNDWI at 20 m |
| `worldcover`, `worldcover_binary` | One 10 m binary band per class; their 100 m means are the fractions |
| `distance_to_water` | `fastDistanceTransform` on a 30 m water mask, converted from squared pixels to metres |
| `terrain` | SRTM elevation and slope, slope computed **after** reprojecting to UTM so the gradient is in metres (D-016) |
| `coordinate_bands` | lon/lat and UTM x/y of cell centres (for blocks and maps only) |
| `build_stack`, `stack_band_names` | Concatenate everything on the grid in a fixed band order |
| `stack_download_url`, `download_stack` | Unmask to the −9999 sentinel (D-017) and download the GeoTIFF |
| `stack_to_frame`, `write_year_table` | Read the GeoTIFF with rasterio, turn −9999 into NaN, drop incomplete cells, write the CSV |
| `scene_counts`, `extract_year` | Log scene counts and run the whole year |

**Facts worth knowing.** Scene counts 2023/24/25: Landsat 8/11/9, Sentinel-2
14/17/14. No cell was dropped in any year: the minimum mean clear-observation
count is 4.35 (2023), well above `MIN_LST_OBS = 3`. `LST_n_obs` is fractional
because it is averaged from 30 m pixels (D-018) and is QC only.

**When / where.** Run once, interactively (notebook 01 or the one-liner in the
README). `scripts/pipeline/run_all.py` starts from `data/raw/`.

---

## 6. Phase 2: Preprocessing and the modelling table

**What.** `data/processed/uhi_pune_clean.parquet`: 245,949 rows (81,983 cells × 3
years) × 28 columns, zero nulls.

**How** (`src/mirageai/preprocess.py`), in the order `build_clean_table` applies them:

1. **`load_all`** concatenates the three yearly CSVs with a `year` column.
2. **`add_cell_id`**: `floor(x/100) × 10⁶ + floor(y/100)`. *Why floor:* cell
   centres sit at half-cell offsets (x = k·100 + 50), so x/100 ends in .5 and
   numpy's round-half-to-even merged neighbouring columns. The consistency test
   caught 7,456 merged cells on its first run (D-022).
3. **`add_blocks`**: the same formula at 1, 2, 4 and 8 km. Ids come from
   coordinates only, so a cell has the same block in every year and its three
   yearly rows can never be split across train and test.
4. **`add_lst_anom`**: `LST_anom = LST − median LST of that year`. *Why (D-021):*
   2023 runs ~1.7 °C hotter than 2024 at city level, and about four fifths of
   that is an artefact of which scene dates each year happened to contain.
   Nothing in the features encodes the year, so the level is unlearnable noise
   and would contaminate the 2025 test. The anomaly keeps the within-year
   spatial contrast, which is what the drivers explain.
5. **`add_urban_flag`**: `is_urban = frac_built ≥ 0.5` (D-019).
6. **`check_block_consistency`** asserts one block per cell per size.

Other functions:

* `missing_report` — per-year rows, nulls, cells below the observation floor.
* `vif_table` — variance inflation factors with an intercept. With all seven
  fractions the design is singular (they sum to 1). Dropping `frac_built` gives
  median VIF 2.2; dropping the rare `frac_bare` leaves 11.8, so `frac_built` is
  the linear model's reference class (D-023). Only linear regression drops it;
  trees keep all seven so SHAP can attribute a built-up effect (D-025).
* `is_water`, `load_model_table` — the **modelling table**: the clean table minus
  water-majority cells (`frac_water ≥ 0.5`): 899 cells × 3 = 2,697 rows removed,
  243,252 left. Reservoir temperature is thermal inertia, not a land-surface
  driver, and was the model's largest error (D-034). The index is reset, and
  that index is what every out-of-fold `row_id` refers to.
* `core_ring_contrast` — the cool-island numbers.

**EDA** (`scripts/pipeline/run_phase2.py`): LST and NDVI maps, target and feature distributions,
the correlation heatmap, VIF, and correlations with the target for all cells vs
urban cells. Within urban cells many correlations collapse, partly through
**range restriction** (the predictor barely varies inside the subset), which is
why RQ5 is answered by the multivariate model, not by subset correlations.

**When / where.** `scripts/pipeline/run_phase2.py`, first step of `scripts/pipeline/run_all.py`.

---

## 7. Phase 3: Validation design

### 7.1 The problem: spatial autocorrelation

Neighbouring cells have almost the same LST (Moran's I of the target ≈ 0.89).
In a random split, most test cells have a near-identical neighbour in training,
so the model can score well by memorising local texture. That is **leakage**:
the score measures interpolation between neighbours, not generalisation to a new
neighbourhood, which is what a planner needs (D-003).

### 7.2 Spatial block CV

Divide the map into square blocks; assign whole blocks to folds; train on four
folds' blocks, test on the fifth. Test cells are then at least partly far from
any training cell.

### 7.3 How big should the blocks be? (variogram)

**What.** The empirical semivariogram γ(h) is half the mean squared difference
between values at distance h. It rises with distance and flattens at the *sill*;
the distance where it reaches 95% of the sill is the *practical range*: beyond it,
values are roughly independent.

**How** (`cv.empirical_variogram`, `cv.practical_range`): sample 20,000 cells,
draw 5 million random pairs, bin squared differences by distance up to 10 km;
take the sill as the mean of the outer quarter of lags and report the first lag
reaching 95% of it.

**The decision (D-026 → D-028).** The raw target's range is 7.65 km, which first
suggested 8 km blocks. But most of that structure is the regional
core-to-periphery gradient, which the features explain. What block CV must
defeat is the autocorrelation the model *cannot* explain, i.e. of its
**residuals**. The residual range of a default LightGBM is ~3.7 km (3.95 km
median after water exclusion), so **4 km** is the primary block size. Every
size (1, 2, 4, 8 km) is still reported, and 8 km is the conservative bound.
`scripts/pipeline/run_phase4.py` re-checks this as a gate before tuning
(`phase4_residual_range.csv`).

### 7.4 The splitters (`src/mirageai/cv.py`)

| Function | Scheme | Detail |
|---|---|---|
| `random_splits` | Random CV | `GroupKFold` grouped on `cell_id`, so a cell's three years stay together and the inflation measured is spatial, not year-duplicate leakage |
| `spatial_splits` | Spatial block CV | `GroupKFold(shuffle=True, random_state=seed)` over the block column; `seed=None` gives the unshuffled layout |
| `spatio_temporal_splits` | Spatial × temporal | For each spatial fold: train on training blocks in 2023–24, test on held-out blocks in 2025 (D-010 amended) |
| `strip_splits` | Stress test | The original unshuffled layout of the full grid, applied to land rows |
| `block_folds` | helper | block-id → fold map, so a subsample (the tuning sample) reuses the full table's layout exactly |
| `require_land_only` | guard | every splitter refuses water rows |
| `fold_report` | diagnostics | per-fold sizes, years and urban share |

**Why spatio-temporal, not purely temporal (D-010).** The same cells in 2025
have byte-identical land cover and terrain and similar indices, so a same-cells
temporal split measures memorisation of place. Phase 5 measured it: LightGBM
scores RMSE 1.598 on seen places vs 1.747 on unseen ones.

**Why five shuffled layouts (D-033).** Unshuffled `GroupKFold` deals
equal-sized groups in id order. Block ids are column-major, and at 4 km every
grid column holds exactly five full blocks, so each fold became a 4 × 32 km
east-west strip. That is not leakage, but it is an arbitrary design that moved
the headline. Shuffling with five seeds (42–46) makes layout choice negligible
(seed-to-seed std ≤ 0.011 °C) and every result becomes mean ± std over 25 runs.
The strip layout is kept as a "latitude extrapolation" stress test (+0.016 °C).

**Fold imbalance (D-027).** Pune's built-up area is concentrated, so some folds
are much more urban than others. It is reported, not engineered away:
stratifying folds by content would break the spatial logic.

**When / where.** `cv.py` is used by Phases 4–7. `scripts/pipeline/run_phase3.py` regenerates the
historical Phase 3 record (variogram, block layout, fold balance).

---

## 8. Phase 4: Modelling and tuning

### 8.1 The models (`src/mirageai/models.py`)

`make_model(name, **params)` builds an unfitted estimator with the project seed:

| Name | Estimator | Why it is in the comparison (D-007) |
|---|---|---|
| `baseline` | `DummyRegressor(strategy="mean")` | the floor; R² ≈ 0 by construction |
| `linear` | `Pipeline([StandardScaler, LinearRegression])` | the linear status quo of earlier studies; scaling happens inside each fold |
| `rf` | `RandomForestRegressor(n_jobs=-1)` | bagging: many deep trees on bootstrap samples, averaged |
| `xgb` | `XGBRegressor(tree_method="hist", device="cpu")` | gradient boosting with regularisation |
| `lgbm` | `LGBMRegressor(device_type="cpu")` | gradient boosting, leaf-wise growth, fast |

`FEATURE_SET` and `features_for(name)` give the linear models 12 features and
the trees 13. `device="cpu"` is explicit because GPU histogram construction is
not bitwise reproducible (D-031).

### 8.2 Tuning inside spatial CV (`tune`)

* `RandomizedSearchCV`, 25 candidates from `SEARCH_SPACES[name]`, scoring
  negative RMSE.
* **Folds:** the seed-42 shuffled 4 km layout of the 2023–24 table, carried into
  a 60,000-row subsample as explicit splits. Tuning on the full 164k rows would
  have taken hours for the forest (D-029).
* **Training years only:** 2025 is never seen during selection, so the
  spatio-temporal test stays clean.
* `n_jobs=1` for the search, `n_jobs=-1` inside each model, `refit=False`: at
  most one fitted forest is ever in memory (D-031).
* **Known bias, stated:** hyperparameters are chosen on the same spatial folds
  that later report the spatial score, so that score is mildly optimistic. The
  2025 spatio-temporal score is the clean check.

Selected parameters (`best_params.json`):

| Model | Parameters | Tuning RMSE |
|---|---|---|
| LightGBM | 1,200 trees, 63 leaves, lr 0.01, `min_child_samples` 10, `colsample_bytree` 0.6, λ 1 | 1.750 |
| XGBoost | 1,200 trees, depth 8, lr 0.01, `min_child_weight` 20, `colsample_bytree` 0.8, λ 1 | 1.750 |
| Random forest | 200 trees, depth 16, `min_samples_leaf` 2, `max_features` 0.3 | 1.767 |

### 8.3 Out-of-fold predictions (`oof_predictions`)

For each fold: clone the model, fit on the training indices, predict the test
indices, delete the model, and keep a long frame of `row_id, cell_id, year,
fold, model, y_true, y_pred`. Every row is predicted by a model that never saw it.

### 8.4 The Phase 4 driver (`scripts/pipeline/run_phase4.py`)

1. Load the clean table, record the water exclusion, build the model table.
2. **Gate:** re-measure the residual range (D-028); stop if 4 km is no longer the rung.
3. Tune LightGBM, XGBoost and RF (or `--skip-tune`).
4. For every model × scheme × seed: out-of-fold predictions, streamed to parquet
   parts (`prepare_parts_dir`, `merge_parts`, which refuses to overwrite saved
   parts and verifies row counts) and fold metrics.
5. Fit the final 2023–24 models (`*_trainyears.joblib`) used for the
   seen-place comparison.
6. Log wall time and peak memory per stage (`PeakMemory`, a 50 ms sampling
   thread, because Windows exposes only a lifetime peak).

Schemes: random; spatial at 1, 2, 4, 8 km; spatio-temporal at 4 km; plus the two
strip stress tests at 4 km. Five models × these schemes × five seeds gives the
~30 million out-of-fold rows in `oof_predictions.parquet`.

**When / where.** `scripts/pipeline/run_phase4.py`, about 30 minutes on 16 GB.

---

## 9. Phase 5: Evaluation

**What.** Model comparison and diagnostics, without fitting anything new.

**How** (`src/mirageai/evaluate.py`, driven by `scripts/pipeline/run_phase5.py`):

| Function | Output | Purpose |
|---|---|---|
| `load_oof` | filtered OOF rows | reads only the needed scheme/model/columns of the 0.7 GB file, adds `resid = y_true − y_pred` |
| `error_metrics` | RMSE, MAE, R², mean residual | one set of rows |
| `model_comparison` | `model_comparison.csv` | mean and std over 25 runs per scheme × model, plus seed-to-seed std |
| `paired_differences` | `phase5_paired_vs_best.csv` | model minus LightGBM on identical (seed, fold) test sets, removing between-fold variance |
| `cv_gap` | `phase5_cv_gap.csv` | each scheme's RMSE minus random CV's: the inflation (RQ3) |
| `seen_place_predictions` | `phase5_temporal_holdout.csv` | the final 2023–24 model on 2025 rows; compared with OOF, isolates place memorisation |
| `hot_decile_flags`, `top_decile_overlap` | `phase5_hot_decile*.csv` | errors in the hottest decile, and the share of the observed hottest decile also predicted there (chance = 10%) |
| `subgroup_fold_metrics`, `summarise_subgroups` | `phase5_urban_vs_rest.csv` | urban vs rest (D-019) |
| `queen_weights`, `morans_i` | `phase5_morans_i.csv` | spatial autocorrelation of residuals |
| `residual_ranges`, `block_rung` | `phase5_residual_range.csv` | the D-028 check on tuned residuals |

**Why paired differences.** Fold std (~0.06–0.11 °C) is far larger than the
differences between tree models (~0.01 °C). Pairing on identical folds asks the
right question: on the same test set, which model is better, and how often?

**Why detection, not only error, in the hottest decile.** Conditioning on the
observed extreme selects cells whose noise was positive, so any imperfect model
under-predicts there. Overlap of observed and predicted top deciles is not biased
that way.

**Moran's I.** I = (n / ΣW) · Σᵢⱼ wᵢⱼ zᵢ zⱼ / Σᵢ zᵢ², with row-standardised
queen-contiguity weights on the 100 m lattice (built as a distance band of one
diagonal step so water holes are handled). Near 0 means spatially random
residuals; MirageAI's ~0.6 means unmodelled spatial drivers.

**Reference model (D-032).** The three trees tie; LightGBM is chosen for
diagnostics and SHAP because it is lowest on the clean 2025 test and cheapest to
explain (2.8 MB vs 115 MB for the forest). The report says "tied", not "best".

---

## 10. Phase 6: Explainability with SHAP

### 10.1 Shapley values in one paragraph

For one prediction, the Shapley value of a feature is its average marginal
contribution over all orders in which features could be "revealed" to the model.
They are **additive**: base value + Σ SHAP = prediction. TreeSHAP computes them
exactly and quickly for tree ensembles.

### 10.2 What is explained (D-035)

A LightGBM with the tuned parameters refitted on **all 243,252 land rows of all
three years** (`lgbm_all.joblib`); XGBoost the same way for a robustness check.
Phase 5 is closed, so the accuracy claim stays the cross-validated one; the
explanation should describe all the data.

### 10.3 Interventional SHAP and its engineering (D-036)

* **Why interventional.** The features are correlated (fractions sum to one;
  NDVI–NDBI r = −0.72). Path-dependent SHAP conditions on the tree's own split
  path and leaks credit between correlated features. Interventional SHAP
  replaces features with values from a background sample, crediting a feature
  only through the model's response to it. It changes the ranking: NDVI first
  (0.95 °C) under interventional, NDBI first under path-dependent.
* **Background and sample.** 1,000 background rows and 10,000 explained rows,
  disjoint, each stratified by year × urban/rest at population shares
  (`explain.stratified_sample`, largest-remainder allocation).
* **float32 exactness.** shap evaluates trees on float32 inputs, while LightGBM
  thresholds sit between float64 values, so the cast moved some rows across
  splits and additivity failed by up to 0.4 °C. `explain.float32_exact` rounds
  the features to float32 *before fitting*, so the cast is lossless; additivity
  holds to 1e-7, and `check_additivity` is never turned off.
* **Every land row.** shap's interventional loop is single-core (~40 ms/row), so
  `interventional_shap(n_jobs=8)` splits rows into chunks across processes
  (29 min for 243,252 rows) and asserts a common base value. The result is
  tested equal to the serial one.
* **Interactions.** Only path-dependent SHAP has interaction values
  (`interaction_values`), so they are used to *rank pairs* on a 2,000-row subsample.

### 10.4 Summaries (`src/mirageai/explain.py`)

| Function | Output |
|---|---|
| `importance` | mean \|SHAP\| per feature (°C) |
| `grouped`, `group_values`, `DRIVER_GROUPS` | per-row SHAP summed within groups: spectral, dry cover, tree cover, built, water proximity, terrain (D-038) |
| `rank_agreement` | Spearman/Kendall and top-k overlap of two rankings (LightGBM vs XGBoost, all vs urban, method vs method) |
| `top_interactions`, `interaction_matrix` | pair totals \|φᵢⱼ + φⱼᵢ\| |
| `breakpoint`, `_hinge_fit` | one-knot segmented regression of SHAP on feature value with a 200-resample bootstrap CI; labels near-linear / saturates / onset / reverses / changes slope (D-037) |

**Why groups (D-038).** Inside the spectral block, NDVI, NDBI and MNDWI swap
credit between models and SHAP methods: the per-cell dominant *feature* agrees
between LightGBM and XGBoost on 79% of rows, the dominant *group* on 95%. The
driver map therefore shows groups.

**Why fitted thresholds.** A knot with a CI can be checked and reproduced;
eyeballing a dependence plot cannot. For zero-inflated fractions (36–98% zeros)
the jump from none to some is reported separately, since a "knot at 1%" is just
the presence of any cover.

### 10.5 Key outputs

* Spectral block 1.49 °C mean |SHAP|, then dry cover 0.72, tree cover 0.38;
  built-up 0.12 (10th of 13 features).
* Dominant group per cell: spectral 66.8%, dry cover 24.5% (42.7% of urban cells).
* Tree cover: knot at 10.2% (CI 9.6–10.6%); cooling per point flattens above it.
* RQ5: built-up fraction's association falls from −0.22 °C per +10 points (raw)
  to −0.029 in the model, an 87% shrink, still negative.
* RQ2: linear regression gives NDBI a negative coefficient, the opposite of the
  trees' response, because collinear indices trade coefficients.

---

## 11. Phase 7: Greening scenarios

**What.** The change in predicted `LST_anom` (ΔLST) when tree cover is added to
the hottest urban cells, under two bounds and three models.

**How** (`src/mirageai/scenarios.py`, driven by `scripts/pipeline/run_phase7.py`):

1. **Targets (`target_cells`, D-039).** Mean *observed* anomaly over 2023–25 per
   cell; among urban cells (≥ 50% built-up), keep the top decile: **2,673 cells**
   (threshold +2.41 °C). Observed rather than predicted, so targeting does not
   inherit the model's 54% detection rate. They sit on the eastern fringe;
   median built-up 78%, median tree cover 0%.
2. **Edits (`add_tree_cover`, `scenario_delta`, D-040).** A: +10 points; B: +20;
   C: raise to the 10.2% knot. New cover is taken from built-up, bare and grass
   in proportion to their shares, so fractions still sum to one (asserted).
   Shortfalls are reported (none occurred).
3. **Two bounds (D-009).**
   * *Lower:* fractions change, indices stay as observed. Understates the
     effect, since more trees with unchanged greenness is not physical.
   * *Realistic:* indices also move, via per-year LightGBM models from land cover
     to each index (`fit_index_models`, spatially validated, R² 0.61–0.72).
     `updated_indices` applies only the *difference* f(new) − f(old), so each
     cell keeps its own residual (irrigation, shade) (D-041).
4. **Three models (D-042).** LightGBM, XGBoost and RF, all refitted on every land
   row; the spread is reported as the model uncertainty.
5. **Support guard (`SupportGuard`, D-043).** Distance to the 10th nearest
   training row in standardised feature space; rows beyond the 99th percentile of
   training rows' own scores are flagged. ≤ 0.04% of scenario rows were flagged.
6. **`delta_lst`** = prediction(scenario) − prediction(observed), on float32-exact
   features.

**The first-point decision.** 78.7% of target cells have zero tree cover. The
model's response to the first point (0 → 1%) is large (−1.43 °C realistic): it
reflects what tree-less cells *are* (dense new development whose roofs and
paving the features cannot see), not what a tree does. The headline therefore
**excludes the first point**: A −0.37 to −1.02 °C, B −0.56 to −1.63 °C, C −0.35
to −0.97 °C (lower to realistic). The step is reported separately.

**Caveat (always attached).** ΔLST is a change in a model's prediction learned
from cross-sectional differences between cells, not a forecast of what planting
would do.

---

## 12. Phase 8: Report, slides and number checks

**What.** `reports/report.md` → `MirageAI_Report.html` (pandoc with the IEEE
template in `reports/ieee/`) → `MirageAI_Report.pdf` (headless Chrome or Edge
prints the two-column IEEE layout, 15 pages), plus an editable `.docx` in the
same IEEE layout; `scripts/build_slides.py` → `MirageAI_Slides.pptx/.pdf` (python-pptx,
19 slides with named presenters in the notes, then PowerPoint for the PDF);
`reports/viva_prep.md`.

**Why (D-044, D-046, D-053, D-054).** The most common way a wrong number reaches a report is
copying and re-rounding. `scripts/check_report_numbers.py` recomputes every
quoted number from the tables at the report's precision and requires it to
appear where it is quoted; it also checks the slides and viva notes, and a
coverage guard flags any number nobody verified. The builds refuse to export if
any check fails. Report checks are keyed to a phrase in the quoting paragraph and
deck checks to the slide title, so reorganising either document keeps every
check. `scripts/verify_references.py` checks each reference against its DOI record.

**`scripts/pipeline/run_all.py`** runs: Phases 2–7, report figures, slide figures, report build,
slide build, final number check. `--from N` resumes; times and memory go to
`outputs/run_all_log.csv`. `scripts/make_submission.py` packs
`MirageAI_Final_Submission.zip` in the project root, from an allow-list
(no `data/`, no models, no OOF parquets).

---

## 13. Results layer and dashboard

**`src/mirageai/results.py` (What/Why).** Read-only loaders and summaries of
`outputs/`: `load_table`, `load_best_params`, `load_model`, `headline`,
`scheme_row`, `block_size_curve`, `hot_decile_detection`, `residual_morans_i`,
`scenario_bracket`, `dose_response`, `load_shap_cells`, `cell_mean_shap`,
`driver_group_values`, `to_grid`. Missing files raise `FileNotFoundError` naming
the driver that creates them. Tested against the stored tables (D-049).

**`app/streamlit_app.py` (How).** Nine pages selected in the sidebar; tables
cached with `st.cache_data`, the model with `st.cache_resource`; plotly charts;
saved PNGs as fallbacks. The *Interactive prediction* page runs the saved
LightGBM on slider inputs, shows the exact per-feature contributions
(`pred_contrib`) and a lower-bound tree-cover what-if through
`scenarios.add_tree_cover`. See `docs/DASHBOARD.md`.

**Notebooks.** Four short executed walkthroughs (`notebooks/0*.ipynb`) that call
`src/` and show the stored tables and figures.

---

## 14. Testing strategy

`pytest` with 114 tests (56 test functions, several parametrised). Tests that
need generated data skip, with the reason, until it exists (`tests/conftest.py`).

| File | Guards |
|---|---|
| `test_cv.py` | no forbidden column is a feature; one block per cell across years; unique cell ids; no block or cell shared between train and test in any scheme; year boundary of the spatio-temporal split; folds partition the data; water rows refused; seeded layouts deterministic and distinct; strip layout latitudinal and shuffled layout not; urban/rural presence per fold |
| `test_evaluate.py` | queen neighbours and Moran's I on known patterns; hot-decile sizes; spatio-temporal test rows are 2025; comparison table coverage; metric values on known inputs |
| `test_explain.py` | exact stratified sampling; disjoint background; float32 cast lossless; SHAP additivity; parallel equals serial; breakpoint recovers a known knot and gives none to a straight line; interaction ranking; driver groups partition the features |
| `test_models.py` | OOF parts are never silently deleted; merges keep every row |
| `test_preprocess.py` | core/ring radii |
| `test_scenarios.py` | target selection; fractions stay valid and proportional; scenario C; unchanged fractions leave indices exactly; support guard flags ~1% of training rows |
| `test_results.py` | dashboard layer agrees with the tables, including the exact dominant-driver counts |

The pattern is to test **invariants** (no leakage, additivity, sums to one)
rather than implementation details. The block-consistency test caught the
rounding bug on its first run; the additivity check caught the float32 issue.

---

## 15. Reproducibility engineering

| Mechanism | Where | Effect |
|---|---|---|
| Fixed seed 42 (and 42–46 for layouts) | `config.SEED`, `CV_SEEDS` | deterministic splits, sampling and models |
| CPU only | `models.make_model` | bit-identical tree predictions across reruns and CPUs (D-031) |
| Pinned direct dependencies, full lock | `requirements.txt`, `requirements-lock.txt` | the direct dependencies at the versions the results were produced with; the lock pins the whole resolved environment (D-055) |
| Editable install | `pip install -e .`, `pyproject.toml` | `import mirageai` works from any directory (D-012) |
| Single entry point | `scripts/pipeline/run_all.py` | one command from `data/raw` to the report (D-048) |
| Number checker | `scripts/check_report_numbers.py` | documents cannot disagree with tables |
| Explicit nodata sentinel | `config.NODATA` | a missing value is never read as 0 (D-017) |
| Memory discipline | one forest at a time, streamed OOF parts | the full run fits in 16 GB (peak 1.30 GB in Phase 4) |

---

## 16. Design decisions and trade-offs

| Decision | Chosen | Trade-off accepted |
|---|---|---|
| Grid | 100 m (thermal sensor's true resolution) | less spatial detail than 30 m, but no false precision |
| Season | pre-monsoon | one season only |
| Target | within-year anomaly | absolute temperatures not predicted |
| Validation | 4 km spatial blocks, 5 layouts | fewer, noisier folds than random CV |
| Block size | from residual range | judgement call between 4 and 8 km; both reported |
| Water | excluded | no predictions for reservoirs |
| Tuning | on the evaluation folds, 60k subsample | mild optimism; 2025 test is the clean check |
| Reference model | LightGBM in a three-way tie | not "the most accurate" |
| SHAP | interventional, groups for the map | slower; interactions only path-dependent |
| Scenarios | bracket, three models, first point excluded | a range, not a single number; still associational |
| Dashboard | read-only over outputs | cannot explore new scenarios at the realistic bound |

---

## 17. Results

| Scheme (LightGBM) | RMSE °C | R² |
|---|---|---|
| Random CV | 1.620 | 0.78 |
| Spatial 1 km | 1.675 | 0.76 |
| Spatial 2 km | 1.706 | 0.75 |
| **Spatial 4 km (headline)** | **1.746 ± 0.060** | **0.73** |
| Spatial 8 km | 1.840 | 0.69 |
| **Spatio-temporal 2025** | **1.747** | **0.72** |

* Trees tie at 4 km (1.746–1.756); linear regression 1.873; baseline 3.451.
* Random-to-4 km inflation: +0.12 to +0.15 °C for trees, +0.023 for linear.
* Hottest-decile detection within urban cells: 54.2% (chance 10%).
* Residual Moran's I ≈ 0.6.
* Drivers: spectral block ≫ dry cover > tree cover > terrain > water > built-up.
* Scenarios (excl. first point): +10 points −0.37 to −1.02 °C; +20 points −0.56
  to −1.63 °C.

---

## 18. Rebuilding the project from scratch

1. **Environment.** Python 3.11 or newer in a fresh `.venv`; `pip install -r requirements.txt`, then `pip install -e .` (the package, editable). `streamlit run app.py` starts the dashboard.
2. **Config.** Write `config.py` with the AOI, CRS, grid, years, collections,
   masks, feature lists, forbidden columns, thresholds, seeds and paths.
3. **Extraction.** Implement `to_grid` and the per-source functions in
   `gee_extract.py`; download one GeoTIFF per year; convert to CSV; check scene
   counts, water cooler than land, fractions summing to one.
4. **Clean table.** Merge years; `floor`-based cell and block ids; the anomaly
   target; the urban flag; assert block consistency; save parquet. EDA and VIF.
5. **Validation.** Variogram of the target and of a first model's residuals;
   choose the block size; implement random (grouped), spatial (shuffled,
   seeded) and spatio-temporal splitters; write the leakage tests first.
6. **Models.** Baseline, scaled linear regression (12 features), RF, XGBoost,
   LightGBM (13 features); tune with `RandomizedSearchCV` on the training years'
   spatial folds; out-of-fold predictions for every model × scheme × seed.
7. **Evaluation.** Comparison table with mean ± std over runs; paired
   differences; CV gap; seen vs unseen 2025; urban vs rest; hottest-decile
   detection; Moran's I.
8. **SHAP.** Refit LightGBM on float32-rounded features of all land rows;
   interventional TreeSHAP with a stratified background; importance, groups,
   thresholds, interactions, per-cell driver map; XGBoost as a check.
9. **Scenarios.** Observed-hottest urban decile; proportional donor edit; lower
   and realistic bounds via index models; three models; support guard; exclude
   the first point from the headline.
10. **Communication.** Generate figures from tables; write the report with every
    number checked by script; build slides; add the dashboard as a read-only view.

---

## 19. Glossary

| Term | Meaning |
|---|---|
| AOI | Area of interest, the study rectangle |
| Anomaly (`LST_anom`) | LST minus that year's city-wide median |
| Block | A square of the grid (1–8 km) held out as a unit in CV |
| Bound (lower / realistic) | Scenario variant without / with the spectral indices responding to land cover |
| Cool island | A city cooler than its surroundings |
| GroupKFold | K-fold CV that keeps each group (cell or block) on one side |
| Interventional SHAP | SHAP that breaks feature dependence using a background sample |
| Knot | Breakpoint of a piecewise-linear fit |
| Leakage | Information about test data reaching training |
| Moran's I | Spatial autocorrelation statistic (−1 to 1) |
| OOF | Out-of-fold: predicted by a model that did not train on that row |
| Practical range | Distance at which the variogram reaches 95% of its sill |
| Range restriction | Correlations shrink when a variable's range is truncated |
| Spatio-temporal CV | Train on some blocks in 2023–24, test on other blocks in 2025 |
| Support | The region of feature space covered by training data |
| VIF | Variance inflation factor, a multicollinearity measure |
