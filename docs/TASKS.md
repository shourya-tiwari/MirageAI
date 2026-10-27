# Tasks

Tick a box when its task is done and verified. Add a CHANGELOG entry whenever you finish a phase or make a notable change.
Task IDs (`T<phase>.<n>`) are stable, so reference them in commits, e.g. `feat(T1.3): add S2 index extraction`.

## Phase 0: Setup
- [x] T0.1 Create repo `mirageai` with the structure in `INSTRUCTIONS.md`
- [x] T0.2 Create venv (Python 3.11) and `requirements.txt`
- [x] T0.3 Register a noncommercial Earth Engine Cloud project
- [x] T0.4 Run `earthengine authenticate`, then verify `ee.Initialize(project=...)`
- [x] T0.5 Write `src/mirageai/config.py` (AOI, CRS, SCALE, YEARS, season, paths, SEED)
- [x] T0.6 Add `.gitignore` (data/, outputs/models/, .venv, credentials)

## Phase 1: Data Extraction
- [x] T1.1 Landsat 8/9 LST: QA_PIXEL cloud mask, ST_B10 scaling to °C, median and count
- [x] T1.2 Sentinel-2: SCL mask, NDVI / NDBI / MNDWI from a median composite
- [x] T1.3 WorldCover land-cover fractions (7 classes)
- [x] T1.4 Distance to water (fastDistanceTransform at a fixed 30 m projection, converted to metres)
- [x] T1.5 SRTM elevation and slope
- [x] T1.6 `to_grid()`: reproject to native scale, then `reduceResolution(mean)` onto the 100 m grid
- [x] T1.7 Add coordinate bands (lon/lat, UTM x/y)
- [x] T1.8 Fetch the CSV + GeoTIFF for 2023, 2024 and 2025 directly into `data/raw/` (see D-020)
- [x] T1.9 Log scene counts, row counts and dropped cells for every year
- [x] T1.10 Visual sanity check of LST and NDVI maps (save to `outputs/figures/`)

## Phase 2: EDA & Preprocessing
- [x] T2.1 Load all years into one table with a `year` column
- [x] T2.2 Missing-value and dropped-cell report (cloud gaps, `LST_n_obs` distribution)
- [x] T2.3 Target distribution per year, plus an LST map per year
- [x] T2.4 Feature distributions and outlier check (LST, `dist_water_m`)
- [x] T2.5 Correlation heatmap and VIF; document the handling of collinear features
- [x] T2.6 Handle the fraction sum-to-1 issue (drop one fraction for LR)
- [x] T2.7 Save `data/processed/uhi_pune_clean.parquet`
- [x] T2.8 Log decisions in `DECISIONS.md`

## Phase 3: Validation Design
- [x] T3.1 Empirical variogram or autocorrelation range of LST to justify block size
- [x] T3.2 Assign spatial blocks (1, 2, 4 and 8 km) from UTM x/y; 8 km is primary (D-026)
- [x] T3.3 Splitters: random (grouped on `cell_id`), spatial GroupKFold, spatio-temporal (D-010 as amended)
- [x] T3.4 Leakage test: assert no block overlap between train and test (`tests/test_cv.py`, 27 tests)
- [x] T3.5 Plot the block layout map

## Phase 4: Modelling
- [x] T4.1 Baseline: predict the mean (sanity floor)
- [x] T4.2 Linear Regression in a Pipeline with StandardScaler
- [x] T4.3 Random Forest
- [x] T4.4 XGBoost
- [x] T4.5 LightGBM
- [x] T4.6 Hyperparameter tuning inside spatial CV (RandomizedSearchCV with the group splitter)
- [x] T4.7 Save best params and fitted models to `outputs/models/`
- [x] T4.8 Final Phase 4 run on the 16 GB machine: full RF search space restored (D-030 resolved), CPU only, one forest in memory at a time (D-031); runtime and peak memory logged per model; provisional results kept with a `_provisional` suffix

## Phase 5: Evaluation & Comparison
- [x] T5.1 RMSE / MAE / R² for every model × CV scheme, saved to `outputs/tables/model_comparison.csv`
- [x] T5.2 Random-vs-spatial CV gap chart
- [x] T5.3 Temporal holdout results (2025)
- [x] T5.4 Predicted vs actual scatter for the best model
- [x] T5.5 Residual map and Moran's I on residuals
- [x] T5.6 Error in the hottest decile, i.e. how well the model performs where it matters
- [x] T5.7 Report metrics separately for urban cells (`frac_built` >= `URBAN_BUILT_THRESHOLD`) vs the rest (see D-019)
- [x] T5.8 Resolve D-033 (striped 4 km folds): shuffled block assignment over 5 seeds (42-46) as headline, strips as a stress test; water-majority cells excluded (D-034); re-tuned and Phase 4/5 re-run, earlier outputs kept as `*_strips`

## Phase 6: Explainability
- [x] T6.1 SHAP TreeExplainer on the best tree model (LightGBM fitted on all land rows, D-035; interventional, 1,000-row background, D-036; XGBoost rank check)
- [x] T6.2 Global importance (beeswarm and bar)
- [x] T6.3 Dependence plots for the top 4 drivers (look for thresholds and saturation; segmented fits with bootstrap CIs, D-037)
- [x] T6.4 Interaction values: top 3 pairs taken from the interaction matrix rather than a pre-chosen pair (NDBI × grass, NDVI × NDBI, NDVI × MNDWI)
- [x] T6.5 SHAP map: dominant driver per cell (from grouped SHAP, D-038; per-feature map supplementary)
- [x] T6.6 One-line interpretation per driver (`reports/phase6_findings.md`)
- [x] T6.7 Repeat global SHAP importance on the urban subset (`frac_built` >= `URBAN_BUILT_THRESHOLD`) and compare driver ranking with the AOI-wide run (see D-019)
- [x] T6.8 Grouped SHAP for the collinear spectral block (NDVI + NDBI + MNDWI summed), reported alongside the individual values (see D-024)

## Phase 7: Greening Scenarios
- [x] T7.1 Decide the scenario model design (D-009 resolved: lower / realistic bracket; D-041 index models; D-042 three models; D-043 support guard)
- [x] T7.2 Identify the hottest-decile cells **within urban cells** (`frac_built` >= `URBAN_BUILT_THRESHOLD`), not AOI-wide (see D-019); on observed 3-year mean LST_anom (D-039): 2,673 cells
- [x] T7.3 Scenarios: +10 and +20 points of tree cover, plus raise-to-knot (C), taken from built/bare/grass in proportion so fractions still sum to 1 (D-040)
- [x] T7.4 ΔLST map and summary table (`07_delta_lst_maps.png`, `phase7_summary.csv`, `07_cooling_per_point.png`)
- [x] T7.5 Write up the associational caveat (D-009, `reports/phase7_findings.md`)

> **T7 note (from Phase 5):** the model places only about half of the observed urban hottest decile in its own predicted urban hottest decile: precision **54%** before the D-033/D-034 re-run, **54.2%** after it (LightGBM, 4 km spatial CV; 53.0% on the 2025 spatio-temporal test; chance is 10%). Quote this as a caveat wherever scenario targets are chosen from predicted heat: the targeted cells are "likely among the hottest", not a precise list (`phase5_hot_decile_overlap.csv`).

## Phase 8: Report, Presentation, Viva
- [x] T8.1 Report outline mapped to rubric sections (`reports/report.md`: title page, abstract, sections 1-9, references, appendices A-B)
- [x] T8.2 Write report: problem, data, methodology, results, discussion, conclusion, future scope, references (draft; `scripts/build_report.py` → `MirageAI_Report.docx` / `.pdf`, 20 pages; 211 number checks pass, D-044)
- [x] T8.3 Verify every reference (see `REFERENCE.md`): 21 cited, 19 verified by DOI, 2 marked [UNVERIFIED] (no DOI), Pune-specific study left as a team placeholder (`reports/reference_check.csv`)
- [x] T8.4 Slide deck: map-first storytelling, ~10–12 slides (13 slides, `reports/MirageAI_Slides.pptx` / `.pdf`, speaker notes ~10.7 min, presenters [P1]-[P4]; D-046)
- [x] T8.5 Viva prep: each member explains every decision in `DECISIONS.md` (`reports/viva_prep.md`: checked numbers cheat sheet + 33 questions grounded in D-numbers; rehearsal by the team still to do)
- [ ] T8.6 Final reproducibility run from a clean clone

## Phase 9: Final polish
- [x] T9.1 Engineering review: unused imports removed, clearer errors for missing inputs, version synced (D-050)
- [x] T9.2 `mirageai.results`: read-only loaders and summaries of `outputs/`, with `tests/test_results.py`
- [x] T9.3 Streamlit dashboard (`app/streamlit_app.py`, `docs/DASHBOARD.md`, D-049)
- [x] T9.4 Notebooks rebuilt as executed walkthroughs that call `src/`
- [x] T9.5 README and `docs/` brought in line with the implementation; `docs/README.md` index
- [x] T9.6 `project_deep_dive.md` (technical guide) and `learning_log.md` (study guide)
- [x] T9.8 Pipeline drivers moved to `scripts/pipeline/` (D-052); one pinned `requirements.txt` (D-051)
- [x] T9.9 README rewritten as a public project page, with Quick Start, Project Demo and screenshots
- [x] T9.7 Names filled everywhere (report, slides, README, `LICENSE`, `CITATION.cff`, `pyproject.toml`); reference placeholders resolved; PRNs not provided, left as a checklist item (`reports/SUBMISSION_CHECKLIST.md`)
- [x] T9.10 Report rewritten as a 15-page IEEE-format conference paper, built from HTML (D-053)
- [x] T9.11 Slide deck redesigned: 19 slides in the viva flow, named presenters, ~13.9 min of notes (D-054)
- [x] T9.12 MIT license, citation file, and `MirageAI_Final_Submission.zip` in the project root (`scripts/make_submission.py`)
- [x] T9.13 Word copy of the report in the IEEE two-column layout (D-056)
- [x] T9.14 Dependencies simplified: direct pins in `requirements.txt`, full `requirements-lock.txt`, `pip install -e .` as its own step, `streamlit run app.py` (D-055)
