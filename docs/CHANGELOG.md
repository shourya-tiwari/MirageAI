# Changelog

All notable changes to MirageAI (formerly Ushna) are recorded here.
Format based on [Keep a Changelog](https://keepachangelog.com/). Sections: **Added**, **Changed**, **Fixed**, **Removed**, **Data**, **Results**.

## [Unreleased]

Phase 8, part 1 (report draft): T8.1-T8.3 done. Part 2 (slides and viva
prep): T8.4-T8.5 done. Part 3: reproducibility tooling and submission
package; the clean-copy run (T8.6) is not yet complete.

### Changed (dependencies and setup, 2026-10-27; D-055)
- `requirements.txt` now lists only the 23 direct dependencies, pinned; no
  transitive pins and no `-e .`. Setup is `pip install -r requirements.txt`,
  then `pip install -e .`, in a standard `python -m venv .venv` (Python 3.11+).
- `pyproject.toml`: runtime dependencies match the imports; pytest, JupyterLab
  and ipykernel are the optional `dev` group.
- The dashboard starts with `streamlit run app.py` (README, docs, the report's
  Table IX, the demo slide).
- Usage lines in every script say `python …` instead of the conda-prefix path.

### Added (dependencies and setup)
- `app.py`: root launcher for the dashboard.
- `requirements-lock.txt`: the full verified environment, for an exact rebuild.

### Removed (dependencies and setup)
- `geemap`, `seaborn`, the `jupyter` metapackage and `geopandas` as direct
  dependencies (never imported), and about 200 hand-pinned transitive packages.
- The conda-prefix installation option.

### Added (final submission, 2026-10-27; D-053, D-054)
- `LICENSE` (MIT, Shourya Tiwari) and `CITATION.cff`; authors and license in
  `pyproject.toml`; README contributors (Shourya Tiwari, Shashwat Kumar, Rishik
  Sinha, Shubhankar Bhan) and a License section with the data terms.
- `reports/ieee/` (pandoc HTML template and print CSS) and
  `reports/MirageAI_Report.html`.
- `outputs/figures/09_pipeline.png`, the pipeline diagram used by the report and
  the deck (`scripts/make_slide_figures.py`).

### Changed (final submission)
- `reports/report.md` rewritten as an IEEE-format conference paper: 18 numbered
  sections from Introduction to Conclusion, 20 figures, 9 tables, an algorithm
  box, 25 numbered references; `MirageAI_Report.pdf` is 15 pages, printed by
  headless Chrome/Edge (`scripts/build_report.py`); the `.docx` is an
  editable copy in the same IEEE two-column layout (D-056,
  `reports/ieee/ieee-docx.lua`).
- `reports/MirageAI_Slides.pptx` / `.pdf` redesigned: 19 slides (Title →
  Thank you), named presenters in the notes, ~13.9 min at 140 wpm.
- `scripts/check_report_numbers.py`: deck checks keyed by slide title (via
  `reports/slides_manifest.csv`); 376 checks, all passing.
- `scripts/verify_references.py`: four new references (24 verified by DOI, 2
  with no registered DOI); UTF-8 console output.
- `scripts/make_submission.py` writes `MirageAI_Final_Submission.zip` in the
  project root and includes `LICENSE` and `CITATION.cff`.

### Removed (final submission)
- `reports/MirageAI_submission.zip` (replaced by the root-level final zip) and
  every `[NAME]`, `[PRN]`, `[TEAM]`, `[DATE]` and `[UNVERIFIED]` placeholder.

### Changed (repository layout and dependencies, 2026-10-27; D-051, D-052)
- `run_all.py` and `run_phase2.py` … `run_phase7.py` moved to `scripts/pipeline/`;
  every reference (errors, tests, notebooks, dashboard, docs, submission zip)
  updated. Run with `python scripts/pipeline/run_all.py`.
- One `requirements.txt`: direct dependencies grouped by purpose, every
  transitive dependency pinned, Streamlit included, and `-e .` last, so
  `pip install -r requirements.txt` is the whole setup. `pyproject.toml` lists
  the direct dependencies itself.
- README rewritten for a public repository: overview, motivation, features,
  architecture, quick start, reproducibility, project demo, screenshots
- Dashboard screenshots added under `docs/images/`

### Removed (repository layout and dependencies)
- `requirements-lock.txt` and `requirements-dashboard.txt` (merged into
  `requirements.txt`, D-051)

### Added (final polish, 2026-10-27; D-049, D-050)
- `app/streamlit_app.py`: nine-page Streamlit dashboard (overview, dataset, EDA,
  models, evaluation, explainability, interactive prediction, results,
  limitations), read-only over `outputs/`; `.streamlit/config.toml`;
  `requirements-dashboard.txt` (Streamlit 1.65.0, outside the lock);
  `docs/DASHBOARD.md`
- `src/mirageai/results.py` (loaders and summaries of the stored outputs) and
  `tests/test_results.py`
- `project_deep_dive.md` and `learning_log.md` at the root; `docs/README.md` index
- The dashboard, guides and dashboard requirements are in the submission zip

### Changed (final polish)
- README rewritten: results at a glance, pipeline, structure, usage,
  dashboard walkthrough, model details, evaluation, results, limitations
- `docs/PROJECT.md` method section, `docs/REFERENCE.md` CV and SHAP notes,
  `docs/INSTRUCTIONS.md` tree and `docs/MIGRATION.md` test count brought in line
  with the implementation
- Notebooks rebuilt as short executed walkthroughs that call `src/` (the old
  stubs still described the removed Drive export)
- Missing inputs raise `FileNotFoundError` naming the driver that creates them;
  unknown scenario names give a clear error
- Package version 0.1.0 → 0.8.0, matching the last release

### Removed (final polish)
- Unused imports in `run_phase7.py`, `scripts/build_slides.py`, `tests/test_explain.py`

### Added (part 3)
- `run_all.py`: single entry point, Phase 2 → 7 from `data/raw`, then the
  report, slides and number checks (D-048)
- `run_phase2.py`, `run_phase3.py`: drivers for the Phase 1-3 tables and
  figures, which had no committed producer; they reproduce the stored tables
  byte for byte
- `phase8_core_ring_land.csv` and `preprocess.core_ring_contrast()`, with
  `config.CORE_RADIUS_M` / `RING_RADIUS_M`; `tests/test_preprocess.py`
- `tests/conftest.py`: data-dependent tests skip, with the reason, until
  their inputs exist, so pytest runs in a clean clone
- `scripts/make_submission.py` → `reports/MirageAI_submission.zip`;
  `reports/SUBMISSION_CHECKLIST.md`; README data and reproduce sections
- D-047 (surface LST, not air temperature; viva Q3 points to it), D-048

### Changed (part 3)
- Cool-island headline recomputed on land cells (D-034), same core/ring
  definition: core minus ring −3.661 / −3.382 / −3.970 °C (2023/24/25), was
  −3.501 / −3.245 / −3.813 on all cells; slide 2, report and viva now say
  3.4-4.0 °C (was 3.2-3.8). 2024 ring NDVI 0.249 (was 0.243).
- `01_*`, `02_*`, `03_*` figures are now drawn by the committed drivers
  (same data and content, redrawn)
- Number checker: clock times are stripped before the coverage guard

### Added (part 2)
- `reports/MirageAI_Slides.pptx` and `.pdf`: 13 slides, speaker notes on every
  slide with a [P1]-[P4] split, ~10.7 min (`reports/slides_manifest.csv`)
- `scripts/build_slides.py` (python-pptx; numbers formatted from the tables;
  ≤ 30 body words per slide, enforced; PowerPoint PDF export after the number
  check), `scripts/make_slide_figures.py` (`09_group_ranking.png`,
  `09_scenario_bracket.png`)
- `reports/viva_prep.md`: numbers-to-know cheat sheet and 33 questions with
  answers grounded in D-numbers, grouped by rubric area
- `scripts/check_report_numbers.py` now also checks the deck (slide text and
  notes) and the viva notes, plus a coverage guard for unchecked numbers:
  371 checks pass
- python-pptx 1.0.2 (with lxml, xlsxwriter) in `requirements.txt` and the lock
- D-046

### Changed (rename, D-045)
- Project renamed from Ushna to **MirageAI**: Pune's expected daytime urban
  heat island turns out to be a mirage. The registered title "Concrete vs.
  Canopy" is unchanged.
- Package `src/ushna` → `src/mirageai` (`pyproject.toml` name `mirageai`);
  all imports in src, tests, run scripts, scripts and notebooks updated;
  editable install reinstalled
- `MIRAGEAI_GEE_PROJECT` replaces `USHNA_GEE_PROJECT`, which still works as a
  deprecated fallback with a warning. The Earth Engine project ID
  `ushna-aucml` is unchanged.
- Report outputs renamed to `reports/MirageAI_Report.{docx,pdf}`; title page
  now carries the registered title, MirageAI and the correct course name
  (Applications and Use Cases of Machine Learning, TE7483)
- Prose in the project instructions, README and docs updated; earlier CHANGELOG and DECISIONS
  entries keep "Ushna" as history

### Added
- `reports/report.md` (single source) → `reports/Ushna_Report.docx` and
  `reports/Ushna_Report.pdf` (20 pages, 10 figures, 3 tables + 1 appendix table)
- `scripts/build_report.py` (number check → pandoc → Word PDF export),
  `scripts/check_report_numbers.py` (211 checks against the tables),
  `scripts/verify_references.py` (DOI verification → `reports/reference_check.csv`),
  `scripts/make_report_figures.py` (`08_cool_island.png`, `08_model_comparison.png`,
  `08_tree_cover_dependence.png`; tables `phase8_feature_correlations`,
  `phase8_index_correlations`, `phase8_residual_range_by_scheme`,
  `phase8_cool_island_bins`)
- D-044 (report build and number / reference checks)

### Fixed
- Three scenario values were rounded twice and wrong in the last digit
  (B lower −0.56, A realistic range [−1.04, −1.00], C lower range
  [−0.38, −0.31]); corrected everywhere they were quoted
- Hard-coded, stale baseline RMSE in the `05_cv_gap.png` caption; it is now
  computed from the comparison table

### Tooling
- pandoc 3.12.1 installed per user (winget); PDF export uses Microsoft Word

---

## [0.8.0] - 2026-10-16

Phase 7 complete (T7.1-T7.5). D-009 is resolved as a lower / realistic
bracket. After review, the headline excludes the 0 → 1% "any canopy" step,
which is confounded with unobserved urban form and is reported separately
(D-040).

### Added
- `src/ushna/scenarios.py`: target selection on observed 3-year means,
  proportional tree-cover edit from built/bare/grass, scenario deltas,
  per-year land-cover → index models and delta-updated indices, ΔLST, and a
  k-NN support guard
- `run_phase7.py`: single Phase 7 command (~14 min)
- `tests/test_scenarios.py`: 5 tests (targets are the hottest urban decile,
  fraction bookkeeping and shortfall, scenario C, unchanged fractions keep
  observed indices, support guard calibration); 106 tests pass in total
- Model `outputs/models/rf_all.joblib` (all land rows, float32-exact)
- Tables `phase7_{target_cells, target_clusters, index_model_cv, summary,
  dose_response, marginal_cooling, marginal_summary}.csv`,
  `phase7_cell_results.parquet`
- Figures `07_target_cells.png`, `07_delta_lst_maps.png`,
  `07_cooling_per_point.png`
- `reports/phase7_findings.md`; D-009 resolved, D-039 to D-043

### Results
- Targets: 2,673 of 26,726 urban cells (observed mean ≥ +2.41 °C), on the
  eastern fringe; median tree cover 0%, 78.7% with none
- Index models (spatial CV R²): MNDWI 0.72, NDVI 0.67, NDBI 0.61
- **Headline (first point excluded), lower to realistic, 3-model mean:**
  A +10 points −0.37 to −1.02 °C; B +20 points −0.56 to −1.63 °C; C to the
  knot −0.35 to −0.97 °C
- Finding: tree-less urban cells (78.7% of targets) run ~1.4 °C hotter (first
  point −1.43 °C realistic, −0.87 °C lower) in ways not attributable to tree
  cover alone
- Including the step (transparency only): A −1.06 to −2.14, B −1.25 to −2.75,
  C −1.03 to −2.09 °C
- `07_delta_lst_maps.png` maps the headline (realistic, first point excluded)
- Diminishing returns: −0.12 °C per point up to the knot, −0.05 from 12% to 30%
- Out of support: ≤ 0.04% of scenario rows (0.57% at +30 points)

---

## [0.7.0] - 2026-10-16

Phase 6 complete (T6.1-T6.8). After review, T6.5 was rebuilt from grouped
SHAP (D-038), and MNDWI is labelled "moisture" throughout.

### Added
- `src/ushna/explain.py`: final-model fit on float32-exact features,
  stratified explained and background samples, interventional TreeSHAP
  (parallel across worker processes, additivity checked), path-dependent SHAP
  and interaction values, importance, rank agreement, grouped SHAP,
  interaction ranking, segmented breakpoint fits with bootstrap CIs
- `run_phase6.py`: single Phase 6 command (~38 min; `--plots-only` reuses the
  saved SHAP values)
- `tests/test_explain.py`: 9 tests (exact stratified sample, disjoint
  background, lossless float32 cast, SHAP additivity, parallel equals serial,
  known-knot recovery, no knot on a straight line, symmetric interaction
  ranking, grouped sum)
- Models `outputs/models/{lgbm,xgb}_all.joblib` (all land rows, 2023-25)
- Tables `phase6_{shap_importance, rank_agreement, grouped_spectral,
  interactions, interaction_matrix, thresholds, linear_coefs,
  rq5_frac_built, rq5_frac_built_bins, dominant_driver,
  path_vs_interventional, runtimes}.csv`; SHAP values
  `phase6_shap_{lgbm,xgb}.parquet`, `phase6_shap_cells_lgbm.parquet` (every
  land row), `phase6_interaction_values_lgbm.npz`
- Figures `06_shap_beeswarm.png`, `06_shap_bar.png`, `06_shap_dependence.png`,
  `06_shap_interactions.png`, `06_shap_grouped_spectral.png`,
  `06_dominant_driver_map.png` (grouped), `06_dominant_driver_map_features.png`
  (supplementary)
- (T6.5) `explain.DRIVER_GROUPS`, `explain.group_values`; tables
  `phase6_dominant_driver_groups.csv`, `phase6_group_importance.csv`,
  `phase6_dominant_stability.csv`; 10 tests in `tests/test_explain.py`
- `reports/phase6_findings.md`: driver table, RQ1 / RQ2 / RQ5 answers, T6.6 lines
- D-035 (SHAP explains the all-years final model), D-036 (interventional SHAP,
  float32 exactness, path-dependent interactions), D-037 (threshold method)

### Results
- Ranking (mean |SHAP|, °C): NDVI 0.95, NDBI 0.73, MNDWI 0.61, grass 0.54,
  tree 0.38, cropland 0.34, elevation 0.25 … built-up 0.12 (10th). Spectral
  block 1.49. Urban: NDVI 0.82, MNDWI 0.80, NDBI 0.54, grass 0.53; built-up 8th
- LightGBM vs XGBoost: Spearman 0.989 (all), 0.973 (urban); same top five,
  order differs inside the spectral block
- Top interactions: NDBI × grass 0.18, NDVI × NDBI 0.16, NDVI × MNDWI 0.14 °C
- Thresholds: tree cover knot 10.2% (step −0.48 °C at onset, then −0.25 →
  −0.07 °C per 10 points); MNDWI onset at −0.39; NDVI slope change at 0.27;
  water cooling reach ~190 m; elevation flat above 590 m
- RQ5: the frac_built association shrinks 87% (−0.22 → −0.029 °C per 10
  points) once vegetation and dryness are modelled, and stays weakly negative
- Dominant driver group: spectral 66.8% of cells, dry cover 24.5% (42.7% of
  urban cells). It is stable across models (95.3% vs 79.1% per feature) and
  across SHAP methods (92.5% vs 64.6%)

---

## [0.6.0] - 2026-10-16

Phase 5 complete (T5.1-T5.8). D-033 decided: five shuffled block layouts as
the headline, the strip layout as a stress test. Water-majority cells
excluded (D-034). Phase 4 re-tuned and re-run on the land-only table, Phase 5
regenerated. Earlier outputs kept with a `_strips` suffix.

### Added
- (T5.1-T5.7) `src/ushna/evaluate.py`: model × scheme comparison over seed ×
  fold runs, paired differences on (seed, fold), CV gap, unseen vs seen-place
  2025 holdout, hottest-decile and urban subgroup metrics, top-decile detection
  overlap, Moran's I on 100 m queen weights, residual range and block rung
- `run_phase5.py`: single Phase 5 command (~7 min, mostly Moran's I with 999
  permutations); reads `oof_predictions.parquet` with column selection and
  row filters, never whole
- (D-033) `config.CV_SEEDS = (42..46)`; seed-aware `cv.random_splits`,
  `spatial_splits`, `spatio_temporal_splits`; `cv.strip_splits` (stress test,
  layout taken from the full grid); `cv.block_folds`
- (D-034) `config.WATER_EXCLUDE_THRESHOLD = 0.5`, `preprocess.is_water`,
  `preprocess.load_model_table`; `cv.require_land_only` guards every splitter
  and `models.tune`
- `run_phase4.py`: residual-range gate before tuning (exits if the D-028 rung
  moves), `--range-check-only`, `--finish-from-parts`, `--force`
- `models.prepare_parts_dir` (a full run refuses to delete saved OOF parts
  unless `--force`) and `models.merge_parts` (streams parts one at a time with
  `ParquetWriter`, verifies the row count, logs peak memory)
- `plotting.cells_to_grid`
- Tests: water guard on every splitter and on tuning, exact removal count,
  seed determinism and distinctness, strip shape, strip leakage and years,
  `block_folds` consistency, queen-weight neighbours, OOF parts safety and
  merge; 91 tests pass
- Tables `phase4_water_exclusion`, `phase4_residual_range`, `phase4_fold_balance`,
  `model_comparison`, `phase5_{paired_vs_best, cv_gap, strip_stress_test,
  temporal_holdout, temporal_holdout_folds, hot_decile, hot_decile_overlap,
  urban_vs_rest, residual_range, fold_profile_strips, morans_i}.csv`
- Figures `05_cv_gap.png`, `05_pred_vs_actual.png`, `05_residual_map.png`
- D-032, D-033 (decided), D-034; re-check notes on D-028 and D-029

### Changed
- Every spatial and spatio-temporal result is now a mean ± std over 5 seeds × 5
  folds; random CV is repeated over the same seeds
- Re-tuned on the seed-42 shuffled 4 km layout, 2023-24, land only (D-029).
  RF now picks `200 / 2 / 0.3 / 16`
- `LST_anom` keeps its D-021 definition (whole-AOI median); a land-only median
  would shift it by +0.045 to +0.051 °C, a constant offset
- The Phase 4 "plateau from 4 to 8 km" stays withdrawn

### Removed
- `run_tuning.py`, `run_tuning_rf.py` (superseded by `run_phase4.py`, and
  incompatible with the water guard)

### Data
- (D-034) 2,697 of 245,949 rows removed (899 cells × 3 years, 0 urban), leaving
  243,252 (81,084 per year)
- 0.5.0 outputs renamed `*_strips` (26 files: tables, figures, models,
  `best_params_strips.json`, `oof_predictions_strips.parquet`)

### Results
- Residual range (D-028 gate, default LightGBM, 2 km CV): median 3,950 m →
  4 km kept. The tuned model's own 4 km residuals give 4,250 m (rung 8 km); this
  is reported, not acted on, because the measured range grows with the CV that
  generates it (random 3,450 → 8 km 5,250 m). 8 km is the conservative bound
- Spatial 4 km (headline): LightGBM 1.746 ± 0.060 (R² 0.732), XGBoost
  1.746 ± 0.064, RF 1.756 ± 0.059, linear 1.873 ± 0.047, baseline 3.451 °C.
  The trees tie (XGBoost − LightGBM −0.000 ± 0.006)
- Strip stress test: LightGBM 1.762 ± 0.113, i.e. +0.016 °C (+0.054 before
  water exclusion)
- RQ3 curve (LightGBM): random 1.620, 1 km 1.675, 2 km 1.706, 4 km 1.746,
  8 km 1.840. Random → 4 km: trees +0.12 to +0.15, linear +0.023
- 2025 spatio-temporal: LightGBM 1.747 ± 0.108 (R² 0.721); seen-place 1.598
  (R² 0.767); memorisation 0.149 (LightGBM) to 0.227 (RF), linear 0.018
- Residual Moran's I 0.59-0.64 (LightGBM, 4 km) against 0.89-0.90 raw
- Hottest decile detection 62.3% AOI-wide, 54.2% within urban (chance 10%)
- Urban vs rest at 4 km: RMSE 1.614 vs 1.805, R² 0.542 vs 0.753
- Merge of 160 OOF parts: 34,055,280 rows, peak 533 MB

---

## [0.5.0] - 2026-10-16

Phase 4 complete. Final run (T4.8) on the new 16 GB machine.

### Added
- `docs/MIGRATION.md`: how to move the project to a new machine and the checks
  that prove nothing changed (clean table, correlations, VIF and a LightGBM fold
  all reproduced exactly)
- `models.PeakMemory`: sampled peak-RSS context manager for per-model memory logging
- Tables `phase4_fold_metrics.csv`, `phase4_runtimes.csv` (stage x scheme x model:
  seconds and peak RSS), `phase4_tuning_{lgbm,xgb,rf}.csv`, `oof_predictions.parquet`
- Models `outputs/models/{baseline,linear,rf,xgb,lgbm}_trainyears.joblib`
- (T4.8) New task for the final Phase 4 run

### Changed
- (D-030 resolved) RF search space restored: `n_estimators` up to 500,
  `min_samples_leaf` from 1
- (D-031) XGBoost and LightGBM pinned to the CPU; the search runs with
  `refit=False`, and each fold's model is freed right after it predicts
- `run_phase4.py` is now the single Phase 4 command (tuning, evaluation, final
  fits); `--skip-tune` reuses `best_params.json`
- pytest 9.1.1 (with pluggy and iniconfig) added to `requirements.txt` and the
  lock; it had been installed outside the lock on the old machine
- `INSTRUCTIONS.md` §2: the env is created from conda-forge, since recent
  Miniconda refuses the `defaults` channels until their ToS is accepted

### Data
- Provisional Phase 4 outputs kept, renamed `best_params_provisional.json` and
  `_oof_parts_provisional/`

### Results
- Spatial CV RMSE at 4 km (primary): RF 1.789 ± 0.147, XGBoost 1.790 ± 0.153,
  LightGBM 1.791 ± 0.154, linear 1.987 ± 0.244, baseline 3.663 ± 0.381 °C
- Spatio-temporal 4 km (test year 2025): LightGBM 1.774 ± 0.127, RF 1.780 ± 0.130,
  XGBoost 1.785 ± 0.128 °C; R² 0.74
- Random CV: XGBoost 1.597, LightGBM 1.622, RF 1.636 °C, i.e. a ~0.19 °C (12%)
  random-to-spatial gap for every tree model
- Tree RMSE plateaus from 4 km to 8 km, consistent with the residual range in D-028

---

## [0.4.0] - 2026-10-08

Phase 3 complete. Leakage-proof validation design.

### Added
- (T3.1-T3.3) `src/ushna/cv.py`: empirical variogram by random pair sampling,
  practical-range estimator, and the random / spatial / spatio-temporal splitters
- (T3.4) `tests/test_cv.py`: 27 tests covering schema guards, year-independent
  ids, splitter leakage, fold partitioning and fold balance
- (T3.5) Figures `03_variogram_lst_anom.png`, `03_block_layout.png`;
  table `phase3_fold_balance.csv`
- `config.BLOCK_COLUMNS`; `config.FEATURES_LINEAR` / `FEATURES_TREE` (D-025)
- (T6.8) New task: grouped SHAP for the collinear spectral block

### Changed
- (D-026) `BLOCK_SIZES_M` is now (1000, 2000, 4000, 8000) and
  `PRIMARY_BLOCK_SIZE_M` is 8000, from the measured 7.65 km variogram range
- (D-010 amended, D-027) The pure temporal split is replaced by a spatio-temporal
  split; a same-cells temporal split tests memorisation, not generalisation
- (D-025) `frac_built` is dropped for Linear Regression only; tree models keep all
  seven fractions so SHAP can attribute a built-up effect
- LEARNING_LOG: the within-urban correlation drop is reframed as suggestive, since
  restricting to `frac_built` >= 0.5 truncates that predictor's range

### Results
- LST anomaly autocorrelation range 7.65 km (7.55 / 7.65 / 7.65 by year); 2 km
  blocks reach only 63% of the sill and would have under-corrected
- Random CV folds are balanced (max 0.7 pts urban deviation); 8 km spatial folds
  are not (max 17.8 pts), a consequence of the city's shape (D-027)

---

## [0.3.0] - 2026-10-08

Phase 2 complete. Clean modelling table built and audited.

### Added
- (T2.1-T2.7) `src/ushna/preprocess.py`: year concatenation, stable `cell_id`,
  year-independent block ids, within-year anomaly target, urban flag, missing
  report, VIF table and the clean-table builder
- `config.TARGET_ANOM`; `FORBIDDEN_FEATURES` widened to cover the anomaly target,
  `LST_year_median`, `cell_id`, `is_urban` and both block columns
- `data/processed/uhi_pune_clean.parquet` (245,949 rows x 26 cols, 18.4 MB)
- Figures: `02_target_distributions.png`, `02_feature_distributions.png`,
  `02_correlation_heatmap.png`; table `phase2_lst_anom_correlations.csv`
- PROJECT.md: the three years are repeated measurements of the same 81,983 cells,
  not independent samples

### Changed
- (D-021) Primary target is `LST_anom`, the within-year anomaly against that
  year's AOI median; absolute `LST` kept for maps and reporting
- (D-023) `FRACTION_DROP_FOR_LINEAR` changed from `frac_bare` to `frac_built`,
  overturning the provisional D-014 default on VIF evidence

### Fixed
- `cell_id` used round-half-to-even on coordinates whose half-cell offsets land
  exactly on .5, merging adjacent grid columns onto one id; 7,456 merged cells
  straddled two 1 km blocks. Caught by the D-022 consistency test on first run.

### Results
- Within urban cells the `frac_built`-LST_anom correlation collapses from -0.192
  to +0.009, which answers RQ5 at the correlation level
- Dropping any single land-cover fraction changes R-squared by 0.0000, confirming
  the sum-to-1 constraint makes the choice one of conditioning, not fit
- NDBI contributes +0.0081 R-squared beyond all other features despite r = -0.70
  with NDVI

---

## [0.2.0] - 2026-10-08

Phase 1 complete. 100 m feature stacks extracted for all three years.

### Added
- (T1.1-T1.7) `src/ushna/gee_extract.py`: Landsat 8/9 LST with QA_PIXEL masking and
  ST_B10 scaling, Sentinel-2 SCL-masked NDVI/NDBI/MNDWI, WorldCover fractions,
  distance to water, SRTM terrain, `to_grid()` aggregation and coordinate bands
- (T1.8/T1.9) `extract_year()`, `stack_to_frame()`, `write_year_table()`: direct
  download into `data/raw/`, no Drive round-trip (D-020)
- (T1.10) `src/ushna/plotting.py`: `map_panel()` and `save_figure()` helpers
- (T5.7/T6.7) New tasks: report metrics and SHAP separately for urban cells (D-019)
- `config.URBAN_BUILT_THRESHOLD = 0.5`; `config.NODATA` sentinel for masked pixels
- RQ5 in PROJECT.md: does the negative `frac_built`-LST correlation survive controls?
- Shastri et al. (2017) added to REFERENCE.md

### Changed
- ROADMAP Phase 1 exit criteria now test data validity only, never agreement with
  the hypothesis
- NDBI is documented as a dryness index everywhere, never as a built-up index
- D-009 and T7.2: scenarios target the hottest decile of *urban* cells, not AOI-wide

### Removed
- `export_to_drive()` and the `DRIVE_FOLDER` / `EXPORT_MAX_PIXELS` constants: an
  unverified duplicate of the verified download path (D-020)

### Data
- 2023: Landsat 8 scenes (LC08 4 / LC09 4), Sentinel-2 14; 81,983 rows, 0 dropped
- 2024: Landsat 11 scenes (LC08 5 / LC09 6), Sentinel-2 17; 81,983 rows, 0 dropped
- 2025: Landsat 9 scenes (LC08 5 / LC09 4), Sentinel-2 14; 81,983 rows, 0 dropped

### Results
- Daytime surface urban **cool** island confirmed in all three years: core minus
  ring LST = -3.50 / -3.25 / -3.81 °C for 2023 / 2024 / 2025
- Mean LST by cover is rank-stable across years: water < tree < built < crop/shrub
  < bare < grass
- `frac_built` correlates negatively with LST every year (-0.193 / -0.187 / -0.196)
  while NDBI correlates positively (+0.662 / +0.634 / +0.613)

---

## [0.1.1] - 2026-10-08

Phase 0 complete. Earth Engine access verified against live data.

### Added
- (T0.2) `requirements-lock.txt`: exact pinned versions of all 178 resolved
  packages in the verified Phase 0 environment, generated from `pip freeze`.
  Setup now installs from the lock; `requirements.txt` remains the
  human-readable list of the 17 direct dependencies. See D-015.
- (T0.3) Noncommercial Earth Engine Cloud project `ushna-aucml` registered
- (T0.4) `earthengine authenticate` run; `init_ee()` verified to succeed

### Changed
- (T0.4) `config.GEE_PROJECT` defaults to `ushna-aucml` when
  `USHNA_GEE_PROJECT` is unset, so the pipeline runs with no env setup
- `INSTRUCTIONS.md` §1: structure tree lists `requirements-lock.txt`
- `INSTRUCTIONS.md` §2: rewritten as the exact reproduction commands for the
  real environment (conda prefix at `./.venv`, install from the lock,
  `pip install -e .`, `earthengine authenticate`), replacing the
  `python3.11 -m venv` instructions that did not match it

### Data
- Live read-only query, no export: 2024 pre-monsoon (2024-03-01 to 2024-06-01)
  over the AOI returned 5 `LANDSAT/LC08/C02/T1_L2` and 6
  `LANDSAT/LC09/C02/T1_L2` scenes, all 11 with `CLOUD_COVER < 30`. Consistent
  with the ~8-day combined Landsat 8+9 revisit across a 92-day window.

---

## [0.1.0] - 2026-10-08

Phase 0 (Setup) complete except for Earth Engine account registration and
authentication, which both need an interactive Google sign-in.

### Added
- (T0.1) Repo skeleton matching `INSTRUCTIONS.md`: `src/ushna/` package with
  per-phase module stubs, four thin notebooks, `data/`, `outputs/`, `reports/`
- (T0.1) `README.md` with setup, layout and the methodological commitments
- (T0.2) `requirements.txt` and a Python 3.11 environment at `.venv/`; all 17
  dependencies installed and import-verified
- (T0.2) `pyproject.toml` for an editable install, so `from ushna import config`
  resolves from any kernel without `sys.path` patching
- (T0.5) `src/ushna/config.py`: AOI, CRS, grid scale, native scales, season
  dates, collection IDs, QA/SCL masks, band scaling, WorldCover class map,
  feature schema, forbidden columns, CV settings, paths, `SEED = 42`
- (T0.5) `init_ee()` in `gee_extract.py`, which fails with a named error when no
  Earth Engine project is configured
- (T0.6) `.gitignore` covering `data/`, `outputs/models/`, `.venv`, credentials
  and caches, while keeping the directory skeleton present in a fresh clone

### Changed
- `INSTRUCTIONS.md`: structure tree now lists `pyproject.toml`; environment
  section covers conda-provisioned 3.11 and the `.venv/python.exe` prefix path;
  Earth Engine section points at `init_ee()` for verification

### Results
- Stack smoke test passed on synthetic data: parquet roundtrip, spatial
  `GroupKFold` with an asserted no-block-overlap check, baseline + LR + RF +
  XGBoost + LightGBM, SHAP `TreeExplainer`, Moran's I, WGS84 to UTM 43N
  reprojection, and a 300 dpi figure write

---

## [0.0.1] - 2026-10-07

### Added
- Project topic finalised: *Concrete vs. Canopy: Explainable ML for Urban Heat Island Drivers in Pune*
- Project named **Ushna**
- Documentation scaffold: PROJECT, ROADMAP, TASKS, INSTRUCTIONS, DECISIONS, REFERENCE, CHANGELOG, LEARNING_LOG
- Team registration submitted (Domain: Smart Cities; SDG 11, SDG 13)

---

<!--
Template for new entries:

## [0.X.0] - YYYY-MM-DD
### Added
- (Tx.y) ...
### Changed
- ...
### Data
- Exported 2024 grid: N rows, M Landsat scenes, K Sentinel-2 scenes
### Results
- Spatial CV RMSE (LightGBM): X.XX ± X.XX °C
-->
