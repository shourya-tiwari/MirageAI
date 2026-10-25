# Dashboard

An interactive Streamlit front end over the pipeline's results, for the viva and
for anyone exploring the project without reading the code.

## Run it

```bash
streamlit run app.py
```

`app.py` in the repository root runs `app/streamlit_app.py`. Streamlit is pinned
in `requirements.txt` with everything else, so no extra install is needed.
Streamlit serves on `http://localhost:8501` (open it in a
browser if it does not open by itself); the theme is in `.streamlit/config.toml`.
The pipeline never imports Streamlit, so it has no effect on any result.

## Design rules

* **Read-only.** The dashboard never refits a model or recomputes a result. It
  reads `outputs/tables`, `outputs/figures` and `outputs/models` through
  `mirageai.results`, so every number shown is the number the report quotes
  (D-049).
* **Logic in the package, UI in the app.** Loading, summarising and
  rasterising live in `src/mirageai/results.py` and are covered by
  `tests/test_results.py`; `app/streamlit_app.py` only lays things out.
* **Degrades gracefully.** The submission zip ships tables and figures but not
  `data/` or the fitted models. A page that needs a missing file says which
  driver creates it (for example `scripts/pipeline/run_phase6.py` for `lgbm_all.joblib`) and
  shows the saved figure instead.
* **Cached.** Tables use `st.cache_data`; the fitted model uses
  `st.cache_resource`, so it loads once per session.
* **Colour follows the entity.** Each model has one fixed colour on every
  chart; maps reuse the report's colour scales (inferno for LST, red–blue
  diverging for SHAP and anomalies).

## Pages

| Page | Content | Needs |
|---|---|---|
| Project overview | Headline metrics (cool island, spatial-CV RMSE, 2025 test, random-CV optimism), pipeline diagram, RQ answers | tables |
| Dataset | Sources, feature dictionary, cleaning, water exclusion; preview and summary statistics of the clean table | tables; `data/` for the preview |
| Exploratory analysis | Interactive maps of LST, anomaly, NDVI, built-up and tree fractions per year; cool-island curve; correlations all vs urban | `data/` for the interactive maps (falls back to figures) |
| Models & validation | Model roles, block sizing, multi-seed CV design, tuned hyperparameters | `best_params.json`, or the shipped `phase4_tuning_*.csv` |
| Evaluation | Comparison table by scheme, paired differences, RMSE vs block size (RQ3), 2025 holdout (seen vs unseen place), urban vs rest, hottest-decile detection, Moran's I | tables |
| Explainability | SHAP importance (LightGBM vs XGBoost, all vs urban), grouped drivers, per-cell maps of the dominant group or any feature's contribution, thresholds, interactions, linear vs tree response | tables, `phase6_shap_cells_lgbm.parquet` |
| Interactive prediction | Sliders describe a cell; the final LightGBM predicts its LST anomaly, shows a per-feature contribution chart, and a lower-bound tree-cover what-if | `lgbm_all.joblib` |
| Results & insights | Scenario bracket (three-model mean and range per bound), dose–response, key findings, scenario maps | tables, figures |
| Limitations & future work | Caveats, future scope, downloads of the report, slides and viva notes | `reports/` |

## Notes on the interactive prediction

* The model predicts `LST_anom`, the cell's difference from that year's
  city-wide median LST, not an absolute temperature.
* The per-input breakdown uses LightGBM's exact path-dependent contributions
  (`pred_contrib=True`), which are instant for one row. The project's global
  rankings use interventional SHAP (D-036), so magnitudes can differ slightly.
* Built-up fraction takes whatever share the other six classes leave, so the
  fractions always sum to one.
* The tree-cover what-if is the **lower bound** of D-009: fractions change,
  spectral indices stay fixed. It reuses `scenarios.add_tree_cover`, so donors
  and the shortfall rule are exactly those of Phase 7. The realistic bound needs
  the per-year index models and is shown, precomputed, on the Results page.
* Inputs with `frac_water` ≥ 0.5 trigger a warning: water-majority cells are
  outside the model's scope (D-034).

## Extending it

Add a summary function to `mirageai/results.py` (with a test), then a section in
the relevant `page_*` function of `app/streamlit_app.py`. Keep numbers coming
from tables rather than typed into the page.
