# Roadmap

Each phase has a clear deliverable and exit criteria. Don't start a phase until the previous one meets its exit criteria. Fill in target dates once the submission deadline is confirmed.

| Phase | Name | Target date | Status |
|---|---|---|---|
| 0 | Setup | 2026-10-08 | ✅ |
| 1 | Data extraction (GEE) | 2026-10-08 | ✅ |
| 2 | EDA & preprocessing | 2026-10-08 | ✅ |
| 3 | Validation design | 2026-10-08 | ✅ |
| 4 | Modelling | 2026-10-16 | ✅ |
| 5 | Evaluation & comparison | 2026-10-16 | ✅ |
| 6 | Explainability (SHAP) | 2026-10-16 | ✅ |
| 7 | Greening scenarios | 2026-10-16 | ✅ |
| 8 | Report, presentation, viva | | 🟨 |
| 9 | Final polish (dashboard, docs) | 2026-10-27 | ✅ |

Status: ⬜ not started · 🟨 in progress · ✅ done

---

## Phase 0: Setup
**Goal:** working repo and environment, with Earth Engine access.
- **Deliverables:** repo skeleton, `requirements.txt`, GEE project registered and authenticated, `src/mirageai/config.py`.
- **Exit criteria:** `python -c "import ee; ee.Initialize(project=...)"` succeeds; repo structure matches `INSTRUCTIONS.md`.
- **Status:** ✅ Done, 2026-10-08. Exit criteria met: `init_ee()` succeeds against
  Earth Engine project `ushna-aucml`, and the repo structure matches
  `INSTRUCTIONS.md`. Live data access was confirmed beyond auth: 11 usable
  Landsat 8+9 scenes over the AOI for the 2024 pre-monsoon season (5 from LC08,
  6 from LC09, all with `CLOUD_COVER < 30`), which matches the expected ~8-day
  combined revisit over a 92-day window.

## Phase 1: Data Extraction
**Goal:** a 100 m grid dataset of features and LST for 2023, 2024 and 2025.
- **Deliverables:** `src/mirageai/gee_extract.py`, `data/raw/uhi_pune_{year}.csv`, `data/raw/uhi_pune_stack_{year}.tif`.
- **Exit criteria:** these test whether the *data* is valid, not whether it agrees
  with any expected spatial pattern. A result that contradicts our hypothesis is a
  finding, not a failed extraction.
  - All three years exported; scene counts logged; row counts sensible (~60k–80k per year).
  - Water cells are cooler than land cells, which confirms the sign and scaling of the LST conversion.
  - Map orientation verified against known landmarks (Khadakwasla reservoir to the south-west, the Mula–Mutha confluence in the centre).
  - Land-cover fractions sum to ~1 in every cell.
  - LST lies in a physically explainable range, and both outlier tails are accounted for by their land cover.
  - No coordinate column appears in `FEATURES`.
- **Status:** ✅ Done, 2026-10-08. All three years extracted and every exit
  criterion met. Scene counts 2023/2024/2025: Landsat 8/11/9, Sentinel-2 14/17/14.
  81,983 rows per year, 0 cells dropped. Driver signs and magnitudes are stable
  across all three years (see `outputs/tables/phase1_cross_year_checks.csv`).

## Phase 2: EDA & Preprocessing
**Goal:** understand the data and produce a clean modelling table.
- **Deliverables:** `notebooks/02_eda.ipynb`, `data/processed/uhi_pune_clean.parquet`, EDA figures.
- **Exit criteria:**
  - Missing-value report done.
  - Target distribution and LST maps plotted.
  - Correlation matrix and VIF computed.
  - Feature-handling decisions logged in `DECISIONS.md`.
- **Status:** ✅ Done, 2026-10-08. 245,949 rows x 26 cols, zero nulls. Target
  switched to the within-year anomaly (D-021); cell and block ids made year-independent
  and leakage-tested (D-022); `frac_built` confirmed as the linear-model reference
  by VIF (D-023); NDBI retained as a dryness feature (D-024).

## Phase 3: Validation Design
**Goal:** a leakage-proof evaluation setup.
- **Deliverables:** `src/mirageai/cv.py` with spatial block assignment and splitters for random, spatial and temporal schemes.
- **Exit criteria:**
  - Block size justified with an empirical variogram or autocorrelation range.
  - Unit check passes: no block appears in both train and test.
- **Status:** ✅ Done, 2026-10-08. Variogram range 7.65 km (stable across years),
  so 8 km blocks are primary (D-026); 1, 2 and 4 km retained for sensitivity. Three
  splitters in `src/mirageai/cv.py`; the pure temporal split is replaced by a
  spatio-temporal one (D-010 amended). 27 tests pass in `tests/test_cv.py`.
  Fold 3 is urban-heavy (+17.8 pts) and is reported rather than corrected (D-027).

## Phase 4: Modelling
**Goal:** four tuned models.
- **Deliverables:** `src/mirageai/models.py`, tuned hyperparameters saved, `outputs/models/`.
- **Exit criteria:**
  - All 4 models train under every CV scheme.
  - Tuning runs inside spatial CV.
  - Seeds are fixed.
  - Results are reproducible on rerun.
- **Status:** ✅ Done, 2026-10-16 (T4.8, one command: `scripts/pipeline/run_phase4.py`). Five
  models x six schemes (random; spatial at 1/2/4/8 km; spatio-temporal at 4 km),
  tuned on 2023-24 at the 4 km primary block (D-028/D-029), with the full RF space
  restored (D-030 resolved). CPU only, one forest in memory at a time (D-031); the
  peak was 1.30 GB and the whole run took ~31 min. Reruns reproduce fold
  predictions bit-for-bit. At 4 km the three tree models tie at RMSE ~1.79 °C
  (R² ~0.74), against 1.60-1.64 under random CV. Provisional results are kept
  as `*_provisional`.

## Phase 5: Evaluation & Comparison
**Goal:** honest, comparative results.
- **Deliverables:** comparison table (4 models × random / spatial / temporal), residual maps, Moran's I on residuals.
- **Exit criteria:** you can state the best model, the size of the random-vs-spatial CV gap, and where errors concentrate.
- **Status:** ✅ Done, 2026-10-16 (0.6.0; `scripts/pipeline/run_phase4.py` then `scripts/pipeline/run_phase5.py`).
  Water-majority cells excluded (D-034, 2,697 rows); 5 shuffled block layouts
  as headline, strips as a stress test (D-033); 4 km kept on the D-028 gate
  (3,950 m), with 8 km as the conservative bound. Best model: a three-way tree
  tie at 1.746-1.756 °C (R² 0.73); LightGBM is the reference (D-032). Gap,
  random to 4 km: +0.12 to +0.15 °C for the trees, +0.023 for linear. 2025
  spatio-temporal: 1.747 °C, R² 0.721. Errors concentrate in the hottest
  decile (54% urban detection) and remain spatially clustered (residual
  Moran's I ≈ 0.6, unmodelled drivers).

## Phase 6: Explainability
**Goal:** answer RQ1 and RQ2.
- **Deliverables:** SHAP summary, dependence plots for the top 4 drivers, interaction plot (e.g., built-up × water distance), driver ranking table.
- **Exit criteria:** each top driver has a one-line, plain-language interpretation, for example: "cooling from tree cover saturates above X%".
- **Status:** ✅ Done, 2026-10-16 (0.7.0; one command: `scripts/pipeline/run_phase6.py`). The dominant-driver map uses grouped SHAP (D-038): spectral 66.8% of cells, dry cover 24.5%. Spectral block first (1.49 °C mean |SHAP|), then grass, tree cover and cropland; LightGBM and XGBoost rankings agree (Spearman 0.99). Tree-cover cooling saturates at ~10% cover. The frac_built cool-island association shrinks 87% once vegetation and dryness are modelled. Every top driver has a one-line interpretation in `reports/phase6_findings.md`.

## Phase 7: Greening Scenarios
**Goal:** answer RQ4.
- **Deliverables:** `src/mirageai/scenarios.py`, ΔLST map, summary table (e.g., +10% / +20% tree cover in the hottest decile).
- **Exit criteria:** the scenario method and its associational caveat are written up in `DECISIONS.md`.
- **Status:** ✅ Done, 2026-10-16 (0.8.0; one command: `scripts/pipeline/run_phase7.py`). D-009 resolved as a lower / realistic bracket over three models. 2,673 target cells, mostly on the eastern fringe. Headline, excluding the 0 → 1% "any canopy" step: +10 points −0.37 to −1.02 °C, +20 points −0.56 to −1.63 °C. Gains per point roughly halve past the 10.2% knot. Tree-less urban cells run ~1.4 °C hotter, reported as a separate finding. The caveat is in D-009 and `reports/phase7_findings.md`.

## Phase 8: Report, Presentation, Viva
**Goal:** convert work into marks.
- **Deliverables:** final report (PDF), slide deck, viva prep notes.
- **Exit criteria:**
  - Report has every rubric section: problem, methodology, dataset, results, conclusion, future scope, references.
  - Every team member can explain every decision in `DECISIONS.md`.
- **Status:** 🟨 Part 1 done, 2026-10-16: report draft (T8.1-T8.3). `scripts/build_report.py` builds `reports/MirageAI_Report.docx` / `.pdf` (20 pages) from `reports/report.md`, after 211 number checks pass. 19 of 21 references verified by DOI. Part 2 done the same day: slides (T8.4, `reports/MirageAI_Slides.pptx` / `.pdf`, 13 slides, ~10.7 min of notes) and viva prep (T8.5, `reports/viva_prep.md`), all numbers checked (371 checks). Next: clean-clone run (T8.6). Report and deck were later rewritten in Phase 9 (IEEE paper, 19 slides; D-053, D-054).

---

## Phase 9: Final polish
**Goal:** make the project easy to understand, demonstrate and defend.
- **Deliverables:** Streamlit dashboard, rewritten README and docs, `project_deep_dive.md`, `learning_log.md`, executed walkthrough notebooks; a clean root (drivers in `scripts/pipeline/`) and one pinned `requirements.txt`.
- **Exit criteria:** every result unchanged (tests pass, outputs untouched); the dashboard runs from the shipped outputs; docs match the code.
- **Status:** ✅ Done, 2026-10-27 (D-049 to D-054): dashboard, docs, IEEE-format report (15 pages), 19-slide deck, MIT license, final submission zip. Remaining team items (PRNs if required, the clean-clone run T8.6, rehearsal) are in `reports/SUBMISSION_CHECKLIST.md`.

---

## If time runs short (minimum viable path)
Do these in order and stop wherever the deadline hits. Each step still gives a complete project:
1. One year only (2024), spatial CV, 4 models, SHAP.
2. Add the random-vs-spatial CV comparison. This is the strongest methodological point.
3. Add greening scenarios.
4. Add the temporal holdout with all three years.
