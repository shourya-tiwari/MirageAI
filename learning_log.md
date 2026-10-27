# MirageAI Learning Guide

A beginner-to-advanced course built entirely on this repository. Each module
teaches one layer of the project: the theory, the intuition, how MirageAI does
it (with file pointers), common mistakes, interview and viva questions, and a
short revision summary.

> This is the study guide. The project's dated research diary (what was found,
> when, and what changed because of it) is `docs/LEARNING_LOG.md`.

**How to use it.** Read the modules in order the first time. Before the viva,
read only the *Revision summary* and the *Viva questions* of each module, and
answer the questions aloud without notes. Decision numbers (D-0xx) refer to
`docs/DECISIONS.md`.

| # | Module | Level |
|---|---|---|
| 0 | [Prerequisites](#module-0-prerequisites) | Beginner |
| 1 | [Problem understanding](#module-1-problem-understanding) | Beginner |
| 2 | [The dataset](#module-2-the-dataset) | Beginner |
| 3 | [Preprocessing](#module-3-preprocessing) | Beginner → Intermediate |
| 4 | [Exploratory data analysis](#module-4-exploratory-data-analysis) | Intermediate |
| 5 | [Feature engineering](#module-5-feature-engineering) | Intermediate |
| 6 | [Core ML concepts](#module-6-core-ml-concepts) | Intermediate |
| 7 | [Validation and training](#module-7-validation-and-training) | Intermediate → Advanced |
| 8 | [Evaluation](#module-8-evaluation) | Advanced |
| 9 | [Explainability with SHAP](#module-9-explainability-with-shap) | Advanced |
| 10 | [Scenarios and causality](#module-10-scenarios-and-causality) | Advanced |
| 11 | [Streamlit and presentation](#module-11-streamlit-and-presentation) | Intermediate |
| 12 | [Repository walkthrough](#module-12-repository-walkthrough) | Intermediate |
| 13 | [Design decisions](#module-13-design-decisions) | Advanced |
| 14 | [Limitations and future scope](#module-14-limitations-and-future-scope) | All |
| | [Final revision sheet](#final-revision-sheet) | All |

---

## Module 0: Prerequisites

### What you need to know

* **Python:** functions, modules, imports, list/dict comprehensions.
* **pandas:** `DataFrame`, `groupby`, `merge`, boolean indexing, `read_csv`/`read_parquet`.
* **numpy:** arrays, vectorised arithmetic, `np.floor`, random generators.
* **scikit-learn basics:** `fit` / `predict`, `Pipeline`, train/test split, cross-validation.
* **Statistics:** mean, median, variance, correlation, linear regression, quantiles.
* **Maps:** latitude/longitude vs projected coordinates in metres.

### Key idea: geographic vs projected coordinates

Latitude/longitude are angles; one degree of longitude is not one degree of
latitude in metres. A projected CRS such as **UTM zone 43N (EPSG:32643)** turns
positions into metres, so distances, areas, slopes and square blocks make sense.
MirageAI does all processing in UTM (`config.CRS`).

### Set up the project

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS / Linux: source .venv/bin/activate)
python -m pip install --upgrade pip
pip install -r requirements.txt   # direct dependencies, pinned
pip install -e .                  # the mirageai package itself, editable
python -m pytest -q
```

### Common mistakes
* Using a system Python 3.13: the geospatial wheels are most fragile on the newest interpreter (D-011).
* Installing packages one by one instead of from `requirements.txt`, which skips the editable
  `mirageai` install and leads to patching `sys.path` in notebooks (D-012).

### Revision summary
Python + pandas + scikit-learn basics, and the difference between degrees and metres.

---

## Module 1: Problem understanding

### Theory
An **urban heat island (UHI)** is a city warmer than its surroundings. The
**surface** UHI is measured with **land surface temperature (LST)** from
satellites; the **air** UHI with weather stations. They are related but not the
same.

### The research gap MirageAI targets
1. Earlier studies correlate LST with one index (often NDVI) using linear
   models: they miss thresholds and interactions.
2. They validate with random splits, which leak information between neighbouring
   pixels and overstate accuracy.

### Intuition: the "mirage"
You would expect concrete to be hottest. In Pune before the monsoon the opposite
holds in the daytime: harvested fields, dry grass and bare soil around the city
have no water to evaporate and heat fast, while the city has trees, irrigated
gardens and shade. The core is **3.4–4.0 °C cooler** than the outer ring. The
project explains *why*, with a model that separates greenness and dryness from
built-up cover.

### Research questions
RQ1 drivers; RQ2 nonlinear vs linear; RQ3 random vs spatial CV; RQ4 cooling from
tree cover; RQ5 the cool island under controls (`docs/PROJECT.md` §3).

### Stakeholders
Pune and Pimpri-Chinchwad municipal corporations, urban planners, public-health
bodies, residents of the hottest wards (SDG 11, SDG 13).

### Common mistakes
* Treating LST as "how hot it feels" (D-047).
* Framing the project as "city vs countryside": within land-cover classes,
  built-up is still hotter than tree cover; what fails is the naive regional framing.

### Interview questions
1. *What is the difference between surface and air UHI?* Surface UHI uses the
   radiative temperature of surfaces from satellites; air UHI uses near-ground
   air temperature from stations. Surface UHI is strongest in the daytime, air
   UHI usually at night.
2. *Why is this an ML problem and not just a correlation study?* Many correlated
   drivers act nonlinearly and interact; a model plus SHAP can separate them.

### Viva questions
* Why Pune, why pre-monsoon? (Peak heat and the clearest skies, D-002.)
* What would a planner do with your results? (Prioritise greening in tree-less
  fringe neighbourhoods, knowing the numbers are associational.)

### Revision summary
Surface (not air) heat; two research gaps (linearity, leaky validation); the
city core is cooler, and the project explains why.

---

## Module 2: The dataset

### Sources (all via Google Earth Engine)

| Source | Gives | Resolution |
|---|---|---|
| Landsat 8/9 Collection 2 Level-2 | LST (target), from band `ST_B10` | 100 m thermal |
| Sentinel-2 SR | NDVI, NDBI, MNDWI | 10–20 m |
| ESA WorldCover v200 (2021) | 7 land-cover fractions, distance to water | 10 m |
| SRTM | elevation, slope | 30 m |

### Structure
81,983 cells × 3 years = 245,949 rows. **The years are repeated measurements of
the same cells**, not independent samples: land cover and terrain are identical
every year; only LST and the indices change. That one fact drives many
decisions (cell-level grouping, block ids from coordinates, spatio-temporal
testing).

### Key concepts
* **Cloud masking:** Landsat `QA_PIXEL` bits (cloud, shadow, cirrus) and the
  Sentinel-2 `SCL` classes mark bad pixels.
* **Median composite:** the per-pixel median over the season is robust to
  leftover cloud and odd scenes (D-005).
* **Scaling:** `LST (°C) = ST_B10 × 0.00341802 + 149.0 − 273.15`.
* **Area-weighted aggregation:** reproject to the source's native scale, then
  `reduceResolution(mean)` onto 100 m (`gee_extract.to_grid`).

### Worked example: a land-cover fraction
A 100 m cell contains 100 WorldCover pixels of 10 m. If 37 are "built-up", the
binary built-up mask averages to `frac_built = 0.37`.

### Common mistakes
* Computing slope in degrees-of-latitude units (EPSG:4326). MirageAI reprojects
  SRTM to UTM first (D-016).
* Reading a nodata value as a real zero. MirageAI writes an explicit −9999 (D-017).
* Forgetting that `filterDate` end is exclusive: Mar–May is `03-01` to `06-01`.

### Interview questions
1. *Why 100 m and not 30 m?* Landsat's thermal sensor is natively ~100 m; the
   30 m product is resampled, so 30 m would be false precision (D-001).
2. *Why is a median composite better than a mean?* The median ignores outliers
   such as residual cloud; the mean is pulled by them.

### Viva questions
* How many scenes per year? (Landsat 8/11/9; Sentinel-2 14/17/14.)
* Were any cells dropped for missing data? (None: the clearest season left every
  cell with at least 4.35 clear observations on average.)

### Revision summary
Four free sources; 100 m grid; three years of the same 81,983 cells; median
composites; everything in UTM.

---

## Module 3: Preprocessing

### Steps in MirageAI (`src/mirageai/preprocess.py`)
1. Merge the years (`load_all`).
2. Cell id from coordinates with `floor` (`add_cell_id`).
3. Block ids at 1, 2, 4, 8 km (`add_blocks`).
4. Target anomaly (`add_lst_anom`).
5. Urban flag (`add_urban_flag`).
6. Consistency check (`check_block_consistency`).
7. Drop water-majority cells for modelling (`load_model_table`, D-034).

### Theory: why `floor`, not `round`
Cell centres are at x = k·100 + 50, so x/100 = k + 0.5 exactly. numpy rounds
half to even: 3647.5 → 3648 and 3648.5 → 3648. Two columns get one id. `floor`
gives 3647 and 3648. The leakage test caught this immediately (D-022).

### Theory: the anomaly target
`LST_anom = LST − median(LST that year)`. The year-to-year level depends on which
scene dates each year contained, which no feature encodes. Removing it lets the
model learn *where* a cell sits within its year (D-021).

### Theory: scaling inside the fold
Linear regression needs standardised features for comparable coefficients; the
`StandardScaler` sits inside a `Pipeline`, so its mean and SD are learned from
the training fold only. Trees need no scaling.

### Common mistakes
* Fitting a scaler or imputer on the full dataset before CV: test statistics leak
  into training.
* Assigning blocks per year: the three copies of a cell could land in different
  folds.
* Dropping a feature "because it's collinear" without checking what it means
  (NDBI, D-024).

### Interview questions
1. *Where should preprocessing happen relative to cross-validation?* Inside each
   fold, ideally via a `Pipeline`, so nothing is learned from test data.
2. *Why remove water cells instead of adding a water flag?* Water temperature is
   governed by thermal inertia, not land-surface drivers, was the largest error,
   and is not a planning lever (D-034).

### Viva questions
* How many rows did water exclusion remove? (2,697: 899 cells × 3 years.)
* Is the anomaly recomputed without water? (No; recomputing would shift all
  anomalies by a constant ~0.05 °C and change no ranking.)

### Revision summary
Stable ids from coordinates (with floor), anomaly target, urban flag, water out,
and every learned transform inside the fold.

---

## Module 4: Exploratory data analysis

### What MirageAI looked at (`scripts/pipeline/run_phase2.py`, notebook 02)
* LST and NDVI maps per year: the core is green and cool; the periphery dry and hot.
* Target distributions: absolute LST is offset between years; the anomaly is not.
* Feature distributions: fractions are zero-inflated (many cells have 0% of a class).
* Correlation matrix and **VIF**.
* Correlations with the target for all cells vs urban cells.

### Theory: VIF
VIF of feature j = 1 / (1 − R²ⱼ), where R²ⱼ is from regressing feature j on all
other features. VIF > 10 signals strong multicollinearity. The seven fractions
sum to 1, so with all of them VIF is infinite.

### Theory: range restriction
Restricting to urban cells (built-up ≥ 50%) truncates `frac_built`, so its
correlation with LST shrinks mechanically. A shrinking correlation in a subset
is therefore *not* evidence that a relationship disappeared (LEARNING_LOG,
2026-10-08).

### Key EDA numbers
* NDBI correlates +0.63 with the anomaly; `frac_built` −0.19.
* The 17 hottest cells are hillside grassland at ~771 m elevation.
* Dropping `frac_built` gives median VIF 2.2; dropping `frac_bare` 11.8.

### Common mistakes
* Concluding "buildings cool the city" from a negative correlation: built-up land
  sits in the greener core (confounding).
* Comparing absolute LST across years without noting the scene-date artefact.

### Interview questions
1. *What is confounding? Give an example from this project.* A third variable
   drives both: greenness/dryness is correlated with both built-up share and LST,
   creating a negative built-up–LST correlation that is not causal.
2. *How do you detect multicollinearity?* Correlation matrix and VIF.

### Viva questions
* Why does NDBI behave as dryness in Pune? (Bare dry soil has high SWIR
  reflectance, like roofs; in a semi-arid periphery it dominates.)

### Revision summary
Maps first; check distributions, collinearity (VIF), and beware confounding and
range restriction.

---

## Module 5: Feature engineering

### The 13 features (`config.FEATURES`)
* Spectral: NDVI, NDBI, MNDWI.
* Fractions: tree, shrub, grass, crop, built-up, bare, water.
* Terrain/water: elevation, slope, distance to water.

### Deliberate exclusions
* **Coordinates** (`lon`, `lat`, `x_utm`, `y_utm`): a model with coordinates
  learns *where* it is hot, not *why*, and coordinates cannot be changed in a
  scenario (D-004).
* **`LST_n_obs`**: quality control, not a property of the land.
* **Block ids, cell ids, the target and the urban flag**: listed in
  `config.FORBIDDEN_FEATURES` and tested.

### The sum-to-one problem
Seven fractions summing to 1 are perfectly collinear: ordinary least squares has
no unique solution. Drop one for linear regression only. MirageAI drops
`frac_built` (best conditioning; coefficients then read "replace built-up with
X", the scenario question) and keeps all seven for trees (D-023, D-025).

### Worked example
With `frac_built` as the reference, a linear coefficient of +0.93 °C per SD on
`frac_grass` means: replacing built-up with grassland is associated with a
warmer cell.

### Feature groups for explanation (D-038)
spectral · dry cover (grass, crop, bare, shrub) · tree cover · built · water
proximity · terrain. Summed SHAP within a group is still additive.

### Common mistakes
* Dropping the rarest category to fix the sum-to-one issue (worst conditioning).
* Removing `frac_built` from tree models too, which makes the title question
  ("concrete vs canopy") unanswerable.
* Replacing correlated indices with a principal component, which destroys
  interpretability.

### Interview questions
1. *Why exclude coordinates from a spatial model?* They let the model memorise
   locations, inflate spatial-CV-free scores and make driver analysis
   meaningless.
2. *What is the dummy-variable trap and how does it relate here?* Including all
   categories plus an intercept makes the design singular; fractions summing to
   one are the continuous version.

### Viva questions
* Why do linear and tree models have different feature counts? (12 vs 13 by
  design; stated in every comparison table.)

### Revision summary
13 physical features, never coordinates; drop one fraction for OLS only, chosen
for conditioning and interpretation.

---

## Module 6: Core ML concepts

### Linear regression
Minimises squared error with a linear function. Fast, interpretable, but cannot
represent thresholds or interactions unless engineered, and its coefficients are
unstable under collinearity.

### Decision trees
Split the feature space into rectangles by thresholds; predict the mean in each
leaf. Capture nonlinearity and interactions; overfit if grown deep.

### Random forest (bagging)
Many deep trees, each on a bootstrap sample and a random subset of features
(`max_features`); average the predictions. Averaging reduces variance.

### Gradient boosting (XGBoost, LightGBM)
Trees added sequentially, each fitting the residual errors of the ensemble so far,
scaled by a learning rate. Reduces bias; regularised by depth/leaves, learning
rate, row/column subsampling and L2 penalties. LightGBM grows leaf-wise
(`num_leaves`), XGBoost level-wise (`max_depth`); both use histogram binning.

| | Bagging (RF) | Boosting (XGB/LGBM) |
|---|---|---|
| Trees built | independently, in parallel | sequentially |
| Mainly reduces | variance | bias |
| Key knobs | trees, depth, `max_features`, leaf size | trees, learning rate, depth/leaves, subsampling |

### Hyperparameter tuning
`RandomizedSearchCV` samples a fixed number of configurations (25 here) from the
search space and scores each by cross-validation. Cheaper than grid search and
nearly as good when few parameters matter.

### Why scaling matters for linear models, not trees
Trees split on thresholds, which are invariant to monotonic rescaling.

### Common mistakes
* Choosing a model by random-CV score (favours the most flexible model; see Module 8).
* Using the GPU for speed and losing bit-for-bit reproducibility (D-031).

### Interview questions
1. *Bagging vs boosting?* See the table.
2. *Why might LightGBM be preferred over random forest here?* Similar accuracy,
   much smaller model (2.8 MB vs 115 MB), and far faster SHAP (D-032).
3. *What does `learning_rate` trade off?* Smaller steps need more trees but
   generalise better.

### Viva questions
* What were LightGBM's tuned parameters? (1,200 trees, 63 leaves, learning rate
  0.01, `min_child_samples` 10, `colsample_bytree` 0.6.)
* Which model is best? (The three tree models tie under spatial CV; LightGBM is
  the reference for explanation, not the clear winner.)

### Revision summary
Linear = interpretable but rigid; RF = bagging; XGB/LGBM = boosting; tune with
randomized search inside the right CV.

---

## Module 7: Validation and training

### Spatial autocorrelation
"Near things are more related than distant things." Neighbouring 100 m cells
have almost the same LST (Moran's I ≈ 0.89).

### Why random CV leaks
A random test cell almost always has a near-twin neighbour in training. The model
can score well by interpolating, which says nothing about predicting a new
neighbourhood.

### Spatial block CV
Hold out whole blocks of the map. MirageAI: 4 km blocks, `GroupKFold` with
shuffling, five seeds (42–46) × five folds = 25 runs per result (`cv.py`).

### Choosing the block size: the variogram
Semivariance γ(h) = ½ mean[(z(x) − z(x+h))²]. The *range* is where γ reaches 95%
of its plateau (sill). Raw target range 7.65 km; residual range ~3.7–4.0 km.
Blocks must defeat the dependence the model **cannot** explain, so the residual
range sets the size: 4 km (D-028). All sizes are still reported.

### Spatio-temporal CV
Train on 2023–24 in some blocks, test on 2025 in the other blocks: unseen in
space *and* time. A pure temporal split would test the same places, whose land
cover is identical, and measure memorisation (D-010).

### Tuning without leakage
Search inside the spatial folds, on training years only, so 2025 is never seen
(D-029). The remaining small optimism (selection and evaluation share folds) is
stated; the 2025 test is the clean check.

### Training at scale
* One fitted forest in memory at a time; out-of-fold predictions streamed to
  parquet parts (D-030, D-031).
* Final models refitted on all applicable rows after CV.

### Worked example: a fold
Seed 42, fold 2: the blocks dealt to fold 2 are the test set (all three years for
spatial CV, only 2025 for spatio-temporal); the rest train.

### Common mistakes
* Unshuffled `GroupKFold` with id-ordered blocks: MirageAI's 4 km folds became
  latitude strips by accident (D-033). A test now checks fold shape.
* Sizing blocks from the raw target's variogram, which over-sizes them.
* Stratifying blocks by content to "balance" folds, which breaks spatial logic (D-027).

### Interview questions
1. *What is data leakage? Give three forms.* Information about the test set
   reaching training: preprocessing on the full data; neighbouring pixels in
   random CV; tuning on the test set.
2. *How do you pick a spatial block size?* At least the autocorrelation range,
   from a variogram, ideally of residuals.
3. *What is GroupKFold?* K-fold where all rows of a group stay in one fold.

### Viva questions
* Why 4 km and not 8 km? (Residual range median 3,950 m → 4 km; 8 km reported as
  the conservative bound, tree RMSE 1.84–1.85 °C.)
* Why five seeds? (One layout is arbitrary; five make layout effects negligible,
  seed-to-seed std ≤ 0.011 °C.)

### Revision summary
Random CV leaks; spatial blocks sized by residual variogram (4 km); five
layouts; spatio-temporal test on 2025; tuning inside training-year folds.

---

## Module 8: Evaluation

### Metrics
| Metric | Formula | Use |
|---|---|---|
| RMSE | √mean(e²) | headline, °C; penalises large errors |
| MAE | mean\|e\| | robust, °C |
| R² | 1 − SS_res/SS_tot | share of variance; depends on the target's spread |
| Moran's I (residuals) | spatial autocorrelation | ~0 = no spatial pattern left |

Reported as **mean ± std over 25 runs**.

### Results
* LightGBM: random 1.620 → 1 km 1.675 → 2 km 1.706 → **4 km 1.746** → 8 km 1.840 °C.
* **2025 spatio-temporal:** 1.747 °C, R² 0.72.
* Trees tie at 4 km (1.746–1.756); linear 1.873; baseline 3.451.
* Inflation random → 4 km: trees +0.12 to +0.15 °C, linear +0.023.

### Key insight: inflation depends on model flexibility
Flexible models gain most from memorising local texture, so random CV not only
inflates scores, it **biases model comparison** toward the most flexible model.
XGBoost wins under random CV and is tied under spatial CV.

### Paired comparison
Differences between models on identical folds remove between-fold variance.
LightGBM vs XGBoost at 4 km: −0.000 ± 0.006 °C (a tie).

### Subgroups and extremes
* Urban RMSE is lower (1.614) but R² too (0.54): the urban range is narrower.
  Quote subgroup accuracy in °C.
* Hottest decile: residuals are biased by selecting on the observed value; use
  detection overlap instead: 54.2% within urban cells (chance 10%).

### Residual Moran's I ≈ 0.6
Errors are spatially clustered: drivers the features miss (building height,
roofing, irrigation). It also means n = 243k rows overstates independent
information, so uncertainty is reported as fold std, never per-cell standard errors.

### Common mistakes
* Comparing subgroup R² with whole-data R².
* Reporting a single split's score without its spread.
* Calling a 0.002 °C difference a win when the fold std is 0.06.

### Interview questions
1. *RMSE vs MAE: when does each mislead?* RMSE is dominated by a few large
   errors; MAE hides them.
2. *Why can a subgroup have lower RMSE and lower R²?* R² divides by the
   subgroup's variance, which is smaller.
3. *What does Moran's I on residuals tell you?* Whether errors are spatially
   structured, i.e. whether spatial information is missing from the model.

### Viva questions
* What is the place-memorisation gap? (Seen-place 2025 RMSE 1.598 vs unseen 1.747
  for LightGBM: about as large as the spatial leak.)
* Why is LightGBM the reference model? (Tie-break on the clean 2025 test and SHAP
  cost, D-032.)

### Revision summary
Mean ± std over 25 runs; error rises with block size; random CV flatters flexible
models; pair comparisons; quote subgroups in °C; residual Moran's I ≈ 0.6.

---

## Module 9: Explainability with SHAP

### Theory
The Shapley value of feature j for one prediction is its average marginal
contribution across all orderings of features. Properties: **additivity** (base
value + Σ SHAP = prediction), symmetry, dummy, consistency. **TreeSHAP** computes
them exactly in polynomial time for trees.

### Path-dependent vs interventional
* *Path-dependent:* follows the tree's own cover statistics; credit can leak
  between correlated features.
* *Interventional:* replaces features with values from a background dataset;
  credits only the model's response. MirageAI uses it (1,000-row background,
  D-036). The choice reorders NDVI and NDBI, so it matters.

### Engineering lessons
* Fit on float32-rounded features, or additivity fails (shap casts to float32).
* Never disable `check_additivity`; it caught a real problem.
* Interventional SHAP is single-core per row; parallelise over chunks (8 workers).

### Reading the plots
* **Bar / mean |SHAP|:** global importance.
* **Beeswarm:** each dot a cell; position = SHAP, colour = feature value. Red on
  the right means high values push LST up.
* **Dependence:** SHAP vs feature value; look for knots and saturation.
* **Interaction values:** the part of the prediction due to a pair acting together.

### MirageAI's findings
* Spectral block 1.49 °C mean |SHAP| (≈2× dry cover, 0.72), then tree cover 0.38;
  built-up 0.12 (10th of 13).
* Tree-cover cooling saturates at a knot of 10.2% (CI 9.6–10.6%).
* Top interaction: NDBI × grassland.
* Built-up association shrinks 87% versus its raw correlation (RQ5).
* Dominant driver group: spectral 66.8% of cells, dry cover 24.5%.

### Worked example
A cell's prediction is +2.0 °C; base value +0.5 °C. SHAP: NDBI +1.2, NDVI +0.6,
tree cover −0.4, others +0.1. Sum: 0.5 + 1.2 + 0.6 − 0.4 + 0.1 = 2.0 °C.

### Common mistakes
* Reading SHAP as causal ("planting trees will cool by X"). SHAP explains the model.
* Ranking individual collinear features confidently; rank the group.
* Reading a "threshold at 1%" in a zero-inflated feature as a real threshold
  (it is the step from none to some, D-037).

### Interview questions
1. *Why SHAP over built-in feature importance?* Consistent, signed, local, and
   not biased toward high-cardinality features (D-008).
2. *What is the additivity property?* Base + contributions = prediction.
3. *When is interventional SHAP preferable?* When features are correlated and you
   want credit only through the model's response.

### Viva questions
* How do you know the SHAP values are correct? (Additivity checked to 1e-7; the
  parallel run equals the serial run; XGBoost ranks agree at Spearman 0.99.)
* Why a driver *group* map? (Per-group agreement across models 95% vs 79% per feature.)

### Revision summary
Shapley = fair additive credit; interventional for correlated features; check
additivity; rank groups; associational language.

---

## Module 10: Scenarios and causality

### What a scenario is here
Edit a cell's features (more tree cover, less built-up/bare/grass), predict
again, report the change ΔLST. This is a **counterfactual on the model**, not on
the world.

### MirageAI's design
* Targets: the 2,673 urban cells in the hottest decile of observed 3-year mean anomaly (D-039).
* Edits: +10 points (A), +20 (B), up to the 10.2% knot (C), taken from
  built-up, bare and grass in proportion (D-040).
* **Lower bound:** indices held fixed. **Realistic bound:** indices updated by
  land-cover → index models, applying only the predicted difference (D-041).
* Three models, report the range (D-042). Support guard flags rows unlike any
  training row (D-043).
* Headline excludes the first point (0 → 1%): A −0.37 to −1.02 °C, B −0.56 to
  −1.63 °C.

### Correlation vs causation
The models learn how LST differs *between* cells with different tree cover.
Those cells also differ in unobserved ways (building density, roofs, irrigation).
The scenario credits tree cover with whatever travels with it in the data. Only
a causal design (e.g., before/after planting with controls) could estimate the
effect of planting.

### Common mistakes
* Presenting ΔLST as "planting trees will cool by X °C".
* Changing tree cover without reducing another class (fractions no longer sum to 1).
* Extrapolating to feature combinations never seen in training.
* Including the zero-to-some step, which reflects the type of place, not a tree.

### Interview questions
1. *How do you check a counterfactual is plausible?* Support/extrapolation checks
   (nearest-neighbour distance), physical constraints (sum to one), and
   co-varying features (indices).
2. *What is confounding in this scenario?* Unobserved urban form correlated with
   tree cover and LST.

### Viva questions
* Why select targets on observed rather than predicted heat? (Avoids the model's
  54% detection limit; deltas compare the model with itself.)
* Why a bracket? (Neither bound alone is realistic; the truth is plausibly between.)

### Revision summary
Scenario = model counterfactual; two bounds × three models; support guard; first
point excluded; never causal.

---

## Module 11: Streamlit and presentation

### Streamlit basics
A Python script re-runs top to bottom on every interaction. Widgets
(`st.slider`, `st.selectbox`, `st.radio`) return values; layout uses
`st.sidebar`, `st.columns`, `st.tabs`; `st.metric` shows KPI cards;
`st.plotly_chart` shows interactive charts.

### Caching
* `@st.cache_data` for data (tables): returns a copy, keyed on arguments.
* `@st.cache_resource` for objects that should be shared and not copied (the
  fitted model).

### MirageAI's dashboard (`app/streamlit_app.py`, `docs/DASHBOARD.md`)
* Read-only: every number from `outputs/` via `mirageai.results` (D-049).
* Logic in the package (tested), UI in the app.
* Missing files degrade to saved figures with a note on how to create them.
* Interactive prediction: sliders → LightGBM → per-feature contributions →
  tree-cover what-if (lower bound).

### Presentation principles used in the project
* Maps first: the story is spatial.
* One fixed colour per model; diverging red–blue for signed effects.
* Every number in the report, slides and viva notes is checked by a script.

### Common mistakes
* Recomputing results in the dashboard (they can drift from the report).
* Loading a large model inside a function without caching (reloads on every click).

### Interview questions
1. *`cache_data` vs `cache_resource`?* Data is serialised and copied per call;
   resources are shared singletons.
2. *How would you deploy this?* Streamlit Community Cloud or a container; ship
   the outputs, not the raw data.

### Revision summary
Streamlit reruns the script; cache data and resources; keep the dashboard
read-only over checked outputs.

---

## Module 12: Repository walkthrough

| Path | Role | Key items |
|---|---|---|
| `src/mirageai/config.py` | constants and schema | `FEATURES`, `FORBIDDEN_FEATURES`, seeds, thresholds, paths |
| `gee_extract.py` | Phase 1 | `to_grid`, `landsat_lst`, `s2_indices`, `extract_year` |
| `preprocess.py` | Phase 2 | `build_clean_table`, `add_lst_anom`, `vif_table`, `load_model_table` |
| `cv.py` | Phase 3 | `empirical_variogram`, `spatial_splits`, `spatio_temporal_splits`, `strip_splits` |
| `models.py` | Phase 4 | `make_model`, `SEARCH_SPACES`, `tune`, `oof_predictions`, `PeakMemory` |
| `evaluate.py` | Phase 5 | `model_comparison`, `paired_differences`, `cv_gap`, `morans_i` |
| `explain.py` | Phase 6 | `interventional_shap`, `DRIVER_GROUPS`, `breakpoint` |
| `scenarios.py` | Phase 7 | `target_cells`, `add_tree_cover`, `updated_indices`, `SupportGuard` |
| `plotting.py` | maps | `cells_to_grid`, `map_panel` |
| `results.py` | read-only views | `headline`, `scenario_bracket`, `load_shap_cells` |
| `scripts/pipeline/run_phase{2..7}.py` | drivers | write `outputs/` |
| `scripts/pipeline/run_all.py` | entry point | runs everything in order |
| `scripts/` | communication | report/slide builds, number checker, reference checker, zip |
| `tests/` | guards | leakage, additivity, sums to one, layout shape |
| `app/` | dashboard | nine pages |

**How to trace a number.** Example: "spatial-CV RMSE 1.746". It is
`model_comparison.csv`, row `spatial_4km, lgbm`, column `RMSE_mean`; written by
`scripts/pipeline/run_phase5.py` via `evaluate.model_comparison` from `phase4_fold_metrics.csv`,
which `scripts/pipeline/run_phase4.py` produced from `cv.spatial_splits` and
`models.oof_predictions`.

### Revision summary
Config → modules → drivers → outputs → reports/dashboard. Every number is traceable
to a table and a function.

---

## Module 13: Design decisions

Memorise the one-liners; the full arguments are in `docs/DECISIONS.md`.

| ID | Decision | One-liner |
|---|---|---|
| D-001 | 100 m grid | Match the thermal sensor's true resolution |
| D-003 | Spatial CV headline | Random CV measures memorisation of nearby pixels |
| D-004 | No coordinates | We want drivers, not a map lookup |
| D-009 | Scenario bracket | Lower and realistic bounds; neither causal |
| D-010 | Spatio-temporal test | A same-cells split tests memory of place |
| D-019 | Urban targets | The hottest box cells are dry hills, not where people live |
| D-021 | Anomaly target | The model cannot know what year it is |
| D-022 | Ids from coordinates | A cell's block is where it is, not when it was measured |
| D-023/25 | Drop `frac_built` for OLS only | Singularity is a linear-algebra problem |
| D-024 | Keep NDBI as dryness | Collinear but not redundant |
| D-028 | 4 km blocks | Sized to what the model cannot explain |
| D-029 | Tune inside spatial CV | 2025 never seen; small optimism stated |
| D-031 | CPU only | A result you cannot reproduce is worth less than a faster one |
| D-032 | LightGBM reference | A tie, broken on the clean test and SHAP cost |
| D-033 | Five shuffled layouts | The first layout was accidental strips |
| D-034 | Exclude water | Reservoirs are cold because water is water |
| D-036 | Interventional SHAP | Correlated features; checked additivity |
| D-038 | Driver groups | Groups agree across models 95% of the time |
| D-040 | First point excluded | "No trees" marks a kind of place |
| D-044 | Number-checked report | Every number re-derived before the build |
| D-047 | LST, not air | Only temperature measured at 100 m everywhere |
| D-049 | Read-only dashboard | Cannot disagree with the report |

### Common viva traps
* "Isn't 4 km arbitrary?" Measured from residuals; every size reported.
* "Why not just use the most accurate model?" They tie; claiming one is best
  would be over-reading noise.
* "Isn't excluding water cherry-picking?" Defined before evaluation by a land-cover
  rule, 1.1% of rows, none urban, reported in a table.

---

## Module 14: Limitations and future scope

### Limitations
1. Surface temperature at ~10:30 AM, not air temperature or comfort.
2. Associational scenarios; confounding by unobserved urban form.
3. Residual Moran's I ≈ 0.6: missing drivers (building height, roofs, irrigation).
4. Static 2021 land cover for 2023–25.
5. Collinear indices; only the block ranks robustly.
6. Mildly optimistic spatial score from tuning on the same folds.
7. One city, one season, daytime.

### Future scope
* Urban-form and irrigation features; neighbourhood (focal) features.
* Night-time and seasonal LST; air-temperature sensors.
* Causal designs around real planting programmes.
* Nested spatial CV; other Indian cities.

### Viva question
*What would you do with six more months?* Add building height and roof material
features, test night-time LST, and evaluate a real greening programme
before/after.

---

## Final revision sheet

**Numbers to know**

| | |
|---|---|
| Cells / rows / land rows | 81,983 / 245,949 / 243,252 |
| Features | 13 (12 for linear regression) |
| Cool island (core − ring) | 3.4–4.0 °C cooler |
| Block size | 4 km (residual range ~3.95 km); raw range 7.65 km |
| CV runs per result | 5 seeds × 5 folds = 25 |
| LightGBM RMSE: random / 4 km / 2025 | 1.620 / 1.746 / 1.747 °C |
| R²: 4 km / 2025 | 0.73 / 0.72 |
| Random-CV inflation | trees +0.12 to +0.15; linear +0.023 °C |
| Urban hottest-decile detection | 54.2% (chance 10%) |
| Residual Moran's I | ≈ 0.6 |
| Spectral block mean \|SHAP\| | 1.49 °C |
| Tree-cover knot | 10.2% |
| Built-up association shrink | 87% |
| Target cells | 2,673 |
| +10 / +20 points ΔLST | −0.37 to −1.02 / −0.56 to −1.63 °C |

**Ten sentences that cover the project**

1. Pune's daytime surface heat island is inverted before the monsoon.
2. The target is the within-year LST anomaly on a 100 m grid.
3. Coordinates are never features.
4. Random CV leaks; spatial blocks sized from residual autocorrelation fix it.
5. Five shuffled layouts and a spatio-temporal 2025 test give honest numbers.
6. The three tree models tie; LightGBM explains.
7. Interventional SHAP, checked for additivity, ranks greenness/dryness/moisture first.
8. Tree-cover cooling saturates near 10%; built-up matters little once dryness is modelled.
9. Greening the hottest urban cells is associated with 0.4–1.6 °C lower LST.
10. Everything is associational, and every number is checked against the tables.
