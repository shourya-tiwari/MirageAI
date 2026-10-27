# Decisions Log

Every non-obvious choice, with its reasoning. **Each entry doubles as a viva answer**, so every team member should be able to explain all of them.

Format: Decision → Why → Alternatives rejected → Viva one-liner.

---

### D-001: Model on a 100 m grid, not 30 m
- **Why:** Landsat 8/9's thermal sensor (TIRS) is natively ~100 m. The 30 m LST product is resampled, so modelling at 30 m claims detail the target doesn't have and inflates spatial autocorrelation.
- **Rejected:** 30 m (false precision); 250 m+ (loses neighbourhood-level variation).
- **Viva:** "We matched the grid to the thermal sensor's true resolution."

### D-002: Pre-monsoon season (Mar–May)
- **Why:** This is Pune's peak heat season, when UHI matters most, and it has the fewest clouds, so there are more valid Landsat observations.
- **Rejected:** Monsoon (cloud-covered); annual composite (mixes seasons with different vegetation states).
- **Viva:** "We study the season when heat stress is highest and data quality is best."

### D-003: Spatial block cross-validation as the headline metric
- **Why:** Neighbouring cells are highly correlated. A random split puts near-duplicates in train and test, which inflates R². Block CV tests whether the model generalises to unseen *areas*.
- **Rejected:** Random KFold as the main metric. We still report it, but only to quantify the inflation.
- **Viva:** "Random CV measures memorisation of nearby pixels; spatial CV measures generalisation to new neighbourhoods."

### D-004: Coordinates excluded from features
- **Why:** With lat/lon, a model can learn *where* it is hot instead of *why*. That defeats driver analysis, and coordinates can't be changed in scenarios.
- **Viva:** "We want drivers, not a map lookup."

### D-005: Cloud-masked median composites
- **Why:** A single scene can contain clouds, haze or anomalies. A per-pixel median over the season is robust to outliers and leftover cloud.
- **Rejected:** Single best scene (noisy, may have gaps); mean (sensitive to residual cloud).

### D-006: ESA WorldCover 2021 as a static land-cover layer
- **Why:** It is the best freely available 10 m land cover. No newer version exists in GEE.
- **Limitation:** It is out of sync with 2023–2025 LST. The yearly Sentinel-2 indices (NDVI/NDBI/MNDWI) partly compensate.

### D-007: Four models (LR, RF, XGBoost, LightGBM)
- **Why:** LR is the interpretable linear baseline that represents "existing studies". RF is a robust bagging ensemble. XGBoost and LightGBM are state-of-the-art boosting for tabular data. Together they test RQ2 (linear vs nonlinear).

### D-008: SHAP for explainability
- **Why:** SHAP values are additive, consistent and local (one per cell), and TreeExplainer is exact and fast for tree models. They allow driver maps, not just a global ranking.
- **Rejected:** Built-in feature importance, which is biased toward high-cardinality features and has no direction or interactions.

### D-009: Scenario model design (**resolved 2026-10-16**: bracket with two bounds)
- **Problem:** NDVI and `frac_tree` are linked. Raising tree cover while holding NDVI fixed is physically unrealistic.
- **Options:**
  - (a) Fit a land-cover-only model (no spectral indices) for scenarios.
  - (b) Update NDVI through a fitted `frac_tree → NDVI` relationship.
- **Caveat either way:** results are associational cooling *potential*, not causal guarantees.
- **Target population (settled 2026-10-08, see D-019):** scenarios apply to the hottest decile of *urban* cells (`frac_built` >= `config.URBAN_BUILT_THRESHOLD`), not the whole-AOI hottest decile.
- **Resolution: neither option alone; the effect is bracketed.**
  - **Lower bound:** change the fractions only and hold NDVI, NDBI and MNDWI at their observed values. It understates the response, because more trees with unchanged greenness, dryness and moisture is not a state the land can be in.
  - **Realistic bound:** change the fractions and update all three indices through land-cover → index models (D-041). Option (b) is generalised to all three indices and to every fraction, not only `frac_tree → NDVI`.
  - Option (a) is rejected: a model without indices would drop the strongest drivers (Phase 6), fit worse, and explain a different model from the one Phase 6 interpreted.
  - Both bounds are reported, under three models (D-042), with an out-of-support guard (D-043). **Neither bound is causal.**
- **The associational caveat (T7.5), to appear wherever a scenario number does:** ΔLST is the change in the *fitted model's prediction* when a cell's features are edited. It is not the temperature change that planting trees would cause. The models learned how LST varies *across* cells that differ in tree cover. Those cells also differ in things the features do not carry: building height and density, roofing material, irrigation, the age and type of neighbourhood (residual Moran's I ≈ 0.6, LEARNING_LOG). A scenario credits the tree-cover edit with whatever of that contrast travels with tree cover in the data. The realistic bound inherits the same confounding twice, once in the LST model and once in the index models. The numbers are best read as **"cells that look like this with more tree cover are, in these data, this much cooler"**: an upper-end planning signal for where greening is most associated with lower surface temperature. They are not a forecast of the cooling a programme would deliver, and say nothing about air temperature or thermal comfort, which LST does not measure.

### D-010: Three years + temporal holdout (**amended 2026-10-08**)
- **Why:** Training on 2023–24 and testing on 2025 checks whether learned drivers hold in an unseen year, which is a stronger generalisation claim than spatial CV alone.
- **Amendment:** The pure temporal split is replaced by a **spatio-temporal** one (D-027). A same-cells temporal split trains and tests on the identical 81,983 locations, whose WorldCover fractions and SRTM terrain are *byte-identical* across years and whose spectral indices barely move. Such a split measures how well the model memorised each place, not whether the drivers generalise, so it would have reported an optimistic number for the wrong reason. The replacement holds out blocks in space *and* restricts the test set to 2025, so the test cells are unseen on both axes.

---

### D-011: Python 3.11 provisioned with conda into `./.venv`
- **Why:** `pyproject.toml` pins `>=3.11`, but 3.11 is not installed as a system interpreter on the dev machine (only 3.13). `conda create -p ./.venv python=3.11` gives the required version at the path every doc already refers to, so no instruction changes ripple through the repo.
- **Rejected:** Python 3.13 (newest wheels for `rasterio`, `libpysal` and `shap` are least battle-tested there, and the geospatial stack is the fragile part); a named conda env such as `-n ushna` (not gitignored by path, and it hides the environment outside the repo).
- **Note:** In a conda prefix the interpreter is `.venv/python.exe`, not `.venv/Scripts/python.exe`. Console scripts (`earthengine`, `jupyter`) are still under `.venv/Scripts/`.
- **Viva:** "The environment is pinned to 3.11 because that is where the geospatial wheels are stable; conda just supplies that interpreter."

### D-012: Editable install via `pyproject.toml` instead of `sys.path` patching
- **Why:** Notebooks must do `from ushna import config` from a fresh kernel. `pip install -e .` makes that work everywhere — notebooks, scripts, a clean clone — with one documented step.
- **Rejected:** `sys.path.insert(...)` at the top of every notebook (repeated boilerplate, breaks if the notebook is moved, and makes imports depend on the working directory); a `.pth` file dropped into site-packages (invisible, and not reproducible from a clean clone, which T8.6 requires).
- **Cost:** One root file (`pyproject.toml`) beyond the structure listed in `INSTRUCTIONS.md`; the tree there was updated to match.
- **Viva:** "The package is installed, not path-hacked, so a clean clone reproduces the imports."

### D-013: `GEE_PROJECT` read from the environment with a config fallback
- **Why:** The Earth Engine project ID is an account identifier, not a secret, but it is also machine-specific. Reading `USHNA_GEE_PROJECT` with a `config.py` default lets each team member point at their own registered project without editing tracked code, and keeps the "never hardcode credentials" rule intact.
- **How:** `init_ee()` in `gee_extract.py` raises a named error naming both places to set it, instead of failing deep inside the Earth Engine client.
- **Viva:** "Each member's Earth Engine project is configuration, not code, so nobody edits a tracked file to run the pipeline."

### D-014: Feature schema and forbidden columns declared in `config.py`
- **Why:** The project's central methodological claim depends on coordinates and `LST_n_obs` never reaching a model. Declaring `FEATURES` and `FORBIDDEN_FEATURES` as constants makes that rule machine-checkable with an assertion rather than a convention someone has to remember.
- **Checked:** `set(FEATURES) & set(FORBIDDEN_FEATURES)` is empty; asserted during Phase 0 setup and to be asserted again inside the modelling pipeline.
- **Provisional, to confirm later:** `FRACTION_DROP_FOR_LINEAR = "frac_bare"` (confirm against VIF in T2.6), `MIN_LST_OBS = 3` (confirm against the `LST_n_obs` distribution in T2.2), and `PRIMARY_BLOCK_SIZE_M = 2000` (confirm against the variogram range in T3.1). These are defaults so the pipeline is runnable, not findings.
- **Viva:** "The hygiene rule is enforced by an assertion against a declared schema, not by discipline."

---

### D-015: Two dependency files — a loose source list and an exact lock (**superseded by D-051**)
- **Why:** The project is graded on reproducibility, and T8.6 requires a clean-clone run that still works. Unpinned requirements do not give that: `pip install -r requirements.txt` in six months resolves different versions, and the fast-moving parts of this stack (`pandas` 3.0.6, `shap` 0.51.0, `numpy` 2.4.6) are exactly where that breaks. `requirements-lock.txt` pins all 178 resolved packages, so a teammate or an examiner reproduces the environment that was actually verified.
- **How:** `requirements.txt` stays the human-readable list of the 17 direct dependencies and is the file to edit. `requirements-lock.txt` is generated from `pip freeze` on the verified environment and is what setup installs. The editable `ushna` install is filtered out of the lock, because pip records it as an absolute machine-specific path (`-e g:\projects\aucml`); `pip install -e .` is a separate setup step instead.
- **Rejected:** Pinning inside `requirements.txt` itself (loses the readable statement of intent — which packages the project actually chose, versus which came along as transitive dependencies); a single unpinned file (not reproducible); `pip-tools` or `uv` (another tool for the team to install, for a dependency set this small).
- **Maintenance:** Edit `requirements.txt`, reinstall, then regenerate the lock with the command in the lock file's header. The two must be regenerated together.
- **Viva:** "The loose file says what we chose; the lock file says exactly what we ran. Reproducing our results uses the lock."

---

### D-016: Slope computed after reprojecting SRTM to the metric CRS
- **Why:** `ee.Terrain.slope()` derives gradients in the image's own projection. SRTM ships in EPSG:4326, whose units are degrees, and a degree of longitude is not a degree of latitude in metres, so slope computed there is distorted and latitude-dependent. Reprojecting to EPSG:32643 at 30 m first makes the gradient a true rise-over-run in metres.
- **Evidence:** Slope ranges 0-40.9 degrees with a mean of 4.1, which is plausible for Pune's plateau-and-hill terrain.
- **Viva:** "Slope is only meaningful in a metric projection, so we pinned the DEM to UTM before computing it."

### D-017: Masked pixels exported as an explicit -9999 sentinel
- **Why:** GeoTIFF nodata handling differs across writers and readers, so an unmasked-but-invalid cell could silently be read as a real 0, which is a plausible value for every fraction band. An explicit sentinel outside the valid range of all 19 bands makes "missing" unambiguous on the way back in.
- **Observed:** For 2024 the sentinel never fired: zero masked cells across all bands, because pre-monsoon cloud cover is low enough that every 100 m cell has at least 7.8 valid Landsat observations out of 11 scenes. The mechanism is still required for other years, which may be cloudier.
- **Viva:** "Missing data has one unambiguous value, so a real zero can never be mistaken for a gap."

### D-018: `LST_n_obs` is aggregated by mean, so it is fractional
- **Why:** The observation count is produced at Landsat's 30 m native scale and then averaged onto the 100 m grid like every other band, so a cell reports the mean number of cloud-free observations across the ~11 30 m pixels it contains, not an integer.
- **Implication:** `MIN_LST_OBS` is a threshold on that mean. It stays a QC column and never becomes a feature.
- **Viva:** "It is the average number of clear looks per cell, which is why it is not a whole number."

---

### D-019: Scenarios and sub-group reporting target urban cells, not the AOI-wide hottest decile
- **Why:** The 2024 extraction showed the AOI-wide hottest decile is dry hillside grassland at ~771 m elevation, not city. Planting trees there would be a land-use change on open hill slopes, which answers a question nobody asked: it is neither a greening intervention a municipal body would make nor a measure of urban heat relief. Restricting to cells with `frac_built` >= 0.5 puts RQ4 back on the population the stakeholders in PROJECT.md section 5 actually govern.
- **Also applies to reporting:** because Pune shows a daytime surface *cool* island, whole-AOI metrics and SHAP mix two physically different regimes (irrigated, tree-shaded city versus dry fallow periphery). Aggregate numbers would describe neither. T5.7 and T6.7 therefore report metrics and SHAP separately for the urban subset.
- **Rejected:** The AOI-wide hottest decile (selects hillside grassland); an LST percentile threshold instead of a land-cover one (circular, since it selects on the target).
- **Threshold:** `config.URBAN_BUILT_THRESHOLD = 0.5`, i.e. a cell is urban when most of its 100 m footprint is built-up. In 2024 that is 26,720 of 81,983 cells (32.6%).
- **Viva:** "The hottest places in the bounding box are dry hills. The hottest places people live are not the same cells, and only the second group is a planning target."

---

### D-020: Direct getDownloadURL fetch instead of a Google Drive export
- **Why:** The original design exported to Drive and relied on a manual download into `data/raw/`. The AOI stack is only ~6 MB (319 x 257 cells, 19 float bands), which is well inside the synchronous download limit, so `getDownloadURL` returns it in about 40 seconds with no batch task, no Drive folder and no manual step. That removes the one part of the pipeline a person had to perform by hand, which matters for the clean-clone reproducibility run in T8.6.
- **Also:** the Drive code path was written but never executed. Keeping unverified code that duplicates a verified path invites someone to run the untested one. It was deleted rather than left in place, along with the now-unused `DRIVE_FOLDER` and `EXPORT_MAX_PIXELS` constants.
- **Rejected:** Keeping both paths (one would stay untested); `ee.batch.Export` with polling (batch-task latency and Drive auth for no benefit at this size).
- **Limit to watch:** this only works because the AOI is small. A larger AOI or a finer grid would exceed the synchronous limit and the Drive export would have to come back.
- **Viva:** "The study area is small enough to download synchronously, so there is no manual export step to get wrong."

---

### D-021: Primary target is the within-year LST anomaly, not absolute LST
- **Decision:** `LST_anom = LST - median(LST of that year over the AOI)`. Absolute `LST` stays in the table for maps and reporting; `LST_anom` is what the models predict.
- **Why:** Nothing in the feature set encodes the year, the scene dates or the weather. The 2023 composite runs ~1.7 degC hotter than 2024 at AOI level, and the Phase 1 follow-up showed roughly four fifths of that gap disappears once all years are restricted to a common day-of-year window, i.e. it is an artefact of which scenes each year happened to contain. A model given absolute LST would be asked to predict a level it has no information about, so the error would be irreducible noise. Worse, it would contaminate the temporal holdout: testing on 2025 would partly measure whether 2025's scene timing resembled 2023-24's, not whether the drivers generalise.
- **Rejected:** Absolute LST with a year dummy (a dummy is unusable at prediction time for an unseen year, which is exactly what the temporal holdout simulates); per-year standardisation by mean and standard deviation (also rescales the spread, which removes real between-year differences in spatial contrast that we want to keep); no adjustment (above).
- **Check:** Per-year anomaly means are +0.075, +0.123 and +0.027 degC with standard deviations 3.90, 3.43 and 3.59, so the offset is removed while the within-year spread is preserved.
- **Viva:** "The model has no way to know what year it is, so we ask it to explain where a cell sits within its own year, not how hot that year was."

### D-022: Cell and block IDs derive from coordinates only, never from the year
- **Decision:** `cell_id` and `block_{1000,2000}m` are computed from `x_utm`/`y_utm` alone, so a given ground cell carries the same ids in all three years. A test asserts that every `cell_id` maps to exactly one block per block size across the whole table.
- **Why:** The three years are repeated measurements of the same 81,983 cells. If blocks were assigned per year, the 2023, 2024 and 2025 copies of one cell could land in different folds, putting near-identical rows on both sides of a split. That is the exact leakage spatial CV exists to prevent, and it would be invisible in the metrics.
- **Implementation note:** `cell_id` uses `floor`, not `round`. Grid centres sit at half-cell offsets (`x = k*100 + 50`), so `x / 100` lands exactly on `.5` and numpy's round-half-to-even merges adjacent columns onto one id. The consistency test caught this on its first run: 7,456 merged cells straddled two 1 km blocks.
- **Viva:** "A cell's block is a property of where it is, not of when it was measured, so the same ground never appears in training and test."

### D-023: frac_built is the dropped reference fraction for linear models (supersedes the D-014 default)
- **Decision:** `FRACTION_DROP_FOR_LINEAR = "frac_built"`, replacing the provisional `frac_bare` that D-014 flagged for confirmation in T2.6.
- **Why (fit):** None. Because the seven fractions sum to 1, any one is exactly reconstructible from the other six, so dropping any single fraction changes model R-squared by 0.0000. The choice is purely about conditioning and interpretation.
- **Why (conditioning):** `frac_bare` is the rarest class (mean 0.026, sd 0.092), so removing it barely perturbs the sum-to-1 constraint and the remaining six fractions stay almost as collinear as before: median VIF 11.8 with 7 features above 10. Dropping the highest-variance fraction, `frac_built` (mean 0.328, sd 0.369), breaks the constraint most effectively: median VIF 2.2, and the only features left above 10 are the three spectral indices, which are collinear with each other rather than with the fractions.
- **Why (interpretation):** The dropped class is the reference, so every remaining fraction coefficient reads as the effect of *replacing built-up land with that class*. That is literally the greening-scenario question in RQ4, so the linear model's coefficients become directly quotable rather than relative to bare ground, which is both rare and not a realistic conversion source.
- **Viva:** "All seven fractions carry the same information, so we chose the reference that both conditions the matrix best and makes the coefficients answer our actual question."

### D-024: NDBI stays in the feature set, interpreted as dryness
- **Decision:** Keep NDBI. Do not drop it for collinearity.
- **Why:** It is the strongest single correlate of the target (+0.635 with `LST_anom` over all cells, +0.659 within urban cells) and it carries the dryness mechanism that RQ5 is about. It also adds information the other indices do not: dropping it costs +0.0081 of R-squared even with all twelve remaining features present, despite correlating -0.70 with NDVI.
- **Caveat recorded:** Its VIF is 18.4, and NDVI's is 29.8 and MNDWI's 18.8. The three indices are mutually collinear, so in the **linear model** their individual coefficients are unstable and must be read as a block, not quoted separately. Tree models are unaffected by collinearity, and SHAP will attribute between them, so the tree-model results are the ones to interpret per index.
- **Rejected:** Dropping NDBI (discards the dryness signal central to RQ5 for a modest VIF gain); replacing the three indices with a principal component (destroys the per-index interpretability that RQ1 needs).
- **Viva:** "It is collinear with NDVI but not redundant, and it measures the thing that makes Pune's periphery hot."

---

### D-025: The dropped fraction applies to Linear Regression only
- **Decision:** `FEATURES_LINEAR` omits `frac_built`; `FEATURES_TREE` keeps all seven fractions. Only the linear model sees the reduced set.
- **Why:** D-023's drop exists solely to break the sum-to-1 singularity, which is a property of ordinary least squares, not of trees. Random Forest, XGBoost and LightGBM split on one feature at a time and are indifferent to exact collinearity; removing `frac_built` from them would buy nothing and cost everything that matters here. SHAP attributes to the features a model actually used, so a tree model trained without `frac_built` could not assign any contribution to built-up surface at all. Since "how much does concrete raise temperature" is the project's title question and the core of RQ1 and RQ5, dropping it from the explainable models would quietly make that question unanswerable.
- **Enforced:** `config.py` asserts `"frac_built" in FEATURES_TREE` at import, so a future edit cannot silently remove it, plus `FRACTION_DROP_FOR_LINEAR not in FEATURES_LINEAR` and a length check.
- **Reporting consequence:** linear and tree models are fitted on different feature counts (12 vs 13). That is stated wherever the comparison table appears, because it is a deliberate design choice rather than an oversight.
- **Viva:** "The singularity is a linear-algebra problem, so only the linear model pays for it. The trees keep concrete, because concrete is what we are trying to explain."

---

### D-026: 8 km spatial blocks, from the measured autocorrelation range (**superseded by D-028**)
- **Decision:** `PRIMARY_BLOCK_SIZE_M = 8000`. 1 km, 2 km and 4 km are retained for the sensitivity analysis REFERENCE section 4 asks for.
- **Evidence:** The empirical variogram of `LST_anom` reaches 95% of its sill at **7.65 km**, and the estimate is stable across years (7.55 / 7.65 / 7.65 km for 2023 / 2024 / 2025). Blocks must be at least the autocorrelation range for held-out blocks to be approximately independent of training blocks, so 2 km — the value provisionally assumed in D-014 — is roughly a quarter of what the data requires. At 2 km, semivariance has reached only 63% of the sill, meaning train and test cells on either side of a block boundary still share substantial signal.
- **Structure of the variogram:** It rises steeply to ~50% of sill within 1 km, then climbs slowly to the sill over the next 7 km. The short leg is local land-cover texture; the long leg is the regional core-to-periphery gradient. Both are real spatial dependence that block CV must defeat, which is why the 95% criterion is applied to the whole curve rather than to the short structure alone.
- **Accepted cost:** The AOI is only ~32 x 26 km, so 8 km blocks yield just 20 blocks and 4 per fold. Fold-level metrics will be noisy, and all results must be reported as mean +/- std across folds rather than as point estimates.
- **Rejected:** Keeping 2 km as primary (under-corrects for the measured range, which is the exact error D-003 exists to prevent); shrinking the blocks to get more of them (optimises the convenience of the estimator at the cost of its validity).
- **Viva:** "We measured how far the temperature signal travels, found about 7.6 km, and made the blocks at least that wide. The alternative is a block CV that still leaks."

### D-027: Fold imbalance is reported, not engineered away
- **Context:** At 8 km blocks, fold 3 of the spatial and spatio-temporal schemes holds out 50.4% urban cells against an overall share of 32.6%, a deviation of +17.8 points that exceeds our 10-point reporting tolerance.
- **Decision:** Report and flag it. Do not stratify, reshuffle or reseed to remove it.
- **Why:** Urban share per 8 km block ranges from 0.6% to 64.6% because Pune's built-up area is centrally concentrated with ribbon sprawl along corridors. With 20 blocks and 4 per fold, no assignment makes every fold representative. Stratifying folds by urban share would mean choosing which blocks sit together on the basis of their content, which breaks the spatial contiguity that makes block CV a valid test and quietly reintroduces a mild form of selection on the outcome. Reseeding until the flag clears would be fitting the validation design to a cosmetic criterion.
- **Consequence carried forward:** fold 3 is the hardest and most urban fold, so the spread across folds is itself informative about where the model struggles. T5.7's urban-subset metrics are what this flag points at, and the hard test asserts only the non-degenerate floor: every fold must contain some urban and some rural cells, or the urban-subset metric would be undefined.
- **Viva:** "The imbalance is a property of the city's shape, not of our seed, so we report it instead of hiding it."

---

### D-028: Primary block size 4 km, set from the RESIDUAL autocorrelation range (amends D-026)
- **Why the raw range was the wrong input:** D-026 sized blocks from the variogram of raw `LST_anom`, which reaches 95% of its sill at 7.65 km. But most of that structure is the regional core-to-periphery gradient, and the features largely explain it. Block CV only needs to defeat the dependence the model cannot already account for, which is the autocorrelation of its *residuals* (Roberts et al. 2017).
- **Measurement:** LightGBM with default parameters, out-of-fold under 2 km spatial CV, residual variogram computed per year with the T3.1 method. Residual variance is 2.97 against 13.28 for the raw target, so the model removes about 78% of the variance before any spatial structure is considered.

| year | nugget | sill | practical range (95%) |
|---|---|---|---|
| 2023 | 1.111 | 3.179 | 4,550 m |
| 2024 | 1.266 | 2.579 | 3,750 m |
| 2025 | 1.169 | 2.810 | 2,750 m |

- **Decision:** `PRIMARY_BLOCK_SIZE_M = 4000`, `STRESS_BLOCK_SIZE_M = 8000`. All four sizes are still fitted and reported.
- **The judgement, stated plainly:** the rule "smallest rung at or above the residual range" gives 4 km on the central estimate (mean 3,683 m, median 3,750 m) but 8 km on the across-year maximum (4,550 m, driven by 2023). 4 km was chosen because the estimate is not precise enough to justify doubling the block on a 14% overshoot in one year: the residual variogram is nearly flat approaching its sill (86.9% of sill at 2 km, 94.7% at 5 km for 2023), so moving the threshold from 95% to 90% gives 2,550 m and to 99% gives 7,350 m for the same year. Re-sampling the pairs changes it by only ~194 m, so the imprecision is in the criterion, not the estimator. Since every block size is reported, the stricter reading remains fully visible in the results rather than being replaced by this choice.
- **Viva:** "We sized the blocks to the autocorrelation the model could not already explain, not to the gradient it explains well, and we report the stricter 8 km case alongside."
- **Re-check after D-034 (2026-10-16):** the same protocol was re-run on the land-only table, with the one change that the 2 km folds use the seed-42 shuffled layout (D-033). It runs as a gate in `run_phase4.py` before tuning and stops the run if the rung moves (`phase4_residual_range.csv`). 95% ranges: 2023 5,050 m, 2024 3,950 m, 2025 2,350 m; median 3,950 m, mean 3,783 m, so **4 km holds**, now by a narrower margin (50 m below the rung against 250 m before). The maximum (5,050 m, again 2023) would give 8 km, as before. Residual variance is 3.34 / 2.75 / 2.91 against target variance 13.48 / 10.42 / 11.36. The 90% and 99% ranges (2,550 / 2,150 / 1,450 m and 6,950 / 5,950 / 5,650 m) keep the criterion-imprecision point above: the choice of sill fraction moves the range more than water exclusion does. The tuned LightGBM's own residuals at 4 km are checked again in Phase 5 (`phase5_residual_range.csv`).
- **Post-tuning check (2026-10-16), does not support 4 km:** the tuned LightGBM's out-of-fold residuals give a median 95% range of 4,250 m at 4 km (seed 42: 5,050 / 4,250 / 4,150 m), which is rung 8 km. Seeds 43-46 give 4,250-4,450 m. Under the D-028 protocol's own 2 km folds the tuned model also gives 4,250 m, against 3,950 m with default parameters, so the shift comes from tuning, not from water exclusion or the fold layout. 2024 is the year that tips it (3,950 → 4,250 m). The range also grows with the CV block that produced the residuals (random 3,450, 1-2 km 4,250, 4 km 4,250, 8 km 5,250 m), which is why D-028 fixed the measurement protocol in the first place.
- **Decision (2026-10-16): 4 km stays primary.** The gate protocol was fixed in advance, and it passes. The measured range depends on the CV used to generate the residuals (random 3,450 m → 8 km 5,250 m), so re-measuring under a different protocol and resizing the blocks to match would be circular: the larger the block, the longer the range it reports. The post-tuning 4,250 m is reported here as measured. 8 km results are already in the RQ3 curve and are reported as the conservative bound (tree RMSE 1.84-1.85 °C, R² 0.68). No re-run.

### D-029: Tuning runs inside spatial CV on training years only
- **Setup:** `RandomizedSearchCV`, 25 candidates, scored on negative RMSE, with `GroupKFold` over 4 km blocks. Fitted on 2023-24 rows only, subsampled to 60,000 for runtime (a full Random Forest search over 164k rows was ~79 minutes by measured single-fit cost; 60k keeps the three searches to roughly 90 minutes combined).
- **Why training years only:** the spatio-temporal scheme tests on 2025. If tuning had seen 2025 rows, the hyperparameters would be selected partly on the test year and that evaluation would no longer be clean.
- **Known optimistic bias, stated rather than hidden:** hyperparameters are selected on the same spatial block folds that later report spatial-CV scores, so the spatial-CV numbers are mildly optimistic. A nested CV would remove it but multiplies cost by the outer fold count for a bias that is small relative to the random-versus-spatial gap we are measuring. **The spatio-temporal result on 2025 is the clean check**, because neither the blocks nor the year were used in selection.
- **Subsampling caveat:** tuning on 60k of 164k rows may favour slightly more regularised settings than the full data would. Final models are refitted on all applicable rows.
- **Viva:** "We tuned inside the spatial folds, which flatters the spatial score a little, so the honest headline is the 2025 spatio-temporal result that selection never touched."
- **Re-tune (2026-10-16, after D-033 and D-034):** same protocol (25 candidates, 60k subsample, 2023-24 only, `refit=False`), on the land-only table, with folds from the **seed-42 shuffled 4 km layout**. The layout is computed on the full 2023-24 table (`cv.block_folds`) and carried into the subsample as explicit splits, so the tuning folds are exactly the seed-42 evaluation folds rather than a re-deal of the subsample's blocks. Results (`best_params.json`): LightGBM 1.7505, XGBoost 1.7498, RF 1.7674. RF now selects `n_estimators=200, min_samples_leaf=2, max_features=0.3, max_depth=16`. The optimistic-bias caveat above now applies to the seed-42 repeat only; seeds 43-46 deal the blocks differently from the layout used in selection.

---

### D-030: The Random Forest search space is bounded by machine memory (**resolved 2026-10-16**, see the end of this entry)
- **What happened:** The first search selected `n_estimators=500, min_samples_leaf=1, max_depth=20`. Fitting that on the full 163,966 training rows was killed by the operating system. The cause is memory, not code: with fully grown leaves a tree reaches ~78,600 nodes on this data, so one tree is ~5 MB and a 500-tree forest needs ~2.5 GB. The development machine has 8.3 GB total and ~1.4 GB free.
- **Why the search did not notice:** tuning ran on the 60,000-row subsample from D-029, where each training fold is ~48k rows and the same configuration needs well under a gigabyte. The subsample made an infeasible setting look cheap. This is the D-029 subsampling caveat biting harder than anticipated: it changed what was *runnable*, not merely what was optimal.
- **Decision:** constrain the RF space to `n_estimators` in {200, 300}, `min_samples_leaf` in {5, 10, 20}, `max_features` in {0.3, 0.5, 0.7}, `max_depth` in {8, 12, 16, 20}, and re-run the RF search. Worst case is then ~0.45 GB. Measured node counts: 78,604 per tree at `min_samples_leaf=1`, 23,562 at 5, 7,284 at 20.
- **Honest statement of the limitation:** RF is therefore searched over a smaller space than XGBoost and LightGBM, so a like-for-like "best tuned model" comparison slightly disadvantages it. This must be stated wherever the model comparison appears. It is a hardware constraint of this project, not evidence about Random Forests.
- **Rejected:** Keeping the infeasible parameters and reporting RF as failed (loses a model the rubric expects); subsampling the training data for RF only (makes its scores incomparable with the other models on a different axis, which is worse); lowering `n_jobs` (the whole forest is retained regardless, so peak memory barely moves).
- **Also changed:** the Phase 4 driver streams out-of-fold predictions to per-run parquet parts and combines them at the end, rather than holding ~6.6M rows (~370 MB) in memory alongside a fitted forest.
- **Viva:** "The best hyperparameters on a sample were unfittable on the full data because of memory, so we restricted the forest's search space to what the machine can hold and said so."
- **Resolution (2026-10-16):** the project moved to an i7-13650HX (14 cores / 20 threads) with 16 GB RAM (see `docs/MIGRATION.md`). The full RF space is restored: `n_estimators` in {200, 300, 500}, `min_samples_leaf` in {1, 2, 5, 10, 20}, `max_features` in {0.3, 0.5, 0.7}, `max_depth` in {8, 12, 16, 20}. The pre-D-030 grid was never written down, so this is a reconstruction. It contains every value the first search selected, and the grid above is the one actually searched in T4.8. RF is now searched on equal terms with XGBoost and LightGBM, so the comparison caveat above no longer applies to the T4.8 results. It still applies to the `_provisional` files, which are kept for comparison. Memory safety now comes from the execution rule in D-031, not from shrinking the space.
- **Outcome (T4.8):** the 25-candidate search picked `n_estimators=500, min_samples_leaf=5, max_features=0.5, max_depth=16` (tuning RMSE 1.7706 ± 0.0569). Peak memory over the whole Phase 4 run was 1.30 GB, during the final RF fit. Five of the 25 candidates had `min_samples_leaf=1` and none ranked in the top 8. The old D-030 optimum (`500 / 1 / 0.3 / 20`) was not among the sampled candidates. Scored separately under the identical folds, it gives 1.7658 ± 0.0523, reproducing its D-030 value exactly. That is 0.005 better, a tenth of the fold standard deviation. **The search result stands:** swapping in a configuration from a different search would be selecting on the score outside the stated protocol. The top eight candidates lie within 0.004 of each other, so the RF optimum is flat.

---

### D-031: CPU only, and parallelism inside models rather than across search fits
- **CPU only:** every model runs on the CPU, including XGBoost and LightGBM, which could use the RTX 5060. GPU histogram construction is not bitwise reproducible against the CPU path or across driver versions, and Phase 4's exit criterion is "results are reproducible on rerun". `device="cpu"` (XGBoost) and `device_type="cpu"` (LightGBM) are set explicitly so that a library default change cannot move work onto the GPU silently. The payoff was visible at migration: LightGBM fold predictions were bit-identical across two different CPUs.
- **Memory rule for 16 GB:** `RandomizedSearchCV(n_jobs=1)` with every model at `n_jobs=-1`. Parallelising across search fits would hold one fitted forest per worker at once, up to ~3 GB each at `min_samples_leaf=1`. Parallelising inside a model uses the same cores and holds a single forest. On top of that, each fold's model is deleted as soon as it has predicted, the search runs with `refit=False` (only `best_params_` is used, and refitting would keep a forest alive into the next search), and out-of-fold predictions are still streamed to parquet parts (D-030).
- **Logged:** wall time and peak resident memory per stage and model, in `outputs/tables/phase4_runtimes.csv`. Peak memory comes from a 50 ms sampling thread, because Windows only exposes a lifetime peak working set that cannot be reset between models.
- **Viva:** "We used the GPU for nothing, because a result we cannot reproduce bit-for-bit is worth less than a faster one."

---

### D-032: LightGBM is the reference model for Phase 5 diagnostics and Phase 6 SHAP
- **The accuracy question is a tie.** On identical folds, the paired RMSE difference against LightGBM at the primary 4 km blocks is −0.002 ± 0.011 (RF) and −0.000 ± 0.006 (XGBoost), and on the spatio-temporal 2025 test it is +0.006 ± 0.016 and +0.011 ± 0.008 (`phase5_paired_vs_best.csv`). At 4 km, RF beats LightGBM on 3 of 5 folds and XGBoost on 4 of 5. Every gap is a few thousandths of a degree against a fold std of 0.13-0.15. XGBoost's clear lead under random CV (−0.025, 5/5 folds) disappears under every spatial scheme, which is LEARNING_LOG's "spatial CV erases the ranking" seen at the fold level.
- **Tie-break, in order:** (1) lowest mean RMSE on the spatio-temporal 2025 test (1.774 against 1.780 RF and 1.785 XGBoost), the one result that neither tuning nor block choice touched (D-029), and lowest there against XGBoost on all 5 folds; (2) SHAP cost: `TreeExplainer` on a 500-tree, depth-16 forest over 82k cells is orders of magnitude slower than on LightGBM, and the saved RF is 168 MB against 2.8 MB.
- **Stated plainly:** this is not a claim that LightGBM is more accurate. The report says "the three tree models are statistically indistinguishable; LightGBM is used for diagnostics and explanation".
- **Rejected:** RF on the 4 km mean (a 0.002 lead that is 1/70 of the fold std); averaging the three into an ensemble (adds a fourth model nobody tuned, and SHAP on an average of three tree models has no exact TreeExplainer path).
- **Viva:** "The three tree models tie under spatial CV, so we picked the one that is best on the clean 2025 test and cheapest to explain, and we say it is a tie."
- **Re-check after D-033/D-034 (0.6.0, 5 seeds × 5 folds):** still a tie, and the tie-break still lands on LightGBM. Paired RMSE differences against LightGBM at 4 km: RF +0.010 ± 0.010 (RF better on 3/25 runs), XGBoost −0.000 ± 0.006 (11/25). On the spatio-temporal 2025 test LightGBM is again lowest, 1.747 against 1.749 RF and 1.750 XGBoost, now better than XGBoost on 17 of 25 runs rather than all 5 folds. The saved RF is 115 MB against 2.8 MB for LightGBM. XGBoost's random-CV lead (−0.025, 25/25 runs) is again erased by spatial blocking.

### D-033: Shuffled block-to-fold assignment, repeated over five seeds; the strip layout is a stress test (**decided 2026-10-16**)
- **What happened:** `GroupKFold` without shuffling sorts groups by size and deals equal-sized groups to folds in block-id order. Block ids are `bx * 1e6 + by` (D-022), so the blocks are dealt column by column. At 4 km each column of the full grid holds exactly five full blocks, the same as `N_SPLITS`, so the k-th full block of every column lands in fold k. Each fold is a ~4 km × 32 km east-west strip: 75-77% of each fold's cells sit in one block row, and the rest are the partial edge blocks. Fold 0 is the northern strip and fold 4 the southern, highest and steepest one. The 1, 2 and 8 km layouts do not have this coincidence and are mixed. It is not leakage (no block is in both train and test, and the T3.4 tests still hold), but in 0.5.0 it moved the headline: on the old table, LightGBM at 4 km gave 1.791 ± 0.154 striped against 1.737 ± 0.055 with one shuffled layout.
- **Decision:** the strip layout is an unintended tie-breaking artefact of `GroupKFold`, not a pre-registered design, and a single shuffled layout would be equally arbitrary. So:
  - Spatial and spatio-temporal CV deal blocks to folds at random (`GroupKFold(shuffle=True)`), **repeated over `CV_SEEDS` = 42-46**, at every block size (the RQ3 curve) and for the 2025 spatio-temporal test. Random CV is repeated over the same seeds. Every result is reported as **mean ± std over all 25 seed × fold runs** (`model_comparison.csv` also gives the std of the five seed means). Paired model differences pair on (seed, fold).
  - The strip layout is kept as a reported **stress test, "structured latitudinal extrapolation"**, at 4 km for spatial and spatio-temporal CV (`*_4km_strips`, seed column −1), never as the headline. It is computed on the full grid and applied to land rows (D-034), since removing water would otherwise scramble it.
  - Tuning uses the seed-42 shuffled 4 km layout (D-029).
  - The Phase 4 "plateau from 4 to 8 km" stays **withdrawn**. With the layout controlled, error rises steadily with block size (LightGBM 1.675 / 1.706 / 1.746 / 1.840 at 1 / 2 / 4 / 8 km).
- **Results (land only, 5 seeds):** LightGBM spatial 4 km 1.746 ± 0.060 against strips 1.762 ± 0.113; spatio-temporal 1.747 ± 0.108 against 1.753 ± 0.121 (`phase5_strip_stress_test.csv`). The strip penalty is now **+0.016 °C** (trees +0.013 to +0.016) against +0.054 before water was removed, but the strips still nearly double the spatial fold std (0.113 against 0.060). Seed-to-seed std of the mean is ≤ 0.011 at 4 km, so five repeats are enough to make layout choice negligible next to fold variance.
- **Outputs of the striped run in 0.5.0** are kept with a `_strips` suffix (tables, figures, models), not deleted.
- **Viva:** "Our first 4 km folds were latitude strips by accident of id ordering, so we replaced one arbitrary layout with five random ones and kept the strips as a stress test of extrapolating across latitude."

---

### D-034: Water-majority cells are excluded from training and evaluation
- **Rule:** rows with `frac_water >= WATER_EXCLUDE_THRESHOLD` (0.5, `config.py`) are dropped before any split, by `preprocess.load_model_table()`. Every splitter and `models.tune` call `cv.require_land_only` and refuse a table that still holds one (tested in `tests/test_cv.py`).
- **Why:** reservoir and river LST is governed by the thermal inertia of open water, not by the land-surface drivers this model is about, and water is not a planning lever for a greening scenario. Phase 5 found water to be the largest error (majority-water cells averaged −1.65 °C, Khadakwasla −3 to −4 °C), and 46% of the water fell in one fold, so a handful of reservoir cells was steering both the error map and fold variance.
- **Removed:** 899 cells × 3 years = **2,697 of 245,949 rows** (1.1%), leaving 243,252 (81,084 per year). None of them is urban, so the urban subgroup (D-019) is unchanged (`phase4_water_exclusion.csv`).
- **Kept:** `frac_water` and `dist_water_m` stay as features. Land cells with some water, and land near water, are exactly where the cooling effect of water on land is learnable.
- **Target unchanged:** `LST_anom` keeps its D-021 definition, the anomaly from the whole-AOI median, water included. Recomputing the median on land only would shift every anomaly by +0.045 to +0.051 °C per year, a constant offset that changes no ranking or error, so the established definition is kept.
- **Strip stress test:** the original striped layout (D-033) is computed on the full grid, water included, and only then applied to land rows. On the land-only table, unshuffled `GroupKFold` is no longer striped (removing water changes block sizes, and the modal-row share drops to 0.20-0.41), so this is the only way to keep the layout that was reported in 0.5.0. Checked: it reproduces the stored strip fold of all 243,252 land rows.
- **Rejected:** keeping water and adding a water flag (the trees already split on `frac_water`, and the error persisted); a lower threshold such as 0.2 (would remove shoreline land, where cooling from water is a real driver).
- **Viva:** "Reservoirs are cold because water is water, not because of anything a planner can change, so they are out of the model, and how many cells that removed is in the table."

---

### D-035: SHAP explains a final LightGBM fitted on every land row of all three years
- **Model:** LightGBM with the tuned parameters (`best_params.json`, D-029), refitted on all 243,252 land rows of 2023, 2024 and 2025 (`outputs/models/lgbm_all.joblib`). It is not the 2023-24 model (`lgbm_trainyears.joblib`) that scored the 2025 holdout.
- **Why all years:** Phase 5 is closed, so 2025 has done its job as the untouched test year. An explanation is meant to describe the relationship in all available data, not to be validated, and adding a third year gives the trees more variation to learn from. The model's accuracy claim stays the one measured in Phase 5 (spatial CV 1.746 °C; 2025 spatio-temporal 1.747 °C), and the report says so.
- **Features rounded to float32** before fitting (D-036). The rounding changes values by at most 6e-8 relative, and in-sample RMSE moves from 1.5487 to 1.5505 °C.
- **Robustness model:** XGBoost with its tuned parameters, fitted the same way (`xgb_all.joblib`), explained on the same rows.
- **Viva:** "SHAP describes one final model trained on everything; its accuracy is the cross-validated number, not its fit to the data it explains."

### D-036: Interventional TreeSHAP with a 1,000-row background; interactions are path-dependent
- **Perturbation:** `TreeExplainer(feature_perturbation="interventional")`, against 1,000 background rows. The features are correlated: the land-cover fractions sum to one, NDVI and NDBI correlate at r = −0.72, and NDVI and MNDWI at −0.49. Path-dependent SHAP conditions on the tree's own split path, so credit leaks between correlated features in ways set by tree structure. Interventional SHAP breaks the dependence and credits a feature only through the model's response to it. **The perturbation choice matters here:** path-dependent SHAP ranks NDBI first (0.97 °C against NDVI 0.69), whereas interventional SHAP ranks NDVI first (0.95 against NDBI 0.73), and the two pick the same dominant driver for only 64.6% of rows. Overall importance still correlates at Spearman 0.989 (`phase6_path_vs_interventional.csv`).
- **Samples (`explain.stratified_sample`):** the 10,000 explained rows and the 1,000 background rows are drawn with each year × urban/rest stratum at its population share (exact allocation, largest remainder), and they are disjoint. 3,297 explained rows are urban. Seeds are 42 (explained), 43 (background) and 44 (2,000-row interaction subsample, drawn from the explained rows).
- **Float32 exactness:** shap's interventional algorithm evaluates trees on float32 inputs, but LightGBM's thresholds sit between float64 values, so the cast moved some cells across splits. The SHAP values failed the additivity check by up to 0.4 °C. Fitting on float32-rounded features (`explain.float32_exact`) makes the cast lossless, and additivity now holds to 1e-7. `check_additivity` stays on everywhere.
- **Every land row is explained, not only the sample.** shap's interventional code is a single-core loop per row (~40 ms per row, almost independent of background size), so it runs in 8 worker processes (`explain.interventional_shap(n_jobs=8)`, 29 min for 243,252 rows). The parallel result is tested to equal the serial one. The per-cell driver map uses these values, and the 10,000-row summaries are a subset of the same run.
- **Interactions (T6.4):** shap has no interventional interaction algorithm (it reports "independent does not support interactions"), so interaction values are exact path-dependent TreeSHAP on the 2,000-row subsample, additivity checked. They are reported on that scale, and given the 64.6% agreement above, they rank pairs rather than give interventional-scale magnitudes.
- **Units:** °C of `LST_anom`. Language is associational throughout: SHAP describes how the fitted model responds, not what a change on the ground would cause.
- **Viva:** "Our features are strongly correlated, so we used interventional SHAP, checked that it adds up to the prediction exactly, and showed how much the ranking would change under the default method."

### D-037: Thresholds are one-knot segmented fits with bootstrap CIs; zero-inflated fractions are split at zero
- **Method (`explain.breakpoint`):** for each feature, a continuous one-knot piecewise-linear fit of SHAP on feature value, over the 10,000 explained rows. The knot is searched over the 5th-95th percentiles and then refined locally. The 95% CI comes from 200 bootstrap resamples. Shape labels: **near-linear** if the knot adds under 2 points of R² (no threshold reported); **saturates** or **onset** if one side's slope is under a quarter of the other's; **reverses** for opposite signs; **changes slope** otherwise.
- **Zero inflation:** 36-98% of cells have exactly zero of a given fraction, so a knot at 1% tree cover is the step from none to some, not a threshold. For fractions with ≥ 20% zeros, the step (mean SHAP in the lowest positive quintile minus mean SHAP at zero) is reported separately, and the knot is refitted on positive values only (`pos_*` columns in `phase6_thresholds.csv`).
- **Why not read thresholds off the plot:** a fitted knot with a CI can be checked and reproduced, and it is tested on synthetic data with a known knot (`tests/test_explain.py`).
- **Viva:** "Each threshold is a fitted breakpoint with a bootstrap interval, and we split off the jump from zero cover so we don't call 'having any trees' a threshold."

### D-038: The dominant-driver map uses driver groups; MNDWI is labelled "moisture"
- **Groups (`explain.DRIVER_GROUPS`):** spectral (NDVI + NDBI + MNDWI), dry cover (grass + crop + bare + shrub), tree cover, built-up, water proximity (`dist_water_m` + `frac_water`), terrain (elevation + slope). Every feature is in exactly one group (asserted), and grouped SHAP is the per-row sum of its members, so additivity is kept.
- **Why:** the spectral members trade credit between themselves across models and SHAP methods (D-036), so a per-feature dominant-driver map flips arbitrarily between NDVI, NDBI and MNDWI. Measured on the 10,000 explained rows, the dominant driver agrees between LightGBM and XGBoost on 79.1% of rows per feature but 95.3% per group, and between interventional and path-dependent SHAP on 64.6% per feature but 92.5% per group (`phase6_dominant_stability.csv`).
- **Result:** spectral dominates 66.8% of land cells, dry cover 24.5% (42.7% of urban cells), terrain 3.3%, tree cover 3.3%, water proximity 2.0%, built-up 0.1%. The per-feature map is kept as a supplementary figure. The map names the top three groups and folds the rest into "Other" with their shares listed, because only three categorical slots stay distinguishable to colour-blind readers on a map.
- **Labels:** MNDWI is a land-surface moisture index here, as NDBI is a dryness index (D-024). Both are labelled that way in every figure and write-up, never as "water" or "built-up".
- **Viva:** "Individual spectral indices swap credit between models, so we map groups of drivers, which agree across models 95% of the time."

---

### D-039: Scenario targets are the hottest decile of urban cells on OBSERVED 3-year mean LST_anom
- **Rule (`scenarios.target_cells`):** per cell, the mean of the observed `LST_anom` over 2023-25. Among urban cells (`frac_built` >= 0.5; land cover is static, so a cell is urban in every year or in none, which is asserted), keep those at or above the 90th percentile.
- **Result:** **2,673 of 26,726 urban cells** (threshold +2.41 °C; mean +3.63 °C), 8,019 cell-years (`phase7_target_cells.csv`). They form 595 clusters of 8-connected cells, and the five largest hold 20% (`phase7_target_clusters.csv`). By quadrant from the AOI centre: north-east 45%, south-east 29%, north-west 22%, south-west 3%; median distance from the centre 11.5 km. They sit on the urban fringe, mostly the eastern edge, not in the old core. Median built-up is 78%, median tree cover 0.0%, and 96% are below the 10.2% tree-cover knot.
- **Why observed, not predicted:** selecting on the model's own predictions would make targeting only as good as the model's 54% urban hottest-decile precision (TASKS, T7 note). Selecting on the observed values removes that caveat. The cost, selecting on the target, does not bias the scenario deltas, which compare the model with itself on the same cells. It does mean the targets are partly cells whose observed heat the model under-predicts, so their predicted baseline is cooler than observed. Only differences are reported, never levels.
- **Viva:** "We targeted the urban cells that were actually hottest over three years, not the ones the model thinks are hottest."

### D-040: Scenarios add tree cover from built, bare and grass, in proportion
- **Scenarios:** (A) +10 points of tree cover; (B) +20 points; (C) raise tree cover to the Phase 6 knot (10.2%, read from `phase6_thresholds.csv`, D-037) in cells below it. C adds 9.2 points on average and changes 2,577 of the 2,673 cells.
- **Donors (`scenarios.add_tree_cover`):** the new cover is taken from `frac_built`, `frac_bare` and `frac_grass`, the dry and hot classes, in proportion to their current shares. Crop, shrub and water are untouched, so the fractions still sum to exactly what they did (asserted). If the donors hold less than the requested amount, all of them converts and the remainder is reported as a shortfall. With a median of 78% built-up, **no target cell has a shortfall** in any scenario.
- **The 0 → 1% step is excluded from the headline (amended after review).** 78.7% of target cells start at exactly 0% tree cover (96% are below the knot). For those cells, the model's response to the first point of cover is the zero-to-some step from D-037: −1.43 °C (realistic) and −0.87 °C (lower) on average. That first point means **"any canopy present"**, one 10 m WorldCover tree pixel in the 100 m cell. It is confounded with unobserved urban form: tree-less cells are disproportionately dense new fringe and industrial development, and the features do not see building height, density, roofing or irrigation. It is therefore not a planting effect and is **reported separately as a finding**: tree-less urban cells run about 1.4 °C hotter, in ways the model cannot attribute to tree cover alone.
- **Headline = the first point excluded, as a lower-realistic bracket** (`dLST_excl_first_point` in `phase7_summary.csv`): **A +10 points −0.37 to −1.02 °C; B +20 points −0.56 to −1.63 °C; C to the knot −0.35 to −0.97 °C.** Including the step, the values would be A −1.06 to −2.14, B −1.25 to −2.75 and C −1.03 to −2.09 °C; these are kept in the table for transparency, never quoted as the result. Beyond the first point, each point is worth −0.12 °C (realistic) or −0.04 °C (lower) from 2% up to the knot, and −0.05 or −0.02 °C from 12% to 30%: gains per point roughly halve past the knot (ratios 0.45 and 0.40; `phase7_marginal_summary.csv`).
- **Viva:** "Trees replace the hot surfaces, mostly roofs and paving, in proportion, and the fractions still add up. We quote the cooling beyond the first pixel of canopy, because 'no trees at all' marks a kind of place, not a missing tree."

### D-041: The realistic bound updates the indices through per-year land-cover → index models
- **Models (`scenarios.fit_index_models`):** one LightGBM per (year, index) for NDVI, NDBI and MNDWI, from the seven fractions plus elevation, slope and distance to water. They use the tuned LST LightGBM parameters, not re-tuned, since the job is to transfer land cover into the indices rather than to win on accuracy. They are fitted per year because the indices shift between years while land cover is static.
- **Validation:** spatial CV at 4 km, 5 seeds (D-033), per year (`phase7_index_model_cv.csv`). Mean R²: **MNDWI 0.716 ± 0.037, NDVI 0.668 ± 0.043, NDBI 0.614 ± 0.042**. Per year: MNDWI 0.752 / 0.708 / 0.690, NDVI 0.698 / 0.671 / 0.635, NDBI 0.616 / 0.632 / 0.593 (2023 / 2024 / 2025).
- **Delta, not level (`scenarios.updated_indices`):** new index = observed index + f(new fractions) − f(old fractions). Each cell keeps its own residual from the index model (irrigation, shade, roofs), and only the land-cover-driven part moves. Unchanged fractions return the observed index exactly (tested).
- **Effect:** under A the indices move by NDVI +0.067, NDBI −0.034 and MNDWI −0.021 on average. MNDWI falls slightly with more trees: across Pune's cells, more tree cover goes with lower MNDWI, and the index model carries that.
- **Caveat:** about a third of index variance is not explained by land cover, and what is explained is a cross-sectional association, so the realistic bound inherits the confounding of D-009 a second time.
- **Viva:** "We let the indices respond to the new land cover through models that were themselves spatially validated, and changed each cell's indices only by the predicted difference."

### D-042: Every scenario runs through three tuned models refitted on all land rows
- **Models:** LightGBM and XGBoost from Phase 6 (`{lgbm,xgb}_all.joblib`), and RF refitted the same way (`rf_all.joblib`: tuned parameters, every land row of 2023-25, float32-exact features, D-035/D-036). The three were tied under spatial CV (D-032), so none has a better claim, and their spread is a direct measure of model uncertainty in the response.
- **Reported:** the mean ΔLST of the three, with the min-max across them. Each model's ΔLST is averaged over target cells within each year, then over 2023-25.
- **Not reported:** a confidence interval from fold variation. The scenario is a counterfactual edit on fitted models, and the model range is the honest spread available. It does not cover confounding (D-009).
- **Viva:** "Three models that tie on accuracy give us three answers, and we report the range rather than pick one."

### D-043: Scenario rows outside the training support are flagged
- **Guard (`scenarios.SupportGuard`):** features standardised by their training mean and SD. A row's score is its distance to the **10th** nearest training row (all 243,252 land rows). The threshold is the 99th percentile of the same score for 20,000 training rows, each queried against the others with itself excluded. k = 10 is above the three near-duplicate rows every cell has (one per year), so a scenario row cannot look supported just because its own cell's observed rows are close.
- **Result:** threshold 1.64. Flagged scenario rows: 0.00% (A lower), 0.01% (A realistic), 0.04% (B both), 0.00% / 0.01% (C), so results with and without flagged rows agree to 0.001 °C. Even at +30 points the flagged share is only 0.57% (lower) and 0.12% (realistic). The scenarios stay inside the data because Pune already has urban cells with 10-30% tree cover at similar built-up shares.
- **Limits:** support in feature space is necessary, not sufficient. A supported scenario row is one the models have seen *something like*, which says nothing about whether the cross-sectional contrast is causal.
- **Viva:** "We checked that the greener versions of our hottest cells look like real cells in the data, and almost all of them do."

---

### D-044: The report is built from one Markdown source, with every key number checked against the tables
- **Source and build (`scripts/build_report.py`):** `reports/report.md` is the single source. pandoc (3.12.1, per-user install via `winget install JohnMacFarlane.Pandoc`) writes the report `.docx` (then named `Ushna_Report.docx`), using `reports/template.docx` as the reference document if it exists. Microsoft Word, driven through COM, exports the PDF. No LaTeX engine is installed, and exporting from Word makes the PDF exactly the .docx layout the team will edit.
- **Numbers (`scripts/check_report_numbers.py`):** 211 checks recompute quoted figures from `outputs/tables/*.csv`, `best_params.json` or the phase findings files. Each check requires the number, at the report's precision, to appear in the paragraph or table row where it is quoted. The build refuses to run if any check fails. On its first run the checker caught three Phase 7 values that had been rounded twice (3-dp printout → 2 dp) and were wrong in the last digit; they are corrected in the report, the findings, D-040, CHANGELOG, ROADMAP and LEARNING_LOG. Numbers no phase table held (index correlations, residual range by CV scheme) were written to `phase8_*.csv` by `scripts/make_report_figures.py` rather than quoted from the decision log.
- **References (`scripts/verify_references.py`):** each entry is resolved through doi.org content negotiation (Crossref or DataCite) and its first author, year (print or online) and venue are compared with the citation; results go to `reports/reference_check.csv`. The two entries with no registered DOI (LightGBM, NeurIPS; scikit-learn, JMLR) are marked **[UNVERIFIED]** in the report. Lundberg & Lee (2017) is cited by its arXiv DOI, with the NeurIPS venue marked unverified. A global SUHI survey was dropped because its DOI record has no abstract, so its content could not be checked. No Pune-specific study could be verified, so the report carries a marked placeholder for the team.
- **Viva:** "Every number in the report is re-derived from the output tables by a script before the document is built, and every reference was checked against its DOI record."
- **Amended by D-053:** the PDF is now printed from HTML in IEEE layout; Word is no longer needed.

### D-045: The project is renamed from Ushna to MirageAI
- **Why:** the central finding is that Pune's expected daytime urban heat island is a mirage. In the pre-monsoon season the city core is cooler than its dry surroundings (Phases 1, 5 and 6), and the project explains why. The new name says what the project found; the old one (Hindi/Sanskrit for "heat") said only what it studied.
- **Unchanged:** the registered title, "Concrete vs. Canopy: Explainable Machine Learning for Urban Heat Island Drivers in Pune", which is also the report's title. The Earth Engine Cloud project ID `ushna-aucml` keeps its name too: it is a registered external resource, and renaming it would mean a new Cloud project and re-registration.
- **What changed:**
  - Package `src/ushna` → `src/mirageai`, distribution name `mirageai` in `pyproject.toml`, and every import in `src/`, tests, `run_phase*.py`, `scripts/` and the notebooks.
  - Environment variable `USHNA_GEE_PROJECT` → `MIRAGEAI_GEE_PROJECT`. The old name is still read as a fallback and raises a `DeprecationWarning`.
  - Prose in the project instructions, README, `docs/*.md` and the report; report outputs `reports/MirageAI_Report.{docx,pdf}`.
- **History kept:** earlier CHANGELOG and DECISIONS entries still say "Ushna". They record what was true when they were written and are not rewritten. The raw file `data/raw/ushna_pune_features.csv` is unreferenced by any code and left as it is.
- **Viva:** "We renamed the project after its main finding: the heat island we expected isn't there in the daytime, and the model explains why."

### D-046: The slide deck is generated, word-limited and number-checked like the report
- **Build (`scripts/build_slides.py`):** python-pptx writes `reports/MirageAI_Slides.pptx` (13 slides, 16:9, one accent colour, a slide number on every slide). Every number is formatted from the output tables at build time, never typed. PowerPoint (via COM) exports the PDF, after the number check passes. python-pptx 1.0.2 (with lxml and xlsxwriter) is added to `requirements.txt` and the lock.
- **Word limit:** body text is counted per slide and the build fails above 30 words. Two exceptions are declared in code: the title slide (team and course block) and the closing slide, which carries the planner sentence from `phase7_findings.md` verbatim (asserted identical) at 51 words.
- **Figures:** everything comes from `outputs/figures`. Two slide-specific figures, the grouped-SHAP ranking and the scenario bracket, are made by `scripts/make_slide_figures.py` from the tables; the pipeline diagram and the model table are native PowerPoint objects.
- **Number checks extended (`scripts/check_report_numbers.py`):** the same script now reads the deck (each slide's text and its notes separately) and `reports/viva_prep.md`. Beyond the targeted checks, a coverage guard flags **any** number in the deck or viva notes that no check verified, unless it is a design constant read from the package (grid size, years, block sizes, seeds, sample sizes, scenario increments). A deliberately corrupted viva number was caught both ways before the files were finalised.
- **Speaker notes:** about 45-55 s per slide at 140 words per minute, ~10.7 min in total, with a four-presenter split marked [P1]-[P4] (slides 1-3, 4-6, 7-10, 11-13). Timing is written to `reports/slides_manifest.csv`. **Amended by D-054:** 19 slides, named presenters, checks keyed by slide title.
- **Viva:** "The slides and the viva notes are built from the same tables as the report, and a script refuses to export them if any number on them is unchecked or wrong."

### D-047: The target is land surface temperature, not air temperature
- **Why:** Landsat 8/9 measures the surface skin temperature, the radiative temperature of roofs, roads, canopy and soil, at its overpass of about 10:30 AM local time. It does so wall-to-wall at the thermal sensor's ~100 m (D-001). Air temperature would need station data or reanalysis: Pune has only a handful of weather stations, and reanalysis grids are kilometres across (ERA5-Land is ~9 km), far coarser than the neighbourhood scale at which planners act. LST is the standard proxy for the *surface* urban heat island (Voogt & Oke, 2003).
- **Limit, stated everywhere it matters:** LST is not what people feel. Air temperature and thermal comfort also depend on humidity, wind, shade and radiation from surrounding surfaces, and they peak in the afternoon, not at a mid-morning overpass. Every result is therefore about daytime surface temperature: the cool island, the drivers and the scenario associations (D-009 caveat; report section 8).
- **Rejected:** Air temperature from stations (too few points for a 100 m model, and no spatial blocks to validate on); reanalysis air temperature (one value across many neighbourhoods); MODIS LST (day and night, but ~1 km, which blurs the urban mosaic the study is about).
- **Viva:** "We model surface temperature because it is the only temperature measured at 100 m across the whole city; it is the surface heat island, not the air people breathe, and we say so."

### D-048: One entry point from data/raw, and the reproducibility test is a clean copy
- **Entry point:** `run_all.py` runs `run_phase2.py` → `run_phase7.py`, the report and slide figures, the report and slide builds and a final number check, each as its own process, stopping at the first failure (`--from N` resumes). Wall time and peak memory per step go to `outputs/run_all_log.csv`, outside `outputs/tables`, so the results directory holds results only. Earth Engine extraction is out of scope: `data/raw` is the input, and a Phase 1 re-run already reproduced the extraction exactly.
- **Phase 2 and 3 drivers rebuilt:** the Phase 1-3 tables and figures had been made interactively on the previous machine, and no committed code produced them. `run_phase2.py` (clean table, the Phase 1 checks that need only `data/raw`, the EDA figures and correlations) and `run_phase3.py` (variogram, block layout, fold balance) now do. Both reproduce the stored `phase1_cross_year_checks.csv`, `phase2_lst_anom_correlations.csv` and `phase3_fold_balance.csv` byte for byte, and the clean parquet exactly. `run_phase3.py` deliberately rebuilds the *historical* Phase 3 layout (full grid, unshuffled block folds; D-026) with `GroupKFold` directly, because the current splitters refuse water rows (D-034). The current fold design is in `phase4_fold_balance.csv`.
- **Tests in a clean clone:** tests that need the clean table or Phase 4 outputs depend on session fixtures in `tests/conftest.py`, which skip them with the reason until `run_all.py` has built their inputs. So pytest runs straight after install (pure tests only), and in full after the pipeline.
- **The test:** copy the project without `.venv`, `data/processed`, `outputs/` or built reports; create the environment from `requirements-lock.txt` (`pip freeze` must equal the lock); pytest; `run_all.py`; compare every CSV in `outputs/tables` with the reference run, byte-identical or within 1e-9 for floats.
- **Submission zip:** `scripts/make_submission.py` leaves out the out-of-fold prediction parquets (~0.8 GB, rebuilt by `run_phase4.py`). Every metric derived from them ships as a CSV.
- **Viva:** "One command rebuilds every result from the raw satellite tables, and a clean copy on a fresh environment has to reproduce every results table exactly."

### D-049: The dashboard is a read-only view over outputs/, with its logic in the package
- **Decision:** `app/streamlit_app.py` shows the project's results and never refits or recomputes them. Every number and chart comes from `outputs/tables`, `outputs/figures` or `outputs/models`, through `mirageai.results`, a small module of loaders and summaries that `tests/test_results.py` checks against the stored tables (for example, the per-cell dominant-driver counts must equal `phase6_dominant_driver_groups.csv` exactly).
- **Why:** the report, slides and viva notes are already number-checked against the tables (D-044, D-046). A dashboard that recomputed anything could show a figure that disagrees with the report in front of an examiner. Reading the same tables makes it agree by construction. Keeping the loaders in `src/` follows the "logic in the package, notebooks thin" rule and makes them testable.
- **Interactive prediction:** the one live computation is a forward pass of the saved final LightGBM (`lgbm_all.joblib`) on slider inputs, with LightGBM's exact `pred_contrib` breakdown, and a lower-bound tree-cover edit through `scenarios.add_tree_cover`. It is labelled as associational, and as path-dependent SHAP rather than the interventional SHAP of the global ranking.
- **Optional dependency:** Streamlit is in `requirements-dashboard.txt`, not in the lock, so the verified pipeline environment (D-015) is unchanged; the pipeline never imports it. The pinned version (1.65.0) accepts the locked pandas 3.0.6 and pyarrow 25.0.1. **Amended by D-051:** Streamlit and its dependencies are now pinned in the single `requirements.txt`; the pipeline still never imports them.
- **Graceful degradation:** the submission zip ships no `data/` and no fitted models (D-048), so pages that need them name the driver that creates them and fall back to the saved figures.
- **Rejected:** recomputing metrics from the out-of-fold parquet in the app (0.7 GB, not shipped, and a second code path to the same numbers); a notebook-only demo (less usable live in a viva).
- **Viva:** "The dashboard reads the same tables the report is checked against, so it cannot disagree with the report."

### D-050: Final polish keeps every result unchanged
- **Decision:** the final clean-up changes only code quality, error messages, documentation and presentation. No change touches a computation, a seed, a feature list, a fold layout or a stored output.
- **What changed:** three unused imports removed; missing input files now raise a `FileNotFoundError` naming the driver that creates them (`preprocess.load_year`, `load_clean`, `evaluate.load_oof`); a clearer error for an unknown scenario; package version set to 0.8.0 to match the last release; the four notebooks rebuilt as short, executed walkthroughs (the old stubs still described the removed Drive export, D-020); `mirageai.results` and the dashboard added; README and docs rewritten to match the implementation; `project_deep_dive.md` and `learning_log.md` added at the root.
- **Check:** the full test suite passes before and after, and no file in `outputs/` was rewritten.
- **Viva:** "The polish changed how the project reads, not what it computes."

### D-051: One pinned `requirements.txt` replaces the source list, the lock and the dashboard file (supersedes D-015)
- **Decision:** `requirements.txt` is the only dependency file. It holds, in order: the 25 packages the project chose, pinned and grouped by purpose (Earth Engine, data, ML, explainability, spatial, plotting, dashboard, report builds, runner/notebooks/tests); every transitive dependency, pinned; and `-e .`, the editable `mirageai` install. `requirements-lock.txt` and `requirements-dashboard.txt` are removed. `pyproject.toml` now declares the direct dependencies itself, unpinned, instead of reading them from `requirements.txt`.
- **Why:** a fresh clone should need one command, `pip install -r requirements.txt`, to run the pipeline, the dashboard, the tests, the notebooks and the report builds. Three files with a separate `pip install -e .` step was the most common way to set the project up wrongly. D-015's two aims survive: every package is still pinned to the verified version, and the packages the project chose are still readable as a separate, grouped block rather than lost among 195 pins.
- **Pins:** the D-015 lock unchanged, plus Streamlit 1.65.0 and the eleven packages it adds (altair, pydeck, starlette, uvicorn, httptools, python-multipart, websockets, itsdangerous, watchdog, toml; the rest were already locked). `pywinpty` carries a `sys_platform == "win32"` marker so the file also installs on Linux and macOS.
- **Check:** a new virtual environment on Python 3.11.15, installed with only `pip install -r requirements.txt` (offline, from wheels of exactly the pinned versions), passes `pip check`, runs the 114 tests and launches the dashboard. The same check on a simulated fresh clone (only the files Git tracks) passes 24 tests with the data-dependent 90 skipped, all 371 number checks and all nine dashboard pages.
- **Rejected:** keeping the lock and adding Streamlit to it (still two files and a separate editable-install step); an unpinned single file (not reproducible, D-015); `pip-tools`/`uv` (another tool to install).
- **Maintenance:** to add or upgrade a package, change its pin (and, for a new direct dependency, `pyproject.toml`), reinstall into a fresh environment, run `pip check` and the tests.
- **Viva:** "One file, every version pinned: the top block says what we chose, the rest says exactly what we ran."

### D-052: Pipeline drivers live in `scripts/pipeline/`
- **Decision:** `run_all.py` and `run_phase2.py` … `run_phase7.py` moved from the repository root to `scripts/pipeline/`. The root now holds only the README, the two guides, `pyproject.toml`, `requirements.txt`, `.gitignore` and the project directories.
- **What changed with them:** `run_all.py` finds the repository root two levels up and runs `scripts/pipeline/run_phase*.py`; every error message, skip message, notebook, dashboard hint and document that names a driver uses the new path; `scripts/make_submission.py` takes the drivers with `scripts/`. The drivers' own code is unchanged: they import `mirageai` (installed) and write through `config` paths, so they do not depend on the working directory.
- **Check (2026-10-27):** `scripts/pipeline/run_all.py` started from scratch on a clean copy (fresh environment, empty `outputs/` and `data/processed/`). Phases 2 and 3 reproduced every output byte for byte, including the clean table; Phase 4 reproduced the tuned hyperparameters and tuning scores of all three models and the first 175 fold results exactly (random-forest per-split tuning scores differ at the 1e-16 level, float summation order). The run was then stopped by the operating system for low memory, with other jobs running alongside, so Phases 5-8 were not repeated here; the full clean-copy run remains T8.6.
- **Historical entries** in this file and in CHANGELOG still name the drivers without the folder; the file names did not change.
- **Viva:** "The root says what the project is; the scripts that run it are one folder down."

### D-053: The report is an IEEE-format conference paper, printed from HTML (amends D-044)
- **Decision:** `reports/report.md` is rewritten as an IEEE conference paper (Abstract and Index Terms; sections I-XVIII from Introduction to Conclusion; reproducibility appendix; numbered references). `scripts/build_report.py` turns it into `reports/MirageAI_Report.html` with pandoc and an IEEE template (`reports/ieee/template.html`, `ieee.css`: A4, two 88.9 mm columns, Times 10 pt, roman-numbered small-caps headings, table captions above, numbered equations), and headless Chrome or Edge prints the PDF (15 pages). pandoc also writes `MirageAI_Report.docx`, an editable Word copy in the same IEEE layout (D-056).
- **Why:** the course asked for an IEEE-style paper of 15-16 pages. Word cannot be driven to a two-column IEEE layout from Markdown without a hand-made template, and no LaTeX engine is installed (D-044). CSS multi-column layout printed by Chromium gives the IEEE geometry from the same single source, with no new Python dependency.
- **Content:** every number-checked paragraph of the previous report is kept (all 371 checks still pass, now 376 with the new deck). New sections (literature review, problem statement, preprocessing, EDA, feature engineering, hyperparameter tuning, evaluation metrics, visualizations) use only facts from `config.py`, the clean table and the output tables. Four references were added for the literature review (Weng 2004, Meyer 2019, Peng 2012, Bowler 2010), each verified by DOI with `scripts/verify_references.py`; the `[UNVERIFIED]` flags are gone because the two no-DOI entries are now cited from their proceedings/journal pages, and the Pune placeholder was dropped (the checklist allowed it). The team names replace the placeholders.
- **Rejected:** LaTeX with IEEEtran (not installed, and a second source to keep in sync); a Word template (single-column output from pandoc).
- **Viva:** "The paper is one Markdown file; a script checks its numbers, then pandoc and a browser lay it out in IEEE format."

### D-054: The deck is redesigned for a 10-15 minute viva, with checks keyed by slide title (amends D-046)
- **Decision:** `scripts/build_slides.py` builds 19 slides in the order Title, Motivation, Problem statement, Objectives, Dataset, Methodology, ML pipeline, EDA, Models, Evaluation, Results (accuracy, drivers, greening), Dashboard demo, Key insights, Limitations, Future scope, Conclusion, Thank you. Each notes page starts with the presenter's name (Shourya 1-5, Shashwat 6-10, Rishik 11-14, Shubhankar 15-19); notes total ~13.9 min at 140 wpm.
- **Checks:** the deck checks in `check_report_numbers.py` now find a slide by its title in `reports/slides_manifest.csv` rather than by a fixed number, so reordering slides keeps every check. The body-word limit is 45 words per slide (the denser problem-statement and model slides), with the title, objectives, conclusion and closing slides declared exceptions. The coverage guard still flags any number on a slide or in its notes that no check verified.
- **Design:** one accent colour, a kicker label above each title, figures from `outputs/figures` (the pipeline diagram is the report's Fig. 1) and the dashboard screenshot in `docs/images`, no placeholders.
- **Viva:** "The deck follows the order of the project; every number on it is still re-checked against the tables before it is exported."

### D-055: `requirements.txt` lists direct dependencies only; the lock and the editable install are separate steps (supersedes D-051)
- **Decision:** `requirements.txt` holds only the 23 packages the project uses directly, pinned to the versions the results were produced with: the 21 imported by `src/`, `app/`, `scripts/` and `tests/` (pytest included), plus JupyterLab and ipykernel to open and run the notebooks. pip resolves everything else. `requirements-lock.txt` is `pip freeze --exclude-editable` of the verified environment, for an exact rebuild. The package is installed by a separate `pip install -e .`. `pyproject.toml` lists the 20 runtime libraries unpinned and pytest/JupyterLab/ipykernel as the optional `dev` group. A root `app.py` runs `app/streamlit_app.py`, so the dashboard starts with `streamlit run app.py`.
- **Why:** D-051's single file pinned about 200 transitive packages by hand, including platform-specific ones and three packages nothing imports (`geemap`, `geopandas` directly, `seaborn`), plus the `jupyter` metapackage. That made the file hard to read and fragile on any interpreter other than the one it was frozen on, and installs failed. Direct pins keep the result-defining libraries (scikit-learn, XGBoost, LightGBM, SHAP, pandas, NumPy) exact; the lock keeps full reproducibility without making it the default path.
- **Removed:** `geemap`, `seaborn` and the `jupyter` metapackage (never imported; replaced by `jupyterlab` + `ipykernel`); `geopandas` as a direct dependency (still installed, because `libpysal`/`esda` need it); every hand-pinned transitive package; the conda-prefix setup (D-011), since a standard `python -m venv` works on Python 3.11 and newer.
- **Rejected:** keeping the D-051 file (the problem being fixed); unpinned direct dependencies (results would drift with new releases); `pip-tools`/`uv` (another tool to install).
- **Maintenance:** change a pin in `requirements.txt` (and `pyproject.toml` for a new package), reinstall into a fresh `.venv`, run `pip check` and the tests, then regenerate the lock with `pip freeze --exclude-editable > requirements-lock.txt`.
- **Check:** in the repository root, a new `.venv` on Python 3.13.15 (Windows) with only the README commands: pip resolves 164 packages with no backtracking, `pip check` is clean, `pip install -e .` succeeds, the 114 tests pass, the three fitted LightGBM models load, and `streamlit run app.py` serves all nine pages in a browser with no exceptions or console errors (every radio, selectbox and slider exercised through Streamlit's `AppTest`). `geopandas` 1.2.0 arrives through `libpysal`/`esda`. The lock has 165 pins.
- **Viva:** "requirements.txt says what we chose and at which version; the lock says exactly what we ran."

### D-056: The Word copy of the report uses the IEEE layout too (amends D-053)
- **Decision:** `MirageAI_Report.docx` is built in the same IEEE layout as the PDF: A4, two columns with the title and author block across the page, Times 10 pt, roman-numbered small-caps headings, numbered equations, 8 pt captions, tables and references. `scripts/build_report.py` generates the Word styles from pandoc's default reference document, and `reports/ieee/ieee-docx.lua` adds the column sections (wide figures and Table V span both columns), heading numbers, equation numbers and the author table.
- **Why:** an editable copy that looks like the submitted paper, for the team or a faculty template, without hand-formatting in Word. No new dependency: pandoc and the standard library only.
- **Check:** opened in Word, 14 pages (Word's line breaking differs slightly from the browser's 15).
- **Viva:** "The Word file comes from the same Markdown as the PDF, so editing either never means retyping numbers."

_Add new decisions below (D-057 onward)._
