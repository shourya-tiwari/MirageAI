# Learning Log

What we learned, what surprised us, and what broke. This is the raw material for the viva and for the report's discussion section.

## Entry Template

```
### YYYY-MM-DD: <short title>  (Phase X, Tx.y)
- **Context:** what we were doing
- **Learned / observed:** the insight or surprise
- **Evidence:** number, plot, or link to output
- **Impact:** what we changed because of it
```

---

## Concepts to Master Before the Viva

Tick each one once every team member can explain it without notes.

**Remote sensing**
- [ ] What LST is and how it differs from air temperature
- [ ] Why the Landsat thermal band is ~100 m natively
- [ ] What NDVI, NDBI and MNDWI measure, and their failure cases
- [ ] Cloud masking with QA bits and the SCL band
- [ ] Median compositing and why it's robust

**ML methodology**
- [ ] Spatial autocorrelation, and why random CV leaks
- [ ] Spatial block CV and how block size is chosen (variogram)
- [ ] Bagging (RF) vs boosting (XGBoost/LightGBM)
- [ ] Why scaling matters for LR but not for trees
- [ ] Multicollinearity, VIF, and the sum-to-1 fraction problem
- [ ] Hyperparameter tuning without leakage

**Evaluation and explainability**
- [ ] RMSE vs MAE vs R², and when each misleads
- [ ] Moran's I on residuals
- [ ] Shapley values: the intuition and the additivity property
- [ ] Reading SHAP beeswarm, dependence and interaction plots
- [ ] Correlation vs causation in the scenario results

---

## Entries

### 2026-10-07: Topic choice and research gap  (Phase 0)
- **Context:** Choosing a non-Kaggle, research-worthy case study.
- **Learned / observed:** Many UHI studies use linear single-index correlation and random-split validation. Those are two concrete, defensible gaps to target.
- **Evidence:** See PROJECT.md §1 and the spatial-validation references in REFERENCE.md.
- **Impact:** Spatial CV and SHAP became the core methodological contributions.

### 2026-10-08: The environment is the first real dependency risk  (Phase 0, T0.2)
- **Context:** Setting up the venv. Python 3.11 turned out not to be installed; only 3.13 was.
- **Learned / observed:** The temptation was to just use 3.13. The geospatial and explainability stack (`rasterio`, `libpysal`, `esda`, `shap`) is exactly where fresh-interpreter wheel gaps bite, and discovering that in Phase 6 would cost far more than provisioning 3.11 now. Separately, "all imports succeed" is not the same as "the stack works": `pandas` resolved to 3.0.6 and `shap` to 0.51.0, both very new, so a smoke test exercising GroupKFold, all four model families, `TreeExplainer`, Moran's I and a `pyproj` reprojection was run before declaring the phase usable.
- **Evidence:** All 17 requirements installed on 3.11.15; smoke test passed every stage, including a `pyproj` check that Pune's centre maps to UTM 43N x=380725 y=2048092 (plausible for zone 43N).
- **Impact:** Pinned 3.11 via conda (D-011), recorded the conda-prefix interpreter path quirk in `INSTRUCTIONS.md`, and kept the smoke test's shape as the template for verifying later phases.

### 2026-10-08: Hygiene rules are worth more as assertions than as prose  (Phase 0, T0.5)
- **Context:** Writing `config.py` with the feature list.
- **Learned / observed:** "Coordinates are never features" is the project's main methodological claim, but as a written rule it only holds as long as everyone remembers it. Declaring `FEATURES` and `FORBIDDEN_FEATURES` side by side turns it into a one-line assertion that fails loudly instead.
- **Evidence:** `set(FEATURES) & set(FORBIDDEN_FEATURES)` is empty, asserted during setup; 13 features, 6 forbidden columns.
- **Impact:** See D-014. The same assertion goes inside the modelling pipeline in Phase 4.

### 2026-10-08: Pune's daytime surface heat island is inverted  (Phase 1, T1.10)
- **Context:** Visual sanity check of the 2024 LST and NDVI maps, which ROADMAP expected to show "hot industrial and dense core areas, cool hills and water".
- **Learned / observed:** Water and rivers are cool as expected, but the dense core is roughly 3.2 degC *cooler* than the outer ring, not hotter, and the hottest cells are dry grassland on hill slopes rather than built-up land. The NDVI map is close to a mirror image of the LST map: the core is the greenest part of the AOI (NDVI 0.35) and the rural periphery is the driest (NDVI 0.24). This is the daytime surface urban cool island reported in semi-arid Indian cities during pre-monsoon: harvested, bare and dry fallow land around the city has almost no evapotranspiration and heats faster than an irrigated, tree-shaded urban core.
- **Evidence:** Mean LST by dominant cover: water 34.2, tree 41.3, built 43.6, crop 45.0, shrub 45.1, bare 45.6, grass 48.2 degC. Core (<5 km) 41.8 degC vs outer ring (>12 km) 45.0 degC. The 17 hottest cells are 90% grassland at a mean 771 m elevation and 12.4 degrees slope.
- **Impact:** The project's "Concrete vs. Canopy" premise still holds *within* land-cover classes, since built-up is 2.3 degC hotter than tree cover. What fails is the naive city-versus-countryside framing. It also sets up the methodological payoff: univariate `frac_built` correlates **negatively** with LST (-0.187) purely because built-up land is concentrated in the greener core, which is a textbook confound that SHAP on a multivariate model should unpick. ROADMAP's Phase 1 exit criterion needs rewording before it can be called met.

### 2026-10-08: NDBI is tracking dry soil, not buildings  (Phase 1, T1.2)
- **Context:** Checking which extracted features correlate with LST.
- **Learned / observed:** NDBI is the single strongest positive correlate of LST (+0.634), far stronger than `frac_built` (-0.187). Since the two disagree in sign, NDBI cannot be measuring built-up surface here. It is picking up the dry, bare, low-moisture periphery, which is the known NDBI failure mode in semi-arid regions and was already flagged as a limitation in PROJECT.md section 12.
- **Evidence:** Correlations above; `frac_bare` and `frac_grass` dominate the high-NDBI, high-LST cells.
- **Impact:** NDBI must be interpreted as a dryness/bareness index in the write-up, not as a built-up index. Worth revisiting in T2.5 when collinearity is assessed.

### 2026-10-08: The cool island is stable across three years, not a 2024 artefact  (Phase 1, T1.8)
- **Context:** 2023 and 2025 were extracted after the inverted spatial pattern showed up in 2024, specifically to test whether it was a one-year anomaly.
- **Learned / observed:** Every sign and nearly every magnitude reproduces. Core-minus-ring LST is -3.50, -3.25 and -3.81 degC for 2023, 2024 and 2025; `frac_built` correlates -0.193, -0.187, -0.196 with LST; NDBI correlates +0.662, +0.634, +0.613. The ranking of mean LST by dominant cover is identical in all three years apart from crop and shrub, which swap while staying within 0.1 degC of each other. 2023 is the hottest year throughout (mean LST 46.18 vs 44.44 and 44.84), which shifts all levels together but changes no relationship.
- **Evidence:** `outputs/tables/phase1_cross_year_checks.csv`; the three LST maps share a colour scale so the 2023 offset is visible directly.
- **Impact:** The cool island is a property of the season and the region, not of one year's compositing. The temporal holdout in D-010 is now a fair test, since 2025 is drawn from the same relationships the training years show.

### 2026-10-08: Cloud-free seasons make the observation count nearly useless as QC  (Phase 1, T1.9)
- **Context:** Checking `LST_n_obs` against `MIN_LST_OBS` for every year.
- **Learned / observed:** No cell in any year falls below the threshold of 3, and no cell was dropped for missing data in any year. The minimum observation count is 4.35 (2023), 7.82 (2024) and 7.00 (2025). The pre-monsoon choice in D-002 removed the cloud-gap problem so completely that the QC column has nothing to screen out, and the count mostly reflects how many scenes the year had rather than how cloudy it was.
- **Evidence:** Scene counts 8 / 11 / 9 track the per-year `LST_n_obs` medians of 7 / 10 / 9 almost exactly.
- **Impact:** `MIN_LST_OBS` stays in `config.py` as a guard for reruns with different seasons or years, but it is doing no filtering here. T2.2's missing-value report will be short, and that is a result worth stating rather than a step to skip.

### 2026-10-08: The 2023 level shift is mostly scene composition, not a later season  (Phase 1 follow-up)
- **Context:** 2023 ran ~1.5 degC hotter than 2024 and 2025 across every land-cover class. The obvious suspect was scene timing: if 2023's Landsat scenes fell later in the pre-monsoon, the composite would sample a hotter part of the season.
- **Learned / observed:** **2023 does not skew later.** Median day-of-year is 104 (2023), 107 (2024) and 93 (2025), so 2024 is the *latest*-sampled year and is also the coolest, which is the opposite of what a timing explanation predicts. But timing still turns out to matter: restricting every year to the day-of-year window all three cover (64-125) collapses the 2023-minus-2024 gap from **+1.74 degC to +0.36 degC**, and the 2025 gap from +0.39 to +0.10 degC. So roughly four fifths of the apparent inter-annual difference comes from the scenes outside the common window and how the median composite weights them, not from a uniformly hotter year. The direction differs per year: adding the late-May scenes raises 2023's AOI mean by 0.81 degC but *lowers* 2024's by 0.58 degC, i.e. late May 2023 was hot and late May 2024 was not.
- **Evidence:** Scene dates 2023 Mar 5 - May 24 (8 scenes), 2024 Mar 7 - May 26 (11), 2025 Mar 2 - May 5 (9). 2025 has no scene after May 5, so its composite never sees peak late-May heat at all. AOI mean LST, full season vs DOY 64-125: 46.17/45.36, 44.43/45.01, 44.82/45.11.
- **Impact:** This is the empirical case for D-021. The year-to-year level is an artefact of which scenes each year happened to contain, and no feature in the model encodes scene timing, so the level is unexplainable by construction and would be pure noise in a temporal holdout. Modelling the within-year anomaly removes it. It also means absolute LST values must never be compared across years in the report without this caveat.

### 2026-10-08: The leakage test caught a bug in the leakage test's own inputs  (Phase 2, T2.1)
- **Context:** D-022 asked for an assertion that every `cell_id` maps to exactly one spatial block across years. It failed immediately: 7,456 cells mapped to two different 1 km blocks.
- **Learned / observed:** The coordinate grids were in fact identical across years, down to zero floating-point spread. The fault was in our own `cell_id`. Grid centres sit at half-cell offsets (`x = 364850, 364950, ...`), so `x / 100` lands exactly on `.5`, and numpy's round-half-to-even sent `3647.5` and `3648.5` both to `3648`, merging adjacent grid columns into a single id. The merged "cell" then genuinely spanned two blocks. Using `floor` fixes it and yields exactly 319 x 257 = 81,983 distinct cells, three rows each.
- **Evidence:** `nunique(cell_id)` went from 41,087 to 81,983; the consistency test now passes for both block sizes.
- **Impact:** The test earned its place on its first run, and against a bug nobody suspected. The general lesson: an invariant check is worth writing even when you are confident, because the thing it catches is usually not the thing you wrote it for. Also a reminder that banker's rounding is the wrong default for grid indexing.

### 2026-10-08: The rarest category is the worst one to drop  (Phase 2, T2.6)
- **Context:** D-014 provisionally chose `frac_bare` as the fraction to drop for linear models, flagged for confirmation against VIF.
- **Learned / observed:** It was the wrong choice, and the reason is not obvious. Because the fractions sum to 1, dropping any one of them leaves model fit *exactly* unchanged (incremental R-squared 0.0000 for all seven), so the decision is entirely about conditioning and interpretation. Dropping the rarest class barely perturbs the constraint: `frac_bare` averages 0.026, so the remaining six still nearly sum to 1 and stay collinear (median VIF 11.8, seven features above 10). Dropping the highest-variance class, `frac_built` at 0.328, breaks the constraint properly (median VIF 2.2). The intuition "drop the one you care about least" is backwards.
- **Evidence:** Full drop-candidate comparison in the T2.6 table; see D-023.
- **Impact:** `FRACTION_DROP_FOR_LINEAR` changed to `frac_built`, which also makes every remaining coefficient read as "replace built-up with this class", i.e. the RQ4 question.

### 2026-10-08: The within-urban correlation drop is suggestive, not an answer  (Phase 2, T2.5)
- **Context:** `frac_built` correlates -0.192 with `LST_anom` across all cells but +0.009 within urban cells (`frac_built` >= 0.5). It is tempting to read that as RQ5 answered: the negative association is a between-regime artefact and vanishes under control.
- **Why that reading is wrong:** Restricting to `frac_built` >= 0.5 truncates the range of the very variable whose correlation is being measured. Correlation shrinks mechanically when a predictor's variance is cut, regardless of whether any underlying relationship changed. The same artefact explains why `frac_water` (-0.365 -> -0.045), `MNDWI` (-0.283 -> +0.069), `dist_water_m` (0.194 -> 0.018) and `slope` (0.127 -> 0.013) all collapse in the urban subset: those features barely vary within majority-built cells. Range restriction, not a change in mechanism.
- **What it does and does not support:** It is consistent with the confounding story and worth reporting as a descriptive contrast, but it is not evidence for it. A univariate correlation on a truncated subsample cannot separate "the effect disappears when controlled" from "the variance disappeared".
- **Impact:** RQ5 is answered in Phases 5-6, by the multivariate model (which holds vegetation and dryness fixed rather than conditioning on the predictor's own range) and by SHAP, where `frac_built`'s attribution is read at fixed values of NDVI and NDBI. The Phase 2 numbers stay in the report as motivation, explicitly labelled as such.

### 2026-10-16: Spatial CV erases the ranking that random CV shows  (Phase 4, T4.8)
- **Context:** First full run of all five models under all six schemes.
- **Learned / observed:** Under random CV, XGBoost is clearly best (RMSE 1.597 against 1.622 for LightGBM and 1.636 for RF, all with fold std < 0.01). At the primary 4 km blocks the three tree models are tied to within 0.002 °C (1.789-1.791), far inside the 0.15 fold std. The extra capacity that random CV rewards is fitting local texture that does not transfer to unseen areas. The random-to-4 km gap is ~0.19 °C (about 12%) for every tree model, and 0.10 for linear regression, which has less local structure to lose.
- **Plateau:** tree RMSE is 1.69-1.70 at 1-2 km and then jumps to 1.79-1.80 at 4 and 8 km, where it flattens. That plateau is the empirical confirmation of D-028: blocks at or above the residual range (~3.7 km) remove the leak, and doubling them to 8 km costs nothing further.
- **Impact:** "best model" must be decided on the spatial and spatio-temporal schemes, where the choice is effectively a tie. The SHAP model should then be chosen on interpretability grounds and stated as such.
- **Update (0.6.0):** the plateau paragraph above is **withdrawn**. It was the striped 4 km layout, not block size (D-033). With five shuffled layouts and water excluded, tree RMSE rises steadily: 1.67-1.69 (1 km), 1.70-1.72 (2 km), 1.75-1.76 (4 km), 1.84-1.85 (8 km). The ranking point stands: XGBoost leads under random CV (1.594) and the trees tie at 4 km (1.746-1.756).

### 2026-10-16: One fold carries most of the 4 km spread, and it is a variance effect, not a failure  (Phase 4, T4.8)
- **Context:** Fold std of RMSE is larger at 4 km (0.15) than at 8 km (0.07-0.09), which looks backwards.
- **Learned / observed:** At 4 km, fold 1 has RMSE 2.01-2.02 for the trees against 1.60-1.83 elsewhere. But its R² (0.78) is among the best. The baseline RMSE for that fold is 4.27 against 3.25-3.68, so the held-out blocks simply have a wider anomaly range, and RMSE scales with it. The genuinely hardest fold is fold 4 (R² 0.68 at 4 km, 0.66 at 8 km), which has the lowest R² under every spatial scheme.
- **Impact:** report R² beside RMSE in Phase 5, because RMSE alone ranks folds by their target variance. T5.5 and T5.7 should look at what fold 4's blocks have in common.
- **Update (0.6.0):** these folds were the latitude strips (D-033). Under the 5-seed shuffled layouts the 4 km fold std is 0.060, below 8 km's 0.116, so the "backwards" ordering was a layout effect. Fold 4 survives as the southern strip of the stress test: R² 0.674 and mean residual −0.45 °C (`phase5_fold_profile_strips.csv`).

### 2026-10-16: Reproducible on rerun is not the same as reproducible across machines  (Phase 4, T4.8)
- **Learned / observed:** On the same machine, reruns of XGBoost, RF and LightGBM folds reproduce the stored predictions bit-for-bit. Across the two machines, RF and LightGBM matched exactly, but XGBoost with identical parameters drifted by ~3e-4 RMSE (random CV 1.5969 → 1.5972; tuning 1.7539 → 1.7531). The likely cause is that its histogram sums depend on the thread count, which went from the old machine's core count to 20.
- **Also found:** in LightGBM, `subsample` (bagging) only takes effect when `subsample_freq > 0`, which the search space never sets, so the `subsample` dimension of the LightGBM search is inert. It was left as is to keep the T4.8 protocol identical to the provisional one.
- **Impact:** quote cross-machine reproducibility to 3 decimals for XGBoost. If the LightGBM search is ever re-opened, add `subsample_freq=1` or drop `subsample`.

### 2026-10-16: The 4 km folds are latitude strips, and that, not the residual range, made 4 km look like a plateau  (Phase 5, T5.5)
- **Context:** The residual map labels each 4 km block with the fold that held it out. The labels form horizontal bands.
- **Learned / observed:** Unshuffled `GroupKFold` deals equal-sized groups to folds in id order. Block ids are column-major, and at 4 km every column has exactly five full blocks, one per fold, so each fold is an east-west strip of 4 km × 32 km. Shuffling the blocks before assignment drops LightGBM's 4 km RMSE from 1.791 ± 0.154 to 1.737 ± 0.055. The Phase 4 entry "tree RMSE jumps at 4 km and flattens to 8 km" was reading this layout effect as a property of block size. With layout controlled, error keeps rising with block size (1.69 → 1.74 → 1.80).
- **Also explains:** the 4 km fold std being larger than the 8 km one ("looks backwards" in the Phase 4 entry), and fold 4's low R². Fold 4 is the southern strip (mean elevation 616 m, the highest) and the model over-predicts its heat (mean residual −0.45 °C), so it is an extrapolation in latitude and terrain, not a noisy fold.
- **Evidence:** `phase5_fold_layout_sensitivity.csv`, `phase5_fold_profile.csv`, `05_residual_map.png`.
- **Impact:** D-033, now decided: five shuffled layouts as the headline, strips as a stress test. Lesson: a splitter's tie-breaking rule is part of the validation design. The leakage tests checked that folds were disjoint, but nothing checked what shape they were. A test now does (`test_strip_layout_is_latitudinal_and_shuffled_layout_is_not`).

### 2026-10-16: Place memorisation costs about as much as spatial leakage  (Phase 5, T5.3)
- **Context:** D-010 replaced the same-cells temporal split with a spatio-temporal one, arguing the former measures memorisation of place. Phase 5 put a number on it.
- **Learned / observed:** On the identical 2025 test rows of each spatio-temporal fold, the final 2023-24 model (which has seen those cells in earlier years) scores RMSE 1.60 / R² 0.79 (LightGBM), against 1.77 / 0.74 when the cells are unseen. The gap is 0.17-0.22 °C for the tree models and 0.09 for linear regression, about the same size as the random-vs-spatial gap (0.15-0.19). Same pattern as RQ3: the more flexible the model, the more it gains from having seen the place.
- **Evidence:** `phase5_temporal_holdout.csv`, `phase5_temporal_holdout_folds.csv`.
- **Impact:** The D-010 amendment is now justified by measurement, not just argument. A pure temporal holdout would have reported R² ≈ 0.79 for the wrong reason.
- **Update (0.6.0, land only, 5 seeds):** unseen 1.747 / R² 0.721 against seen-place 1.598 / R² 0.767 for LightGBM. The gaps are 0.149 ± 0.050 (LightGBM), 0.187 (XGBoost), 0.227 (RF) and 0.018 (linear). See the RQ3 entry below for why the ordering matters.

### 2026-10-16: The residuals still have strong 100 m structure, and water is where the model fails most  (Phase 5, T5.5)
- **Context:** Moran's I (queen contiguity, row-standardised, 100 m lattice) on out-of-fold residuals, per year.
- **Learned / observed:** The raw target has I ≈ 0.90. Tree-model residuals under 4 km spatial CV keep I ≈ 0.60-0.67, and linear 0.64-0.69. So the features explain about a third of the neighbour-to-neighbour similarity and leave the rest. Even random-CV residuals keep I ≈ 0.52-0.60, so this is not a CV artefact. Two plausible sources: Landsat's ~100 m thermal footprint and resampling blur heat across cell edges, so neighbouring LST values share signal the in-cell features cannot carry; and heat genuinely spreads between neighbouring surfaces. The largest residuals are water. Majority-water cells average −1.65 °C (observed colder than predicted), and Khadakwasla and the river corridors reach −3 to −4 °C. That matches the hexbin's lower tail sitting above the 1:1 line. Water cells are rare (2,697 rows, 1.1%) and clustered: fold 1 alone holds 46% of them, so when it is held out the model has seen only about half the water it could learn from.
- **Evidence:** `phase5_morans_i.csv`, `05_residual_map.png`.
- **Impact:** Every per-cell error is strongly dependent on its neighbours, so n = 245,949 overstates the information content. Uncertainty is reported only as fold std, never as a per-cell standard error. Neighbourhood (focal-mean) features go in future scope.
- **Update (0.6.0):** the water finding led to D-034, which removes water-majority cells (2,697 rows). The Moran's I numbers for the land-only re-run are in the limitation entry below.

### 2026-10-16: "Error in the hottest decile" is biased by selecting on the observed value  (Phase 5, T5.6)
- **Context:** T5.6 asks how well the model does where heat matters most.
- **Learned / observed:** Inside the observed AOI-wide hottest decile, LightGBM under-predicts by +1.60 °C on average (urban hottest decile: +2.06 °C). Part of that is guaranteed by the selection itself: conditioning on the observed extreme selects the cells whose noise happened to be positive, so even an unbiased model shows a positive mean residual there, and the baseline shows +6.1 °C. The fairer question is detection: 63% of the observed AOI-wide hottest decile is also in the predicted hottest decile (54% within urban cells; chance is 10%). Within urban cells the trees beat linear regression by 5-7 points (0.54 vs 0.48).
- **Evidence:** `phase5_hot_decile.csv`, `phase5_hot_decile_overlap.csv`.
- **Impact:** Report both. Phase 7 should treat predicted-hottest-decile cells as "likely among the hottest", not as a precise list, since about half of them are not in the observed top decile.
- **Update (0.6.0):** essentially unchanged. LightGBM detection at 4 km is 62.3% AOI-wide and **54.2% within urban cells** (53.0% on the 2025 spatio-temporal test). The mean residual inside the observed decile is +1.60 °C AOI-wide and +2.02 °C within urban. The 54% is now a mandatory caveat on Phase 7 targeting (TASKS, T7 note).

### 2026-10-16: Urban cells have lower RMSE and lower R², and both are true  (Phase 5, T5.7)
- **Learned / observed:** In urban cells (`frac_built` >= 0.5), LightGBM's 4 km RMSE is 1.64 against 1.86 elsewhere, but R² is 0.51 against 0.77. The urban anomaly spread is smaller (baseline RMSE 2.62 vs 4.05), so the same absolute error is a larger share of it. This is the Phase 2 range-restriction point again, now on the error side.
- **Impact:** Quote urban performance as RMSE/MAE in °C. R² inside a subgroup cannot be compared with whole-AOI R².
- **Update (0.6.0):** urban 1.614 ± 0.043 / R² 0.542 against rest 1.805 ± 0.080 / R² 0.753 at 4 km; 2025 spatio-temporal urban 1.587 / 0.546 against 1.815 / 0.739. Water exclusion removed no urban rows, so the urban side moved only through the re-tune and the new layouts.

### 2026-10-16: Residual Moran's I ≈ 0.6 points to drivers the features do not carry  (Phase 5, T5.5; limitation)
- **Learned / observed:** On the land-only re-run, LightGBM residuals under 4 km spatial CV keep Moran's I 0.64 / 0.59 / 0.59 (2023 / 2024 / 2025), and the tree range across models and schemes is 0.52-0.66, against 0.89-0.90 for the raw target. Removing water did not change it, so the structure is not a reservoir artefact. Neighbouring cells share an error the 13 features cannot see. Beyond the thermal-footprint blur noted above, the plausible missing drivers are **building height and urban form** (shading, canyon geometry), **roofing material** (albedo and emissivity of tin, concrete and tile), and **irrigation** (watered lawns, fields and parks cool far more than their NDVI alone implies).
- **Impact:** listed as a limitation and as future scope. **No new features are added now:** none of them is available on a consistent 100 m grid for 2023-25 within this project, and adding a feature after seeing residuals would be fitting the test design. Treat the SHAP drivers in Phase 6 as the best account the available features give, not a complete one.

### 2026-10-16: The strip penalty was mostly water  (Phase 5, T5.8)
- **Learned / observed:** In 0.5.0 the striped 4 km layout cost LightGBM +0.054 °C RMSE against one shuffled layout (1.791 vs 1.737). On the land-only table it costs **+0.016 °C** against the 5-seed mean (1.762 vs 1.746), and +0.006 on the 2025 spatio-temporal test. Water was clustered: 46% of it sat in one strip, which then had to be predicted from about half the water the model could learn from. Once reservoir cells are out, extrapolating across latitude is only mildly harder than predicting scattered blocks. The strips still nearly double the fold std (0.113 vs 0.060).
- **Impact:** most of the "latitude extrapolation" penalty was a water effect. The remaining +0.016 is the honest size of the structured-extrapolation stress test.

### 2026-10-16: How much random CV flatters a model depends on how flexible it is  (Phase 5, T5.2/T5.3; RQ3)
- **Learned / observed:** From random to 4 km spatial CV, linear regression loses +0.023 °C RMSE (+1.3%) while the trees lose +0.12 to +0.15 (RF 0.118, LightGBM 0.126, XGBoost 0.151, i.e. +7 to +10%). The temporal contrast shows the same split: the place-memorisation gap on 2025 is 0.018 for linear against 0.15-0.23 for the trees. XGBoost, the most flexible on random CV, loses the most and drops from first to tied.
- **Impact:** RQ3's answer is not a single inflation number. Random CV inflates accuracy in proportion to how much local, place-specific structure a model can fit. Comparing models under random CV therefore biases the comparison toward the most flexible model, not just the absolute scores.

### 2026-10-16: SHAP's additivity check caught a float32 cast inside shap  (Phase 6, T6.1)
- **Learned / observed:** The first interventional run failed the additivity check: one cell's SHAP values summed to −4.38 °C against a prediction of −4.00. Path-dependent SHAP and LightGBM's own `pred_contrib` were exact (1e-13), so the trees were parsed correctly. The cause is that shap's interventional code evaluates trees on float32 inputs, while LightGBM splits between float64 values. Casting the data to float32 changed LightGBM's own predictions by up to 0.40 °C. Fitting on float32-rounded features (a relative change of at most 6e-8) makes additivity exact to 1e-7.
- **Impact:** never turn `check_additivity` off to make an error go away. Here it was the only thing standing between the report and SHAP values that did not describe the model (D-036).

### 2026-10-16: With correlated indices, the SHAP method changes which driver comes first  (Phase 6, T6.1/T6.2)
- **Learned / observed:** Interventional SHAP ranks NDVI first (0.95 °C) and NDBI second (0.73). Path-dependent SHAP on the same rows and model reverses them (NDBI 0.97, NDVI 0.69), and the two methods pick the same dominant driver for only 65% of rows. Overall importance still correlates at Spearman 0.99. XGBoost under interventional SHAP agrees with LightGBM on the top-five set, but it also reshuffles NDVI, NDBI and MNDWI between themselves (urban: MNDWI 0.93 vs NDVI 0.91).
- **Impact:** inside the spectral block the order is not robust to the explanation method or the model, so the report ranks the **block** (1.49 °C, 2.75× the next driver) with confidence and its members only loosely. Everything from grass fraction downward ranks consistently.

### 2026-10-16: Linear regression gets NDBI's sign wrong, not just its shape  (Phase 6, RQ2)
- **Learned / observed:** NDBI correlates +0.70 with the anomaly, and the trees respond to it positively (+0.90 °C per SD). Linear regression gives it −0.84 °C per SD and inflates NDVI to −3.33 (trees −1.15). Its NDBI contribution is anti-correlated with the tree SHAP at r = −0.97, while its spectral-block total still tracks the trees' (r = 0.96). With NDVI-NDBI at r = −0.72, least squares can trade one coefficient against the other at almost no cost in fit.
- **Impact:** RQ2's answer is not only "trees find thresholds". The linear coefficients of collinear indices are not interpretable one at a time, so standardised coefficients must never be presented as a driver ranking.

### 2026-10-16: Tree cover's association saturates near 10%, which matters for the +10/+20% scenarios  (Phase 6, T6.3)
- **Learned / observed:** Going from no tree cover to a little is associated with −0.48 °C. Beyond that, the association is −0.25 °C per 10 points up to a knot at 10.2% (CI 9.6-10.6%), then −0.07 °C per 10 points. 31% of urban cells already exceed 10% tree cover, and the urban median is 3%.
- **Impact:** in Phase 7, a +10 or +20 point scenario will move most urban cells across the knot, so the modelled cooling should be markedly sub-linear in the increment and largest for cells starting near zero cover. This is a property of the fitted model, and the scenario write-up must stay associational (D-009, T7.5).

### 2026-10-16: The cool-island association shrinks by 87% once vegetation and dryness are in the model, but does not flip  (Phase 6, RQ5)
- **Learned / observed:** Raw, built-up fraction is associated with −0.22 °C per +10 points (r = −0.24). In the model it gets −0.029 °C per +10 points (mean |SHAP| 0.12 °C, 10th of 13), still negative (Spearman of value vs SHAP −0.80). Within urban cells the raw association is nil (r = +0.009), and the model's small negative attribution there is model-dependent: XGBoost gives built-up fraction half LightGBM's weight.
- **Impact (RQ5):** the Phase 2 suspicion (LEARNING_LOG 2026-10-08, range restriction) is now answered in the multivariate model. The inverted daytime island is mostly a vegetation and dryness contrast: the built core is less dry than the pre-monsoon grass, cropland and bare land around it. It is not a cooling effect of buildings.

### 2026-10-16: Half the scenario cooling is the step from no trees to one percent  (Phase 7, T7.3)
- **Context:** The Phase 6 dependence plot showed a −0.48 °C step at the onset of tree cover (D-037). Phase 7's targets made it decisive: 78.7% of the hottest urban cells have no tree cover at all.
- **Learned / observed:** Adding tree cover one point at a time, the first point (0 → 1%) is associated with −1.43 °C under the realistic bound and −0.87 °C under the lower bound. Each later point is worth −0.12 °C up to the knot and −0.05 °C beyond it. So about half of scenario A's realistic −2.14 °C, and two thirds of its lower −1.06 °C, is that single step. Physically, one 10 m tree pixel in a hectare cannot cool it by 1.4 °C. The trees have learned what zero-tree cells *are* (a land-cover state that travels with unmeasured drivers), not what a tree *does*.
- **Impact:** the conservative estimate excludes the first point (A −1.02 °C, B −1.63 °C realistic) and is the one to quote to planners. Lesson: a fitted threshold at a boundary of the data (zero cover) carries the composition of the cells at that boundary. A scenario that starts most cells there inherits all of it.
- **Update (review):** the headline now excludes the step: A −0.37 to −1.02 °C, B −0.56 to −1.63 °C, C −0.35 to −0.97 °C (lower to realistic). The step is reported separately as a finding ("any canopy present"; tree-less urban cells run ~1.4 °C hotter in ways the model cannot attribute to tree cover alone).

### 2026-10-16: The hottest urban cells are tree-less fringe, not the old core  (Phase 7, T7.2)
- **Learned / observed:** Selecting on observed 3-year mean anomaly, the 2,673 target cells sit a median 11.5 km from the AOI centre: 45% north-east, 29% south-east, 3% south-west. Median built-up is 78%, median tree cover 0%. This is the same east-west contrast as the Phase 5 residual map, and consistent with the inverted daytime island: the dense core is cooler than the dry, recently built fringe.
- **Impact:** RQ4's target population is a planning frontier (where the city is growing) rather than a retrofit problem in the old city. That shapes how the scenarios should be framed in the report.

### 2026-10-16: Rounding a rounded number is how wrong figures get into reports  (Phase 8, T8.2)
- **Context:** The first run of `scripts/check_report_numbers.py` against the draft report.
- **Learned / observed:** Three scenario values were wrong in the last digit (B lower −0.57 → −0.56; A realistic range −1.05 → −1.04; C lower range −0.32 → −0.31). They had been copied from a 3-decimal printout and rounded again to 2 decimals (−0.5650 printed as −0.565, then rounded up), and the mistake had spread to the findings, D-040, CHANGELOG, ROADMAP and LEARNING_LOG. The checker reformats from the full-precision table value, so it could not repeat the error.
- **Impact:** quote numbers only from full-precision sources and let the checker gate the build. A 0.01 °C slip would not change a conclusion, but a reviewer who finds one stops trusting the rest.

### 2026-10-17: Water cells were diluting the cool-island contrast, not inflating it  (Phase 8, slide 2 headline)
- **Context:** The 3.2-3.8 °C core-minus-ring headline came from the Phase 1 table, which used every cell. The models and every later number exclude water-majority cells (D-034), so the headline was recomputed on land cells with the same core/ring definition (core < 5 km, ring > 12 km from the centre of the grid's UTM extent; `config.CORE_RADIUS_M`, `RING_RADIUS_M`).
- **Learned / observed:** Land only, the core is 3.661, 3.382 and 3.970 °C cooler than the ring (2023/24/25), against 3.501, 3.245 and 3.813 on all cells, so the rounded headline moves from 3.2-3.8 °C to 3.4-4.0 °C. Removing water takes out 102 core and 617 ring cells per year, and the ring warms more than the core (2024: +0.21 °C against +0.07 °C), because the larger water bodies, such as Khadakwasla reservoir, lie in the ring, while the core's water is mostly narrow river corridor. The 2024 core NDVI is unchanged (0.349); the ring's rises slightly (0.243 → 0.249).
- **Impact:** the expected direction of the bias was wrong. Cold water cells sit mostly in the ring, so including them made the countryside look cooler and shrank the cool island. The land-only number is the consistent one, and it is what the report, slides and viva notes quote now (`phase8_core_ring_land.csv`); the Phase 1 all-cells table is kept as the record of that phase.

### 2026-10-27: A map of "dominant drivers" depends on when you average  (Phase 9, T9.2)
- **Context:** Rebuilding the Phase 6 dominant-driver shares in `mirageai.results` for the dashboard, from the shipped per-row SHAP table.
- **Learned / observed:** Taking the largest |grouped SHAP| per **cell-year** and then counting gives spectral 67.5% and dry cover 23.8%. The published shares (66.8% and 24.5%) come from averaging each cell's SHAP over the three years **first** and then taking the largest group. The shares differ by under a point, but the per-row labels differ far more: 12.0% of cell-years have a dominant group different from their cell's three-year label, and 27.8% of cells change dominant group between years (mostly spectral ↔ dry cover, which sit close together where both matter). Averaging first matches what the map shows, one colour per place, and hides that year-to-year instability, which is worth stating when the map is presented.
- **Evidence:** `tests/test_results.py::test_per_cell_driver_groups_reproduce_the_phase6_map` asserts the per-cell counts equal `phase6_dominant_driver_groups.csv` exactly.
- **Impact:** the dashboard uses the same order as Phase 6. Lesson: "argmax then aggregate" and "aggregate then argmax" are different statistics; a reproduced number that is off by a percent is usually an order-of-operations difference, not noise.

### 2026-10-27: "It works on my machine" hid three ways a fresh clone would fail  (Phase 9, T9.8)
- **Context:** Moving the drivers to `scripts/pipeline/`, merging the dependency files, and testing the README's Quick Start on a simulated fresh clone (only the files Git would track) in a brand-new virtual environment.
- **Learned / observed:** Three failures were invisible in the working copy. (1) `run_all.py` located the repository as "the folder I am in", which stopped being true after the move; its own data check caught it at once ("data/raw is incomplete") because it fails loudly before running anything. (2) The two out-of-fold parquets (688 MB and 133 MB) were not gitignored; GitHub rejects any file over 100 MB, so the first push would have failed. (3) `check_report_numbers.py` read `outputs/models/best_params.json`, which is gitignored, so the README's checker command crashed on a clone; the shipped tuning tables already hold the same winners (`mirageai.results.load_best_params`, tested equal). Also: a pinned `pip freeze` from Windows includes `pywinpty`, which does not install on Linux or macOS, so it needs a platform marker; and `setuptools` cannot read a requirements file that contains `-e .`, so `pyproject.toml` now lists its dependencies itself.
- **Evidence:** fresh clone: `pip install -r requirements.txt` then `pip check` clean, 24 tests pass and 90 skip with their "run this first" message, 371/371 number checks, all nine dashboard pages render; full repo: 114 tests pass.
- **Impact:** reproducibility has to be tested from what a stranger receives, not from the working copy. The working copy always has the files that `.gitignore` hides.

### 2026-10-27: Checks keyed to positions break when the document is reorganised  (Phase 9, T9.10-T9.11)
- **Context:** Rewriting the report as an IEEE paper and the deck as 19 slides, with the number checker in place.
- **Learned / observed:** The report's checks survived a complete restructuring because each one is keyed to a phrase in the paragraph that quotes the number ("residual variogram", "memorisation gap"), not to a section number; all 371 passed on the first build of the new paper. The deck's checks were keyed to slide numbers ("[slide 7]") and would all have failed on the new order, so they now look the slide up by its title. The coverage guard also caught a slide-number pattern (`n / N`) that was silently deleting digits inside decimals such as "0.10 / 0.30" in the viva notes. A second lesson came from layout: in a two-column page, figures that span both columns cannot float, so a wide figure that does not fit leaves half a page blank; making the two-panel figures single-column and keeping the wide ones for maps and the model table removed most of the gaps.
- **Impact:** anchor automated checks to meaning (keywords, titles), not to position; and look at the rendered pages, because a build that passes every check can still look broken.

### 2026-10-28: A failed install is not always a dependency conflict  (Phase 9, T9.14)
- **Context:** Validating the slimmed `requirements.txt` (D-055) from a fresh `.venv` in the repository root.
- **Learned / observed:** The first attempt ended in "No matching distribution found for pandas==3.0.6" and then "resolution too deep", which reads like a version conflict. It was the network: PyPI requests were timing out at ~15 kB/s, so pip saw an empty version list and backtracked through everything else. Once the downloads went through, pip resolved all 164 packages with no backtracking at all, and `pip check` was clean. The pins never needed to change.
- **Evidence:** the successful log shows a single `Collecting` per package; the lock (`requirements-lock.txt`, 165 pins) was frozen from that environment.
- **Impact:** before loosening a pin to "fix a conflict", confirm that pip actually saw the versions (a timeout and a missing version print the same error). Direct pins plus a lock keep the resolver's job small enough that a real conflict would show up immediately.
