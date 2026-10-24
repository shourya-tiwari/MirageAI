# MirageAI: viva preparation

Every answer is grounded in a decision in `docs/DECISIONS.md` (cited as D-xxx).
Every number on this page is checked against `outputs/tables/` by
`scripts/check_report_numbers.py`; if a table changes, the check fails until
this page is updated. Language is associational throughout: say "is associated
with", never "causes" or "reduces".

## Numbers to know

| Area | Quantity | Value |
|---|---|---|
| Data | Land cells per year | 81,084 |
| Data | Land cell-years (2023-25) | 243,252 |
| Data | Water-majority rows removed | 2,697 |
| Data | Features (no coordinates) | 13 |
| Data | Core cooler than outer ring, by day (land cells) | 3.4-4.0 °C |
| Validation | Residual range, pre-registered gate (median) | 3,950 m |
| Validation | Residual range, tuned model at 4 km | 4,250 m |
| Validation | Residual range: random CV vs 8 km CV | 3,450 m vs 5,250 m |
| Models | Spatial 4 km RMSE: LightGBM / XGBoost / RF / linear | 1.746 / 1.746 / 1.756 / 1.873 °C |
| Models | Random CV RMSE, XGBoost | 1.594 °C |
| Models | 2025 unseen-place RMSE, LightGBM | 1.747 °C (R² 0.721) |
| Models | 2025 seen-place RMSE, LightGBM | 1.598 °C |
| Models | Memorisation gap: LightGBM / XGBoost / RF / linear | 0.149 / 0.187 / 0.227 / 0.018 °C |
| Models | Random → 4 km inflation: linear vs trees | 0.023 vs 0.118-0.151 °C |
| Models | Strip-layout stress penalty, LightGBM | 0.016 °C |
| Errors | Urban vs rest RMSE (LightGBM, 4 km) | 1.614 vs 1.805 °C |
| Errors | Urban vs rest R² | 0.542 vs 0.753 |
| Errors | Hottest-decile detection: urban / all cells | 54.2% / 62.3% |
| Errors | Residual Moran's I, LightGBM 4 km (raw target) | 0.583-0.651 (0.889-0.898) |
| SHAP | Mean abs SHAP: NDVI / dryness / moisture | 0.955 / 0.728 / 0.612 °C |
| SHAP | Mean abs SHAP: grass / tree / built-up | 0.541 / 0.382 / 0.125 °C |
| SHAP | Spectral block (summed per cell) | 1.488 °C |
| SHAP | Spectral contributions that cancel within the block | 35% |
| SHAP | LightGBM-XGBoost rank agreement (Spearman) | 0.989 |
| SHAP | Same dominant driver, LightGBM vs XGBoost: group / feature | 95.3% / 79.1% |
| SHAP | Same dominant driver, interventional vs path-dependent: group / feature | 92.5% / 64.6% |
| SHAP | NDVI-dryness correlation | −0.72 |
| Thresholds | Tree-cover knot (95% CI) | 10.2% (9.6-10.6%) |
| Thresholds | Tree-cover step at first canopy | −0.48 °C |
| Thresholds | Tree cover per 10 points, below / above knot | −0.25 / −0.07 °C |
| Thresholds | Moisture-index onset | −0.39 |
| RQ5 | Built-up per +10 points: raw vs model | −0.22 vs −0.029 °C |
| RQ5 | Shrinkage of the built-up association | 87% |
| RQ5 | Raw built-up correlation within urban cells | +0.009 |
| Scenarios | Target cells (hottest urban decile) | 2,673 |
| Scenarios | Targets with no tree cover | 78.7% |
| Scenarios | A +10 points: lower / realistic | −0.37 / −1.02 °C |
| Scenarios | B +20 points: lower / realistic | −0.56 / −1.63 °C |
| Scenarios | C to the knot: lower / realistic | −0.35 / −0.97 °C |
| Scenarios | First point of canopy: realistic / lower | −1.43 / −0.87 °C |
| Scenarios | Per point, realistic: 2% to knot / 12-30% | −0.12 / −0.05 °C |
| Scenarios | Index-model R²: moisture / NDVI / dryness | 0.716 / 0.668 / 0.614 |
| Scenarios | Scenario rows outside training support (max) | 0.04% |

## Problem

**1. What problem does MirageAI address, and why does it matter for Pune?**
Heat plans need to know where greening is most associated with cooler surfaces, and most LST models are validated in ways that flatter them. We map Pune's pre-monsoon daytime LST at 100 m, validate it honestly with spatial blocks, explain it with SHAP and bracket greening scenarios for the hottest urban cells (D-002, D-003, D-019).

**2. Why the name MirageAI?**
The daytime urban heat island we expected is a mirage: the core is 3.4-4.0 °C cooler than the outer ring by day. The name says what we found; the registered title is unchanged (D-045).

**3. Why surface LST and not air temperature?**
Landsat measures the surface skin temperature at about 10:30 AM, wall-to-wall at 100 m (D-047, D-001). Air temperature would need station or reanalysis data, which is far too coarse for neighbourhood-scale modelling. LST is the standard proxy for the surface heat island, but it isn't what people feel, so we never claim air temperature or comfort; that is stated as a limitation and in the scenario caveat (D-009).

**4. Why only the pre-monsoon season?**
It is Pune's peak heat season and the clearest, so Landsat has the most valid observations (D-002). The cool-island result is therefore a pre-monsoon daytime finding; night and other seasons are future scope.

## Data

**5. Where does the data come from, and why a 100 m grid?**
Landsat 8/9 surface temperature, Sentinel-2 indices, ESA WorldCover fractions and SRTM terrain, composited in Earth Engine as cloud-masked seasonal medians (D-005). The grid is 100 m because that is the thermal sensor's native resolution; 30 m would claim detail the target does not have (D-001).

**6. Why the anomaly target instead of absolute LST?**
Absolute LST shifts between years with weather and scene timing, and no feature encodes the year, so the level is not learnable and would add noise to the 2025 test (D-021). The anomaly from each year's median keeps the spatial pattern, which is what the drivers can explain.

**7. Why exclude water-majority cells?**
Reservoir temperature is set by the thermal inertia of water, not by land-surface drivers, and water is not a planning lever (D-034). It removed 2,697 rows, none urban; water fraction and distance to water stay as features.

**8. Why is NDBI not a "built-up" index here?**
NDBI correlates with heat in the opposite direction to built-up fraction, and it peaks in the least-built cells, because dry grass, fallow and bare soil score as high as roofs (D-024). We keep it and call it the dryness index; MNDWI is used on land as a moisture index (D-038).

**9. Why one WorldCover map for all three years?**
WorldCover 2021 is the best free 10 m land cover and no newer version was available in Earth Engine (D-006). Land-cover change between 2021 and 2025 is therefore not represented, which we list as a limitation.

## Methodology

**10. Why not plain random cross-validation?**
Neighbouring cells are nearly identical, so randomly held-out cells leak information from training (D-003). Random CV is reported only to measure that inflation (RQ3); spatial blocks are the headline.

**11. Why 4 km blocks and not 8?**
Blocks are sized to the autocorrelation the model cannot explain: the pre-registered residual range is 3,950 m, so 4 km (D-028). The tuned model's own residuals give 4,250 m, but the measured range grows with the CV used to produce it (3,450 m under random CV vs 5,250 m under 8 km), so re-measuring would be circular. The 8 km results are reported as the conservative bound.

**12. Why shuffled, multi-seed folds?**
Unshuffled assignment accidentally made every 4 km fold an east-west strip, and one shuffled layout would be just as arbitrary (D-033). We deal blocks at random over 5 seeds and report mean ± std over all 25 runs. The strip layout survives as a stress test; it costs LightGBM only 0.016 °C.

**13. Why are coordinates never features?**
A model that knows where a cell is can memorise it instead of learning drivers, which defeats spatial validation (D-004). The exclusion is an assertion in the pipeline, not a convention (D-014).

**14. Why a spatio-temporal 2025 test instead of a plain temporal holdout?**
A temporal holdout on the same cells mostly measures memorisation of place, because static features repeat across years (D-010). We test on 2025 in blocks never seen in training; scoring the same rows with a model that saw the places adds a memorisation gap of 0.149 °C for LightGBM.

**15. How do you tune without leakage?**
Tuning runs inside spatial CV on 2023-24 only, so 2025 is never seen during selection (D-029). It slightly flatters the seed-42 spatial layout it was tuned on, which is why the 2025 test is the clean check.

**16. Why these four models?**
Linear regression stands for the "existing studies" baseline, Random Forest for bagging, and XGBoost and LightGBM for boosting; comparing them answers RQ2 (D-007).

**17. Why drop built-up fraction only for linear regression?**
The fractions sum to one, which makes the linear design singular, so one must be the reference; built-up conditions it best (D-023). Trees do not have that problem and must keep it, or SHAP could not attribute anything to built-up surface (D-025).

## Implementation

**18. How does SHAP handle the correlated indices?**
We use interventional TreeSHAP with a 1,000-row background, which credits a feature only through the model's response to it (D-036). It matters: path-dependent SHAP picks the same dominant driver only 64.6% of the time, so we rank the spectral block as a whole and its members loosely.

**19. Why does SHAP explain a model fitted on all three years?**
After Phase 5, 2025 had done its job as the untouched test; the explanation should describe all the data (D-035). Its accuracy claim stays the cross-validated one, 1.747 °C on unseen 2025 places.

**20. How did you find thresholds rather than eyeballing plots?**
One-knot segmented fits of SHAP on feature value, with 200-resample bootstrap intervals (D-037). Fractions with many zeros are split into a step at zero and a fit on positive values, which gives the 10.2% tree-cover knot.

**21. Why is the driver map built from groups?**
The individual indices swap credit between models and SHAP methods (D-038). Groups agree across models for 95.3% of cells against 79.1% for single features, so the map does not flip arbitrarily.

**22. How do you know the numbers in the report and slides are right?**
A script recomputes every key number from the output tables and the build refuses to run if one is missing or different (D-044). It caught three values that had been rounded twice. References were checked against their DOI records the same way.

## Results

**23. Which model is best? Why LightGBM if the models are tied?**
Under spatial CV the three tree models are tied (LightGBM and XGBoost 1.746 °C, RF 1.756 °C). LightGBM is the reference because it is lowest on the clean 2025 test and cheapest to explain with SHAP, not because it is more accurate (D-032).

**24. How much does random CV overstate accuracy (RQ3)?**
From random to 4 km blocks, linear regression loses 0.023 °C and the trees 0.118-0.151 °C (D-033). The inflation depends on model flexibility, so random CV would also have crowned the wrong model (XGBoost at 1.594 °C).

**25. What is most associated with daytime heat (RQ1)?**
The spectral block (1.488 °C mean abs SHAP): NDVI, dryness and moisture, then grass, tree cover and cropland; built-up fraction ranks only 10th (D-036, D-038). LightGBM and XGBoost agree with Spearman 0.989.

**26. Does the negative built-up correlation vanish (RQ5)?**
It shrinks by 87%, from −0.22 to −0.029 °C per +10 points, once vegetation and dryness are modelled, and it does not flip (D-024, D-036). The cool core is mostly a vegetation and dryness contrast, not an effect of buildings.

**27. What does the residual Moran's I of about 0.6 mean?**
Neighbouring residuals still resemble each other (0.583-0.651 against 0.889-0.898 for the raw target), so the features explain about a third of the spatial similarity (D-028 sizes blocks from this residual structure). It points to drivers we cannot see, such as building height, roofing and irrigation, and it is why uncertainty is reported as fold spread, not per-cell error.

## Defense

**28. Are the scenarios causal?**
No. They are the change in a fitted model's prediction when a cell's features are edited, learned from comparing different places (D-009). Places with more trees also differ in unmeasured ways, so the numbers are planning signals, not forecasts.

**29. What is the first-point step, and why exclude it?**
For the 78.7% of targets with no canopy, the first pixel of tree cover is associated with −1.43 °C (realistic) or −0.87 °C (lower), far more than any later point (D-040). One 10 m pixel in a hectare cannot do that; the step reflects what tree-less places are, so the headline excludes it and reports it as a finding.

**30. Why target observed rather than predicted heat?**
Selecting on predictions would inherit the model's 54.2% precision at finding the hottest urban cells (D-039). Observed 3-year means pick the 2,673 cells that were actually hottest.

**31. How do you know the scenarios are not extrapolating?**
A nearest-neighbour check flags edited rows farther from the training data than 99% of real rows are from each other (D-043). At most 0.04% of scenario rows are flagged, because Pune already has urban cells with that much tree cover.

**32. Why a bracket instead of one number?**
How much greenness, dryness and moisture respond to new trees is uncertain, so the lower bound holds them fixed and the realistic bound moves them through validated index models (D-009, D-041). Three tied models give the model spread (D-042); +10 points is −0.37 to −1.02 °C.

**33. What would you do with more time?**
Add building height, roofing and irrigation layers, validate against air temperature, extend to night and other seasons, and use land-cover change over time to move beyond cross-sectional comparison (D-009, D-006).
