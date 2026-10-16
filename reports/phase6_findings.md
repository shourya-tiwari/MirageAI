# Phase 6 findings: drivers of daytime LST anomaly (SHAP)

Model: LightGBM, tuned parameters, fitted on all 243,252 land rows of 2023-25
(D-035). SHAP: interventional TreeSHAP, 1,000-row background, 10,000 explained
rows stratified by year × urban/rest (D-036). Units: °C of `LST_anom`. All
statements are associational: they describe the fitted model's response to a
feature, not the effect of changing it on the ground. Source tables are in
`outputs/tables/phase6_*.csv`, figures in `outputs/figures/06_*.png`.

## Driver ranking (mean |SHAP|, °C)

| Rank | Feature | All cells | Urban | Rank urban | XGBoost all | XGBoost urban |
|---|---|---|---|---|---|---|
| 1 | NDVI | 0.955 | 0.819 | 1 | 1.098 (1) | 0.907 (2) |
| 2 | NDBI (dryness) | 0.728 | 0.539 | 3 | 0.588 (3) | 0.487 (4) |
| 3 | MNDWI (moisture) | 0.612 | 0.797 | 2 | 0.747 (2) | 0.934 (1) |
| 4 | Grass fraction | 0.541 | 0.535 | 4 | 0.555 (4) | 0.545 (3) |
| 5 | Tree cover fraction | 0.382 | 0.305 | 5 | 0.346 (5) | 0.287 (5) |
| 6 | Cropland fraction | 0.342 | 0.289 | 6 | 0.343 (6) | 0.285 (6) |
| 7 | Elevation | 0.253 | 0.226 | 7 | 0.261 (7) | 0.239 (7) |
| 8 | Slope | 0.156 | 0.124 | 9 | 0.153 (9) | 0.124 (8) |
| 9 | Distance to water | 0.156 | 0.106 | 10 | 0.171 (8) | 0.122 (9) |
| 10 | Built-up fraction | 0.125 | 0.167 | 8 | 0.089 (10) | 0.090 (10) |
| 11 | Shrub fraction | 0.078 | 0.078 | 11 | 0.077 (11) | 0.087 (11) |
| 12 | Bare fraction | 0.037 | 0.032 | 12 | 0.035 (12) | 0.028 (12) |
| 13 | Water fraction | 0.019 | 0.008 | 13 | 0.020 (13) | 0.008 (13) |
| — | Spectral block (NDVI+NDBI+MNDWI, summed per cell) | 1.488 | 1.022 | 1 | 1.489 | 1.030 |

LightGBM vs XGBoost rank agreement: Spearman 0.989 (Kendall 0.949) on all
cells, 0.973 (0.897) on urban cells. The top-five set is identical in both, but
the order differs inside the spectral block. On all cells XGBoost puts MNDWI (moisture)
above NDBI (#2/#3 swapped). On urban cells it puts MNDWI (moisture) above NDVI (0.934 vs
0.907) and grass above NDBI.

## Dominant driver per cell (T6.5, grouped)

The map is built from grouped SHAP (D-038): spectral (NDVI + NDBI + MNDWI), dry
cover (grass + crop + bare + shrub), tree cover, built-up, water proximity
(dist_water_m + frac_water) and terrain (elevation + slope). Shares of the
81,084 land cells, with shares of urban cells in brackets:

| Group | All cells | Urban cells | Mean grouped |SHAP| (°C) |
|---|---|---|---|
| Spectral | 66.8% | 49.8% | 1.49 |
| Dry cover | 24.5% | 42.7% | 0.72 |
| Terrain | 3.3% | 3.4% | 0.29 |
| Tree cover | 3.3% | 3.0% | 0.38 |
| Water proximity | 2.0% | 0.7% | 0.17 |
| Built-up | 0.1% | 0.3% | 0.12 |

Grouping makes the map stable. The dominant group is the same under XGBoost
for 95.3% of explained rows (79.1% at feature level) and the same under
path-dependent SHAP for 92.5% (64.6% at feature level). The per-feature map is
kept as a supplementary figure (`06_dominant_driver_map_features.png`).
Dry-cover dominance concentrates in the peri-urban fringe and in parts of the
urban fabric with grass or bare plots: 42.7% of urban cells.

## RQ1: Which factors drive daytime LST, and how strongly?

Across Pune's land cells, the model's predicted anomaly is associated most
strongly with the three spectral indices. NDVI carries a mean |SHAP| of
0.95 °C, and its contribution spans 5.0 °C between the 1st and 99th
percentiles. NDBI, which here tracks surface dryness rather than buildings
(D-024), carries 0.73 °C (span 4.3 °C), and MNDWI (moisture) 0.61 °C (span 3.0 °C). Summed
per cell, the spectral block contributes 1.49 °C on average, 2.75 times the
next driver. 35% of its members' separate contributions cancel within the
block (53% in urban cells), so the indices partly carry the same signal. Land
cover follows: grass fraction (0.54 °C) and cropland (0.34 °C) are associated
with warmer surfaces in this dry pre-monsoon season, and tree cover (0.38 °C)
with cooler ones. Terrain comes next: elevation (0.25 °C), slope (0.16 °C) and
distance to water (0.16 °C). Built-up fraction ranks 10th (0.12 °C), and shrub,
bare and water fractions are each below 0.08 °C. For scale, the anomaly's
standard deviation is about 3.4 °C. XGBoost gives the same ranking
(Spearman 0.99). In urban cells, moisture (MNDWI 0.80 °C) rises to nearly
NDVI's weight (0.82 °C), and built-up fraction moves from 10th to 8th
(0.17 °C).

## RQ2: Do the trees capture what linear regression misses?

They do, in three ways.

**Thresholds and saturation.** The dependence curves bend where a straight
line cannot:
- **Tree cover:** any tree cover at all is associated with a step of −0.48 °C.
  Beyond that, about −0.25 °C per 10 points of cover up to a knot at 10.2%
  (95% CI 9.6-10.6%), then only −0.07 °C per 10 points.
- **MNDWI (moisture):** no association below −0.39 (CI −0.40 to −0.39), then −1.4 °C per
  +0.1 above it.
- **NDVI:** −1.2 °C per +0.1 below 0.27 (CI 0.27-0.29), then −0.7 °C.
- **Distance to water:** the cooling associated with water reaches about 190 m
  (CI 185-192 m) and is flat beyond.
- **Elevation:** the association flattens above 590 m.

For tree cover, a straight line explains 69% of the SHAP variation and the
one-knot fit 95%.

**Interactions.** The largest are within the dryness-vegetation system:
NDBI × grass (0.18 °C), NDVI × NDBI (0.16 °C) and NDVI × MNDWI (moisture) (0.14 °C).
These are path-dependent values, so they rank pairs rather than give
interventional-scale magnitudes. Dryness is associated with more warming in
grass-dominated cells. Wet but sparsely vegetated cells are associated with
extra cooling. NDVI and NDBI partly offset each other, consistent with the
within-block cancellation.

**Unstable credit under collinearity.** This turned out to matter as much as
nonlinearity. Standardised linear-regression coefficients split the spectral
signal very differently from the trees:

| Index | Linear regression (°C per SD) | Trees' linearised response (°C per SD) |
|---|---|---|
| NDVI | −3.33 | −1.15 |
| MNDWI (moisture) | −2.12 | −0.68 |
| NDBI | −0.84 | +0.90 |

The linear NDBI coefficient has the opposite sign to both its raw correlation
with LST (+0.70) and the tree response. Its contribution is anti-correlated
with the tree SHAP at r = −0.97, compensated by an inflated NDVI. The linear
model's spectral-block total still tracks the trees' (r = 0.96; 1.69 vs
1.49 °C). So its individual spectral coefficients cannot be read as driver
effects, while the trees' per-index shapes can. Together these cost linear
regression 0.13 °C of spatial-CV RMSE (1.873 vs 1.746 °C).

## RQ5: Does the negative frac_built–LST correlation vanish once vegetation and dryness are controlled for?

Raw, Pune shows a daytime surface cool-island association: built-up fraction
correlates with the LST anomaly at r = −0.24 (Spearman −0.26), or −0.22 °C per
+10 points built. With vegetation, dryness, moisture, land cover and terrain in
the model, the response attributed to built-up fraction shrinks to −0.029 °C
per +10 points. That is 87% smaller, with mean |SHAP| 0.12 °C, 10th of 13.

**It does not flip to warming.** The remaining association stays weakly
negative: Spearman between frac_built and its SHAP is −0.80. Mean SHAP is
+0.08 °C for cells under 10% built and −0.17 °C for cells over 75% built, a
total span of about 0.25 °C that mostly saturates by ~19% built.

Within urban cells the raw association is already nil (r = +0.009). There, the
model attributes a small cooling (−0.018 °C per +10 points; mean SHAP
−0.16 °C; rank 8), and XGBoost gives built-up fraction about half LightGBM's
weight (0.09 vs 0.17 °C). That residual is therefore model-dependent and should
not be over-read.

So most of the cool island is carried by vegetation and dryness: in the
pre-monsoon, the built core is less dry than the grass, cropland and bare land
around it. Interventional SHAP varies built-up fraction while holding the other
fractions fixed, which cannot happen on the ground because fractions sum to
one, so this is a conditional association, not an effect of building.

## T6.6 One line per driver

- **NDVI:** greener cells are associated with cooler surfaces, about −1.2 °C
  per +0.1 NDVI up to NDVI ≈ 0.27, then a flatter −0.7 °C per +0.1.
- **NDBI (dryness):** drier surfaces are associated with warmer LST, nearly
  linearly, at roughly +0.7 to +1.1 °C per +0.1 NDBI, with no threshold. It
  measures dryness here, not buildings.
- **MNDWI (moisture):** no association while surfaces are dry
  (MNDWI < −0.39). Above that, wetter cells run cooler, about −1.4 °C per +0.1.
- **Grass fraction:** in the dry season grass behaves as a warm surface. Cells
  with some grass run about 0.35 °C warmer than cells with none, rising by
  about +0.4 °C per 10 points up to ~12% grass and +0.13 °C per 10 points
  beyond.
- **Tree cover:** any tree cover is associated with about 0.5 °C cooler
  surfaces. Further cooling is about −0.25 °C per 10 points up to ~10% cover,
  then flattens to −0.07 °C per 10 points.
- **Cropland fraction:** fields in the pre-monsoon (largely fallow) are
  associated with about 0.3 °C warmer surfaces, rising gently with more
  cropland.
- **Elevation:** higher cells run warmer up to about 590 m (+0.16 °C per
  10 m), with no further association above that.
- **Distance to water:** cells beside water run about 1.1 °C cooler than cells
  190 m or more away; beyond ~190 m there is no further association.
- **Built-up fraction:** a small negative association (cooler) that is mostly
  in place by ~19% built, worth about 0.25 °C in total. See RQ5.
