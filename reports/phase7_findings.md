# Phase 7 findings: greening scenarios for Pune's hottest urban cells (RQ4)

**Read this first (T7.5, D-009).** Every ΔLST below is the change in a fitted
model's prediction when a cell's features are edited. It is not the
temperature change that planting trees would cause. The models learned how land
surface temperature varies across cells that differ in tree cover. Those cells
also differ in things the features do not carry: building height and density,
roofing material, irrigation, and the age and type of neighbourhood (residual
Moran's I ≈ 0.6). A scenario credits the tree-cover edit with whatever of that
contrast travels with tree cover in the data. Read the numbers as "cells that
look like this with more tree cover are, in these data, this much cooler". They
are a planning signal for where greening is most associated with lower
*surface* temperature, not a forecast of what a programme would deliver, and
they say nothing about air temperature or thermal comfort.

Sources: `outputs/tables/phase7_*.csv`, `outputs/figures/07_*.png`, D-009 and
D-039 to D-043.

## Target cells

These are urban cells (built-up ≥ 50%) in the hottest decile of their
**observed** 2023-25 mean LST anomaly: **2,673 of 26,726 urban cells**
(threshold +2.41 °C; mean +3.63 °C), 8,019 cell-years. They lie on the urban
fringe, mostly to the east:

| Quadrant from the AOI centre | Share of target cells |
|---|---|
| North-east | 45% |
| South-east | 29% |
| North-west | 22% |
| South-west | 3% |

The median distance from the centre is 11.5 km. They form 595 clusters, and the
five largest hold 20%:

| Cluster | Cells | Location (lat, lon) | Direction from the centre |
|---|---|---|---|
| 1 | 171 | 18.581, 73.920 | north-east |
| 2 | 132 | 18.468, 73.957 | south-east |
| 3 | 110 | 18.635, 73.836 | north |
| 4 | 68 | 18.452, 73.970 | south-east |
| 5 | 62 | 18.576, 73.965 | east |

Median built-up is 78%, median tree cover 0.0%. 78.7% of the cells have no tree
cover at all, and 96% are below the 10.2% knot.

## Bracketing (D-009)

- **Lower bound:** tree cover replaces built, bare and grass in proportion to
  their shares; NDVI, NDBI (dryness) and MNDWI (moisture) are held fixed.
- **Realistic bound:** the same edit, with the three indices updated by
  Δindex = f(new fractions) − f(old fractions). f is a LightGBM per year and
  index, from fractions and terrain, validated by spatial CV:

| Index | R² mean ± std | 2023 | 2024 | 2025 |
|---|---|---|---|---|
| MNDWI (moisture) | 0.716 ± 0.037 | 0.752 | 0.708 | 0.690 |
| NDVI | 0.668 ± 0.043 | 0.698 | 0.671 | 0.635 |
| NDBI (dryness) | 0.614 ± 0.042 | 0.616 | 0.632 | 0.593 |

## Results: mean ΔLST over target cells (°C)

**Headline: the 0 → 1% "any canopy" step is excluded** (D-040), and the effect is
quoted as a lower-realistic bracket:

| Scenario | Lower bound | Realistic bound | Headline (lower to realistic) |
|---|---|---|---|
| A: +10 points | −0.37 [−0.42, −0.33] | −1.02 [−1.04, −1.00] | **−0.37 to −1.02 °C** |
| B: +20 points | −0.56 [−0.65, −0.49] | −1.63 [−1.67, −1.60] | **−0.56 to −1.63 °C** |
| C: to the 10.2% knot (avg +9.2 points) | −0.35 [−0.38, −0.31] | −0.97 [−1.00, −0.94] | **−0.35 to −0.97 °C** |

Values are the mean of LightGBM, XGBoost and RF, with the range across the three
models in brackets. Per-year means are averaged over 2023-25.

For transparency, the full table including the step, never quoted as the
result:

| Scenario | Bound | ΔLST including the step | Excluding flagged cells | % flagged |
|---|---|---|---|---|
| A: +10 points | lower | −1.06 [−1.32, −0.88] | −1.06 | 0.00 |
| A: +10 points | realistic | −2.14 [−2.29, −2.02] | −2.14 | 0.01 |
| B: +20 points | lower | −1.25 [−1.55, −1.04] | −1.25 | 0.04 |
| B: +20 points | realistic | −2.75 [−2.91, −2.62] | −2.75 | 0.04 |
| C: to the knot | lower | −1.03 [−1.28, −0.87] | −1.03 | 0.00 |
| C: to the knot | realistic | −2.09 [−2.23, −1.98] | −2.09 | 0.01 |

- Per-model values are in `phase7_summary.csv`.
- No target cell had too few donor fractions in any scenario.
- Flagged rows are almost absent, so including or excluding them changes
  nothing. Even at +30 points only 0.57% (lower) and 0.12% (realistic) of rows
  fall outside the training support.

## Finding: tree-less urban cells run about 1.4 °C hotter

78.7% of the target cells have no tree cover at all. For them, the first point
of cover, i.e. **any canopy present** (one 10 m WorldCover tree pixel in the
100 m cell), is associated with −1.43 °C (realistic) and −0.87 °C (lower). That
is far more than any later point, and one tree pixel in a hectare cannot do
it. The step is confounded with unobserved urban form: tree-less cells are
disproportionately dense new fringe and industrial development, and the
features do not see building height, density, roofing or irrigation. **So
tree-less urban cells run about 1.4 °C hotter in ways the model cannot
attribute to tree cover alone.** It is reported as a finding about where the
heat is, not as a cooling effect of planting, and it is excluded from the
scenario headline.

## What the numbers say

1. **About 0.4-1.0 °C for raising tree-sparse cells to ~10% cover.** A
   (+10 points) and C (to the knot) give almost the same bracket. C reaches
   about 95% of A's cooling with 92% of its tree cover (9.2 vs 10 points).
2. **Gains per point roughly halve past the knot.** Beyond the first point,
   each point of tree cover is associated with −0.12 °C (realistic) or
   −0.04 °C (lower) from 2% up to the 10.2% knot, and −0.05 or −0.02 °C from
   12% to 30% (ratios 0.45 and 0.40). B doubles the added cover for about 60%
   more cooling (realistic −1.63 vs −1.02 °C).
3. **The bounds differ by a factor of about three.** How much greenness,
   dryness and moisture respond to new tree cover is the largest single
   uncertainty, far larger than the choice of model (model ranges ≤ 0.16 °C
   once the step is excluded).
4. **The models attribute less to tree cover than the raw data do.** Urban
   cells with 60-90% built-up and 10-15% tree cover are 1.0 °C cooler on
   average than those with 5-10%, against the models' ~0.05-0.12 °C per point.
   Holding vegetation, dryness, moisture and the other land cover fixed absorbs
   most of the raw contrast, as it should. The raw contrast is itself
   confounded, so this is a plausibility check, not validation.

## Planner-facing takeaway

Pune's hottest urban surfaces are tree-sparse fringe developments, mostly on
the eastern edge. In these data, raising them to ~10% tree cover is associated
with ~0.4-1.0 °C cooler daytime surfaces, with gains per point roughly halving
beyond ~10-12% cover. These are cross-sectional associations, not measured
effects of planting.
