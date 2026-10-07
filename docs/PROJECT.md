# MirageAI — Concrete vs. Canopy

> **Concrete vs. Canopy: Explainable Machine Learning for Urban Heat Island Drivers in Pune**

| | |
|---|---|
| **Course** | Applications and Use Cases of Machine Learning (TE7483), Semester 5 |
| **Institute** | Symbiosis Institute of Technology, Pune (AIML, batch 2024–28) |
| **Assessment** | ESE Case Study Project, 30 marks (60% weight) |
| **Domain / SDGs** | Smart Cities · SDG 11 (Sustainable Cities) · SDG 13 (Climate Action) |
| **Project name** | MirageAI (renamed from Ushna, D-045) |

Pune's expected daytime urban heat island turns out to be a mirage: the city core is cooler than its dry surroundings, and MirageAI explains why.

---

## 1. Problem Statement

Rapid urbanisation in Pune is replacing vegetation and water bodies with concrete and asphalt. This intensifies the Urban Heat Island (UHI) effect and increases heat stress, energy demand, and public-health risk.

Most existing UHI studies have two weaknesses:
1. **Linear, single-index analysis.** They correlate land surface temperature (LST) with one index, usually NDVI, and miss nonlinear effects and interactions between land-cover drivers.
2. **Inflated accuracy.** They validate with random train–test splits. Neighbouring pixels are nearly identical (spatial autocorrelation), so test pixels "leak" information from training pixels and the reported accuracy is overstated.

MirageAI builds an explainable ML framework that:
- predicts LST on a 100 m grid from multi-source satellite features
- evaluates models honestly with spatial block cross-validation
- quantifies each driver's contribution with SHAP
- simulates greening scenarios to estimate achievable cooling

## 2. Objectives

1. Build a 100 m multi-source dataset for Pune covering the pre-monsoon season (Mar–May) of 2023, 2024 and 2025.
2. Compare Linear Regression, Random Forest, XGBoost and LightGBM for LST prediction under spatial block CV.
3. Quantify how much random CV inflates accuracy compared with spatial CV.
4. Explain LST drivers with SHAP, including nonlinear thresholds and feature interactions.
5. Estimate the cooling potential of greening scenarios in Pune's hottest zones.

## 3. Research Questions

- **RQ1:** Which land-cover and terrain factors drive daytime LST in Pune, and how strongly?
- **RQ2:** Do nonlinear models capture driver effects that linear models miss (thresholds, saturation, interactions)?
- **RQ3:** By how much does random CV overstate performance compared with spatial block CV?
- **RQ4:** How much cooling is associated with realistic increases in tree cover in the hottest cells?
- **RQ5:** Does Pune show a daytime surface urban cool island in the pre-monsoon season, and does the negative `frac_built`–LST correlation vanish once vegetation and dryness are controlled for?

## 4. Scope

**In scope**
- Daytime land surface temperature (Landsat overpass, ~10:30 AM local time)
- Pre-monsoon season (Mar–May), 2023–2025
- Pune Municipal Corporation + PCMC core area
- Associational driver analysis and what-if scenarios

**Out of scope**
- Air temperature or nighttime UHI
- Causal claims (results are associations, not proven causes)
- Real-time or operational systems
- Cities other than Pune

## 5. Stakeholders

- **Pune Municipal Corporation (PMC) and PCMC:** urban planning and green-cover policy
- **Urban planners and architects:** siting parks, cool roofs, tree planting
- **Public-health bodies:** identifying heat-vulnerable zones
- **Residents of the hottest wards:** the people who carry the heat burden

## 6. Study Area

Bounding box (WGS84): lon 73.72–74.02, lat 18.42–18.65. This covers the PMC + PCMC core (~800 km², ~80,000 cells at 100 m). Replace it with an official PMC boundary shapefile if one is available.

Projection for all processing: **EPSG:32643** (UTM zone 43N, metres).

## 7. Data Sources (all free, official, non-Kaggle)

| Source | Role | Resolution | GEE ID |
|---|---|---|---|
| Landsat 8/9 Collection 2 Level-2 | **Target:** LST from thermal band ST_B10 | 30 m product (100 m native thermal) | `LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2` |
| Sentinel-2 Surface Reflectance | NDVI, NDBI, MNDWI | 10–20 m | `COPERNICUS/S2_SR_HARMONIZED` |
| ESA WorldCover v200 (2021) | Land-cover fractions, water mask | 10 m | `ESA/WorldCover/v200` |
| SRTM | Elevation, slope | 30 m | `USGS/SRTMGL1_003` |

All data is accessed through Google Earth Engine (free noncommercial account).

## 8. Features

| Feature | Description | Source |
|---|---|---|
| `NDVI` | Vegetation index | Sentinel-2 |
| `NDBI` | Dryness / bare-surface index (see §12) | Sentinel-2 |
| `MNDWI` | Water index | Sentinel-2 |
| `frac_tree`, `frac_shrub`, `frac_grass`, `frac_crop`, `frac_built`, `frac_bare`, `frac_water` | Fraction (0–1) of each land-cover class inside the 100 m cell | WorldCover |
| `dist_water_m` | Distance to nearest water pixel (m) | WorldCover |
| `elevation`, `slope` | Terrain | SRTM |
| **`LST`** (target) | Median pre-monsoon LST (°C) | Landsat 8/9 |
| `LST_n_obs` | Number of cloud-free observations (QC only, not a feature) | Landsat 8/9 |
| `lon`, `lat`, `x_utm`, `y_utm` | Coordinates (CV blocks and maps only, **never features**) | — |

## 9. Methodology

```mermaid
flowchart LR
    A[GEE extraction<br/>Landsat, S2, WorldCover, SRTM] --> B[100 m grid<br/>CSV + GeoTIFF per year]
    B --> C[EDA & preprocessing<br/>missing, outliers, VIF]
    C --> D[Spatial blocks<br/>4 km primary; 1, 2, 8 km sensitivity]
    D --> E[Models<br/>LR, RF, XGB, LGBM]
    E --> F[Evaluation<br/>random vs spatial CV<br/>temporal holdout]
    F --> G[SHAP<br/>global, dependence, interactions]
    G --> H[Greening scenarios<br/>ΔLST in hottest cells]
    H --> I[Report, slides, viva]
```

1. **Extraction:** cloud-masked median composites, aggregated by area-weighted mean onto a 100 m grid (`gee_extract.py`).
2. **Preprocessing:** merge the years, derive year-independent cell and block ids (D-022), model the within-year anomaly `LST_anom` rather than absolute LST (D-021), check distributions, outliers and multicollinearity (VIF). Land-cover fractions sum to 1, so linear regression drops `frac_built` (D-023, D-025). Water-majority cells are excluded before any split (D-034).
3. **Validation design** (`cv.py`):
   - spatial block CV with **4 km** blocks, sized from the autocorrelation range of model residuals (D-028); 1, 2 and 8 km are reported for sensitivity
   - blocks dealt to folds at random, repeated over five seeds (42-46): every result is mean ± std over 25 runs (D-033)
   - random CV (grouped by cell) run alongside, only to measure inflation
   - spatio-temporal test: train on 2023 + 2024 in some blocks, test on 2025 in the others (D-010)
4. **Modelling:** a mean baseline plus four models, each tuned *inside* spatial CV on the training years only (D-029).
5. **Evaluation:** RMSE, MAE and R² for each CV scheme, paired model differences, urban vs rest, hottest-decile detection, residual maps and Moran's I.
6. **Explainability:** interventional TreeSHAP on the final LightGBM (D-035, D-036), grouped drivers (D-038), fitted thresholds with bootstrap CIs (D-037) and interaction values.
7. **Scenarios:** raise tree cover in the hottest decile of *urban* cells (`frac_built` ≥ 0.5) and predict ΔLST under a lower and a realistic bound with three models (D-039 to D-043). Results are reported as associational cooling potential.

### Structure of the dataset

The three years are **repeated measurements of the same 81,983 cells, not
independent samples.** The grid, the WorldCover fractions and the SRTM terrain are
identical in every year; only the Landsat LST and the Sentinel-2 indices vary. The
table therefore holds 3 × 81,983 = 245,949 rows but only 81,983 distinct locations.

Two consequences, both enforced in code:
- Spatial blocks are derived from UTM coordinates alone, so a cell falls in the
  same block in every year and no cell can appear on both sides of a split (D-022).
- Effective sample size for anything spatial is far below the row count, so
  significance from n = 245,949 would be badly overstated.

## 10. Outputs (all delivered)

- Clean 100 m dataset (3 years) and GeoTIFF stacks
- Model comparison table (5 models × 8 validation schemes, 25 runs each): `outputs/tables/model_comparison.csv`
- LST, prediction and residual maps of Pune
- SHAP driver ranking, dependence and interaction plots
- Greening-scenario cooling map and summary table
- Final report, presentation deck, viva notes, reproducible code repo
- Interactive Streamlit dashboard (`app/streamlit_app.py`)

## 11. Rubric Mapping (ESE, 30 marks)

| Rubric component | Marks | Where MirageAI covers it |
|---|---|---|
| Problem refinement & use-case relevance | 3 | §1–5: research gap, RQs, stakeholders, SDGs |
| Dataset understanding, preprocessing, feature handling | 4 | §7–8, Phase 2: sources, sizes, features, missing values, VIF, validation strategy |
| Methodology & ML pipeline design | 5 | §9, `DECISIONS.md`: every choice justified |
| Implementation / project execution | 5 | Reproducible repo, `src/` modules, configs, seeds |
| Results, metrics, comparative analysis | 5 | 4 models × 3 CV schemes, RMSE/MAE/R², residual maps |
| Final report quality | 3 | Report with all required sections + references |
| Communication & presentation | 2 | Map-heavy deck (maps tell this story best) |
| Viva / project defense | 3 | `DECISIONS.md` (each decision = a viva answer), `LEARNING_LOG.md` |

## 12. Known Limitations

- WorldCover is a 2021 snapshot, while LST and indices are per-year.
- Landsat LST is daytime surface temperature, not the air temperature people feel.
- Cloud gaps remove some cells; pre-monsoon is chosen partly to minimise this.
- **NDBI acts as a dryness index here, not a built-up index.** In semi-arid Pune it
  tracks bare, low-moisture surfaces; in 2024 it correlates +0.63 with LST while
  `frac_built` correlates −0.19. It must not be described as a built-up index anywhere
  in this project.
- **WorldCover `crop` is harvested fallow during Mar–May.** The pre-monsoon cropland
  class is mostly bare dry soil, not standing vegetation, which is why it behaves
  thermally like bare ground rather than like canopy.
- Scenario results are associations from a predictive model, not causal estimates.
