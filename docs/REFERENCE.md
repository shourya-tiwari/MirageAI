# Reference

Technical reference for implementation, plus the bibliography.

## 1. Data Processing Details

### Landsat 8/9 Collection 2 Level-2 (target)
- Collections: `LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2` (merge both)
- Pre-filter: `CLOUD_COVER < 30`
- **LST (°C)** = `ST_B10 × 0.00341802 + 149.0 − 273.15`
- Mask fill values: `ST_B10 > 0`
- **QA_PIXEL** bits to mask (bit = 1 means bad):

| Bit | Meaning |
|---|---|
| 1 | Dilated cloud |
| 2 | Cirrus |
| 3 | Cloud |
| 4 | Cloud shadow |

- Overpass is ~10:30 AM local time, so this is *daytime surface* temperature.

### Sentinel-2 SR (features)
- Collection: `COPERNICUS/S2_SR_HARMONIZED`, pre-filter `CLOUDY_PIXEL_PERCENTAGE < 30`
- Reflectance = DN / 10000
- **SCL** classes to mask: 3 (cloud shadow), 8 (cloud medium probability), 9 (cloud high probability), 10 (thin cirrus), 11 (snow)
- Bands used: B3 (green), B4 (red), B8 (NIR), B11 (SWIR1, 20 m)

### Spectral indices
| Index | Formula (Sentinel-2 bands) | Meaning |
|---|---|---|
| NDVI | (B8 − B4) / (B8 + B4) | Vegetation |
| NDBI | (B11 − B8) / (B11 + B8) | **Dryness / bare surface in semi-arid Pune.** Named a built-up index in the literature, but here it tracks dry bare ground, not buildings — see PROJECT.md §12 |
| MNDWI | (B3 − B11) / (B3 + B11) | Water |

### ESA WorldCover v200 (2021)
`ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')`

| Code | Class | Feature |
|---|---|---|
| 10 | Tree cover | `frac_tree` |
| 20 | Shrubland | `frac_shrub` |
| 30 | Grassland | `frac_grass` |
| 40 | Cropland | `frac_crop` |
| 50 | Built-up | `frac_built` |
| 60 | Bare / sparse vegetation | `frac_bare` |
| 80 | Permanent water | `frac_water` |

The remaining classes (70 snow, 90 wetland, 95 mangroves, 100 moss) are negligible in Pune.

### SRTM
`USGS/SRTMGL1_003`, band `elevation`; slope via `ee.Terrain.slope()`.

## 2. GEE Pitfalls (read before Phase 1)

1. **Composites lose their projection.** `.median()` returns a 1° WGS84 default projection. Always `.reproject(crs, native_scale)` *before* `reduceResolution()`.
2. **`reduceResolution` pattern:**
   ```python
   img.reproject(crs=CRS, scale=native) \
      .reduceResolution(ee.Reducer.mean(), maxPixels=1024) \
      .reproject(crs=CRS, scale=100)
   ```
3. **`fastDistanceTransform` returns squared distance in *pixels*.** Reproject the water mask to a fixed 30 m first, then compute `.sqrt().multiply(30)` to get metres. A neighbourhood of 256 px caps distances at ~7.7 km.
4. **WorldCover v200 is an ImageCollection.** Use `.first()`.
5. **`filterDate` end is exclusive.** For Mar–May use `('YYYY-03-01', 'YYYY-06-01')`.
6. **Avoid `.getInfo()` on large objects.** Use exports for data and `.getInfo()` only for small counts.
7. **Export CRS and scale explicitly** (`EPSG:32643`, 100 m) on both table and image exports.

## 3. Metrics

| Metric | Formula | Notes |
|---|---|---|
| RMSE | √(mean((y − ŷ)²)) | °C, penalises large errors |
| MAE | mean(\|y − ŷ\|) | °C, robust and interpretable |
| R² | 1 − SS_res / SS_tot | Share of variance explained |
| Moran's I (residuals) | Spatial autocorrelation | Near 0 means residuals are spatially random; positive means unmodelled spatial structure |

Report every metric as **mean ± std across CV folds**.

## 4. Spatial CV Notes
- Block ID: `floor(x_utm / size) * 1e6 + floor(y_utm / size)` (`preprocess.block_id`), from coordinates only, so a cell keeps its block in every year (D-022). `floor`, not `round`: cell centres sit on half-cell offsets.
- Splitter: `GroupKFold(n_splits=5, shuffle=True, random_state=seed)` over the block column, repeated for seeds 42-46 (D-033). Unshuffled `GroupKFold` deals equal-sized blocks in id order, which at 4 km makes every fold a latitude strip; that layout is kept only as a stress test.
- Block size: at least the autocorrelation range of the model's **residuals** (D-028): ~3.7-4.0 km → 4 km primary. The raw target's range (7.65 km) mostly reflects the regional gradient the features already explain. 1, 2 and 8 km are reported for sensitivity.
- Random CV is grouped on `cell_id`, so a cell's three yearly rows stay together and the measured inflation is spatial, not year-duplicate leakage.
- Every splitter refuses water-majority rows (`cv.require_land_only`, D-034).

## 5. SHAP Notes
- `shap.TreeExplainer(model, data=background, feature_perturbation="interventional")`, 1,000 background rows, `check_additivity=True` (D-036). Interventional SHAP is used because the features are correlated.
- Fit the explained model on float32-rounded features (`explain.float32_exact`), or additivity fails: shap evaluates trees on float32 inputs.
- Interventional SHAP is a single-core loop (~40 ms per row); `explain.interventional_shap(n_jobs=8)` parallelises it over row chunks with identical results.
- Interaction values exist only for path-dependent TreeSHAP; use them to rank pairs, not for interventional-scale magnitudes.
- Summaries use a 10,000-row sample stratified by year × urban/rest (`explain.stratified_sample`); the per-cell map uses every land row.

## 6. Bibliography

> ⚠️ **Verify every entry** (authors, year, venue, DOI) on Google Scholar before citing. Also add 3–5 recent Indian/Pune UHI studies found on Scholar. Do not cite any paper you haven't opened.
>
> **T8.3 status (2026-10-16):** the report's reference list is verified against DOI records by `scripts/verify_references.py` (results in `reports/reference_check.csv`). LightGBM (Ke et al., 2017) and scikit-learn (Pedregosa et al., 2011) have no registered DOI and are marked [UNVERIFIED] in the report. Added after verification: Siddiqui et al. (2021) and Mohammad & Goswami (2020, EGU abstract) for Indian SUHI context; Voogt & Oke (2003), Lundberg et al. (2020), Drusch et al. (2012), Farr et al. (2007), Moran (1950) and the USGS Landsat C2 L2 dataset. No Pune-specific study could be verified yet; the report has a marked placeholder. The list below is the original planning list; `reports/report.md` is authoritative.

**Core methods**
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1).
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS*.

**Spatial validation**
- Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8).
- Ploton, P., et al. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. *Nature Communications*, 11.
- Valavi, R., et al. (2019). blockCV: An R package for generating spatially or environmentally separated folds for k-fold cross-validation of species distribution models. *Methods in Ecology and Evolution*, 10(2).

**UHI and remote sensing**
- Oke, T. R. (1982). The energetic basis of the urban heat island. *Quarterly Journal of the Royal Meteorological Society*, 108(455).
- Zha, Y., Gao, J., & Ni, S. (2003). Use of normalized difference built-up index in automatically mapping urban areas from TM imagery. *International Journal of Remote Sensing*, 24(3).
- Xu, H. (2006). Modification of normalised difference water index (NDWI) to enhance open water features in remotely sensed imagery. *International Journal of Remote Sensing*, 27(14).
- Gorelick, N., et al. (2017). Google Earth Engine: Planetary-scale geospatial analysis for everyone. *Remote Sensing of Environment*, 202.
- Zanaga, D., et al. (2022). ESA WorldCover 10 m 2021 v200. *Zenodo*.
- Shastri, H., Barik, B., Ghosh, S., Venkataraman, C., & Sadavarte, P. (2017). Flip flop of day-night and summer-winter surface urban heat island intensity in India. *Scientific Reports*, 7, 40178. — documents the daytime surface urban **cool** island in Indian cities during the pre-monsoon season, which is the effect MirageAI observes (see LEARNING_LOG, 2026-10-08). Still subject to the T8.3 verification pass.

**To find (Pune / India context)**
- [ ] Pune UHI / LST studies
- [ ] Indian city UHI + ML studies
- [ ] Recent SHAP-for-UHI studies (any city)
