"""Single source of truth for MirageAI: study area, grid, data sources, paths and seeds."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

# ----------------------------------------------------------------------------
# Earth Engine
# ----------------------------------------------------------------------------
# Noncommercial Earth Engine Cloud project ID. Set it here or override with the
# MIRAGEAI_GEE_PROJECT environment variable. Never put credentials in this file.
# The ID itself keeps its registered name; renaming the project (D-045) does not
# rename the Cloud project.
# Deprecated: USHNA_GEE_PROJECT, the variable's name before the rename, is still
# read as a fallback and warns; remove it once no environment sets it.
if "MIRAGEAI_GEE_PROJECT" not in os.environ and "USHNA_GEE_PROJECT" in os.environ:
    warnings.warn("USHNA_GEE_PROJECT is deprecated; set MIRAGEAI_GEE_PROJECT instead",
                  DeprecationWarning, stacklevel=2)
GEE_PROJECT: str = os.environ.get(
    "MIRAGEAI_GEE_PROJECT", os.environ.get("USHNA_GEE_PROJECT", "ushna-aucml")
)

# ----------------------------------------------------------------------------
# Study area and grid
# ----------------------------------------------------------------------------
# Pune Municipal Corporation + PCMC core, WGS84 bounding box.
AOI_BOUNDS: tuple[float, float, float, float] = (73.72, 18.42, 74.02, 18.65)

# Projected CRS for all processing: UTM zone 43N, units metres.
CRS: str = "EPSG:32643"

# Modelling grid, matched to the native resolution of the Landsat thermal sensor.
SCALE: int = 100

# Native resolutions of the source products, used before reduceResolution().
NATIVE_SCALE: dict[str, int] = {
    "landsat": 30,
    "s2_10m": 10,
    "s2_20m": 20,
    "worldcover": 10,
    "srtm": 30,
}

# ----------------------------------------------------------------------------
# Temporal extent
# ----------------------------------------------------------------------------
YEARS: tuple[int, ...] = (2023, 2024, 2025)
TRAIN_YEARS: tuple[int, ...] = (2023, 2024)
TEST_YEAR: int = 2025

# Pre-monsoon season, Mar-May. filterDate's end is exclusive, hence 06-01.
SEASON_START_MMDD: str = "03-01"
SEASON_END_MMDD: str = "06-01"


def season_dates(year: int) -> tuple[str, str]:
    """Return the (start, end) date strings of the pre-monsoon season for a year."""
    return f"{year}-{SEASON_START_MMDD}", f"{year}-{SEASON_END_MMDD}"


# ----------------------------------------------------------------------------
# Collections and masking
# ----------------------------------------------------------------------------
LANDSAT_COLLECTIONS: tuple[str, ...] = (
    "LANDSAT/LC08/C02/T1_L2",
    "LANDSAT/LC09/C02/T1_L2",
)
S2_COLLECTION: str = "COPERNICUS/S2_SR_HARMONIZED"
WORLDCOVER_COLLECTION: str = "ESA/WorldCover/v200"
SRTM_IMAGE: str = "USGS/SRTMGL1_003"

MAX_CLOUD_COVER: int = 30  # Landsat CLOUD_COVER, percent
MAX_CLOUDY_PIXEL_PERCENTAGE: int = 30  # Sentinel-2 CLOUDY_PIXEL_PERCENTAGE, percent

# Landsat Collection 2 Level-2 surface temperature scaling: K = DN * mult + add.
ST_B10_MULT: float = 0.00341802
ST_B10_ADD: float = 149.0
KELVIN_TO_CELSIUS: float = -273.15

# QA_PIXEL bits to mask out (a set bit means the pixel is bad).
QA_PIXEL_BITS: dict[str, int] = {
    "dilated_cloud": 1,
    "cirrus": 2,
    "cloud": 3,
    "cloud_shadow": 4,
}

# Sentinel-2 SCL classes to mask out.
SCL_MASK_CLASSES: tuple[int, ...] = (3, 8, 9, 10, 11)

# Sentinel-2 reflectance scaling: reflectance = DN / 10000.
S2_REFLECTANCE_SCALE: float = 1e-4

# ----------------------------------------------------------------------------
# WorldCover land-cover classes
# ----------------------------------------------------------------------------
# Only the classes that are non-negligible in Pune. Snow (70), wetland (90),
# mangroves (95) and moss (100) are dropped.
WORLDCOVER_CLASSES: dict[int, str] = {
    10: "frac_tree",
    20: "frac_shrub",
    30: "frac_grass",
    40: "frac_crop",
    50: "frac_built",
    60: "frac_bare",
    80: "frac_water",
}
WATER_CLASS: int = 80

# fastDistanceTransform returns squared distance in pixels. Compute it on a
# fixed 30 m grid, then sqrt and multiply by 30 to get metres. A 256 px
# neighbourhood caps distances at ~7.7 km.
DIST_TRANSFORM_SCALE: int = 30
DIST_TRANSFORM_NEIGHBOURHOOD: int = 256

# Sentinel value written for masked pixels in GeoTIFF exports. Outside the valid
# range of every band (LST, indices, fractions, distance, terrain).
NODATA: float = -9999.0


# ----------------------------------------------------------------------------
# Modelling schema
# ----------------------------------------------------------------------------
TARGET: str = "LST"

# Primary modelling target: within-year anomaly against that year's AOI median.
# No feature encodes the year or scene timing, so the absolute level is not
# learnable and would only add noise to the temporal holdout (D-021).
TARGET_ANOM: str = "LST_anom"

FRACTION_FEATURES: tuple[str, ...] = tuple(WORLDCOVER_CLASSES.values())
SPECTRAL_FEATURES: tuple[str, ...] = ("NDVI", "NDBI", "MNDWI")
TERRAIN_FEATURES: tuple[str, ...] = ("elevation", "slope", "dist_water_m")

FEATURES: tuple[str, ...] = SPECTRAL_FEATURES + FRACTION_FEATURES + TERRAIN_FEATURES

# Fractions sum to 1, so one is dropped for linear models to avoid singularity.
# frac_built is the reference category: it has the largest variance, so dropping
# it conditions the design matrix best (median VIF 2.2 vs 11.8 for frac_bare),
# and it makes every remaining coefficient read as "replace built-up with X",
# which is exactly the greening-scenario question (D-023).
FRACTION_DROP_FOR_LINEAR: str = "frac_built"

# The drop applies to Linear Regression ONLY. Tree models are indifferent to the
# sum-to-1 singularity and must keep every fraction, or SHAP cannot attribute any
# effect to built-up surface at all (D-025).
FEATURES_LINEAR: tuple[str, ...] = tuple(
    f for f in FEATURES if f != FRACTION_DROP_FOR_LINEAR
)
FEATURES_TREE: tuple[str, ...] = FEATURES

assert "frac_built" in FEATURES_TREE, "tree models must retain frac_built (D-025)"
assert FRACTION_DROP_FOR_LINEAR not in FEATURES_LINEAR
assert len(FEATURES_LINEAR) == len(FEATURES_TREE) - 1

# Coordinates are used for CV blocks and maps only, never as features.
COORD_COLUMNS: tuple[str, ...] = ("lon", "lat", "x_utm", "y_utm")

# Quality-control column: number of cloud-free Landsat observations. Never a feature.
QC_COLUMNS: tuple[str, ...] = ("LST_n_obs",)

# Cells with fewer than this many valid observations are dropped.
MIN_LST_OBS: int = 3

# Columns that must never reach a model, asserted in the pipeline.
# Block-ID columns are appended once BLOCK_SIZES_M is defined, further down.
FORBIDDEN_FEATURES: tuple[str, ...] = (
    COORD_COLUMNS
    + QC_COLUMNS
    + (TARGET, TARGET_ANOM, "LST_year_median", "cell_id", "is_urban")
)

# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
SEED: int = 42
N_SPLITS: int = 5
# Primary block size comes from the autocorrelation range of model RESIDUALS,
# not of raw LST_anom: the raw range (7.65 km) is dominated by the regional
# core-to-periphery gradient that the features already explain. The residual
# range is ~3.7 km centrally (3.95 km on the land-only re-check), so 4 km is the
# smallest rung at or above it.
# 8 km is retained as a stress-test scheme and every size is reported (D-028).
BLOCK_SIZES_M: tuple[int, ...] = (1000, 2000, 4000, 8000)
PRIMARY_BLOCK_SIZE_M: int = 4000
STRESS_BLOCK_SIZE_M: int = 8000

# Block IDs are grouping keys for CV, never model inputs.
BLOCK_COLUMNS: tuple[str, ...] = tuple(f"block_{size}m" for size in BLOCK_SIZES_M)
FORBIDDEN_FEATURES = FORBIDDEN_FEATURES + BLOCK_COLUMNS
HOT_DECILE: float = 0.9  # quantile that defines the "hottest decile" of cells

# A cell counts as urban when most of its 100 m footprint is built-up. Scenarios
# and sub-group reporting use this instead of the AOI-wide hottest decile (D-019).
URBAN_BUILT_THRESHOLD: float = 0.5

# Water-majority cells are excluded from training and evaluation: reservoir LST
# is governed by thermal inertia, not land-surface drivers, and water is not a
# planning lever. frac_water and dist_water_m stay as features (D-034).
WATER_EXCLUDE_THRESHOLD: float = 0.5

# Phase 1 cool-island contrast: the core is every cell within CORE_RADIUS_M of
# the centre of the grid's UTM extent, the outer ring every cell beyond
# RING_RADIUS_M. The headline uses land cells only (D-034).
CORE_RADIUS_M: int = 5000
RING_RADIUS_M: int = 12000

# Spatial and spatio-temporal CV deal blocks to folds at random, repeated over
# these seeds; results are mean +/- std over all seed x fold runs. Unshuffled
# GroupKFold is kept only as the latitude-strip stress test at 4 km (D-033).
CV_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parents[2]

DATA_RAW: Path = ROOT / "data" / "raw"
DATA_PROCESSED: Path = ROOT / "data" / "processed"
OUTPUTS: Path = ROOT / "outputs"
FIGURES: Path = OUTPUTS / "figures"
TABLES: Path = OUTPUTS / "tables"
MODELS: Path = OUTPUTS / "models"
REPORTS: Path = ROOT / "reports"

CLEAN_TABLE: Path = DATA_PROCESSED / "uhi_pune_clean.parquet"


def raw_csv(year: int) -> Path:
    """Path to the downloaded per-year GEE CSV export."""
    return DATA_RAW / f"uhi_pune_{year}.csv"


def raw_tif(year: int) -> Path:
    """Path to the downloaded per-year GEE GeoTIFF stack export."""
    return DATA_RAW / f"uhi_pune_stack_{year}.tif"


def figure(phase: int, name: str) -> Path:
    """Path for a figure, following the outputs/figures/<phase>_<name>.png convention."""
    return FIGURES / f"{phase:02d}_{name}.png"


def table(name: str) -> Path:
    """Path for a CSV table in outputs/tables/."""
    return TABLES / f"{name}.csv"


# ----------------------------------------------------------------------------
# Figure conventions
# ----------------------------------------------------------------------------
FIG_DPI: int = 300
LST_CMAP: str = "inferno"
DIVERGING_CMAP: str = "RdBu_r"


def ensure_dirs() -> None:
    """Create the data and output directories if they do not exist."""
    for path in (DATA_RAW, DATA_PROCESSED, FIGURES, TABLES, MODELS, REPORTS):
        path.mkdir(parents=True, exist_ok=True)
