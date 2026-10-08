"""Phase 1: Earth Engine extraction of LST, spectral indices, land cover and terrain."""

from __future__ import annotations

from pathlib import Path

import ee

from mirageai import config


def init_ee() -> None:
    """Initialise Earth Engine with the configured Cloud project."""
    if not config.GEE_PROJECT:
        raise RuntimeError(
            "No Earth Engine project configured. Set GEE_PROJECT in "
            "src/mirageai/config.py or the MIRAGEAI_GEE_PROJECT environment variable."
        )
    ee.Initialize(project=config.GEE_PROJECT)


def aoi() -> ee.Geometry:
    """Study-area rectangle in WGS84."""
    return ee.Geometry.Rectangle(
        list(config.AOI_BOUNDS), proj="EPSG:4326", geodesic=False
    )


# ---------------------------------------------------------------------------
# T1.6 Grid aggregation
# ---------------------------------------------------------------------------
def to_grid(img: ee.Image, native_scale: int) -> ee.Image:
    """Aggregate an image onto the 100 m grid by area-weighted mean.

    Composites carry a default 1-degree projection, so the native scale must be
    imposed before reduceResolution or the aggregation is meaningless.
    """
    return (
        img.reproject(crs=config.CRS, scale=native_scale)
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
        .reproject(crs=config.CRS, scale=config.SCALE)
    )


# ---------------------------------------------------------------------------
# T1.1 Landsat 8/9 land surface temperature (target)
# ---------------------------------------------------------------------------
def _mask_landsat(img: ee.Image) -> ee.Image:
    """Cloud-mask one Landsat C2 L2 scene and convert ST_B10 to degrees Celsius."""
    qa = img.select("QA_PIXEL")
    clear = ee.Image.constant(1)
    for bit in config.QA_PIXEL_BITS.values():
        clear = clear.And(qa.bitwiseAnd(1 << bit).eq(0))

    raw = img.select("ST_B10")
    return (
        raw.updateMask(clear)
        .updateMask(raw.gt(0))  # drop fill values before scaling
        .multiply(config.ST_B10_MULT)
        .add(config.ST_B10_ADD)
        .add(config.KELVIN_TO_CELSIUS)
        .rename("LST")
    )


def landsat_collection(year: int) -> ee.ImageCollection:
    """Merged, pre-filtered Landsat 8 + 9 collection for the season."""
    start, end = config.season_dates(year)
    parts = [
        ee.ImageCollection(cid)
        .filterBounds(aoi())
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUD_COVER", config.MAX_CLOUD_COVER))
        for cid in config.LANDSAT_COLLECTIONS
    ]
    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part)
    return merged


def landsat_lst(year: int) -> ee.Image:
    """Median seasonal LST in degrees Celsius plus the cloud-free observation count."""
    masked = landsat_collection(year).map(_mask_landsat)
    return masked.median().rename("LST").addBands(masked.count().rename("LST_n_obs"))


# ---------------------------------------------------------------------------
# T1.2 Sentinel-2 spectral indices
# ---------------------------------------------------------------------------
def _mask_s2(img: ee.Image) -> ee.Image:
    """Mask cloud, shadow, cirrus and snow in one Sentinel-2 SR scene via SCL."""
    scl = img.select("SCL")
    clear = ee.Image.constant(1)
    for cls in config.SCL_MASK_CLASSES:
        clear = clear.And(scl.neq(cls))
    return img.updateMask(clear)


def s2_collection(year: int) -> ee.ImageCollection:
    """Pre-filtered Sentinel-2 SR collection for the season."""
    start, end = config.season_dates(year)
    return (
        ee.ImageCollection(config.S2_COLLECTION)
        .filterBounds(aoi())
        .filterDate(start, end)
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE", config.MAX_CLOUDY_PIXEL_PERCENTAGE
            )
        )
    )


def s2_indices(year: int) -> tuple[ee.Image, ee.Image]:
    """NDVI (10 m) and NDBI + MNDWI (20 m) from a cloud-masked median composite.

    Returned separately because they resolve at different native scales.
    """
    bands = ["B3", "B4", "B8", "B11"]
    med = (
        s2_collection(year)
        .map(_mask_s2)
        .select(bands)
        .median()
        .multiply(config.S2_REFLECTANCE_SCALE)
    )
    ndvi = med.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndbi = med.normalizedDifference(["B11", "B8"]).rename("NDBI")
    mndwi = med.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    return ndvi, ndbi.addBands(mndwi)


# ---------------------------------------------------------------------------
# T1.3 / T1.4 WorldCover fractions and distance to water
# ---------------------------------------------------------------------------
def worldcover() -> ee.Image:
    """The single ESA WorldCover v200 map; the collection holds one image."""
    return ee.ImageCollection(config.WORLDCOVER_COLLECTION).first().select("Map")


def worldcover_binary() -> ee.Image:
    """One 10 m binary band per modelled land-cover class."""
    wc = worldcover()
    return ee.Image.cat(
        [wc.eq(code).rename(name) for code, name in config.WORLDCOVER_CLASSES.items()]
    )


def distance_to_water() -> ee.Image:
    """Distance to the nearest permanent-water pixel, in metres.

    fastDistanceTransform returns squared distance in pixels, so the water mask
    is pinned to a fixed 30 m projection first and the result converted.
    """
    water = worldcover().eq(config.WATER_CLASS).unmask(0)
    pinned = water.reproject(crs=config.CRS, scale=config.DIST_TRANSFORM_SCALE)
    squared_px = pinned.fastDistanceTransform(config.DIST_TRANSFORM_NEIGHBOURHOOD)
    return (
        squared_px.sqrt().multiply(config.DIST_TRANSFORM_SCALE).rename("dist_water_m")
    )


# ---------------------------------------------------------------------------
# T1.5 Terrain
# ---------------------------------------------------------------------------
def terrain() -> ee.Image:
    """SRTM elevation (m) and slope (degrees), computed in the metric project CRS."""
    srtm = ee.Image(config.SRTM_IMAGE).select("elevation")
    pinned = srtm.reproject(crs=config.CRS, scale=config.NATIVE_SCALE["srtm"])
    return pinned.rename("elevation").addBands(
        ee.Terrain.slope(pinned).rename("slope")
    )


# ---------------------------------------------------------------------------
# T1.7 Coordinate bands
# ---------------------------------------------------------------------------
def coordinate_bands() -> ee.Image:
    """Cell-centre lon/lat and UTM x/y on the 100 m grid. Never model features."""
    lonlat = ee.Image.pixelLonLat().rename(["lon", "lat"])
    xy = ee.Image.pixelCoordinates(ee.Projection(config.CRS)).rename(
        ["x_utm", "y_utm"]
    )
    return lonlat.addBands(xy).reproject(crs=config.CRS, scale=config.SCALE)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build_stack(year: int) -> ee.Image:
    """The full 100 m feature stack for one year, in the documented band order."""
    ns = config.NATIVE_SCALE
    ndvi, ndbi_mndwi = s2_indices(year)
    return ee.Image.cat(
        [
            to_grid(landsat_lst(year), ns["landsat"]),
            to_grid(ndvi, ns["s2_10m"]),
            to_grid(ndbi_mndwi, ns["s2_20m"]),
            to_grid(worldcover_binary(), ns["worldcover"]),
            to_grid(distance_to_water(), config.DIST_TRANSFORM_SCALE),
            to_grid(terrain(), ns["srtm"]),
            coordinate_bands(),
        ]
    ).reproject(crs=config.CRS, scale=config.SCALE)


def stack_band_names() -> list[str]:
    """Band order produced by build_stack(), for reading the exported GeoTIFF."""
    return [
        "LST",
        "LST_n_obs",
        "NDVI",
        "NDBI",
        "MNDWI",
        *config.WORLDCOVER_CLASSES.values(),
        "dist_water_m",
        "elevation",
        "slope",
        *config.COORD_COLUMNS,
    ]


def scene_counts(year: int) -> dict[str, int]:
    """Number of scenes passing the pre-filters, per source collection."""
    counts: dict[str, int] = {}
    for cid in config.LANDSAT_COLLECTIONS:
        key = cid.split("/")[1].lower()
        counts[key] = (
            ee.ImageCollection(cid)
            .filterBounds(aoi())
            .filterDate(*config.season_dates(year))
            .filter(ee.Filter.lt("CLOUD_COVER", config.MAX_CLOUD_COVER))
            .size()
            .getInfo()
        )
    counts["landsat_total"] = sum(counts.values())
    counts["s2"] = s2_collection(year).size().getInfo()
    return counts


# ---------------------------------------------------------------------------
# T1.8 / T1.9 Export and download
# ---------------------------------------------------------------------------
def stack_download_url(year: int) -> str:
    """Synchronous GeoTIFF download URL for the year's stack."""
    return (
        build_stack(year)
        .unmask(config.NODATA)
        .getDownloadURL(
            {
                "region": aoi(),
                "scale": config.SCALE,
                "crs": config.CRS,
                "format": "GEO_TIFF",
                "filePerBand": False,
            }
        )
    )


def download_stack(year: int) -> Path:
    """Fetch the year's stack GeoTIFF into data/raw/ and return its path."""
    import urllib.request

    config.ensure_dirs()
    dest = config.raw_tif(year)
    with urllib.request.urlopen(stack_download_url(year)) as resp:
        payload = resp.read()
    dest.write_bytes(payload)
    return dest


def stack_to_frame(year: int):
    """Read the downloaded GeoTIFF into a DataFrame, one row per grid cell."""
    import numpy as np
    import pandas as pd
    import rasterio

    names = stack_band_names()
    with rasterio.open(config.raw_tif(year)) as src:
        if src.count != len(names):
            raise ValueError(f"expected {len(names)} bands, found {src.count}")
        arr = src.read().astype("float64")

    frame = pd.DataFrame({n: arr[i].ravel() for i, n in enumerate(names)})
    frame = frame.replace(config.NODATA, np.nan)
    frame.insert(0, "year", year)
    return frame


def write_year_table(year: int) -> tuple[Path, dict[str, int]]:
    """Drop cells with any missing band, write the year's CSV, return path and counts."""
    frame = stack_to_frame(year)
    n_grid = len(frame)
    clean = frame.dropna()
    ordered = ["year", *config.COORD_COLUMNS, config.TARGET, *config.QC_COLUMNS, *config.FEATURES]
    clean = clean[ordered]

    dest = config.raw_csv(year)
    clean.to_csv(dest, index=False)
    return dest, {
        "grid_cells": n_grid,
        "rows": len(clean),
        "dropped": n_grid - len(clean),
    }


def extract_year(year: int) -> dict:
    """Run the full Phase 1 pipeline for one year and return a summary."""
    tif = download_stack(year)
    csv, counts = write_year_table(year)
    return {
        "year": year,
        "scenes": scene_counts(year),
        "tif": tif,
        "csv": csv,
        **counts,
    }
