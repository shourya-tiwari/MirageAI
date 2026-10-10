"""Shared map and figure helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np

from mirageai import config


def raster_extent_km(transform, width: int, height: int) -> tuple[float, ...]:
    """Axis extent in kilometres (UTM easting/northing) for imshow."""
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    return (left / 1000, right / 1000, bottom / 1000, top / 1000)


def cells_to_grid(
    x_utm: np.ndarray, y_utm: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Rasterise cell-centre values onto the 100 m grid; returns (grid, extent in km)."""
    col = np.floor(x_utm / config.SCALE).astype("int64")
    row = np.floor(y_utm / config.SCALE).astype("int64")
    c0, r1 = col.min(), row.max()
    grid = np.full((r1 - row.min() + 1, col.max() - c0 + 1), np.nan)
    grid[r1 - row, col - c0] = values
    left, top = c0 * config.SCALE, (r1 + 1) * config.SCALE
    extent = (
        left / 1000,
        (left + grid.shape[1] * config.SCALE) / 1000,
        (top - grid.shape[0] * config.SCALE) / 1000,
        top / 1000,
    )
    return grid, extent


def map_panel(
    ax,
    grid: np.ndarray,
    extent: tuple[float, ...],
    *,
    cmap: str,
    cbar_label: str,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Draw one north-up raster map with a labelled colorbar and metric axes."""
    im = ax.imshow(
        grid,
        extent=extent,
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("UTM 43N easting (km)", fontsize=9)
    ax.set_ylabel("UTM 43N northing (km)", fontsize=9)
    ax.tick_params(labelsize=8, length=3, width=0.6)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("0.6")

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.040, pad=0.025)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8, length=2, width=0.6)
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor("0.6")


def save_figure(fig: mpl.figure.Figure, path: Path) -> Path:
    """Save a figure at the project's standard dpi and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight", facecolor="white")
    return path
