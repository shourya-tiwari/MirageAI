"""Guards for the descriptive cool-island contrast (Phase 1 definition, land-only headline)."""

from __future__ import annotations

import pandas as pd
import pytest

from mirageai import config, preprocess


def test_core_ring_contrast_uses_the_configured_radii():
    """Cells inside CORE_RADIUS_M are core, beyond RING_RADIUS_M ring, the band between neither."""
    r_core, r_ring = config.CORE_RADIUS_M, config.RING_RADIUS_M
    dist = [0, r_core - 1, r_core + 1, r_ring - 1, r_ring + 1, 2 * r_ring]
    lst = [30.0, 32.0, 99.0, 99.0, 40.0, 44.0]
    df = pd.DataFrame({"year": 2024, "x_utm": dist, "y_utm": 0.0, config.TARGET: lst, "NDVI": 0.5})
    out = preprocess.core_ring_contrast(df, centre=(0.0, 0.0)).iloc[0]
    assert (out.n_core, out.n_ring) == (2, 2)
    assert out.core_LST == pytest.approx(31.0)
    assert out.ring_LST == pytest.approx(42.0)
    assert out.core_minus_ring == pytest.approx(-11.0)
