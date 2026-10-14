"""Guards for the Phase 5 evaluation: neighbour weights, subgroup definitions, holdout hygiene."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mirageai import config, cv, evaluate, preprocess


@pytest.fixture(scope="module")
def df(clean_table_ready):
    """The land-only modelling table (D-034)."""
    return preprocess.load_model_table()


@pytest.fixture(scope="module")
def year_weights(df):
    """One year's land cells with their queen weights."""
    yr = df[df["year"] == config.TEST_YEAR].reset_index(drop=True)
    return yr, evaluate.queen_weights(yr["x_utm"].to_numpy(), yr["y_utm"].to_numpy())


def test_queen_neighbours_are_adjacent_cells(year_weights):
    """Moran's I is only meaningful if neighbours are real 100 m neighbours."""
    cells, w = year_weights
    xy = cells[["x_utm", "y_utm"]].to_numpy()
    rng = np.random.default_rng(config.SEED)
    for i in rng.integers(0, len(cells), 500):
        nbrs = w.neighbors[int(i)]
        assert len(nbrs) <= 8
        for j in nbrs:
            dx, dy = np.abs(xy[i] - xy[j])
            assert max(dx, dy) == config.SCALE, (i, j, dx, dy)


def test_interior_cells_have_eight_neighbours(year_weights):
    cells, w = year_weights
    counts = np.array([w.cardinalities[i] for i in range(len(cells))])
    assert np.median(counts) == 8


def test_moran_detects_structure_and_its_absence(year_weights):
    cells, w = year_weights
    smooth = evaluate.morans_i(cells["y_utm"].to_numpy(), w, permutations=0)
    noise = np.random.default_rng(config.SEED).normal(size=len(cells))
    rand = evaluate.morans_i(noise, w, permutations=0)
    assert smooth["morans_I"] > 0.9
    assert abs(rand["morans_I"]) < 0.02


def test_hot_deciles_are_a_tenth_of_their_population(df):
    hot = evaluate.hot_decile_flags(df)
    for year, part in hot.groupby(df["year"]):
        assert part["hot_aoi"].mean() == pytest.approx(1 - config.HOT_DECILE, abs=0.005)
        urban = df.loc[part.index, "is_urban"]
        assert not part.loc[~urban, "hot_urban"].any()
        assert part.loc[urban, "hot_urban"].mean() == pytest.approx(
            1 - config.HOT_DECILE, abs=0.005
        )


def test_spatio_temporal_test_rows_are_2025_only(df):
    for _, te in cv.spatio_temporal_splits(df, config.PRIMARY_BLOCK_SIZE_M):
        assert (df["year"].to_numpy()[te] == config.TEST_YEAR).all()


def test_comparison_covers_every_scheme_and_model(phase4_ready):
    fm = pd.read_csv(config.table("phase4_fold_metrics"))
    comp = evaluate.model_comparison(fm)
    assert len(comp) == len(evaluate.SCHEME_ORDER) * 5
    headline = comp[~comp.scheme.astype(str).str.endswith(evaluate.STRIP_SUFFIX)]
    assert (headline["n_seeds"] == len(config.CV_SEEDS)).all()
    assert (headline["n_runs"] == len(config.CV_SEEDS) * config.N_SPLITS).all()
    assert comp[["RMSE_mean", "MAE_mean", "R2_mean"]].notna().all().all()
    assert (comp.loc[comp.model == "linear", "n_features"] == 12).all()
    assert (comp.loc[comp.model == "lgbm", "n_features"] == 13).all()


def test_error_metrics_known_values():
    m = evaluate.error_metrics(np.array([0.0, 2.0]), np.array([1.0, 1.0]))
    assert m["RMSE"] == pytest.approx(1.0)
    assert m["MAE"] == pytest.approx(1.0)
    assert m["R2"] == pytest.approx(0.0)
    assert m["mean_resid"] == pytest.approx(0.0)
