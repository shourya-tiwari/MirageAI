"""Leakage and balance tests for the validation design (T3.4)."""

from __future__ import annotations

import numpy as np
import pytest

from mirageai import config, cv, preprocess


@pytest.fixture(scope="module")
def df(clean_table_ready):
    """The land-only modelling table (D-034)."""
    return preprocess.load_model_table()


@pytest.fixture(scope="module")
def full(clean_table_ready):
    """The clean table with water cells, i.e. the full 100 m grid."""
    return preprocess.load_clean()


LAYOUT_SEEDS = list(config.CV_SEEDS) + [None]


# ---------------------------------------------------------------------------
# Schema guards
# ---------------------------------------------------------------------------
def test_no_forbidden_column_is_a_feature():
    assert not set(config.FEATURES) & set(config.FORBIDDEN_FEATURES)


def test_coordinates_are_never_features():
    for col in config.COORD_COLUMNS:
        assert col not in config.FEATURES
        assert col not in config.FEATURES_LINEAR
        assert col not in config.FEATURES_TREE


def test_tree_models_keep_frac_built():
    """D-025: SHAP cannot attribute a concrete effect if the model never saw it."""
    assert "frac_built" in config.FEATURES_TREE
    assert config.FRACTION_DROP_FOR_LINEAR not in config.FEATURES_LINEAR
    assert len(config.FEATURES_LINEAR) == len(config.FEATURES_TREE) - 1


def test_block_columns_are_not_features():
    for col in config.BLOCK_COLUMNS:
        assert col not in config.FEATURES


# ---------------------------------------------------------------------------
# D-022: ids are year-independent
# ---------------------------------------------------------------------------
def test_each_cell_maps_to_one_block_across_years(df):
    preprocess.check_block_consistency(df)


def test_cell_id_is_unique_per_location(df):
    per_cell = df.groupby("cell_id").size()
    assert per_cell.min() == per_cell.max() == len(config.YEARS)


# ---------------------------------------------------------------------------
# Splitter leakage
# ---------------------------------------------------------------------------
def test_random_split_keeps_a_cell_whole(df):
    """Random CV must not put a cell's 2023 row in train and its 2024 row in test."""
    for train, test in cv.random_splits(df):
        assert not set(df.iloc[train]["cell_id"]) & set(df.iloc[test]["cell_id"])


@pytest.mark.parametrize("seed", LAYOUT_SEEDS)
@pytest.mark.parametrize("block_size", config.BLOCK_SIZES_M)
def test_spatial_split_has_no_shared_block_or_cell(df, block_size, seed):
    col = cv.block_column(block_size)
    for train, test in cv.spatial_splits(df, block_size, seed=seed):
        assert not set(df.iloc[train][col]) & set(df.iloc[test][col])
        assert not set(df.iloc[train]["cell_id"]) & set(df.iloc[test]["cell_id"])


@pytest.mark.parametrize("seed", LAYOUT_SEEDS)
@pytest.mark.parametrize("block_size", config.BLOCK_SIZES_M)
def test_spatio_temporal_split_has_no_shared_block_or_cell(df, block_size, seed):
    col = cv.block_column(block_size)
    for train, test in cv.spatio_temporal_splits(df, block_size, seed=seed):
        assert not set(df.iloc[train][col]) & set(df.iloc[test][col])
        assert not set(df.iloc[train]["cell_id"]) & set(df.iloc[test]["cell_id"])


@pytest.mark.parametrize("block_size", config.BLOCK_SIZES_M)
def test_spatio_temporal_split_respects_the_year_boundary(df, block_size):
    """Train sees only 2023-24; test sees only 2025 (D-010 as amended)."""
    for train, test in cv.spatio_temporal_splits(df, block_size):
        assert set(df.iloc[train]["year"]) == set(config.TRAIN_YEARS)
        assert set(df.iloc[test]["year"]) == {config.TEST_YEAR}


def test_every_split_is_non_empty(df):
    for name, fn in cv.SCHEMES.items():
        for train, test in fn(df):
            assert len(train) > 0, name
            assert len(test) > 0, name


def test_spatial_folds_partition_the_data(df):
    """Every row is tested exactly once across the folds."""
    seen = np.concatenate([test for _, test in cv.spatial_splits(df)])
    assert len(seen) == len(df)
    assert len(np.unique(seen)) == len(df)


# ---------------------------------------------------------------------------
# D-034: water never reaches a splitter
# ---------------------------------------------------------------------------
def test_model_table_has_no_water_majority_rows(df):
    assert not (df["frac_water"] >= config.WATER_EXCLUDE_THRESHOLD).any()
    assert "frac_water" in config.FEATURES and "dist_water_m" in config.FEATURES


def test_model_table_removes_exactly_the_water_rows(df, full):
    water = preprocess.is_water(full)
    assert len(df) == len(full) - int(water.sum())
    assert water.sum() > 0


@pytest.mark.parametrize(
    "make",
    [
        lambda t: cv.random_splits(t),
        lambda t: cv.spatial_splits(t, config.PRIMARY_BLOCK_SIZE_M),
        lambda t: cv.spatial_splits(t, config.PRIMARY_BLOCK_SIZE_M, seed=None),
        lambda t: cv.spatio_temporal_splits(t, config.PRIMARY_BLOCK_SIZE_M),
        lambda t: cv.strip_splits(t, t),
        lambda t: cv.block_folds(t, config.PRIMARY_BLOCK_SIZE_M),
    ],
    ids=["random", "spatial", "spatial_unshuffled", "spatio_temporal", "strips", "block_folds"],
)
def test_every_splitter_refuses_water_rows(full, make):
    with pytest.raises(ValueError, match="water-majority"):
        make(full)


def test_tuning_refuses_water_rows(full):
    from mirageai import models

    with pytest.raises(ValueError, match="water-majority"):
        models.tune(full, "lgbm", n_iter=1, subsample=1000)


# ---------------------------------------------------------------------------
# D-033: fold layouts
# ---------------------------------------------------------------------------
def _fold_of_row(splits, n):
    fold = np.full(n, -1)
    for k, (_, test) in enumerate(splits):
        fold[test] = k
    return fold


def test_seeded_layouts_are_deterministic_and_distinct(df):
    a = _fold_of_row(cv.spatial_splits(df, seed=42), len(df))
    b = _fold_of_row(cv.spatial_splits(df, seed=42), len(df))
    c = _fold_of_row(cv.spatial_splits(df, seed=43), len(df))
    assert (a == b).all()
    assert (a != c).any()


def _modal_row_share(df, splits):
    row = np.floor(df["y_utm"].to_numpy() / config.PRIMARY_BLOCK_SIZE_M)
    return [float(np.unique(row[te], return_counts=True)[1].max() / len(te)) for _, te in splits]


def test_strip_layout_is_latitudinal_and_shuffled_layout_is_not(df, full):
    """The stress test must actually be strips, and the headline must not be (D-033)."""
    strips = cv.strip_splits(df, full[[cv.block_column(config.PRIMARY_BLOCK_SIZE_M)]])
    assert min(_modal_row_share(df, strips)) > 0.7
    for seed in config.CV_SEEDS:
        assert max(_modal_row_share(df, cv.spatial_splits(df, seed=seed))) < 0.6


def test_strip_layout_has_no_shared_block_and_respects_years(df, full):
    layout = full[[cv.block_column(config.PRIMARY_BLOCK_SIZE_M)]]
    col = cv.block_column(config.PRIMARY_BLOCK_SIZE_M)
    for temporal in (False, True):
        for train, test in cv.strip_splits(df, layout, temporal=temporal):
            assert not set(df.iloc[train][col]) & set(df.iloc[test][col])
            if temporal:
                assert set(df.iloc[test]["year"]) == {config.TEST_YEAR}
                assert set(df.iloc[train]["year"]) == set(config.TRAIN_YEARS)


def test_block_folds_matches_spatial_splits(df):
    mapping = cv.block_folds(df, config.PRIMARY_BLOCK_SIZE_M)
    col = cv.block_column(config.PRIMARY_BLOCK_SIZE_M)
    fold = _fold_of_row(cv.spatial_splits(df), len(df))
    assert (df[col].map(mapping).to_numpy() == fold).all()


# ---------------------------------------------------------------------------
# Fold balance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scheme", sorted(cv.SCHEMES))
def test_every_fold_contains_both_urban_and_rural_cells(df, scheme):
    """Hard floor: an all-rural fold makes the urban-subset metrics of T5.7 undefined."""
    splits = cv.SCHEMES[scheme](df)
    report = cv.fold_report(df, splits, scheme)
    assert report["urban_share"].min() > 0.05, report.to_string()
    assert report["urban_share"].max() < 0.95, report.to_string()


@pytest.mark.parametrize("scheme", sorted(cv.SCHEMES))
def test_fold_urban_balance_is_reported(df, scheme, capsys):
    """Report and flag fold imbalance rather than failing on it.

    At the primary 8 km block size the city is a contiguous patch only two to
    three blocks wide, so urban cells cannot spread evenly across 20 blocks and
    some fold is always city-heavy. Stratifying to fix it would break the spatial
    contiguity that makes block CV a valid test, so the imbalance is reported and
    carried into the results as fold-level variance instead of being engineered
    away (D-026).
    """
    splits = cv.SCHEMES[scheme](df)
    report = cv.fold_report(df, splits, scheme)
    flagged = report.loc[report["urban_dev_pts"].abs() > cv.URBAN_BALANCE_TOLERANCE_PTS]
    with capsys.disabled():
        if len(flagged):
            print(f"\n  [flag] {scheme}: {len(flagged)} fold(s) exceed "
                  f"{cv.URBAN_BALANCE_TOLERANCE_PTS:.0f}-point urban-share tolerance")
            for _, r in flagged.iterrows():
                print(f"         fold {int(r.fold)}: urban {r.urban_share*100:.1f}% "
                      f"({r.urban_dev_pts:+.1f} pts)")
    assert len(report) == config.N_SPLITS
