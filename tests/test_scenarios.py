"""Guards for Phase 7: target selection, fraction bookkeeping, index deltas, support check."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mirageai import config, preprocess, scenarios as S


@pytest.fixture(scope="module")
def df(clean_table_ready):
    return preprocess.load_model_table()


def test_targets_are_the_hottest_urban_decile_on_observed_means(df):
    t = S.target_cells(df)
    assert t.is_urban.all() and (t.n_years == len(config.YEARS)).all()
    n_urban = t.attrs["n_urban_cells"]
    assert len(t) == pytest.approx(n_urban * (1 - config.HOT_DECILE), abs=2)
    urban_mean = df[df.is_urban].groupby("cell_id")[config.TARGET_ANOM].mean()
    assert urban_mean.drop(t.index).max() < t.mean_anom.min()


def test_tree_increase_keeps_fractions_valid_and_proportional():
    fr = pd.DataFrame({f: [0.0, 0.0] for f in config.FRACTION_FEATURES})
    fr.loc[0, ["frac_built", "frac_bare", "frac_grass", "frac_tree"]] = [0.6, 0.1, 0.1, 0.2]
    fr.loc[1, ["frac_built", "frac_crop", "frac_tree"]] = [0.05, 0.85, 0.10]
    new, short = S.add_tree_cover(fr, 0.2)
    assert np.allclose(new.sum(axis=1), fr.sum(axis=1))
    # row 0: donors 0.8 give 0.2 in proportion 6:1:1
    assert new.loc[0, "frac_tree"] == pytest.approx(0.4)
    assert new.loc[0, "frac_built"] == pytest.approx(0.6 - 0.15)
    assert new.loc[0, "frac_bare"] == pytest.approx(0.1 - 0.025)
    assert short[0] == 0
    # row 1: only 0.05 of donors; crop untouched; the rest is a shortfall
    assert new.loc[1, "frac_tree"] == pytest.approx(0.15)
    assert new.loc[1, "frac_crop"] == pytest.approx(0.85)
    assert short[1] == pytest.approx(0.15)
    assert (new.to_numpy() >= 0).all()


def test_scenario_c_only_raises_cells_below_the_knot():
    d = S.scenario_delta("C", np.array([0.0, 0.05, 0.102, 0.3]), 0.102)
    assert np.allclose(d, [0.102, 0.052, 0.0, 0.0])


def test_unchanged_fractions_leave_observed_indices_exactly(df):
    part = df[df.year == config.TEST_YEAR].sample(3000, random_state=config.SEED)
    models = S.fit_index_models(part, {"n_estimators": 20})
    out = S.updated_indices(part, part, models)
    for idx in S.INDEX_TARGETS:
        assert np.array_equal(out[idx].to_numpy(), part[idx].to_numpy())


def test_support_guard_flags_about_one_percent_of_training_rows_and_far_points(df):
    feats = list(config.FEATURES_TREE)
    train = df.sample(20_000, random_state=config.SEED)
    g = S.SupportGuard(train, feats, ref_sample=5_000)
    assert np.mean(g.ref_scores > g.threshold) == pytest.approx(0.01, abs=0.004)
    far = train.head(5).copy()
    far[feats] = (train[feats].mean() + 20 * train[feats].std()).to_numpy()
    assert g.flag(far).all()
