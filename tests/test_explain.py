"""Guards for Phase 6: the explained sample, exact SHAP additivity, and the threshold finder."""

from __future__ import annotations

import numpy as np
import pytest

from mirageai import config, explain as X, models as M, preprocess


@pytest.fixture(scope="module")
def df(clean_table_ready):
    return preprocess.load_model_table()


def test_sample_is_exact_stratified_and_deterministic(df):
    idx = X.stratified_sample(df, X.N_EXPLAIN)
    assert len(idx) == X.N_EXPLAIN and idx.is_unique
    assert idx.equals(X.stratified_sample(df, X.N_EXPLAIN))
    pop = df.groupby(list(X.STRATA)).size() / len(df)
    got = df.loc[idx].groupby(list(X.STRATA)).size() / len(idx)
    assert np.abs(got - pop).max() < 1 / len(idx) + 1e-12


def test_background_is_disjoint_from_explained_rows(df):
    idx = X.stratified_sample(df, X.N_EXPLAIN)
    bg = X.stratified_sample(df, X.N_BACKGROUND, seed=config.SEED + 1, exclude=idx)
    assert len(bg) == X.N_BACKGROUND
    assert not set(bg) & set(idx)


def test_float32_rounding_makes_the_cast_lossless(df):
    f = X.float32_exact(df[list(config.FEATURES_TREE)].head(5000))
    assert (f.to_numpy().astype(np.float32).astype(np.float64) == f.to_numpy()).all()
    assert X.float32_exact(f).equals(f)


def test_interventional_shap_sums_to_the_prediction(df):
    feats = M.features_for("lgbm")
    sub = df.sample(6000, random_state=config.SEED)
    model = M.make_model("lgbm", n_estimators=50).fit(X.float32_exact(sub[feats]), sub[config.TARGET_ANOM])
    Xs = X.float32_exact(sub[feats].iloc[:200])
    vals, base = X.interventional_shap(model, Xs, X.float32_exact(sub[feats].iloc[200:400]))
    assert np.abs(vals.sum(axis=1) + base - model.predict(Xs)).max() < 1e-4


def test_parallel_shap_equals_serial(df):
    feats = M.features_for("lgbm")
    sub = df.sample(3000, random_state=config.SEED)
    model = M.make_model("lgbm", n_estimators=30).fit(X.float32_exact(sub[feats]), sub[config.TARGET_ANOM])
    Xs, bg = X.float32_exact(sub[feats].iloc[:300]), X.float32_exact(sub[feats].iloc[300:350])
    serial, b1 = X.interventional_shap(model, Xs, bg)
    par, b2 = X.interventional_shap(model, Xs, bg, n_jobs=2, chunk=100)
    assert b1 == pytest.approx(b2)
    assert np.allclose(serial, par, atol=1e-9)


def test_breakpoint_recovers_a_known_saturation():
    rng = np.random.default_rng(config.SEED)
    x = rng.uniform(0, 1, 4000)
    y = -4 * np.minimum(x, 0.3) + rng.normal(0, 0.05, x.size)
    bp = X.breakpoint(x, y, n_boot=50)
    assert bp["knot_lo"] <= 0.3 <= bp["knot_hi"]
    assert abs(bp["knot"] - 0.3) < 0.03
    assert bp["shape"] == "saturates"


def test_straight_line_is_not_given_a_threshold():
    rng = np.random.default_rng(config.SEED)
    x = rng.uniform(0, 1, 4000)
    assert X.breakpoint(x, 2 * x + rng.normal(0, 0.05, x.size), n_boot=20)["shape"] == "near-linear"


def test_interaction_ranking_uses_the_symmetric_pair_total():
    feats = ["a", "b", "c"]
    inter = np.zeros((10, 3, 3))
    inter[:, 0, 2] = inter[:, 2, 0] = 0.5  # a x c: pair total 1.0
    inter[:, 0, 1] = inter[:, 1, 0] = 0.1  # a x b: pair total 0.2
    top = X.top_interactions(inter, feats, k=2)
    assert list(zip(top.feature_a, top.feature_b)) == [("a", "c"), ("a", "b")]
    assert top.mean_abs_interaction.iloc[0] == pytest.approx(1.0)


def test_driver_groups_partition_the_features_and_preserve_the_total():
    feats = list(config.FEATURES_TREE)
    vals = np.random.default_rng(config.SEED).normal(size=(50, len(feats)))
    g = X.group_values(vals, feats)
    assert g.shape == (50, len(X.DRIVER_GROUPS))
    assert np.allclose(g.sum(axis=1), vals.sum(axis=1))


def test_grouped_shap_is_the_member_sum():
    feats = ["NDVI", "frac_tree", "NDBI", "MNDWI"]
    vals = np.arange(8, dtype=float).reshape(2, 4)
    assert np.allclose(X.grouped(vals, feats), vals[:, [0, 2, 3]].sum(axis=1))
