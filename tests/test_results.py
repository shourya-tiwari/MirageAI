"""Guards for the read-only results layer the dashboard uses: it must agree with the tables."""

from __future__ import annotations

import numpy as np
import pytest

from mirageai import config, evaluate as E, results as R


@pytest.fixture(scope="module")
def comparison():
    if not config.table("model_comparison").exists():
        pytest.skip("needs outputs/tables/model_comparison.csv: run scripts/pipeline/run_phase5.py or scripts/pipeline/run_all.py first")
    return R.load_table("model_comparison")


def test_missing_table_names_its_producer():
    with pytest.raises(FileNotFoundError, match="scripts/pipeline/run_phase7.py"):
        R.load_table("phase7_no_such_table")


def test_headline_reads_the_primary_and_spatio_temporal_rows(comparison):
    core_ring = R.load_table("phase8_core_ring_land")
    h = R.headline(comparison, core_ring)
    spatial = R.scheme_row(comparison, E.PRIMARY_SCHEME)
    assert h["spatial_rmse"] == spatial.RMSE_mean
    assert h["tree_rmse_min"] <= h["spatial_rmse"] <= h["tree_rmse_max"]
    assert h["random_rmse"] < h["spatial_rmse"], "random CV should look better than spatial CV"
    assert 0 < h["cool_island_min"] <= h["cool_island_max"]


def test_block_size_curve_covers_random_and_every_block_size(comparison):
    curve = R.block_size_curve(comparison)
    lgbm = curve[curve.model == "lgbm"]
    assert list(lgbm.block_km) == [0] + [s // 1000 for s in config.BLOCK_SIZES_M]


def test_scenario_bracket_has_one_row_per_scenario_and_bound():
    if not config.table("phase7_summary").exists():
        pytest.skip("needs outputs/tables/phase7_summary.csv: run scripts/pipeline/run_phase7.py or scripts/pipeline/run_all.py first")
    b = R.scenario_bracket(R.load_table("phase7_summary"))
    assert len(b) == 6 and set(b.scenario) == {"A", "B", "C"}
    assert (b.dLST_lo <= b.dLST).all() and (b.dLST <= b.dLST_hi).all()
    # Excluding the 0 -> 1% step can only shrink the cooling (D-040).
    assert (b.dLST > b.dLST_incl_first_point).all()


def test_per_cell_driver_groups_reproduce_the_phase6_map():
    path = config.TABLES / "phase6_shap_cells_lgbm.parquet"
    if not path.exists():
        pytest.skip("needs phase6_shap_cells_lgbm.parquet: run scripts/pipeline/run_phase6.py or scripts/pipeline/run_all.py first")
    cells = R.load_shap_cells()
    assert set(cells.year) == set(config.YEARS)
    counts = R.driver_group_values(R.cell_mean_shap(cells)).dominant.value_counts()
    stored = R.load_table("phase6_dominant_driver_groups").set_index("dominant").cells
    assert (counts.reindex(stored.index).to_numpy() == stored.to_numpy()).all()


def test_best_params_fallback_matches_the_saved_json(monkeypatch, tmp_path):
    saved = config.MODELS / "best_params.json"
    if not saved.exists() or not config.table("phase4_tuning_lgbm").exists():
        pytest.skip("needs best_params.json and the phase4_tuning tables: run scripts/pipeline/run_phase4.py first")
    from_json = R.load_best_params()
    monkeypatch.setattr(config, "MODELS", tmp_path)  # as in the submission: no models directory
    rebuilt = R.load_best_params()
    for name, info in rebuilt.items():
        assert info["best_params"] == from_json[name]["best_params"]
        assert info["best_cv_rmse"] == pytest.approx(from_json[name]["best_cv_rmse"])


def test_to_grid_places_cells_north_up():
    import pandas as pd

    cells = pd.DataFrame({"x_utm": [50.0, 150.0, 50.0], "y_utm": [150.0, 150.0, 50.0]})
    grid, extent = R.to_grid(cells, [1.0, 2.0, 3.0])
    assert grid.shape == (2, 2)
    assert grid[0, 0] == 1.0 and grid[0, 1] == 2.0 and grid[1, 0] == 3.0 and np.isnan(grid[1, 1])
    assert extent == (0.0, 0.2, 0.0, 0.2)
