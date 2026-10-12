"""Guards for the Phase 4 out-of-fold parts: no silent deletion, no lost rows on merge."""

from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq
import pytest

from mirageai import models as M


def _write_part(path, n, start=0):
    pd.DataFrame({"row_id": range(start, start + n), "y_pred": 0.5}).to_parquet(path, index=False)


def test_full_run_refuses_to_delete_saved_parts(tmp_path):
    parts = tmp_path / "_oof_parts"
    parts.mkdir()
    _write_part(parts / "a.parquet", 3)
    with pytest.raises(FileExistsError, match="--finish-from-parts"):
        M.prepare_parts_dir(parts)
    assert (parts / "a.parquet").exists()


def test_force_clears_saved_parts(tmp_path):
    parts = tmp_path / "_oof_parts"
    parts.mkdir()
    _write_part(parts / "a.parquet", 3)
    M.prepare_parts_dir(parts, force=True)
    assert parts.is_dir() and not any(parts.iterdir())


def test_fresh_or_empty_parts_dir_is_fine(tmp_path):
    parts = tmp_path / "_oof_parts"
    M.prepare_parts_dir(parts)
    M.prepare_parts_dir(parts)
    assert parts.is_dir()


def test_merge_keeps_every_row_in_order(tmp_path):
    paths = [tmp_path / f"p{i}.parquet" for i in range(3)]
    for i, p in enumerate(paths):
        _write_part(p, 10 + i, start=100 * i)
    out = tmp_path / "merged.parquet"
    info = M.merge_parts(paths, out)
    assert info == {"n_parts": 3, "rows_in_parts": 33, "rows_written": 33}
    merged = pq.read_table(out).to_pandas()
    expected = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    pd.testing.assert_frame_equal(merged, expected)
    assert not out.with_name(out.name + ".tmp").exists()
