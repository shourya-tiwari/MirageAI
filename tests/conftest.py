"""Shared guards: tests that read generated data skip, with the reason, until it exists.

A clean clone has data/raw only. The pure tests run straight after install; the
data tests run once scripts/pipeline/run_all.py (or scripts/pipeline/run_phase2.py, scripts/pipeline/run_phase4.py) has built
their inputs.
"""

from __future__ import annotations

import pytest

from mirageai import config


@pytest.fixture(scope="session")
def clean_table_ready():
    if not config.CLEAN_TABLE.exists():
        pytest.skip("needs data/processed/uhi_pune_clean.parquet: run scripts/pipeline/run_phase2.py or scripts/pipeline/run_all.py first")


@pytest.fixture(scope="session")
def phase4_ready():
    if not config.table("phase4_fold_metrics").exists():
        pytest.skip("needs outputs/tables/phase4_fold_metrics.csv: run scripts/pipeline/run_phase4.py or scripts/pipeline/run_all.py first")
