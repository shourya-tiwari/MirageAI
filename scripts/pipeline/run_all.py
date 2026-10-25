"""Single entry point: Phase 2 -> 7 from data/raw, then the report, slides and viva checks.

    python scripts/pipeline/run_all.py            # everything (~1 h on 16 GB, CPU only)
    python scripts/pipeline/run_all.py --from 6   # resume at a step number (see STEPS)

Input is data/raw/uhi_pune_{2023,2024,2025}.csv. Earth Engine extraction
(Phase 1, src/mirageai/gee_extract.py) is out of scope here: data/raw is the
input, and a Phase 1 re-run was already shown to reproduce the extraction
exactly. Phase 1's offline checks (cross-year table, LST/NDVI maps) are rebuilt
from data/raw by scripts/pipeline/run_phase2.py.

Every step runs as its own process, in order, and the run stops at the first
failure. The report and slide builds refuse to export if any number in the
report, deck or viva notes disagrees with the tables (check_report_numbers.py),
and the checks run once more at the end. Wall time and peak memory per step are
written to outputs/run_all_log.csv (not to outputs/tables, which holds results
only).

Needs pandoc on PATH, Chrome or Edge for the report PDF, and PowerPoint for the slides PDF.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[2]  # scripts/pipeline/ -> repository root
STEPS = [
    ("Phase 2: clean table, EDA, Phase 1 offline checks", "scripts/pipeline/run_phase2.py"),
    ("Phase 3: variogram, block layout, fold balance", "scripts/pipeline/run_phase3.py"),
    ("Phase 4: tuning and CV for every model and scheme", "scripts/pipeline/run_phase4.py"),
    ("Phase 5: evaluation and comparison", "scripts/pipeline/run_phase5.py"),
    ("Phase 6: SHAP explainability", "scripts/pipeline/run_phase6.py"),
    ("Phase 7: greening scenarios", "scripts/pipeline/run_phase7.py"),
    ("Report figures and tables", "scripts/make_report_figures.py"),
    ("Slide figures", "scripts/make_slide_figures.py"),
    ("Report: number checks, DOCX, PDF", "scripts/build_report.py"),
    ("Slides: number checks, PPTX, PDF", "scripts/build_slides.py"),
    ("Final number checks: report, slides, viva notes", "scripts/check_report_numbers.py"),
]


def peak_bytes(proc: psutil.Process) -> int:
    """Peak resident memory so far; Windows tracks it, elsewhere use the current RSS."""
    info = proc.memory_info()
    return getattr(info, "peak_wset", info.rss)


def run(script: str) -> tuple[int, float, int]:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    t0 = time.perf_counter()
    child = subprocess.Popen([sys.executable, str(ROOT / script)], cwd=ROOT, env=env)
    proc, peak = psutil.Process(child.pid), 0
    while child.poll() is None:
        try:
            peak = max(peak, peak_bytes(proc), *(peak_bytes(k) for k in proc.children(recursive=True)))
        except psutil.Error:
            pass
        time.sleep(0.5)
    return child.returncode, time.perf_counter() - t0, peak


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=1, help="first step number to run")
    args = ap.parse_args()

    missing = [y for y in (2023, 2024, 2025) if not (ROOT / "data" / "raw" / f"uhi_pune_{y}.csv").exists()]
    if missing:
        sys.exit(f"data/raw is incomplete (missing {missing}); see README, 'Data'")

    log, t_all = [], time.perf_counter()
    for n, (label, script) in enumerate(STEPS, 1):
        if n < args.start:
            continue
        print(f"\n=== [{n}/{len(STEPS)}] {label}  ({script})", flush=True)
        code, secs, peak = run(script)
        log.append({"step": n, "label": label, "script": script, "exit_code": code,
                    "seconds": round(secs, 1), "peak_mem_gb": round(peak / 1e9, 2)})
        pd.DataFrame(log).to_csv(ROOT / "outputs" / "run_all_log.csv", index=False)
        print(f"=== [{n}/{len(STEPS)}] exit {code} in {secs / 60:.1f} min, peak {peak / 1e9:.2f} GB", flush=True)
        if code:
            sys.exit(f"step {n} ({script}) failed; resume with --from {n}")
    total = time.perf_counter() - t_all
    peak = max(r["peak_mem_gb"] for r in log)
    print(f"\nall steps passed in {total / 60:.1f} min; peak memory {peak:.2f} GB (single step)")


if __name__ == "__main__":
    main()
