"""Build MirageAI_Final_Submission.zip (project root): code, tests, docs, results, report and slides.

    python scripts/make_submission.py

Included: src/, tests/, scripts/ (with the pipeline drivers in scripts/pipeline/),
notebooks/, docs/, app/ and app.py (dashboard), .streamlit/, README.md, project_deep_dive.md,
learning_log.md, LICENSE, CITATION.cff, pyproject.toml, requirements.txt,
requirements-lock.txt, .gitignore, outputs/figures/,
outputs/tables/, and reports/ (sources, IEEE template, viva notes, findings, reference check, built
HTML/PDF/DOCX/PPTX).

Excluded: .venv/, data/ (see README, "Data"), outputs/models/, every root-level
file not listed in INCLUDE_FILES (root files are an allow-list), anything
gitignored (caches, egg-info), the team checklist, any zip, and the
out-of-fold prediction parquets (~0.8 GB, rebuilt by scripts/pipeline/run_phase4.py; every
metric derived from them is in the shipped CSVs).
"""

from __future__ import annotations

import fnmatch
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "MirageAI_Final_Submission.zip"
TOP = "MirageAI"

INCLUDE_DIRS = ["src", "tests", "scripts", "notebooks", "docs", "app", ".streamlit", "outputs/figures",
                "outputs/tables", "reports"]
INCLUDE_FILES = ["README.md", "project_deep_dive.md", "learning_log.md", "LICENSE", "CITATION.cff",
                 "pyproject.toml", "requirements.txt", "requirements-lock.txt", "app.py", ".gitignore"]
EXCLUDE = [
    "*/__pycache__/*", "*.pyc", "*.egg-info/*", "*/.pytest_cache/*", "*/.ipynb_checkpoints/*",
    "outputs/tables/oof_predictions*.parquet", "outputs/tables/_oof_parts*/*",
    "*.zip", "reports/SUBMISSION_CHECKLIST.md", "reports/~$*", "*/.gitkeep",
]
REQUIRED = ["reports/MirageAI_Report.pdf", "reports/MirageAI_Report.docx",
            "reports/MirageAI_Slides.pptx", "reports/MirageAI_Slides.pdf", "README.md", "LICENSE"]


def files() -> list[Path]:
    out = [ROOT / f for f in INCLUDE_FILES]
    for d in INCLUDE_DIRS:
        out += [p for p in (ROOT / d).rglob("*") if p.is_file()]
    keep = []
    for p in out:
        rel = p.relative_to(ROOT).as_posix()
        if not any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch("/" + rel, pat) for pat in EXCLUDE):
            keep.append(p)
    return sorted(set(keep))


def main() -> None:
    missing = [r for r in REQUIRED if not (ROOT / r).exists()]
    if missing:
        sys.exit(f"missing {missing}: build the report and slides first")
    chosen = files()
    banned = [p for p in chosen if p.relative_to(ROOT).parts[0] in (".venv", "data")
              or p.relative_to(ROOT).parts[:2] == ("outputs", "models")
              or (p.parent == ROOT and p.name not in INCLUDE_FILES)]
    assert not banned, banned
    OUT.unlink(missing_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in chosen:
            z.write(p, f"{TOP}/{p.relative_to(ROOT).as_posix()}")
    with zipfile.ZipFile(OUT) as z:
        assert z.testzip() is None
        infos = z.infolist()
    raw = sum(i.file_size for i in infos)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(infos)} files, {raw / 1e6:.1f} MB uncompressed, "
          f"{OUT.stat().st_size / 1e6:.1f} MB zipped")


if __name__ == "__main__":
    main()
