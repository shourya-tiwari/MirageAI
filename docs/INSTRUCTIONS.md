# Instructions

Working rules for the team. **Read this before writing any code.**

## 1. Repo Structure

```
mirageai/
├── README.md                  project front page: results, install, usage, dashboard
├── project_deep_dive.md       full technical guide (what / why / how / when / where)
├── learning_log.md            study guide: concepts, viva and interview questions
├── app.py                     dashboard launcher: streamlit run app.py
├── requirements.txt           direct dependencies, pinned
├── requirements-lock.txt      the full resolved environment (exact rebuild)
├── pyproject.toml             package metadata; editable install, so `import mirageai` works anywhere
├── .streamlit/config.toml     dashboard theme
├── app/
│   └── streamlit_app.py       dashboard (read-only over outputs/)
├── docs/                      these files (index: docs/README.md)
├── src/mirageai/
│   ├── __init__.py
│   ├── config.py              single source of truth: AOI, CRS, SCALE, YEARS, features, paths, SEED
│   ├── gee_extract.py         Phase 1
│   ├── preprocess.py          Phase 2
│   ├── cv.py                  Phase 3
│   ├── models.py              Phase 4
│   ├── evaluate.py            Phase 5
│   ├── explain.py             Phase 6
│   ├── scenarios.py           Phase 7
│   ├── plotting.py            shared map/figure helpers
│   └── results.py             read-only loaders and summaries of outputs/ (dashboard, notebooks)
├── scripts/
│   ├── pipeline/
│   │   ├── run_all.py         single entry point (Phase 2 -> 7, report, slides, checks)
│   │   └── run_phase{2..7}.py one driver per phase
│   └── *.py                   report/slide figures and builds, number and reference checks, zip
├── notebooks/                 thin: call src/, show stored results
│   ├── 01_data_extraction.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_modelling.ipynb
│   └── 04_explain_scenarios.ipynb
├── data/
│   ├── raw/                   GEE exports (gitignored)
│   └── processed/             clean table (gitignored)
├── tests/                     leakage, schema and method guards (pytest)
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── models/                gitignored
└── reports/                   report, slides, viva notes and their sources
```

## 2. Environment

Python 3.11 or newer (`pyproject.toml` enforces it; the results were produced
on 3.11.15). From the repo root:

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS / Linux: source .venv/bin/activate)
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
earthengine authenticate          # only to re-extract data/raw/
```

Moving to a new machine: follow `docs/MIGRATION.md`, which also lists the checks
that prove the move changed nothing.

- **`requirements.txt` lists only direct dependencies** (D-055): the packages
  the code imports, plus JupyterLab and ipykernel for the notebooks, pinned to
  the versions the results were produced with. pip resolves the rest. To add or
  upgrade a package, change its pin there, reinstall, re-run the tests and
  regenerate the lock.
- **`requirements-lock.txt`** is `pip freeze` of the verified environment
  (without the editable package). Install from it instead of
  `requirements.txt` to rebuild that environment exactly.
- **`pip install -e .`** installs `mirageai` itself in editable mode. That is
  what makes `from mirageai import config` resolve from a notebook kernel or a
  script without `sys.path` patching (D-012), so run it from the repo root.
- The ML pipeline never imports Streamlit; the dashboard is just one more pinned
  dependency.
- `earthengine authenticate` opens a browser once. See §3 for registering the
  Cloud project and setting `GEE_PROJECT`.

## 3. Earth Engine Setup (one-time)

1. Create or select a Google Cloud project and register it for noncommercial Earth Engine use (code.earthengine.google.com → register).
2. Run `earthengine authenticate` locally.
3. Put the project ID in `src/mirageai/config.py` as `GEE_PROJECT`, or export it as
   `MIRAGEAI_GEE_PROJECT`. Never hardcode credentials.
4. Verify with `python -c "from mirageai.gee_extract import init_ee; init_ee()"`.
5. Extraction downloads straight into `data/raw/` via `gee_extract.extract_year(year)`;
   there is no Google Drive round-trip and no manual download step (see D-020).

## 4. ML Hygiene Rules (non-negotiable)

These protect the project's main methodological claim. Breaking any one of them invalidates the results.

1. **Coordinates are never features.** `lon`, `lat`, `x_utm` and `y_utm` are used only for CV blocks and maps.
2. **Spatial CV is the headline metric.** Random KFold is reported only to show inflation.
3. **All preprocessing fits inside the fold.** Use sklearn `Pipeline`, and never scale or impute on the full dataset.
4. **Tuning happens inside spatial CV.** Pass the group splitter to `RandomizedSearchCV`.
5. **The temporal test year (2025) stays untouched** until final evaluation. No tuning on it.
6. **`LST_n_obs` is QC only**, never a feature.
7. **Fixed seeds everywhere** (`config.SEED = 42`).
8. **No feature derived from the target.**

## 5. Code Conventions

- Business logic lives in `src/mirageai/`; notebooks and the dashboard only import and display.
- The dashboard never refits a model: it reads `outputs/` through `mirageai.results` (D-049).
- All constants come from `config.py`. No magic numbers in modules.
- Functions have type hints and a one-line docstring.
- Figures are saved to `outputs/figures/<phase>_<name>.png` at 300 dpi with labelled axes, units (°C, m) and a colorbar on every map.
- Tables are saved as CSV to `outputs/tables/`.
- Keep changes small and focused; one task ID per commit where possible.

## 6. Workflow

1. Pick the next unchecked task in `TASKS.md`.
2. Implement and verify it (run it, look at the output).
3. Tick the box in `TASKS.md`.
4. If a non-obvious choice was made, add an entry to `DECISIONS.md`.
5. If something was learned, add an entry to `LEARNING_LOG.md`.
6. When a phase completes, add a `CHANGELOG.md` entry and update the status in `ROADMAP.md`.

Commit format: `type(Tx.y): message`, where type is one of `feat`, `fix`, `data`, `docs`, `refactor` or `exp`.

## 7. Definition of Done (per task)

- Code runs end-to-end from a fresh kernel.
- Output has been checked visually or numerically, not just "it ran".
- Docs are updated (TASKS, plus DECISIONS or LEARNING_LOG if relevant).
