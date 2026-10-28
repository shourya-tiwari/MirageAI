# MirageAI: pre-submission checklist

The items below need the team. The pipeline cannot do them. Tick each one, then
rebuild (`python scripts/build_report.py` and
`python scripts/build_slides.py`; both re-run every number check) and
re-make the zip (`python scripts/make_submission.py`).

## 1. Team names and PRNs
- [x] **Report author block:** the four names are in the `author:` block of
  `reports/report.md`, with affiliation "Symbiosis Institute of Technology, Pune".
- [x] **Slides:** names on slide 1 and slide 19, and in the deck's file properties
  (`TEAM` in `scripts/build_slides.py`).
- [x] **README contributors, `LICENSE`, `CITATION.cff`, `pyproject.toml`:** names filled.
- [ ] **PRNs:** not added anywhere. If the college requires them, add them to the
  `author:` entries in `reports/report.md` (for example `"Shourya Tiwari (PRN …)"`),
  to `TEAM` in `scripts/build_slides.py`, and to the README contributors table.
- [x] **Presenter split:** the notes name the presenter at the start of every slide
  (Shourya 1–5, Shashwat 6–10, Rishik 11–14, Shubhankar 15–19). To change it, edit
  `TEAM` order or the `P1`–`P4` tags in `scripts/build_slides.py`.

## 2. References
- [x] Pune-specific `[TEAM]` placeholder removed.
- [x] Ke et al. (2017) and Pedregosa et al. (2011) cited from their proceedings /
  journal pages (no DOI is registered for either); Lundberg & Lee (2017) cited at
  NeurIPS 30 with its arXiv DOI.
- [x] Four references added for the literature review (Weng 2004, Meyer 2019,
  Peng 2012, Bowler 2010), each verified by DOI: `reports/reference_check.csv`
  has 24 VERIFIED and the 2 no-DOI entries above.

## 3. College template
- [ ] Is there a required SIT / TE7483 cover page, declaration or certificate
  page? The report follows the IEEE conference layout (`reports/ieee/`). Add any
  mandatory front-matter page to the `.docx` copy, or as a separate PDF.
- [ ] If the slides need a college template, apply it in
  `scripts/build_slides.py` (colours, logo) and re-run the build so the number
  checks still pass.

## 4. Unreferenced raw file
- [ ] `data/raw/ushna_pune_features.csv` (5.2 MB, dated 2026-10-08) is not read by
  any code (D-045). Decide whether to keep it as history or delete it. `data/` is
  not in the submission zip either way.

## 5. Rehearsal
- [ ] Full timed run-through of the 19-slide deck. The notes are ~13.9 min at
  140 wpm (`reports/slides_manifest.csv`); with the live dashboard demo, aim for
  12–15 min, and trim notes rather than rush.
- [ ] Viva drill: each member answers questions from `reports/viva_prep.md`
  outside their own section, and can explain every decision in `docs/DECISIONS.md`.
- [ ] Learn the "numbers to know" table at the top of `viva_prep.md`. It is checked
  against the tables, so it is safe to quote.

## 6. Final package
- [ ] After any edit, rebuild the report and slides. Both builds must end with
  "0 flagged".
- [ ] Re-make `MirageAI_Final_Submission.zip` (project root) and check that it opens
  and holds the report PDF/DOCX/HTML and the slides PPTX/PDF.
- [ ] When the repository is public, add its URL to `CITATION.cff`
  (`repository-code:`) and to the README.
