"""T8.3: verify every report reference against its DOI record (doi.org content negotiation).

    python scripts/verify_references.py

For each entry, the DOI's registered metadata (Crossref or DataCite) is fetched
and compared with what the report cites: first-author family name, year and
venue. Results go to reports/reference_check.csv. An entry with no DOI, or whose
record does not match, is reported as UNVERIFIED and must be marked so in the
report; nothing is filled in from memory.
"""

from __future__ import annotations

import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "reference_check.csv"

# key, cited first author, cited year, cited venue (substring to match), DOI or None
REFERENCES = [
    ("breiman2001", "Breiman", 2001, "Machine Learning", "10.1023/A:1010933404324"),
    ("chen2016", "Chen", 2016, "Knowledge Discovery and Data Mining", "10.1145/2939672.2939785"),
    ("ke2017", "Ke", 2017, "Neural Information Processing Systems", None),
    # Cited as the arXiv record; its NeurIPS 2017 publication has no DOI.
    ("lundberg2017", "Lundberg", 2017, "arXiv", "10.48550/arXiv.1705.07874"),
    ("lundberg2020", "Lundberg", 2020, "Nature Machine Intelligence", "10.1038/s42256-019-0138-9"),
    ("roberts2017", "Roberts", 2017, "Ecography", "10.1111/ecog.02881"),
    ("ploton2020", "Ploton", 2020, "Nature Communications", "10.1038/s41467-020-18321-y"),
    ("valavi2019", "Valavi", 2019, "Methods in Ecology and Evolution", "10.1111/2041-210X.13107"),
    ("oke1982", "Oke", 1982, "Quarterly Journal of the Royal Meteorological Society", "10.1002/qj.49710845502"),
    ("voogt2003", "Voogt", 2003, "Remote Sensing of Environment", "10.1016/S0034-4257(03)00079-8"),
    ("shastri2017", "Shastri", 2017, "Scientific Reports", "10.1038/srep40178"),
    ("siddiqui2021", "Siddiqui", 2021, "Sustainable Cities and Society", "10.1016/j.scs.2021.103374"),
    # EGU General Assembly abstract (not peer reviewed); the record carries the publisher only.
    ("mohammad2020", "Mohammad", 2020, "Copernicus", "10.5194/egusphere-egu2020-6444"),
    ("zha2003", "Zha", 2003, "International Journal of Remote Sensing", "10.1080/01431160304987"),
    ("xu2006", "Xu", 2006, "International Journal of Remote Sensing", "10.1080/01431160600589179"),
    ("gorelick2017", "Gorelick", 2017, "Remote Sensing of Environment", "10.1016/j.rse.2017.06.031"),
    ("zanaga2022", "Zanaga", 2022, "Zenodo", "10.5281/zenodo.7254221"),
    ("drusch2012", "Drusch", 2012, "Remote Sensing of Environment", "10.1016/j.rse.2011.11.026"),
    ("usgs_c2l2", "Earth Resources Observation", 2013, "Geological Survey", "10.5066/P9OGBGM6"),
    ("farr2007", "Farr", 2007, "Reviews of Geophysics", "10.1029/2005RG000183"),
    ("moran1950", "Moran", 1950, "Biometrika", "10.1093/biomet/37.1-2.17"),
    ("pedregosa2011", "Pedregosa", 2011, "Journal of Machine Learning Research", None),
    ("weng2004", "Weng", 2004, "Remote Sensing of Environment", "10.1016/j.rse.2003.11.005"),
    ("meyer2019", "Meyer", 2019, "Ecological Modelling", "10.1016/j.ecolmodel.2019.108815"),
    ("peng2012", "Peng", 2012, "Environmental Science", "10.1021/es2030438"),
    ("bowler2010", "Bowler", 2010, "Landscape and Urban Planning", "10.1016/j.landurbplan.2010.05.006"),
]


def fetch(doi: str) -> dict:
    req = urllib.request.Request(
        f"https://doi.org/{doi}",
        headers={"Accept": "application/vnd.citationstyles.csl+json",
                 "User-Agent": "mirageai-reference-check/1.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def years_of(m: dict) -> list[int]:
    """Every year the record carries (issued, print, online): a journal article
    is conventionally cited by its print-issue year, which can differ from the
    online-first year that `issued` reports."""
    out = []
    for k in ("issued", "published-print", "published-online"):
        parts = (m.get(k) or {}).get("date-parts") or [[None]]
        if parts and parts[0] and parts[0][0]:
            out.append(int(parts[0][0]))
    return out


def venue_of(m: dict) -> str:
    v = m.get("container-title") or m.get("publisher") or ""
    return v[0] if isinstance(v, list) and v else str(v)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # record titles carry non-ASCII dashes
    rows = []
    for key, author, year, venue, doi in REFERENCES:
        row = {"key": key, "doi": doi or "", "cited_author": author, "cited_year": year,
               "cited_venue": venue}
        if not doi:
            rows.append({**row, "status": "UNVERIFIED", "note": "no DOI registered for this venue"})
            continue
        try:
            m = fetch(doi)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
            rows.append({**row, "status": "UNVERIFIED", "note": f"lookup failed: {err}"})
            continue
        authors = m.get("author") or []
        first = (authors[0].get("family") or authors[0].get("literal") or "") if authors else ""
        years, rec_venue = years_of(m), venue_of(m)
        rec_year = "/".join(str(y) for y in sorted(set(years)))
        if not rec_venue or venue.lower() not in rec_venue.lower():
            rec_venue = rec_venue or str(m.get("publisher", ""))
        checks = {
            "author": author.lower() in first.lower(),
            "year": year in years,
            "venue": venue.lower() in rec_venue.lower() or venue.lower() in str(m.get("publisher", "")).lower(),
        }
        title = m.get("title") or ""
        title = title[0] if isinstance(title, list) else title
        status = "VERIFIED" if all(checks.values()) else (
            "PARTIAL" if checks["author"] and checks["year"] else "MISMATCH")
        rows.append({**row, "status": status,
                     "record_first_author": first, "record_year": rec_year,
                     "record_venue": rec_venue, "record_title": title,
                     "record_authors": "; ".join(
                         f"{a.get('family', a.get('literal', ''))}, {a.get('given', '')}".strip(", ")
                         for a in authors),
                     "record_volume": m.get("volume", ""), "record_issue": m.get("issue", ""),
                     "record_pages": m.get("page", ""),
                     "note": ", ".join(k for k, ok in checks.items() if not ok)})
        time.sleep(0.5)
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "key", k))
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['status']:<10} {r['key']:<14} {r.get('record_title', '')[:70]}"
              f"  [{r.get('record_venue', '')} {r.get('record_year', '')}] {r.get('note', '')}")
    counts = {s: sum(r["status"] == s for r in rows) for s in ("VERIFIED", "PARTIAL", "MISMATCH", "UNVERIFIED")}
    print(counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
