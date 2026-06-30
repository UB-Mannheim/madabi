"""
Build a paper lookup table from MADOC metadata (no PDF parsing).

Inputs:
- data/madoc.csv                          (harvested MADOC metadata)
- data/pdf_urls.txt                       (URLs that were used for downloading)
- data/missing_pdf_url_to_filename.csv    (URL->local filename overrides for long names)
- data/pdf/                               (downloaded PDFs)

Output:
- data/from_papers/paper_lookup_madoc.csv
  Columns:
    filename, url, madoc_identifier, title, creators, year, citation

Notes:
- This constructs a lightweight citation string (not strict Chicago).
- Filenames are the actual local PDF filenames (including hashed-shortened ones).
"""

from __future__ import annotations

import ast
import csv
import re
from pathlib import Path
from urllib.parse import urlsplit


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

MADOC = DATA_DIR / "madoc.csv"
PDF_URLS = DATA_DIR / "pdf_urls.txt"
MISSING_MAP = DATA_DIR / "missing_pdf_url_to_filename.csv"
PDF_DIR = DATA_DIR / "pdf"

OUT = DATA_DIR / "from_papers" / "paper_lookup_madoc.csv"


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def _extract_year(s: str) -> str:
    m = re.search(r"\b(18\d{2}|19\d{2}|20\d{2})\b", (s or "").strip())
    return m.group(1) if m else ""


def _parse_title(v: str) -> str:
    s = (v or "").strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            d = ast.literal_eval(s)
            if isinstance(d, dict):
                t = str(d.get("title") or "").strip()
                if t and t.lower() != "none":
                    return t
        except Exception:
            pass
    return s


def _parse_creators(v: str) -> str:
    s = (v or "").strip()
    if not s:
        return ""
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = ast.literal_eval(s)
            if isinstance(arr, list):
                names: list[str] = []
                for item in arr:
                    if isinstance(item, dict):
                        nm = str(item.get("creator_name") or item.get("creatorName") or "").strip()
                        if nm:
                            names.append(nm)
                return "; ".join(names)
        except Exception:
            pass
    return s


def _format_citation(creators: str, year: str, title: str, identifier: str) -> str:
    parts: list[str] = []
    if creators:
        parts.append(creators)
    if year:
        parts.append(f"({year})")
    if title:
        parts.append(f"“{title}.”")
    if identifier:
        parts.append(identifier)
    return " ".join(parts).strip()


def main() -> int:
    if not MADOC.exists():
        raise SystemExit(f"Missing {MADOC}")
    if not PDF_URLS.exists():
        raise SystemExit(f"Missing {PDF_URLS}")
    if not PDF_DIR.exists():
        raise SystemExit(f"Missing {PDF_DIR}")

    urls = [l.strip() for l in PDF_URLS.read_text(encoding="utf-8").splitlines() if l.strip()]

    url_to_local: dict[str, str] = {}
    if MISSING_MAP.exists():
        for row in _read_csv_rows(MISSING_MAP):
            u = (row.get("url") or "").strip()
            fn = (row.get("filename") or "").strip()
            if u and fn:
                url_to_local[u] = fn

    # Build local filename -> URL (based on basename, overridden by missing-map when present)
    filename_to_url: dict[str, str] = {}
    for u in urls:
        local = url_to_local.get(u)
        if not local:
            local = Path(urlsplit(u).path).name
        if local:
            filename_to_url[local] = u

    # Build URL -> MADOC row by matching against the `file` field (which can be a list-like string)
    madoc_rows = _read_csv_rows(MADOC)
    url_to_row: dict[str, dict[str, str]] = {}
    for row in madoc_rows:
        fv = (row.get("file") or "").strip()
        if not fv:
            continue
        candidates: list[str] = []
        if fv.startswith("[") and fv.endswith("]"):
            try:
                parsed = ast.literal_eval(fv)
                if isinstance(parsed, list):
                    candidates = [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                candidates = [fv]
        else:
            candidates = [fv]
        for u in candidates:
            if u and u not in url_to_row:
                url_to_row[u] = row

    # Only include rows that exist locally
    local_pdfs = {p.name for p in PDF_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"}

    out_rows: list[dict[str, str]] = []
    for fn in sorted(local_pdfs):
        u = filename_to_url.get(fn, "")
        row = url_to_row.get(u, {}) if u else {}
        title = _parse_title(row.get("titles") or row.get("citationTitle") or "")
        creators = _parse_creators(row.get("creators") or "")
        year = _extract_year(row.get("date") or "")
        identifier = (row.get("identifier") or "").strip()
        citation = _format_citation(creators=creators, year=year, title=title, identifier=identifier)
        out_rows.append(
            {
                "filename": fn,
                "url": u,
                "madoc_identifier": identifier,
                "title": title,
                "creators": creators,
                "year": year,
                "citation": citation,
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["filename", "url", "madoc_identifier", "title", "creators", "year", "citation"],
        )
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

