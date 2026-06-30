"""
Build the unified librarian CSV for BOTH OSF and Figshare in one pass.

There is one row-builder per source (`build_osf_rows`, `build_figshare_rows`) since each
has a different raw metadata shape, but they share the common helpers (employee map,
citation/MADOC/DOI lookups, provenance assembly) and emit the same schema.

Scoring is done upstream (code/from_papers/scoring.py); this script only formats the
metadata and carries the scores into the final file.

Inputs (data/from_papers/):
- osf_unimannheim_scored_potential.csv               (OSF scores, from scoring.py)
- figshare_unimannheim_scored_potential.csv          (Figshare scores, from scoring.py)
- osf_metadata/osf_<id>.json                          (OSF metadata)
- figshare_metadata/figshare_<id>.json               (Figshare metadata)
- extracted_repositories_with_papers_2026-06.csv     (which paper links each item)
- paper_lookup_madoc.csv                              (paper citation + MADOC link)
- results.csv                                         (paper DOIs, fallback for provenance)
- UniMa_Employee_List_llm_enriched_normalized_orcid_api.csv

Output:
- data/from_papers/unified_osf_figshare_v5.csv

Run: python code/from_papers/build_unified_metadata.py
"""

from __future__ import annotations

import ast
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[2]
FROM_PAPERS = BASE_DIR / "data" / "from_papers"

# Inputs
SCORED_OSF = FROM_PAPERS / "osf_unimannheim_scored_potential.csv"
SCORED_FIGSHARE = FROM_PAPERS / "figshare_unimannheim_scored_potential.csv"
OSF_METADATA_DIR = FROM_PAPERS / "osf_metadata"
FIGSHARE_DIR = FROM_PAPERS / "figshare_metadata"
EMPLOYEE_CSV = FROM_PAPERS / "UniMa_Employee_List_llm_enriched_normalized_orcid_api.csv"
PAPER_MENTIONS = FROM_PAPERS / "extracted_repositories_with_papers_2026-06.csv"
PAPER_LOOKUP = FROM_PAPERS / "paper_lookup_madoc.csv"
RESULTS_CSV = FROM_PAPERS / "results.csv"

# Output
OUT = FROM_PAPERS / "unified_osf_figshare_v5.csv"

HEADER = [
    "Source",
    "Source DOI",
    "Source URL",
    "Type",
    "Title",
    "Creators",
    "Affiliations",
    "Description",
    "Year",
    "License",
    "Date",
    "Provenance",
    "Confidence",
    "Verdict",
    "Confidence_Evidence",
]

ORCID_RE = re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[0-9X]\b")
OSF_ID_ANYWHERE_RE = re.compile(r"(?:osf\.io/|10\.17605/OSF\.IO/)([0-9a-z]{5})", re.IGNORECASE)
FIGSHARE_DOI_RE = re.compile(r"10\.6084/m9\.figshare\.(\d+)(?:\.v\d+)?", re.IGNORECASE)
FIGSHARE_ID_URL_RE = re.compile(r"figshare\.com/(?:articles|datasets)/.+?/(\d+)(?:[/?#].*)?$", re.IGNORECASE)


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------
def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            return [dict(r) for r in csv.DictReader(f)]
    except UnicodeDecodeError:
        with path.open(newline="", encoding="cp1252") as f:
            return [dict(r) for r in csv.DictReader(f)]


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="cp1252"))


def as_str(x: Any) -> str:
    return "" if x is None else str(x)


def scored_col(row: dict[str, str], *names: str) -> str:
    for name in names:
        val = row.get(name)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def norm_orcid(x: str) -> str:
    m = ORCID_RE.search((x or "").strip())
    return m.group(0) if m else ""


def norm_name(x: str) -> str:
    s = (x or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\-'.]", "", s)
    return s


def strip_html(s: str) -> str:
    s2 = re.sub(r"<[^>]+>", " ", s or "")
    return re.sub(r"\s+", " ", s2).strip()


def year_from_iso(dt: str) -> str:
    m = re.match(r"^(\d{4})-", (dt or "").strip())
    return m.group(1) if m else ""


def read_employee_maps() -> tuple[dict[str, list[str]], set[str]]:
    """ORCID -> [employee names], and a set of normalized employee names (name fallback)."""
    rows = read_csv_rows(EMPLOYEE_CSV)
    if not rows:
        return {}, set()
    orcid_cols = [c for c in ["orcid_api", "orcid_m", "orcid_id_llm"] if c in rows[0]]
    orcid_to_names: dict[str, list[str]] = {}
    names_norm: set[str] = set()
    for row in rows:
        emp_name = (row.get("full_name_m") or "").strip() or (
            ((row.get("first_name_llm") or "").strip() + " " + (row.get("last_name_llm") or "").strip()).strip()
        )
        if emp_name:
            names_norm.add(norm_name(emp_name))
        for c in orcid_cols:
            o = norm_orcid(row.get(c) or "")
            if not o:
                continue
            orcid_to_names.setdefault(o, [])
            if emp_name and emp_name not in orcid_to_names[o]:
                orcid_to_names[o].append(emp_name)
    return orcid_to_names, names_norm


def build_filename_to_citation() -> dict[str, str]:
    out: dict[str, str] = {}
    for row in read_csv_rows(PAPER_LOOKUP):
        fn = (row.get("filename") or "").strip()
        cit = (row.get("citation") or "").strip()
        if fn and cit and fn not in out:
            out[fn] = cit
    return out


def build_filename_to_madoc() -> dict[str, str]:
    out: dict[str, str] = {}
    for row in read_csv_rows(PAPER_LOOKUP):
        fn = (row.get("filename") or "").strip()
        link = (row.get("madoc_identifier") or "").strip()  # already the clean MADOC URL
        if fn and link and fn not in out:
            out[fn] = link
    return out


def _parse_results_doi(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    m = re.search(r"\b10\.\d{4,9}/[^\s\"'<>]+", s)
    if m:
        return m.group(0)
    try:
        v = ast.literal_eval(s)
        if isinstance(v, dict):
            for k in ("doi", "DOI"):
                if k in v and v.get(k):
                    return str(v.get(k)).strip()
    except Exception:
        pass
    return s


def build_filename_to_paper_doi() -> dict[str, str]:
    out: dict[str, str] = {}
    for row in read_csv_rows(RESULTS_CSV):
        fn = (row.get("filename") or "").strip()
        doi = _parse_results_doi((row.get("02 DOI") or "").strip())
        if fn and doi:
            out[fn] = doi
    return out


def paper_filename_candidates(fn: str) -> list[str]:
    """Generate filename variants to tolerate minor differences across pipeline steps."""
    s0 = (fn or "").strip()
    if not s0:
        return []
    cands: list[str] = []

    def add(x: str) -> None:
        x2 = (x or "").strip()
        if x2 and x2 not in cands:
            cands.append(x2)

    add(s0)
    if s0.lower().endswith("_.pdf"):
        add(s0[:-5] + ".pdf")
    if s0.lower().endswith(".pdf"):
        add(re.sub(r"\s*\(\d+\)(?=\.pdf$)", "", s0, flags=re.IGNORECASE))
        add(re.sub(r"%20%28\d+%29(?=\.pdf$)", "", s0, flags=re.IGNORECASE))
        m = re.match(r"^(.*?)(-\d+)?\.pdf$", s0, flags=re.IGNORECASE)
        if m:
            base, suffix = m.group(1), (m.group(2) or "")
            add(f"{base}-1.pdf" if not suffix else f"{base}{suffix}-1.pdf")
    return cands


def provenance_entries(filenames: list[str], f2cit: dict[str, str], f2link: dict[str, str], f2doi: dict[str, str]) -> str:
    """Assemble 'citation - MADOC link' entries for the citing paper(s); shared by OSF + Figshare."""
    entries: list[str] = []
    for fn in filenames:
        cit = ""
        link = ""
        for cand in paper_filename_candidates(fn):
            if not cit:
                cit = f2cit.get(cand, "").strip()
            if not link:
                link = f2link.get(cand, "").strip()
            if cit and link:
                break
        if cit and link and link not in cit:
            entries.append(f"{cit} - {link}")
        elif cit:
            entries.append(cit)
        elif link:
            entries.append(link)
    entries = list(dict.fromkeys(entries))  # de-dupe (same paper under multiple filename variants)
    if entries:
        return "; ".join(entries)
    # fallback: paper DOIs, then bare filenames
    dois = list(dict.fromkeys(d for fn in filenames for d in [next((f2doi.get(c, "").strip() for c in paper_filename_candidates(fn) if f2doi.get(c, "").strip()), "")] if d))
    if dois:
        return "; ".join(dois)
    return "; ".join(dict.fromkeys(filenames))


# --------------------------------------------------------------------------------------
# OSF row builder
# --------------------------------------------------------------------------------------
def _osf_id_to_filenames() -> dict[str, list[str]]:
    """osf_id -> [paper filenames]; scans every cell of the mentions CSV for OSF ids."""
    out: dict[str, list[str]] = {}
    for row in read_csv_rows(PAPER_MENTIONS):
        filename = (row.get("filename") or "").strip()
        if not filename:
            continue
        for v in row.values():
            m = OSF_ID_ANYWHERE_RE.search(str(v or ""))
            if not m:
                continue
            oid = m.group(1).lower()
            out.setdefault(oid, [])
            if filename not in out[oid]:
                out[oid].append(filename)
    return out


def _person_mentions_mannheim(person: dict[str, Any]) -> bool:
    hay: list[str] = []
    for k in ("employment", "education"):
        v = person.get(k)
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    hay += [vv.strip() for vv in item.values() if isinstance(vv, str) and vv.strip()]
        elif isinstance(v, dict):
            hay += [vv.strip() for vv in v.values() if isinstance(vv, str) and vv.strip()]
    txt = " | ".join(hay).lower()
    return ("university of mannheim" in txt) or ("universität mannheim" in txt) or ("universitaet mannheim" in txt)


def _osf_confidence_evidence(r: dict[str, str], combined_obj: dict[str, Any], emp_orcid_to_names: dict[str, list[str]]) -> str:
    sentences: list[str] = []
    matched = scored_col(r, "unima_matched_contributors", "matched_contributors")
    n_matched = scored_col(r, "unima_n_matched_contributors", "n_matched_contributors")
    n_total = scored_col(r, "osf_n_contributors", "n_contributors")
    match_method = scored_col(r, "unima_match_method", "match_method")

    block = combined_obj.get("contributors") if isinstance(combined_obj, dict) else None
    people = block.get("people") if isinstance(block, dict) else None

    orcid_matches: list[str] = []
    if isinstance(people, list) and emp_orcid_to_names:
        for p in people:
            if not isinstance(p, dict):
                continue
            o = norm_orcid(as_str(p.get("orcid")))
            names = emp_orcid_to_names.get(o) or [] if o else []
            if names:
                who = (p.get("full_name") or "").strip() or "unknown"
                orcid_matches.append(f"{who} (ORCID {o}) matches Uni Mannheim employee record: {', '.join(names)}")
    if orcid_matches:
        sentences.append("ORCID match: " + "; ".join(orcid_matches) + ".")

    any_profile = scored_col(r, "osf_profile_mentions_mannheim", "any_mannheim_profile_match").lower() in {"true", "t", "1", "yes", "y"}
    snippets = scored_col(r, "osf_profile_mannheim_snippets", "mannheim_profile_snippets")
    mannheim_people: list[str] = []
    if any_profile and isinstance(people, list):
        for p in people:
            if isinstance(p, dict) and _person_mentions_mannheim(p):
                nm = (p.get("full_name") or "").strip()
                if nm and nm not in mannheim_people:
                    mannheim_people.append(nm)
    if any_profile and snippets:
        sn = " ".join(snippets.split())
        if len(sn) > 240:
            sn = sn[:237].rstrip() + "..."
        if mannheim_people:
            sentences.append(f"OSF profile lists University of Mannheim for: {', '.join(mannheim_people)} (e.g., “{sn}”).")
        else:
            sentences.append(f"OSF profile mentions Mannheim affiliation (e.g., “{sn}”).")
    elif any_profile:
        if mannheim_people:
            sentences.append(f"OSF profile lists University of Mannheim for: {', '.join(mannheim_people)}.")
        else:
            sentences.append("OSF profile mentions Mannheim affiliation.")

    if matched:
        if "orcid" in match_method and "full_name" in match_method:
            how = "by contributor name and ORCID"
        elif "orcid" in match_method:
            how = "by ORCID"
        elif "given_family" in match_method:
            how = "by given+family name"
        elif "full_name" in match_method:
            how = "by full name"
        elif match_method:
            how = f"by {match_method.replace('_', ' ')}"
        else:
            how = "by contributor information"
        if n_matched and n_total:
            count_str = f" ({n_matched} of {n_total} contributors matched)"
        elif n_matched:
            count_str = f" ({n_matched} contributor(s) matched)"
        else:
            count_str = ""
        sentences.append(f"Matched contributors to Uni Mannheim employee list {how}: {matched}{count_str}.")

    paper_years = scored_col(r, "citing_paper_years", "paper_years")
    tenure_status = scored_col(r, "unima_tenure_status", "tenure_status")
    tenure_ok = scored_col(r, "unima_tenure_ok_contributors", "tenure_ok_contributors")
    tenure_mismatch = scored_col(r, "unima_tenure_mismatch_contributors", "tenure_mismatch_contributors")
    year_part = f" ({paper_years.replace(';', ', ')})" if paper_years else ""
    if tenure_ok:
        sentences.append(f"Mannheim employment covers the citing paper year{year_part} for: {tenure_ok}.")
    if tenure_mismatch:
        sentences.append(f"Employee matched but Mannheim employment does not cover the citing paper year{year_part} for: {tenure_mismatch}.")
    elif tenure_status == "paper_year_unknown" and matched:
        sentences.append("Tenure vs citing paper year not verified (paper year missing in lookup).")

    if not sentences:
        ev = scored_col(r, "unima_confidence_evidence", "evidence")
        return f"Match evidence: {ev}" if ev else ""
    return " ".join(sentences)


def _osf_license(obj: dict[str, Any]) -> str:
    lic = obj.get("license")
    if isinstance(lic, dict):
        name = as_str(lic.get("name")).strip()
        url = as_str(lic.get("url")).strip()
        return f"{name} ({url})" if (name and url) else (name or url)
    res = obj.get("resource") if isinstance(obj.get("resource"), dict) else {}
    node_lic = res.get("node_license") if isinstance(res, dict) else None
    if isinstance(node_lic, dict):
        holders = node_lic.get("copyright_holders")
        year = as_str(node_lic.get("year")).strip()
        holders_str = "; ".join(as_str(h).strip() for h in holders if as_str(h).strip()) if isinstance(holders, list) else ""
        if holders_str and year:
            return f"Copyright {year}: {holders_str}"
        if year:
            return f"Copyright {year}"
        if holders_str:
            return f"Copyright holders: {holders_str}"
    return ""


def _osf_year(date_str: str) -> str:
    s = (date_str or "").strip()
    if not s:
        return ""
    m = re.match(r"^(\d{4})-\d{2}-\d{2}", s)
    if m:
        return m.group(1)
    try:
        return str(datetime.fromisoformat(s.replace("Z", "+00:00")).year)
    except Exception:
        return ""


def _join_people_names(people: Any) -> str:
    if not isinstance(people, list):
        return ""
    seen: set[str] = set()
    out: list[str] = []
    for p in people:
        if not isinstance(p, dict):
            continue
        nm = (p.get("full_name") or "").strip()
        if nm and nm.lower() not in seen:
            seen.add(nm.lower())
            out.append(nm)
    return "; ".join(out)


def build_osf_rows(emp_orcid_to_names: dict[str, list[str]], f2cit, f2link, f2doi) -> list[dict[str, str]]:
    rows = read_csv_rows(SCORED_OSF)
    # De-dup by OSF id (same id can appear under multiple URL variants); prefer the row with a DOI.
    best: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for r in rows:
        oid = (r.get("osf_id") or "").strip().lower()
        if not oid:
            continue
        if oid not in best:
            best[oid] = r
            order.append(oid)
        elif (r.get("osf_doi") or "").strip() and not (best[oid].get("osf_doi") or "").strip():
            best[oid] = r
    rows = [best[o] for o in order]

    osf_id_to_filenames = _osf_id_to_filenames()

    out: list[dict[str, str]] = []
    for r in rows:
        osf_id = (r.get("osf_id") or "").strip().lower()
        jp = OSF_METADATA_DIR / f"osf_{osf_id}.json"
        obj = read_json(jp) if jp.exists() else {}
        resource = obj.get("resource") if isinstance(obj.get("resource"), dict) else {}

        title = (scored_col(r, "osf_title", "title") or resource.get("title") or "").strip()
        description = as_str(resource.get("description")).strip()
        html_url = as_str((resource.get("urls") or {}).get("html")).strip() if isinstance(resource.get("urls"), dict) else ""
        doi_val = scored_col(r, "osf_doi", "doi")
        identifier = doi_val if doi_val.lower().startswith("http") else (f"https://doi.org/{doi_val}" if doi_val else "")

        creators = ""
        block = obj.get("contributors") if isinstance(obj, dict) else None
        if isinstance(block, dict):
            creators = _join_people_names(block.get("people"))

        any_profile = scored_col(r, "osf_profile_mentions_mannheim", "any_mannheim_profile_match").lower() in {"true", "t", "1", "yes", "y"}
        affiliations = "Universität Mannheim" if any_profile else ""

        dates = resource.get("dates") if isinstance(resource.get("dates"), dict) else {}
        date_str = (
            as_str(dates.get("date_registered")).strip()
            or as_str(dates.get("date_created")).strip()
            or as_str(obj.get("fetched_at")).strip()
        )

        paper_fns = osf_id_to_filenames.get(osf_id, [])
        prov = provenance_entries(paper_fns, f2cit, f2link, f2doi) if paper_fns else ""
        provenance = f"OSF dataset was taken from this paper: {prov}" if prov else ""

        out.append(
            {
                "Source": "OSF",
                "Source DOI": identifier,
                "Source URL": html_url,
                "Type": "Dataset",
                "Title": title,
                "Creators": creators,
                "Affiliations": affiliations,
                "Description": description,
                "Year": _osf_year(date_str),
                "License": _osf_license(obj) if isinstance(obj, dict) else "",
                "Date": date_str,
                "Provenance": provenance,
                "Confidence": scored_col(r, "unima_confidence_score", "score"),
                "Verdict": scored_col(r, "verdict"),
                "Confidence_Evidence": _osf_confidence_evidence(r, obj if isinstance(obj, dict) else {}, emp_orcid_to_names),
            }
        )
    return out


# --------------------------------------------------------------------------------------
# Figshare row builder
# --------------------------------------------------------------------------------------
def _figshare_article_to_filenames() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in read_csv_rows(PAPER_MENTIONS):
        repo = (row.get("repository") or "").strip().lower()
        if repo != "figshare":
            continue
        fn = (row.get("filename") or "").strip()
        url = (row.get("url") or "").strip()
        if not fn or not url:
            continue
        m = FIGSHARE_DOI_RE.search(url) or FIGSHARE_ID_URL_RE.search(url)
        if not m:
            continue
        aid = m.group(1)
        out.setdefault(aid, [])
        if fn not in out[aid]:
            out[aid].append(fn)
    return out


def _figshare_type(article: dict[str, Any]) -> str:
    t = (as_str(article.get("defined_type_name")) or "").strip().lower()
    if t in {"figure", "media", "poster", "presentation", "image"}:
        return "Image"
    return "Dataset"


def _figshare_confidence_fallback(article: dict[str, Any], emp_orcid_to_names, emp_names_norm) -> tuple[str, str, str]:
    """Local heuristic used only if the notebook Figshare score is missing."""
    authors = article.get("authors")
    if not isinstance(authors, list):
        return "0.0", "No Mannheim evidence found", "Match evidence: no_employee_match"
    orcid_hits, name_hits = [], []
    for a in authors:
        if not isinstance(a, dict):
            continue
        nm = as_str(a.get("full_name")).strip()
        o = norm_orcid(as_str(a.get("orcid_id")))
        if o and o in emp_orcid_to_names:
            orcid_hits.append(f"{nm} (ORCID {o}) -> {', '.join(emp_orcid_to_names[o])}")
        elif nm and norm_name(nm) in emp_names_norm:
            name_hits.append(nm)
    if orcid_hits:
        return "1.0", "Very likely Mannheim", "ORCID match to Uni Mannheim employee list: " + "; ".join(orcid_hits)
    if name_hits:
        return "0.7", "Likely Mannheim - verify (name match)", "Author name match to Uni Mannheim employee list: " + "; ".join(sorted(set(name_hits)))
    return "0.0", "No Mannheim evidence found", "Match evidence: no_employee_match"


def build_figshare_rows(emp_orcid_to_names, emp_names_norm, f2cit, f2link) -> list[dict[str, str]]:
    figshare_scores = {(_sr.get("figshare_id") or "").strip(): _sr for _sr in read_csv_rows(SCORED_FIGSHARE) if (_sr.get("figshare_id") or "").strip()}
    article_to_filenames = _figshare_article_to_filenames()

    out: list[dict[str, str]] = []
    if not FIGSHARE_DIR.exists():
        return out
    for jp in sorted(FIGSHARE_DIR.glob("figshare_*.json")):
        obj = read_json(jp)
        article = obj.get("article") if isinstance(obj, dict) else None
        if not isinstance(article, dict):
            continue
        article_id = str(obj.get("article_id") or "").strip()

        doi = as_str(article.get("doi")).strip()
        if doi and not doi.lower().startswith("http"):
            doi = f"https://doi.org/{doi}"
        url = as_str(article.get("url_public_html") or article.get("figshare_url") or "").strip()
        title = as_str(article.get("title")).strip()
        published = as_str(article.get("published_date") or article.get("created_date")).strip()

        lic = article.get("license") if isinstance(article.get("license"), dict) else {}
        lic_name, lic_url = as_str(lic.get("name")).strip(), as_str(lic.get("url")).strip()
        license_str = f"{lic_name} ({lic_url})" if (lic_name and lic_url) else (lic_name or lic_url)

        creators = "; ".join(
            as_str(a.get("full_name")).strip()
            for a in (article.get("authors") or [])
            if isinstance(a, dict) and as_str(a.get("full_name")).strip()
        )

        # Prefer the notebook score (same logic as OSF); fall back to the local heuristic.
        srow = figshare_scores.get(article_id)
        if srow is not None:
            conf = (srow.get("unima_confidence_score") or "").strip()
            verdict = (srow.get("verdict") or "").strip()
            conf_ev = (srow.get("why") or "").strip() or (srow.get("unima_confidence_evidence") or "").strip()
        else:
            conf, verdict, conf_ev = _figshare_confidence_fallback(article, emp_orcid_to_names, emp_names_norm)

        paper_fns = article_to_filenames.get(article_id, []) if article_id else []
        prov = provenance_entries(paper_fns, f2cit, f2link, {}) if paper_fns else ""
        provenance = f"Figshare item was taken from this paper: {prov}" if prov else ""

        out.append(
            {
                "Source": "Figshare",
                "Source DOI": doi,
                "Source URL": url,
                "Type": _figshare_type(article),
                "Title": title,
                "Creators": creators,
                "Affiliations": "",
                "Description": strip_html(as_str(article.get("description"))),
                "Year": year_from_iso(published),
                "License": license_str,
                "Date": published,
                "Provenance": provenance,
                "Confidence": conf,
                "Verdict": verdict,
                "Confidence_Evidence": conf_ev,
            }
        )
    return out


def main() -> int:
    emp_orcid_to_names, emp_names_norm = read_employee_maps()
    f2cit = build_filename_to_citation()
    f2link = build_filename_to_madoc()
    f2doi = build_filename_to_paper_doi()

    rows = build_osf_rows(emp_orcid_to_names, f2cit, f2link, f2doi)
    rows += build_figshare_rows(emp_orcid_to_names, emp_names_norm, f2cit, f2link)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        with OUT.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=HEADER)
            w.writeheader()
            w.writerows(rows)
    except PermissionError:
        raise SystemExit(f"PermissionError: cannot write to {OUT} (close it in Excel and re-run).")

    n_osf = sum(1 for r in rows if r["Source"] == "OSF")
    n_fig = sum(1 for r in rows if r["Source"] == "Figshare")
    print(f"Wrote {len(rows)} rows to {OUT}  (OSF: {n_osf}, Figshare: {n_fig})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
