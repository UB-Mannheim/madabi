"""
Unified Uni-Mannheim scoring for OSF + Figshare "potential datasets".

Both sources are scored with the SAME logic (`score_record`): match an item's people
(OSF contributors / Figshare authors) to the Uni Mannheim employee list (ORCID -> name)
and check whether a citing MADOC paper appeared during that person's Mannheim employment
(the "tenure" check). The only source-specific bit is the label used in the reason
sentence ("OSF project" vs "Figshare item").

Used by notebooks/explore_mannheim_contributors.ipynb:

    from scoring import run_all
    osf_scored, figshare_scored = run_all()

…and runnable directly:

    python code/from_papers/scoring.py

Outputs:
- data/from_papers/osf_unimannheim_scored_potential.csv
- data/from_papers/figshare_unimannheim_scored_potential.csv
"""

from __future__ import annotations

import calendar
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
FROM_PAPERS = BASE_DIR / "data" / "from_papers"

# Inputs
OSF_METADATA_DIR = FROM_PAPERS / "osf_metadata"
OSF_METADATA_MANIFEST = OSF_METADATA_DIR / "manifest.csv"
EMPLOYEE_CSV = FROM_PAPERS / "UniMa_Employee_List_llm_enriched_normalized_orcid_api.csv"
PAPER_LINKS_CSV = FROM_PAPERS / "extracted_repositories_with_papers_2026-06.csv"
PAPER_LOOKUP_CSV = FROM_PAPERS / "paper_lookup_madoc.csv"
FIGSHARE_DIR = FROM_PAPERS / "figshare_metadata"

# Outputs
OSF_OUT = FROM_PAPERS / "osf_unimannheim_scored_potential.csv"
FIGSHARE_OUT = FROM_PAPERS / "figshare_unimannheim_scored_potential.csv"


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
OSF_DOI_RE = re.compile(r"10\.17605/OSF\.IO/([0-9a-z]{5})", re.IGNORECASE)
OSF_ID_RE = re.compile(r"^/([0-9a-z]{5})(?:/|$)", re.IGNORECASE)
OSF_FILES_SUBID_RE = re.compile(r"/files/([0-9a-z]{5})(?:/|$)", re.IGNORECASE)
FIGSHARE_DOI_RE = re.compile(r"10\.6084/m9\.figshare\.(\d+)(?:\.v\d+)?", re.IGNORECASE)
FIGSHARE_ID_URL_RE = re.compile(r"figshare\.com/(?:articles|datasets)/.+?/(\d+)(?:[/?#].*)?$", re.IGNORECASE)
ORCID_RE = re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[0-9X]\b")
YEAR_RE = re.compile(r"\b(18\d{2}|19\d{2}|20\d{2})\b")
MANNHEIM_RE = re.compile(
    r"\b(university\s+of\s+mannheim|universit[aä]t\s+mannheim|uni-?mannheim|mannheim)\b",
    re.IGNORECASE,
)


def is_true(x: Any) -> bool:
    return str(x or "").strip().lower() in {"true", "t", "1", "yes", "y"}


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV as strings, tolerating a UTF-8 BOM or occasional cp1252."""
    try:
        return pd.read_csv(path, dtype=str)
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="cp1252")


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="cp1252"))


def norm_name(s: Any) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def norm_pair(g: str, f: str) -> str:
    return f"{norm_name(g)}|{norm_name(f)}"


def norm_orcid(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    m = ORCID_RE.search(str(x).strip())
    return m.group(0) if m else ""


def extract_osf_id(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    m = OSF_DOI_RE.search(u)
    if m:
        return m.group(1).lower()
    try:
        from urllib.parse import urlparse

        p = urlparse(u)
        if p.netloc.lower().endswith("osf.io"):
            mf = OSF_FILES_SUBID_RE.search(p.path or "")
            if mf:
                return mf.group(1).lower()
            m2 = OSF_ID_RE.match(p.path or "")
            if m2:
                return m2.group(1).lower()
    except Exception:
        return ""
    return ""


def extract_figshare_id(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    m = FIGSHARE_DOI_RE.search(u) or FIGSHARE_ID_URL_RE.search(u)
    return m.group(1) if m else ""


def parse_year_month(raw: Any):
    s = str(raw or "").strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    m = re.match(r"^(\d{4})(?:-(\d{1,2}))?", s)
    if m:
        return int(m.group(1)), int(m.group(2) or 1)
    ym = YEAR_RE.search(s)
    return (int(ym.group(1)), 1) if ym else None


def stint_bounds(start_ym, end_ym):
    if not start_ym:
        return None
    sy, sm = start_ym
    start = date(sy, sm, 1)
    if end_ym:
        ey, em = end_ym
        last = calendar.monthrange(ey, em)[1]
        end = date(ey, em, last)
    else:
        end = date.today()
    return start, end


def ranges_overlap(a_start, a_end, b_start, b_end) -> bool:
    return a_start <= b_end and b_start <= a_end


def tenure_covers_paper_year(stints, paper_year: int) -> bool:
    if not stints or paper_year is None:
        return False
    p0, p1 = date(int(paper_year), 1, 1), date(int(paper_year), 12, 31)
    for bounds in stints:
        if bounds and ranges_overlap(bounds[0], bounds[1], p0, p1):
            return True
    return False


def iter_text_fields(x):
    if x is None:
        return
    if isinstance(x, str):
        yield x
        return
    if isinstance(x, dict):
        for v in x.values():
            yield from iter_text_fields(v)
        return
    if isinstance(x, list):
        for v in x:
            yield from iter_text_fields(v)


def fmt_windows(windows) -> str:
    out = []
    for b in windows:
        if not b:
            continue
        s, e = b
        out.append(f"{s.year}-{e.year}")
    seen: set[str] = set()
    uniq = []
    for w in out:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return ", ".join(uniq)


def join_names(names, max_show: int = 4) -> str:
    uniq = list(dict.fromkeys([n for n in names if n]))
    if not uniq:
        return ""
    if len(uniq) <= max_show:
        return "; ".join(uniq)
    return "; ".join(uniq[:max_show]) + f"; +{len(uniq) - max_show} more"


def format_citing_papers(papers) -> str:
    """One readable entry per citing Mannheim paper: 'Title (Year) - <madoc link>'."""
    parts = []
    for p in papers or []:
        title = (p.get("title") or "").strip() or "(untitled)"
        year = p.get("year")
        link = (p.get("madoc_link") or "").strip()
        label = f"{title} ({year})" if year else title
        if link:
            label += f" - {link}"
        parts.append(label)
    return " | ".join(parts)


def fix_mojibake(s: Any) -> str:
    s = "" if s is None else str(s)
    if "Ã" not in s and "Â" not in s:
        return s
    try:
        return s.encode("latin1").decode("utf-8")
    except Exception:
        return s


# --------------------------------------------------------------------------------------
# Context: everything loaded once and shared by both OSF and Figshare scoring
# --------------------------------------------------------------------------------------
@dataclass
class Context:
    orcid_stints: dict
    full_name_stints: dict
    pair_stints: dict
    filename_to_year: dict
    filename_to_paper: dict
    osf_id_to_years: dict
    osf_id_to_papers: dict
    figshare_id_to_years: dict
    figshare_id_to_papers: dict
    manifest_df: pd.DataFrame


def _build_employee_stints(emp: pd.DataFrame):
    orcid_stints: dict[str, list] = {}
    full_name_stints: dict[str, list] = {}
    pair_stints: dict[str, list] = {}

    start_cols = [
        ("uni_start_date_m_normalized", "uni_end_date_m_normalized"),
        ("uni_start_date_m_normalized_2", "uni_end_date_m_normalized_2"),
        ("uni_start_date_llm_norm", "uni_end_date_llm"),
    ]
    name_cols = [c for c in ["full_name_m", "full_name", "name", "name_normalized"] if c in emp.columns]
    first_col = next((c for c in ["first_name_llm", "given_name", "first_name"] if c in emp.columns), None)
    last_col = next((c for c in ["last_name_llm", "family_name", "last_name"] if c in emp.columns), None)
    orcid_cols = [c for c in ["orcid_api", "orcid_m", "orcid_id_llm"] if c in emp.columns]

    for _, erow in emp.iterrows():
        stints = []
        for sc, ec in start_cols:
            if sc not in emp.columns:
                continue
            end_val = erow.get(ec) if ec in emp.columns else ""
            bounds = stint_bounds(parse_year_month(erow.get(sc)), parse_year_month(end_val))
            if bounds:
                stints.append(bounds)
        if not stints:
            continue
        for o in {norm_orcid(erow.get(c)) for c in orcid_cols} - {""}:
            orcid_stints.setdefault(o, []).extend(stints)
        for n in {norm_name(erow.get(c)) for c in name_cols} - {""}:
            full_name_stints.setdefault(n, []).extend(stints)
        if first_col and last_col:
            pair = norm_pair(erow.get(first_col) or "", erow.get(last_col) or "")
            if pair != "|":
                pair_stints.setdefault(pair, []).extend(stints)
    return orcid_stints, full_name_stints, pair_stints


def _build_paper_maps(paper_links: pd.DataFrame, paper_lookup: pd.DataFrame):
    filename_to_year: dict[str, int] = {}
    filename_to_paper: dict[str, dict] = {}
    for _, row in paper_lookup.iterrows():
        fn = str(row.get("filename") or "").strip()
        if not fn:
            continue
        ym = parse_year_month(str(row.get("year") or ""))
        if ym:
            filename_to_year[fn] = ym[0]
        filename_to_paper[fn] = {
            "title": str(row.get("title") or "").strip(),
            "year": ym[0] if ym else None,
            # madoc_identifier is already the clean MADOC landing-page URL
            "madoc_link": str(row.get("madoc_identifier") or "").strip(),
        }

    def _accumulate(kind_repo: str, id_fn):
        id_to_years: dict[str, set] = {}
        id_to_papers: dict[str, list] = {}
        for _, row in paper_links.iterrows():
            repo = str(row.get("repository") or "").strip()
            raw = str(row.get("url") or "").strip()
            if kind_repo == "Figshare":
                if repo != "Figshare" and "figshare" not in raw.lower():
                    continue
            elif repo != kind_repo:
                continue
            rid = id_fn(raw)
            fn = str(row.get("filename") or "").strip()
            if not rid or not fn:
                continue
            y = filename_to_year.get(fn)
            if y is not None:
                id_to_years.setdefault(rid, set()).add(y)
            paper = filename_to_paper.get(fn)
            if paper and (paper.get("title") or paper.get("madoc_link")):
                bucket = id_to_papers.setdefault(rid, [])
                key = (paper.get("title"), paper.get("madoc_link"))
                if key not in {(p.get("title"), p.get("madoc_link")) for p in bucket}:
                    bucket.append(paper)
        return id_to_years, id_to_papers

    osf_id_to_years, osf_id_to_papers = _accumulate("OSF", extract_osf_id)
    figshare_id_to_years, figshare_id_to_papers = _accumulate("Figshare", extract_figshare_id)
    return (
        filename_to_year,
        filename_to_paper,
        osf_id_to_years,
        osf_id_to_papers,
        figshare_id_to_years,
        figshare_id_to_papers,
    )


def _load_manifest() -> pd.DataFrame:
    osf_manifest = read_csv(OSF_METADATA_MANIFEST)
    osf_manifest["has_potential_data_files_bool"] = osf_manifest.get("has_potential_data_files", "").map(is_true)
    manifest_df = osf_manifest[osf_manifest["has_potential_data_files_bool"]].copy()
    # Drop duplicate OSF ids (same id can appear under multiple input URL variants, e.g.
    # with/without a view_only token); keep one row per id, preferring the one with a DOI.
    if "osf_id" in manifest_df.columns:
        manifest_df["_has_doi"] = manifest_df.get("doi", "").fillna("").astype(str).str.strip().ne("")
        manifest_df = (
            manifest_df.sort_values("_has_doi", ascending=False, kind="stable")
            .drop_duplicates(subset="osf_id", keep="first")
            .drop(columns=["_has_doi"])
        )
    return manifest_df


def build_context() -> Context:
    emp = read_csv(EMPLOYEE_CSV)
    paper_links = read_csv(PAPER_LINKS_CSV)
    paper_lookup = read_csv(PAPER_LOOKUP_CSV)

    orcid_stints, full_name_stints, pair_stints = _build_employee_stints(emp)
    (
        filename_to_year,
        filename_to_paper,
        osf_id_to_years,
        osf_id_to_papers,
        figshare_id_to_years,
        figshare_id_to_papers,
    ) = _build_paper_maps(paper_links, paper_lookup)

    return Context(
        orcid_stints=orcid_stints,
        full_name_stints=full_name_stints,
        pair_stints=pair_stints,
        filename_to_year=filename_to_year,
        filename_to_paper=filename_to_paper,
        osf_id_to_years=osf_id_to_years,
        osf_id_to_papers=osf_id_to_papers,
        figshare_id_to_years=figshare_id_to_years,
        figshare_id_to_papers=figshare_id_to_papers,
        manifest_df=_load_manifest(),
    )


def employee_stints_for_person(person: dict, ctx: Context):
    o = norm_orcid(person.get("orcid") or "")
    if o and o in ctx.orcid_stints:
        return ctx.orcid_stints[o], "orcid"
    full = norm_name(person.get("full_name") or "")
    if full and full in ctx.full_name_stints:
        return ctx.full_name_stints[full], "full_name"
    pair = norm_pair(person.get("given_name") or "", person.get("family_name") or "")
    if pair in ctx.pair_stints:
        return ctx.pair_stints[pair], "given_family"
    return [], ""


# --------------------------------------------------------------------------------------
# The one scorer used for BOTH OSF and Figshare
# --------------------------------------------------------------------------------------
def score_record(contributors, paper_years: set, ctx: Context, citing_papers=None, kind: str = "OSF project") -> dict:
    matched: list[str] = []
    methods: set[str] = set()
    any_mannheim_profile_match = False
    mannheim_snippets: list[str] = []
    ok_people: list[dict] = []      # employed at Mannheim when a citing paper appeared
    fail_people: list[dict] = []    # matched, but citing paper is outside their employment window
    noyear_people: list[dict] = []  # matched, but no citing-paper year was available to check

    for p in contributors:
        if not isinstance(p, dict):
            continue
        for fieldname in ("employment", "education"):
            for s in iter_text_fields(p.get(fieldname)):
                if MANNHEIM_RE.search(s):
                    any_mannheim_profile_match = True
                    methods.add("osf_profile_mannheim")
                    mannheim_snippets.append(s)

        stints, match_kind = employee_stints_for_person(p, ctx)
        if not stints:
            continue
        label = (p.get("full_name") or match_kind).strip()
        matched.append(label)
        methods.add(match_kind)
        person = {"name": label, "method": match_kind, "windows": stints}
        if not paper_years:
            noyear_people.append(person)
        elif any(tenure_covers_paper_year(stints, y) for y in paper_years):
            ok_people.append(person)
        else:
            fail_people.append(person)

    tenure_ok_names = [p["name"] for p in ok_people]
    tenure_fail_names = [p["name"] for p in fail_people]
    employee_match_no_year = [p["name"] for p in noyear_people]
    paper_years_str = ";".join(str(y) for y in sorted(paper_years)) if paper_years else ""

    def match_phrase(people):
        if any(p["method"] == "orcid" for p in people):
            return True, "matched by ORCID"
        return False, "matched by name"

    titles = [(p.get("title") or "").strip() for p in (citing_papers or []) if (p.get("title") or "").strip()]

    def cite_clause():
        if not titles:
            return f"a citing paper ({paper_years_str})"
        extra = f" (and {len(titles) - 1} more)" if len(titles) > 1 else ""
        return f'a Mannheim paper that links to it ("{titles[0]}"{extra}, {paper_years_str})'

    if ok_people:
        score, evidence = 1.0, "employee_match_tenure_ok"
        is_orcid, how = match_phrase(ok_people)
        verdict = "Very likely Mannheim" if is_orcid else "Likely Mannheim - verify (name match)"
        why = (
            f"Confirmed Mannheim staff: {join_names(tenure_ok_names)} ({how}). "
            f"This {kind} is linked from {cite_clause()}, which was published within their "
            f"recorded Mannheim employment - so they were at Mannheim when this work came out."
        )
    elif fail_people:
        score, evidence = 0.8, "employee_match_tenure_mismatch"
        is_orcid, how = match_phrase(fail_people)
        verdict = "Possibly Mannheim - verify (timing mismatch)"
        windows = fmt_windows([w for p in fail_people for w in p["windows"]])
        why = (
            f"{join_names(tenure_fail_names)} matches Mannheim staff ({how}), "
            f"but {cite_clause()} is OUTSIDE their recorded Mannheim employment ({windows}). "
            f"The dataset may pre- or post-date their time here - please verify."
        )
    elif noyear_people:
        score, evidence = 0.8, "employee_match_paper_year_unknown"
        is_orcid, how = match_phrase(noyear_people)
        verdict = "Possibly Mannheim - verify (timing unknown)"
        why = (
            f"{join_names(employee_match_no_year)} matches Mannheim staff ({how}), "
            f"but no citing-paper year was available to confirm they were at Mannheim at the time "
            f"- please verify."
        )
    elif any_mannheim_profile_match:
        score, evidence = 0.5, "osf_profile_mannheim_only"
        snip = (sorted(set(mannheim_snippets)) or [""])[0]
        verdict = "Possibly Mannheim - verify (profile only)"
        why = (
            "No contributor matched the official Mannheim staff list, but a contributor's OSF "
            f'profile mentions Mannheim: "{snip[:200]}" - please verify.'
        )
    else:
        score, evidence = 0.0, "no_employee_match"
        verdict = "No Mannheim evidence found"
        if len(contributors) == 0:
            why = (
                f"This {kind} lists no contributors and mentions Mannheim nowhere, "
                "so there is no evidence linking it to Mannheim staff."
            )
        else:
            why = (
                f"None of the {len(contributors)} contributor(s) matched the Mannheim staff list, "
                "and no contributor's OSF profile mentions Mannheim."
            )

    if ok_people:
        tenure_status = "ok"
    elif fail_people:
        tenure_status = "mismatch"
    elif matched:
        tenure_status = "paper_year_unknown"
    else:
        tenure_status = "no_employee_match"

    return {
        "n_contrib": len(contributors),
        "n_matched": len(set(matched)),
        "matched": matched,
        "methods": methods,
        "mannheim_snippets": mannheim_snippets,
        "any_mannheim_profile_match": any_mannheim_profile_match,
        "paper_years": paper_years_str,
        "tenure_ok_contributors": "; ".join(sorted(set(tenure_ok_names)))[:2000],
        "tenure_mismatch_contributors": "; ".join(sorted(set(tenure_fail_names)))[:2000],
        "tenure_status": tenure_status,
        "score": score,
        "evidence": evidence,
        "verdict": verdict,
        "why": why,
    }


def _apply_mojibake(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].map(fix_mojibake)
    return df


# --------------------------------------------------------------------------------------
# OSF scoring
# --------------------------------------------------------------------------------------
def score_osf(ctx: Context | None = None, write: bool = True) -> pd.DataFrame:
    ctx = ctx or build_context()
    out_rows = []
    for _, r in ctx.manifest_df.iterrows():
        osf_id = str(r.get("osf_id") or "").strip().lower()
        json_path = OSF_METADATA_DIR / f"osf_{osf_id}.json"
        obj = read_json(json_path) if json_path.exists() else None

        contributors = []
        if isinstance(obj, dict):
            block = obj.get("contributors")
            contributors = block.get("people") if isinstance(block, dict) else []
            if not isinstance(contributors, list):
                contributors = []

        citing_papers = ctx.osf_id_to_papers.get(osf_id, [])
        scored = score_record(
            contributors, ctx.osf_id_to_years.get(osf_id, set()), ctx, citing_papers, kind="OSF project"
        )

        out_rows.append(
            {
                "osf_id": osf_id,
                "verdict": scored["verdict"],
                "why": scored["why"],
                "citing_papers": format_citing_papers(citing_papers),
                "osf_title": r.get("title") or "",
                "osf_doi": r.get("doi") or "",
                "osf_original_url": r.get("original_url") or "",
                "osf_html_url": r.get("html_url") or "",
                "osf_api_url": r.get("api_url") or "",
                "osf_harvest_status": r.get("status") or "",
                "osf_harvest_error": r.get("error") or "",
                "osf_inventory_error": r.get("inventory_error") or "",
                "osf_total_files": r.get("total_files") or "",
                "osf_total_bytes": r.get("total_bytes") or "",
                "citing_paper_years": scored["paper_years"],
                "unima_tenure_status": scored["tenure_status"],
                "osf_n_contributors": scored["n_contrib"],
                "unima_n_matched_contributors": scored["n_matched"],
                "unima_matched_contributors": "; ".join(sorted(set(scored["matched"])))[:2000],
                "unima_tenure_ok_contributors": scored["tenure_ok_contributors"],
                "unima_tenure_mismatch_contributors": scored["tenure_mismatch_contributors"],
                "unima_match_method": "+".join(sorted(scored["methods"])),
                "osf_profile_mentions_mannheim": scored["any_mannheim_profile_match"],
                "osf_profile_mannheim_snippets": " | ".join(sorted(set(scored["mannheim_snippets"])))[:2000],
                "unima_confidence_score": scored["score"],
                "unima_confidence_evidence": scored["evidence"],
            }
        )

    out = pd.DataFrame(out_rows)
    if not out.empty:
        out = out.sort_values(
            ["unima_confidence_score", "unima_n_matched_contributors", "osf_id"],
            ascending=[False, False, True],
        )
    preferred = [
        "osf_id", "verdict", "why", "osf_title", "osf_html_url", "osf_original_url", "osf_doi",
        "citing_paper_years", "citing_papers", "unima_matched_contributors", "osf_n_contributors",
        "unima_n_matched_contributors", "osf_profile_mannheim_snippets", "unima_confidence_score",
        "unima_confidence_evidence", "unima_tenure_status", "unima_tenure_ok_contributors",
        "unima_tenure_mismatch_contributors", "unima_match_method", "osf_profile_mentions_mannheim",
        "osf_api_url", "osf_harvest_status", "osf_harvest_error", "osf_inventory_error",
        "osf_total_files", "osf_total_bytes",
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    out = out[cols]
    out = _apply_mojibake(
        out,
        ["why", "citing_papers", "unima_matched_contributors", "unima_tenure_ok_contributors",
         "unima_tenure_mismatch_contributors", "osf_title"],
    )
    if write:
        out.to_csv(OSF_OUT, index=False, encoding="utf-8-sig")
        print(f"Wrote {len(out)} OSF rows to {OSF_OUT}")
    return out


# --------------------------------------------------------------------------------------
# Figshare scoring (same scorer; Figshare authors play the role of OSF contributors)
# --------------------------------------------------------------------------------------
def _figshare_authors_as_contributors(article: dict) -> list:
    out = []
    for a in (article.get("authors") or []):
        if not isinstance(a, dict):
            continue
        out.append(
            {
                "full_name": a.get("full_name") or "",
                "orcid": a.get("orcid_id") or "",
                "given_name": "",
                "family_name": "",
                "employment": None,  # Figshare API exposes no employment/education profile
                "education": None,
            }
        )
    return out


def score_figshare(ctx: Context | None = None, write: bool = True) -> pd.DataFrame:
    ctx = ctx or build_context()
    out_rows = []
    for jp in sorted(FIGSHARE_DIR.glob("figshare_*.json")) if FIGSHARE_DIR.exists() else []:
        obj = read_json(jp)
        article = obj.get("article") if isinstance(obj, dict) else None
        if not isinstance(article, dict):
            continue
        fid = str(obj.get("article_id") or "").strip()

        contributors = _figshare_authors_as_contributors(article)
        citing_papers = ctx.figshare_id_to_papers.get(fid, [])
        scored = score_record(
            contributors, ctx.figshare_id_to_years.get(fid, set()), ctx, citing_papers, kind="Figshare item"
        )

        doi = str(article.get("doi") or "").strip()
        if doi and not doi.lower().startswith("http"):
            doi = f"https://doi.org/{doi}"
        url = str(article.get("url_public_html") or article.get("figshare_url") or "").strip()

        out_rows.append(
            {
                "figshare_id": fid,
                "verdict": scored["verdict"],
                "why": scored["why"],
                "citing_papers": format_citing_papers(citing_papers),
                "figshare_title": article.get("title") or "",
                "figshare_doi": doi,
                "figshare_url": url,
                "citing_paper_years": scored["paper_years"],
                "unima_tenure_status": scored["tenure_status"],
                "figshare_n_authors": scored["n_contrib"],
                "unima_n_matched_authors": scored["n_matched"],
                "unima_matched_authors": "; ".join(sorted(set(scored["matched"])))[:2000],
                "unima_tenure_ok_contributors": scored["tenure_ok_contributors"],
                "unima_tenure_mismatch_contributors": scored["tenure_mismatch_contributors"],
                "unima_match_method": "+".join(sorted(scored["methods"])),
                "unima_confidence_score": scored["score"],
                "unima_confidence_evidence": scored["evidence"],
            }
        )

    out = pd.DataFrame(out_rows)
    if not out.empty:
        out = out.sort_values(
            ["unima_confidence_score", "unima_n_matched_authors", "figshare_id"],
            ascending=[False, False, True],
        )
    out = _apply_mojibake(out, ["why", "citing_papers", "unima_matched_authors", "figshare_title"])
    if write:
        out.to_csv(FIGSHARE_OUT, index=False, encoding="utf-8-sig")
        print(f"Wrote {len(out)} Figshare rows to {FIGSHARE_OUT}")
    return out


def run_all(write: bool = True):
    """Score OSF + Figshare with one shared context. Returns (osf_df, figshare_df)."""
    ctx = build_context()
    return score_osf(ctx, write=write), score_figshare(ctx, write=write)


if __name__ == "__main__":
    osf_df, fig_df = run_all()
    print("\nOSF verdicts:")
    print(osf_df["verdict"].value_counts().to_string())
    print("\nFigshare verdicts:")
    print(fig_df["verdict"].value_counts().to_string())
