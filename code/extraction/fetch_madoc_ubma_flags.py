"""
Fetch UB Mannheim EPrints flags (ubma_external, ubma_unibiblio) from native XML
export and merge them into the MADOC harvest CSV.

Run after `code/harvester/madoc.py` (and optionally after PDF downloads). OAI
OpenAIRE metadata does not include these fields; they only appear in:

  https://madoc.bib.uni-mannheim.de/cgi/export/eprint/<id>/XML/madoc.xml
"""
from __future__ import annotations

import argparse
import ast
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DEFAULT_INPUT = DATA_DIR / "madoc.csv"
DEFAULT_MERGED = DATA_DIR / "madoc_with_ubma.csv"
DEFAULT_CACHE = DATA_DIR / "madoc_ubma_flags_cache.csv"

EXPORT_URL = "https://madoc.bib.uni-mannheim.de/cgi/export/eprint/{eprint_id}/XML/madoc.xml"
MADOC_ID_RE = re.compile(
    r"https?://madoc\.bib\.uni-mannheim\.de/(\d+)(?:/|$)", re.IGNORECASE
)


def _local_tag(elem: ET.Element) -> str:
    t = elem.tag
    return t.split("}", 1)[-1] if "}" in t else t


def _iter_file_urls(v: object):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return
    s = str(v).strip()
    if not s:
        return
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                for u in parsed:
                    u2 = str(u).strip()
                    if u2:
                        yield u2
                return
        except (SyntaxError, ValueError, TypeError):
            pass
    yield s


def madoc_eprint_id_from_row(identifier: object, file_val: object) -> int | None:
    if identifier is not None and not (isinstance(identifier, float) and pd.isna(identifier)):
        m = MADOC_ID_RE.search(str(identifier).strip())
        if m:
            return int(m.group(1))
    if file_val is not None and not (isinstance(file_val, float) and pd.isna(file_val)):
        for url in _iter_file_urls(file_val):
            m = MADOC_ID_RE.search(url)
            if m:
                return int(m.group(1))
    return None


def _parse_bool_text(text: str | None) -> bool | None:
    if text is None:
        return None
    t = text.strip().upper()
    if t == "TRUE":
        return True
    if t == "FALSE":
        return False
    return None


def parse_ubma_flags(xml_bytes: bytes) -> tuple[bool | None, bool | None]:
    root = ET.fromstring(xml_bytes)
    ext: bool | None = None
    uni: bool | None = None
    for elem in root.iter():
        tag = _local_tag(elem)
        if tag == "ubma_external":
            ext = _parse_bool_text(elem.text)
        elif tag == "ubma_unibiblio":
            uni = _parse_bool_text(elem.text)
    return ext, uni


def fetch_flags(
    session: requests.Session, eprint_id: int, timeout: float
) -> tuple[bool | None, bool | None, str | None]:
    url = EXPORT_URL.format(eprint_id=eprint_id)
    try:
        r = session.get(url, timeout=timeout)
        if r.status_code == 404:
            return None, None, "404"
        r.raise_for_status()
        ext, uni = parse_ubma_flags(r.content)
        return ext, uni, None
    except requests.RequestException as e:
        return None, None, str(e)


def _error_cell_empty(val: object) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return True
    return str(val).strip() == ""


def load_cache(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype={"eprint_id": int})
    out: dict[int, dict] = {}
    for _, row in df.iterrows():
        eid = int(row["eprint_id"])
        err = row.get("ubma_xml_error")
        out[eid] = {
            "eprint_id": eid,
            "ubma_external": row.get("ubma_external"),
            "ubma_unibiblio": row.get("ubma_unibiblio"),
            "ubma_xml_error": None if _error_cell_empty(err) else err,
        }
    return out


def cache_row(eprint_id: int, ext, uni, err) -> dict:
    return {
        "eprint_id": eprint_id,
        "ubma_external": ext,
        "ubma_unibiblio": uni,
        "ubma_xml_error": err,
    }


def save_cache(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch MADOC ubma_* flags from EPrints XML export.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Harvest CSV from madoc.py")
    p.add_argument("--output-merged", type=Path, default=DEFAULT_MERGED, help="Output CSV with flag columns")
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="Per-eprint cache CSV (resume / dedupe)")
    p.add_argument("--sleep", type=float, default=0.15, help="Seconds between HTTP requests")
    p.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout per request")
    p.add_argument("--refresh", action="store_true", help="Re-fetch all IDs (ignore cache values)")
    p.add_argument("--limit", type=int, default=0, help="Only process first N distinct eprint IDs (debug)")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    for col in ("ubma_external", "ubma_unibiblio", "ubma_xml_error"):
        if col in df.columns:
            df = df.drop(columns=[col])
    df["eprint_id"] = df.apply(
        lambda r: madoc_eprint_id_from_row(r.get("identifier"), r.get("file")), axis=1
    )

    ids = sorted({int(x) for x in df["eprint_id"].dropna().unique()})
    if args.limit and args.limit > 0:
        ids = ids[: args.limit]

    rows_by_id: dict[int, dict] = load_cache(args.cache)
    if args.refresh:
        for eid in ids:
            rows_by_id.pop(eid, None)

    def needs_fetch(eid: int) -> bool:
        if eid not in rows_by_id:
            return True
        err = rows_by_id[eid].get("ubma_xml_error")
        if err is None or (isinstance(err, float) and pd.isna(err)):
            return False
        return str(err).strip() != ""

    to_fetch = [eid for eid in ids if needs_fetch(eid)]
    print(f"Distinct eprint IDs in input: {len(ids)}; HTTP fetches this run: {len(to_fetch)}")

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "MadabiMadocFlagFetcher/1.0 (metadata harvest; institutional research)"}
    )

    for n, eid in enumerate(tqdm(to_fetch, desc="ubma XML"), start=1):
        ext, uni, err = fetch_flags(session, eid, args.timeout)
        rows_by_id[eid] = cache_row(eid, ext, uni, err)
        if args.sleep > 0:
            time.sleep(args.sleep)
        if n % 200 == 0:
            save_cache(args.cache, [rows_by_id[i] for i in sorted(rows_by_id)])

    save_cache(args.cache, [rows_by_id[i] for i in sorted(rows_by_id)])

    flag_df = pd.DataFrame([rows_by_id[i] for i in sorted(rows_by_id)])
    merged = df.merge(flag_df, on="eprint_id", how="left")
    args.output_merged.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_merged, index=False)
    print(f"Wrote {len(merged)} rows to {args.output_merged}")
    print(f"Cache: {args.cache} ({len(rows_by_id)} eprint IDs)")


if __name__ == "__main__":
    main()
