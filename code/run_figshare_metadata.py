"""
Fetch Figshare metadata for Figshare URLs listed in extracted_repositories.csv.

Supports:
- https://figshare.com/articles/.../<article_id>
- https://figshare.com/datasets/.../<article_id>
- https://doi.org/10.6084/m9.figshare.<article_id>[.vN]

For each unique Figshare article_id, calls the public Figshare API v2:
- Article details: https://api.figshare.com/v2/articles/{article_id}
- Article files:   https://api.figshare.com/v2/articles/{article_id}/files

Writes one JSON per article under data/from_papers/figshare_metadata/ and appends to
data/from_papers/figshare_metadata/manifest.csv.

API base: https://api.figshare.com/v2
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


FIGSHARE_DOI_RE = re.compile(r"10\.6084/m9\.figshare\.(\d+)(?:\.v\d+)?", re.IGNORECASE)
FIGSHARE_ID_IN_URL_RE = re.compile(r"figshare\.com/(?:articles|datasets)/.+?/(\d+)(?:[/?#].*)?$", re.IGNORECASE)


def extract_figshare_article_id(url: str) -> Optional[str]:
    u = (url or "").strip()
    if not u:
        return None

    m = FIGSHARE_DOI_RE.search(u)
    if m:
        return m.group(1)

    m = FIGSHARE_ID_IN_URL_RE.search(u)
    if m:
        return m.group(1)

    return None


@dataclass(frozen=True)
class FigshareResult:
    status: str  # ok | not_found | forbidden | error
    api_url: str
    error: str = ""
    data: Any = None


def figshare_get_json(api_url: str, timeout: int) -> requests.Response:
    # Public endpoints don't require auth; send an explicit UA to be polite.
    headers = {"User-Agent": "madabi-figshare-metadata/1.0"}
    return requests.get(api_url, headers=headers, timeout=timeout)


def fetch_figshare_article(article_id: str, timeout: int) -> FigshareResult:
    api_url = f"https://api.figshare.com/v2/articles/{article_id}"
    try:
        resp = figshare_get_json(api_url, timeout=timeout)
    except requests.RequestException as e:
        return FigshareResult(status="error", api_url=api_url, error=f"request_error: {e.__class__.__name__}: {e}")

    if resp.status_code == 200:
        try:
            return FigshareResult(status="ok", api_url=api_url, data=resp.json())
        except Exception as e:
            return FigshareResult(status="error", api_url=api_url, error=f"json_decode_error: {e}")

    if resp.status_code in (401, 403):
        return FigshareResult(status="forbidden", api_url=api_url, error=f"http_{resp.status_code}")
    if resp.status_code == 404:
        return FigshareResult(status="not_found", api_url=api_url, error="http_404")
    return FigshareResult(status="error", api_url=api_url, error=f"http_{resp.status_code}")


def fetch_figshare_files(article_id: str, timeout: int) -> FigshareResult:
    api_url = f"https://api.figshare.com/v2/articles/{article_id}/files"
    try:
        resp = figshare_get_json(api_url, timeout=timeout)
    except requests.RequestException as e:
        return FigshareResult(status="error", api_url=api_url, error=f"request_error: {e.__class__.__name__}: {e}")

    if resp.status_code == 200:
        try:
            return FigshareResult(status="ok", api_url=api_url, data=resp.json())
        except Exception as e:
            return FigshareResult(status="error", api_url=api_url, error=f"json_decode_error: {e}")

    if resp.status_code in (401, 403):
        return FigshareResult(status="forbidden", api_url=api_url, error=f"http_{resp.status_code}")
    if resp.status_code == 404:
        return FigshareResult(status="not_found", api_url=api_url, error="http_404")
    return FigshareResult(status="error", api_url=api_url, error=f"http_{resp.status_code}")


def safe_filename(article_id: str) -> str:
    return f"figshare_{article_id}.json"


def load_figshare_ids(csv_path: Path) -> list[tuple[str, str]]:
    """
    Returns list of (original_url, article_id), deduped by article_id.
    """
    seen: set[str] = set()
    out: list[tuple[str, str]] = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            repo = (row.get("repository") or "").strip()
            raw = (row.get("url") or "").strip()
            if not raw:
                continue
            if repo != "Figshare" and "figshare" not in raw.lower():
                continue
            article_id = extract_figshare_article_id(raw)
            if not article_id:
                continue
            if article_id in seen:
                continue
            seen.add(article_id)
            out.append((raw, article_id))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch Figshare metadata for Figshare links.")
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("data/from_papers/extracted_repositories.csv"),
        help="CSV from analyze_from_papers (repository,url,...).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/from_papers/figshare_metadata"),
        help="Output directory for per-article JSON and manifest.csv.",
    )
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds (default 60).")
    ap.add_argument("--delay", type=float, default=1.0, help="Delay seconds between requests (default 1.0).")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N article ids (0 = all).")
    ap.add_argument("--skip-existing", action="store_true", help="Skip if output JSON already exists.")
    ap.add_argument("--dry-run", action="store_true", help="Write manifest rows without calling Figshare API.")
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"Input CSV not found: {args.input.resolve()}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.csv"
    write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0

    try:
        mf = manifest_path.open("a", newline="", encoding="utf-8")
    except PermissionError:
        print(f"Cannot write to {manifest_path} (file likely open/locked). Close it and re-run.", file=sys.stderr)
        return 3

    rows = load_figshare_ids(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]

    fieldnames = [
        "original_url",
        "article_id",
        "api_article_url",
        "api_files_url",
        "output_json",
        "status",
        "error",
        "title",
        "item_type",
        "published_date",
    ]

    with mf:
        w = csv.DictWriter(mf, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        for original_url, article_id in tqdm(rows, desc="Figshare API"):
            out_path = args.out_dir / safe_filename(article_id)
            out_rel = out_path.as_posix()

            if args.skip_existing and out_path.is_file():
                w.writerow(
                    {
                        "original_url": original_url,
                        "article_id": article_id,
                        "api_article_url": "",
                        "api_files_url": "",
                        "output_json": out_rel,
                        "status": "skipped_exists",
                        "error": "",
                        "title": "",
                        "item_type": "",
                        "published_date": "",
                    }
                )
                mf.flush()
                continue

            if args.dry_run:
                w.writerow(
                    {
                        "original_url": original_url,
                        "article_id": article_id,
                        "api_article_url": "",
                        "api_files_url": "",
                        "output_json": out_rel,
                        "status": "dry_run",
                        "error": "",
                        "title": "",
                        "item_type": "",
                        "published_date": "",
                    }
                )
                mf.flush()
                continue

            article_res = fetch_figshare_article(article_id, timeout=args.timeout)
            files_res = fetch_figshare_files(article_id, timeout=args.timeout)

            status = "ok" if (article_res.status == "ok") else article_res.status
            error = article_res.error

            payload = {
                "article_id": article_id,
                "original_url": original_url,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "api": {
                    "article_url": article_res.api_url,
                    "files_url": files_res.api_url,
                },
                "article_status": article_res.status,
                "files_status": files_res.status,
                "article_error": article_res.error,
                "files_error": files_res.error,
                "article": article_res.data,
                "files": files_res.data,
            }

            if article_res.status == "ok":
                out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            title = ""
            item_type = ""
            published_date = ""
            if isinstance(article_res.data, dict):
                title = str(article_res.data.get("title") or "")
                item_type = str(article_res.data.get("defined_type") or article_res.data.get("item_type") or "")
                published_date = str(article_res.data.get("published_date") or "")

            w.writerow(
                {
                    "original_url": original_url,
                    "article_id": article_id,
                    "api_article_url": article_res.api_url,
                    "api_files_url": files_res.api_url,
                    "output_json": out_rel,
                    "status": status,
                    "error": error,
                    "title": title,
                    "item_type": item_type,
                    "published_date": published_date,
                }
            )
            mf.flush()

            if args.delay > 0:
                time.sleep(args.delay)

    print(f"Wrote manifest: {manifest_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

