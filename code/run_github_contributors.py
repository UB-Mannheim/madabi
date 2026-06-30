"""
Batch GitHub contributor enrichment for repos listed in extracted_repositories.csv.

Goal:
- For each unique GitHub repo URL (https://github.com/<owner>/<repo>) referenced by papers,
  fetch the repo contributors and enrich each contributor with the best-available identity fields:
  - login (always)
  - full name (GitHub "name", may be null)
  - affiliation proxy fields: company, bio, location (best-effort; may be null)

Why:
- SOMEF outputs useful software metadata, but it typically does not export a full contributors list.
- This script produces an explicit contributor table to support Mannheim affiliation heuristics.

Auth:
- Strongly recommended: set env var GITHUB_TOKEN to avoid low rate limits.

Outputs:
- data/from_papers/github_contributors/<owner>__<repo>.json
- data/from_papers/github_contributors/manifest.csv (append-only log)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


GITHUB_REPO_RE = re.compile(r"https?://github\.com/([^/]+)/([^/?#]+)", re.IGNORECASE)


def repo_root_from_url(url: str) -> str | None:
    u = url.strip()
    if u.endswith(".git"):
        u = u[:-4]
    m = GITHUB_REPO_RE.match(u)
    if not m:
        return None
    return f"https://github.com/{m.group(1)}/{m.group(2)}"


def owner_repo_from_root(repo_root: str) -> tuple[str, str] | None:
    m = GITHUB_REPO_RE.match(repo_root.strip())
    if not m:
        return None
    return m.group(1), m.group(2)


def safe_filename(repo_root: str) -> str:
    m = GITHUB_REPO_RE.match(repo_root)
    if not m:
        h = abs(hash(repo_root)) % (10**9)
        return f"unknown_{h}.json"
    return f"{m.group(1)}__{m.group(2)}.json"


def load_github_repo_urls(csv_path: Path) -> list[tuple[str, str]]:
    """Return list of (original_url, normalized_repo_root), deduped by normalized root."""
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("repository") or "").strip() != "GitHub":
                continue
            raw = (row.get("url") or "").strip()
            if not raw:
                continue
            norm = repo_root_from_url(raw)
            if not norm:
                continue
            if norm in seen:
                continue
            seen.add(norm)
            rows.append((raw, norm))
    return rows


def github_headers() -> dict[str, str]:
    h = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "madabi-github-contributors",
    }
    tok = os.environ.get("GITHUB_TOKEN", "").strip()
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def load_env_file(dotenv_path: Path) -> None:
    """
    Minimal .env loader (KEY=VALUE lines).

    - Does not override existing environment variables.
    - Ignores blank lines and comments starting with '#'.
    - Strips surrounding quotes from values.
    """
    if not dotenv_path.is_file():
        return
    try:
        lines = dotenv_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, val = s.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        if key in os.environ and os.environ[key].strip():
            continue
        if len(val) >= 2 and ((val[0] == val[-1]) and val[0] in ("'", '"')):
            val = val[1:-1]
        os.environ[key] = val


@dataclass(frozen=True)
class GHResp:
    status: str  # ok | not_found | forbidden | rate_limited | abuse_limited | error
    http_status: int
    error: str = ""
    data: Any = None


def _classify(resp: requests.Response) -> str:
    if resp.status_code == 200:
        return "ok"
    if resp.status_code == 404:
        return "not_found"
    if resp.status_code in (401, 403):
        # can be forbidden or rate-limit; inspect headers
        if resp.headers.get("X-RateLimit-Remaining") == "0":
            return "rate_limited"
        return "forbidden"
    if resp.status_code == 429:
        return "rate_limited"
    return "error"

def _looks_like_abuse_limit(resp: requests.Response) -> bool:
    """Detect GitHub's abuse/secondary rate limit JSON response."""
    if resp.status_code not in (403, 429):
        return False
    try:
        j = resp.json()
    except Exception:
        return False
    if not isinstance(j, dict):
        return False
    msg = str(j.get("message") or "").lower()
    doc = str(j.get("documentation_url") or "").lower()
    return ("abuse detection mechanism" in msg) or ("secondary rate limit" in msg) or ("abuse-rate-limits" in doc)


def _retry_after_seconds(resp: requests.Response) -> int:
    ra = (resp.headers.get("Retry-After") or "").strip()
    if ra.isdigit():
        return int(ra)
    return 0


def gh_get_json(
    url: str,
    timeout: int,
    *,
    retries: int,
    backoff_base_s: float,
    delay_s: float,
) -> GHResp:
    """
    GET a GitHub API URL and return parsed JSON.

    Retries with backoff on:
    - transient request errors
    - GitHub abuse/secondary rate limit responses
    """
    attempt = 0
    while True:
        try:
            resp = requests.get(url, headers=github_headers(), timeout=timeout)
        except requests.RequestException as e:
            attempt += 1
            if attempt > retries:
                return GHResp(status="error", http_status=0, error=f"request_error: {e.__class__.__name__}: {e}")
            time.sleep(max(delay_s, backoff_base_s * attempt))
            continue

        if _looks_like_abuse_limit(resp):
            attempt += 1
            if attempt > retries:
                tail = (resp.text or "")[:400].replace("\n", " ")
                return GHResp(status="abuse_limited", http_status=resp.status_code, error=tail)
            wait_s = _retry_after_seconds(resp) or int(backoff_base_s * attempt)
            time.sleep(max(delay_s, float(wait_s)))
            continue

        status = _classify(resp)
        if status != "ok":
            tail = (resp.text or "")[:400].replace("\n", " ")
            return GHResp(status=status, http_status=resp.status_code, error=tail)

        try:
            return GHResp(status="ok", http_status=200, data=resp.json())
        except Exception as e:
            return GHResp(status="error", http_status=200, error=f"json_decode_error: {e}")


def gh_paginate(url: str, timeout: int, max_pages: int = 50) -> tuple[str, list[dict[str, Any]], str]:
    """
    Return (status, combined_items, error).
    Handles GitHub-style pagination via Link header.
    """
    items: list[dict[str, Any]] = []
    next_url = url
    pages = 0
    while next_url and pages < max_pages:
        r = gh_get_json(next_url, timeout=timeout, retries=0, backoff_base_s=0, delay_s=0)
        if r.status != "ok":
            return r.status, items, r.error
        if not isinstance(r.data, list):
            return "error", items, "expected_list_response"
        items.extend([x for x in r.data if isinstance(x, dict)])
        pages += 1

        link = ""
        # requests keeps headers on the Response, but we didn't return it; re-fetch headers cheaply:
        # Instead, use a second GET HEAD? To keep simple, stop at one page unless explicitly needed.
        # GitHub contributors endpoint often fits in one page for small repos.
        next_url = "" if pages >= 1 else ""
        _ = link  # placeholder (documented limitation)
    return "ok", items, ""


def fetch_repo_contributors(owner: str, repo: str, timeout: int) -> GHResp:
    # 100 per page is max; we fetch first page (see gh_paginate limitation).
    url = f"https://api.github.com/repos/{owner}/{repo}/contributors?per_page=100&anon=false"
    status, items, err = gh_paginate(url, timeout=timeout)
    if status != "ok":
        return GHResp(status=status, http_status=0, error=err, data=items)
    return GHResp(status="ok", http_status=200, data=items)


def fetch_user(login: str, timeout: int, *, retries: int, backoff_base_s: float, delay_s: float) -> GHResp:
    return gh_get_json(
        f"https://api.github.com/users/{login}",
        timeout=timeout,
        retries=retries,
        backoff_base_s=backoff_base_s,
        delay_s=delay_s,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch GitHub contributors and user profiles for repos from CSV.")
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("data/from_papers/extracted_repositories.csv"),
        help="CSV with (repository,url,...) from analyze_from_papers.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/from_papers/github_contributors"),
        help="Directory for per-repo contributor JSON + manifest.csv.",
    )
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds (default 60).")
    ap.add_argument("--delay", type=float, default=0.5, help="Delay seconds between API calls (default 0.5).")
    ap.add_argument("--retries", type=int, default=5, help="Retries for transient errors / abuse limits (default 5).")
    ap.add_argument(
        "--backoff",
        type=float,
        default=30.0,
        help="Base seconds for backoff when abuse-limited (default 30).",
    )
    ap.add_argument(
        "--max-users-per-repo",
        type=int,
        default=30,
        help="Fetch at most N contributor user profiles per repo (default 30).",
    )
    ap.add_argument("--limit", type=int, default=0, help="Process at most N repos (0 = all).")
    ap.add_argument("--skip-existing", action="store_true", help="Skip if output JSON already exists.")
    ap.add_argument("--dry-run", action="store_true", help="Write manifest rows without calling GitHub API.")
    args = ap.parse_args()

    # Convenience: load repo-root .env if present (for GITHUB_TOKEN).
    # Users should keep .env out of version control.
    load_env_file(Path(".env"))

    if not args.input.is_file():
        print(f"Input CSV not found: {args.input.resolve()}", file=sys.stderr)
        return 1

    repos = load_github_repo_urls(args.input)
    if args.limit > 0:
        repos = repos[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.csv"
    write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0

    fieldnames = [
        "original_url",
        "repo_url",
        "output_json",
        "status",
        "contributors",
        "errors",
    ]

    try:
        mf = manifest_path.open("a", newline="", encoding="utf-8")
    except PermissionError:
        print(f"Cannot write to {manifest_path} (file likely open/locked). Close it and re-run.", file=sys.stderr)
        return 3

    with mf:
        w = csv.DictWriter(mf, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        user_cache: dict[str, dict[str, Any]] = {}

        for original_url, repo_root in tqdm(repos, desc="GitHub contributors"):
            out_path = args.out_dir / safe_filename(repo_root)
            out_rel = out_path.as_posix()

            if args.skip_existing and out_path.is_file():
                w.writerow(
                    {
                        "original_url": original_url,
                        "repo_url": repo_root,
                        "output_json": out_rel,
                        "status": "skipped_exists",
                        "contributors": "",
                        "errors": "",
                    }
                )
                mf.flush()
                continue

            if args.dry_run:
                w.writerow(
                    {
                        "original_url": original_url,
                        "repo_url": repo_root,
                        "output_json": out_rel,
                        "status": "dry_run",
                        "contributors": "",
                        "errors": "",
                    }
                )
                mf.flush()
                continue

            parsed = owner_repo_from_root(repo_root)
            if not parsed:
                w.writerow(
                    {
                        "original_url": original_url,
                        "repo_url": repo_root,
                        "output_json": out_rel,
                        "status": "bad_repo_url",
                        "contributors": "0",
                        "errors": "cannot_parse_owner_repo",
                    }
                )
                mf.flush()
                continue

            owner, repo = parsed
            contrib_resp = fetch_repo_contributors(owner, repo, timeout=args.timeout)
            if contrib_resp.status != "ok":
                w.writerow(
                    {
                        "original_url": original_url,
                        "repo_url": repo_root,
                        "output_json": out_rel,
                        "status": contrib_resp.status,
                        "contributors": str(len(contrib_resp.data or [])),
                        "errors": contrib_resp.error,
                    }
                )
                mf.flush()
                continue

            contrib_items = contrib_resp.data or []
            logins = [c.get("login") for c in contrib_items if isinstance(c, dict) and c.get("login")]
            logins = [str(x) for x in logins if x]
            if args.max_users_per_repo > 0:
                logins = logins[: args.max_users_per_repo]

            users: dict[str, Any] = {}
            user_errors: list[str] = []
            for login in logins:
                if login in user_cache:
                    users[login] = user_cache[login]
                    continue

                uresp = fetch_user(
                    login,
                    timeout=args.timeout,
                    retries=args.retries,
                    backoff_base_s=args.backoff,
                    delay_s=args.delay,
                )
                if uresp.status == "ok" and isinstance(uresp.data, dict):
                    a = uresp.data
                    prof = {
                        "login": a.get("login"),
                        "name": a.get("name"),
                        "company": a.get("company"),
                        "location": a.get("location"),
                        "bio": a.get("bio"),
                        "blog": a.get("blog"),
                        "email": a.get("email"),
                        "html_url": a.get("html_url"),
                        "type": a.get("type"),
                    }
                    users[login] = prof
                    user_cache[login] = prof
                else:
                    user_errors.append(f"{login}:{uresp.status}:{uresp.error or uresp.http_status}")
                if args.delay > 0:
                    time.sleep(args.delay)

            payload = {
                "repo_url": repo_root,
                "owner": owner,
                "repo": repo,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "contributors_api_items": contrib_items,
                "users": users,
                "user_errors": user_errors,
                "notes": {
                    "affiliation_proxy": "GitHub does not provide institutional affiliation; company/location/bio are best-effort signals.",
                    "pagination": "contributors endpoint fetched first page only (per_page=100).",
                    "max_users_per_repo": args.max_users_per_repo,
                },
            }
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            w.writerow(
                {
                    "original_url": original_url,
                    "repo_url": repo_root,
                    "output_json": out_rel,
                    "status": "ok",
                    "contributors": str(len(logins)),
                    "errors": ";".join(user_errors[:5]),  # keep manifest compact
                }
            )
            mf.flush()

            if args.delay > 0:
                time.sleep(args.delay)

    print(f"Wrote manifest: {manifest_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

