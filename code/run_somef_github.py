"""
Batch SOMEF extraction for GitHub URLs listed in extracted_repositories.csv.

Reads repository == "GitHub", normalizes each URL to https://github.com/{owner}/{repo},
runs `somef describe` per repo, and writes one JSON per repo plus a manifest CSV.

Prerequisite (once per machine / venv):  somef configure -a

Uses the `somef` console script next to the current Python (e.g. .venv/Scripts/somef.exe).
Do not use `python -m somef` with somef 0.8.x — the package __main__ does not invoke the CLI.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


GITHUB_REPO_RE = re.compile(
    r"https?://github\.com/([^/]+)/([^/?#]+)",
    re.IGNORECASE,
)


def repo_root_from_url(url: str) -> str | None:
    """Return canonical https://github.com/owner/repo or None if not a repo URL."""
    u = url.strip()
    if u.endswith(".git"):
        u = u[:-4]
    m = GITHUB_REPO_RE.match(u)
    if not m:
        return None
    owner, repo = m.group(1), m.group(2)
    return f"https://github.com/{owner}/{repo}"


def safe_filename(repo_url: str) -> str:
    """owner__repo.json"""
    m = GITHUB_REPO_RE.match(repo_url)
    if not m:
        h = abs(hash(repo_url)) % (10**9)
        return f"unknown_{h}.json"
    return f"{m.group(1)}__{m.group(2)}.json"


def is_usable_somef_json(path: Path) -> bool:
    """True if the file looks like a successful SOMEF output (non-empty JSON object/array)."""
    if not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    if not text or text == "null":
        return False
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return False
    if data is None:
        return False
    if isinstance(data, dict) and len(data) == 0:
        return False
    if isinstance(data, list) and len(data) == 0:
        return False
    return True


def find_somef_executable() -> Path | None:
    root = Path(sys.executable).resolve().parent
    name = "somef.exe" if sys.platform == "win32" else "somef"
    candidate = root / name
    if candidate.is_file():
        return candidate
    import shutil

    w = shutil.which("somef")
    return Path(w) if w else None


def load_github_repo_urls(csv_path: Path) -> list[tuple[str, str]]:
    """Return list of (original_url, normalized_repo_url), deduped by normalized URL."""
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("repository", "").strip() != "GitHub":
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SOMEF on GitHub repos from CSV.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/from_papers/extracted_repositories.csv"),
        help="CSV from analyze_from_papers (repository, url, ...).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/from_papers/somef_github"),
        help="Directory for per-repo JSON and manifest.csv.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.6,
        help="SOMEF classification threshold (passed to somef describe -t).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between repos (helps with unauthenticated GitHub rate limits).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Max seconds to allow SOMEF per repo (default 3600).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N repos (0 = all).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if output JSON exists and looks like a successful SOMEF run (not null, "
        "non-empty object/array). Retries repos with missing, null, empty, or invalid JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List repos that would be processed, do not call somef.",
    )
    parser.add_argument(
        "--readme-only",
        action="store_true",
        help="Pass SOMEF -ro: only fetch README (avoids full ZIP extract on Windows when "
        "repos contain paths invalid on NTFS, e.g. filenames with carriage returns).",
    )
    parser.add_argument(
        "--fallback-readme-only",
        action="store_true",
        default=True,
        help="If full-repo extraction fails, retry the same repo with SOMEF -ro (README-only). "
        "Enabled by default; pass --no-fallback-readme-only to disable.",
    )
    parser.add_argument(
        "--no-fallback-readme-only",
        action="store_false",
        dest="fallback_readme_only",
        help="Disable README-only fallback retry after a failure.",
    )
    args = parser.parse_args()

    somef = find_somef_executable()
    if not somef and not args.dry_run:
        print(
            "Could not find `somef` executable. Install with: pip install somef\n"
            "Then run once: somef configure -a",
            file=sys.stderr,
        )
        return 2

    if not args.input.is_file():
        print(f"Input CSV not found: {args.input.resolve()}", file=sys.stderr)
        return 1

    pairs = load_github_repo_urls(args.input)
    if args.limit > 0:
        pairs = pairs[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.csv"

    fieldnames = [
        "original_url",
        "repo_url",
        "output_json",
        "somef_mode",
        "status",
        "error",
    ]

    def classify_failure(output_text: str) -> str:
        t = output_text.lower()
        if "repository name is private or incorrect" in t:
            return "private_or_incorrect"
        if "api rate limit exceeded" in t or "rate limit" in t:
            return "rate_limited"
        if "oserror" in t and "invalid argument" in t and "zipfile" in t:
            return "zip_invalid_path"
        if "404" in t or "not found" in t:
            return "not_found"
        return "error"

    def run_somef(repo_url: str, out_path: Path, readme_only: bool) -> tuple[int, str]:
        cmd = [
            str(somef),
            "describe",
            "-r",
            repo_url,
            "-o",
            str(out_path),
            "-t",
            str(args.threshold),
            "-p",
        ]
        if readme_only:
            cmd.append("-ro")
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return proc.returncode, combined

    def is_null_json(p: Path) -> bool:
        try:
            return p.read_text(encoding="utf-8").strip() == "null"
        except Exception:
            return False

    # Keep one stable manifest.csv and append on each run.
    # If the file is open/locked (common with Excel on Windows), fail fast instead of creating many manifests.
    write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0
    try:
        mf = manifest_path.open("a", newline="", encoding="utf-8")
    except PermissionError:
        print(
            f"Cannot write to {manifest_path} (file is likely open/locked). "
            "Close it (e.g. Excel) and re-run.",
            file=sys.stderr,
        )
        return 3

    with mf:
        writer = csv.DictWriter(mf, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for raw, norm in tqdm(pairs, desc="SOMEF GitHub"):
            out_name = safe_filename(norm)
            out_path = args.out_dir / out_name

            out_rel = out_path.as_posix()

            if args.skip_existing and is_usable_somef_json(out_path):
                writer.writerow(
                    {
                        "original_url": raw,
                        "repo_url": norm,
                        "output_json": out_rel,
                        "somef_mode": "",
                        "status": "skipped_exists",
                        "error": "",
                    }
                )
                mf.flush()
                continue

            if args.dry_run:
                writer.writerow(
                    {
                        "original_url": raw,
                        "repo_url": norm,
                        "output_json": out_rel,
                        "somef_mode": "",
                        "status": "dry_run",
                        "error": "",
                    }
                )
                continue

            try:
                # Prefer richer extraction (full repo). Force README-only only when requested,
                # or when falling back after a failure.
                mode = "readme_only" if args.readme_only else "full"
                rc, output = run_somef(norm, out_path, readme_only=args.readme_only)
            except subprocess.TimeoutExpired:
                err = f"timeout after {args.timeout}s"
                writer.writerow(
                    {
                        "original_url": raw,
                        "repo_url": norm,
                        "output_json": out_rel,
                        "somef_mode": "",
                        "status": "error",
                        "error": err,
                    }
                )
                mf.flush()
                continue

            out_tail = (output or "")[-4000:]
            # One line for CSV tools (full traceback is multiline); keep the tail with the real error.
            out_one_line = " ".join(out_tail.splitlines()) if out_tail else ""

            needs_fallback = (
                (rc != 0 or not out_path.is_file() or is_null_json(out_path))
                and (not args.readme_only)
                and args.fallback_readme_only
            )
            if needs_fallback:
                # Retry with README-only to avoid ZIP extraction issues / heavy repo fetch.
                mode = "fallback_readme_only"
                rc2, output2 = run_somef(norm, out_path, readme_only=True)
                rc, output = rc2, output2
                out_tail = (output or "")[-4000:]
                out_one_line = " ".join(out_tail.splitlines()) if out_tail else ""

            status = "ok"
            if rc != 0 or not out_path.is_file() or is_null_json(out_path):
                status = classify_failure(output or "")

            if status != "ok":
                writer.writerow(
                    {
                        "original_url": raw,
                        "repo_url": norm,
                        "output_json": out_rel,
                        "somef_mode": mode,
                        "status": status,
                        "error": out_one_line or f"exit {rc}",
                    }
                )
            else:
                writer.writerow(
                    {
                        "original_url": raw,
                        "repo_url": norm,
                        "output_json": out_rel,
                        "somef_mode": mode,
                        "status": "ok",
                        "error": "",
                    }
                )
            mf.flush()

            if args.delay > 0:
                time.sleep(args.delay)

    print(f"Wrote manifest: {manifest_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
