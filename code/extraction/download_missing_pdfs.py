"""
Download a small set of PDFs from a URL list.

Why: A few MADOC PDF URLs can fail in aria2c on Windows (e.g., due to very long
filenames or WinTLS issues). This helper downloads the remaining URLs using
Python + requests and writes them into `data/pdf/` with a safe filename.

Input:
- data/missing_pdf_urls.txt  (one URL per line)

Output:
- data/pdf/<safe_filename>.pdf
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import urlsplit

import requests


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdf"
IN_PATH = DATA_DIR / "missing_pdf_urls.txt"
MAP_PATH = DATA_DIR / "missing_pdf_url_to_filename.csv"


def _safe_filename_from_url(url: str) -> str:
    """
    Convert URL basename to a Windows-safe filename.
    If it is too long, shorten and append a hash.
    """
    name = Path(urlsplit(url).path).name
    name = name.strip()
    if not name.lower().endswith(".pdf"):
        name = name + ".pdf"

    # Replace problematic characters for Windows filesystems
    name = re.sub(r'[<>:"/\\\\|?*]', "_", name)
    name = re.sub(r"\s+", " ", name).strip()

    # Keep filenames comfortably below typical Windows path limits
    MAX = 140
    if len(name) > MAX:
        h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        stem = name[:-4]
        stem = stem[: MAX - len(h) - 1 - 4].rstrip(" ._")
        name = f"{stem}_{h}.pdf"
    return name


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)


def main() -> int:
    if not IN_PATH.exists():
        raise SystemExit(f"Missing {IN_PATH}. Create it first (e.g. from the missing URL check).")

    urls = [l.strip() for l in IN_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not urls:
        print(f"No URLs in {IN_PATH}")
        return 0

    ok = 0
    failed: list[str] = []
    mappings: list[tuple[str, str]] = []
    for url in urls:
        fn = _safe_filename_from_url(url)
        out_path = PDF_DIR / fn
        mappings.append((url, fn))
        try:
            if out_path.exists() and out_path.stat().st_size > 0:
                ok += 1
                continue
            print(f"Downloading {url} -> {out_path.name}")
            download(url, out_path)
            ok += 1
        except Exception as e:
            failed.append(f"{url}\t{e.__class__.__name__}: {e}")

    print(f"Downloaded/available: {ok}/{len(urls)}")
    # Write mapping (URL -> local filename) for downstream joins.
    MAP_PATH.write_text(
        "url,filename\n" + "\n".join([f"{u},{fn}" for u, fn in mappings]) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote URL->filename map to {MAP_PATH}")
    if failed:
        fail_path = DATA_DIR / "missing_pdf_download_failures.txt"
        fail_path.write_text("\n".join(failed) + "\n", encoding="utf-8")
        print(f"Failures written to {fail_path}")
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

