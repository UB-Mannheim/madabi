import subprocess
import ast
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdf"
MADOC_CSV = DATA_DIR / "madoc_with_ubma.csv"
PDF_URLS = DATA_DIR / "pdf_urls.txt"
ARIA2_LOG = DATA_DIR / "aria2c_download.log"

PDF_DIR.mkdir(parents=True, exist_ok=True)

# Source metadata harvested via code/harvester/madoc.py
if not MADOC_CSV.exists():
    raise FileNotFoundError(
        f"Required enriched MADOC table not found: {MADOC_CSV}. "
        "Run code/extraction/fetch_madoc_ubma_flags.py first."
    )

print(f"Using MADOC table: {MADOC_CSV.name}")
df = pd.read_csv(MADOC_CSV)
if "ubma_unibiblio" not in df.columns:
    raise KeyError(
        f"{MADOC_CSV} does not contain 'ubma_unibiblio'. "
        "Please regenerate it with code/extraction/fetch_madoc_ubma_flags.py."
    )

sdf = df[(df.rights == "open access") & (df.resourceType == "journal article")]

def _flag_is_true(v: object) -> bool:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    if isinstance(v, bool):
        return v is True
    return str(v).strip().lower() == "true"


# Keep only records that are explicitly marked as part of the university bibliography.
before = len(sdf)
sdf = sdf[sdf["ubma_unibiblio"].map(_flag_is_true)]
skipped = before - len(sdf)
print(
    "Keeping only rows with ubma_unibiblio=True "
    f"and skipping {skipped} other rows."
)

def _iter_urls(v: object):
    """
    MADOC's `file` field is usually a single URL string but can occasionally
    be a stringified python list like "['url1', 'url2']".
    """
    if v is None:
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
        except Exception:
            pass
    yield s


# Export URLs (one URL per line)
urls: list[str] = []
for v in sdf["file"].dropna().tolist():
    for u in _iter_urls(v):
        urls.append(u.strip())

PDF_URLS.write_text("\n".join(urls) + "\n", encoding="utf-8")

# aria2c command with robust flags.
# Note: if some downloads fail, aria2c exits non-zero; we still keep partial successes
# and write a log file so you can retry/resume.
cmd = [
    'aria2c',
    # Laptop-friendly concurrency defaults (server runs can bump these if needed)
    '-x', '8',
    '-s', '8',
    '-j', '2',
    '-i', str(PDF_URLS),
    '-d', str(PDF_DIR),
    '-l', str(ARIA2_LOG),
    # Work around occasional WinTLS/SChannel handshake issues on Windows.
    # MADOC is a public HTTPS endpoint; this avoids spurious aborts for a small
    # number of URLs that otherwise return HTTP 200 via other clients.
    '--check-certificate=false',
    '--timeout=30',
    '--connect-timeout=15',
    '--max-tries=3',
    '--retry-wait=5',
    '--always-resume=true',
    # If a file exists but its .aria2 control file doesn't, aria2 otherwise aborts
    # (errorCode=13) to avoid truncating an existing file to 0 bytes.
    '--allow-overwrite=true',
    '--auto-file-renaming=false'
]

try:
    res = subprocess.run(cmd, check=False)
    if res.returncode == 0:
        print("✅ All downloads completed.")
    else:
        print(
            "⚠️ aria2c finished with errors (some downloads may have failed). "
            f"Exit code: {res.returncode}. See log: {ARIA2_LOG}"
        )
except FileNotFoundError:
    print("❌ aria2c not found. Please install aria2 and ensure aria2c is on PATH.")

