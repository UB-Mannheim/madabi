import subprocess
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdf"
MADOC_CSV = DATA_DIR / "madoc.csv"
PDF_URLS = DATA_DIR / "pdf_urls.txt"

PDF_DIR.mkdir(parents=True, exist_ok=True)

# Source metadata harvested via code/harvester/madoc.py
df = pd.read_csv(MADOC_CSV)
sdf = df[(df.rights == "open access") & (df.resourceType == "journal article")]

# Export URLs from DataFrame
sdf['file'].dropna().to_csv(PDF_URLS, index=False, header=False)

# aria2c command with robust flags
cmd = [
    'aria2c',
    '-x', '16',
    '-s', '16',
    '-j', '8',
    '-i', str(PDF_URLS),
    '-d', str(PDF_DIR),
    '--timeout=30',
    '--connect-timeout=15',
    '--max-tries=3',
    '--retry-wait=5',
    '--always-resume=true',
    '--auto-file-renaming=false'
]

try:
    subprocess.run(cmd, check=True)
    print("✅ All downloads completed.")
except subprocess.CalledProcessError as e:
    print("❌ Download encountered an error:", e)

