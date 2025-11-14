import pandas as pd
import re
from bs4 import BeautifulSoup
import ast
import html

# Load files
metadata_df = pd.read_csv('../data/unified_mannheim_metadata.csv')

# Precompiled regex patterns
_RE_CTRL          = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
_RE_BSL_ESC       = re.compile(r"\\[nrt]")
_RE_ZWSP          = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060]")
_RE_MD_HEADINGS   = re.compile(r"(?m)^\s*#{1,6}\s*")
_RE_MD_EMPH       = re.compile(r"[*_~`]+")
_RE_WS            = re.compile(r"\s+")
_RE_LISTY_STR     = re.compile(r"^\s*\[\s*['\"]")
_MAX_LEN_DEFAULT  = 10000

_MISSING_TOKEN = "NA"

def _is_missing_after_clean(s: str) -> bool:
    """Decide if a cleaned string should be treated as missing."""
    t = s.strip()
    return t == "" or t == "-"

def clean_description(text, *, max_len: int = _MAX_LEN_DEFAULT) -> str:
    """Cleaning for 'Description'"""
    # Treat None/NaN as missing
    if text is None or pd.isna(text):
        return _MISSING_TOKEN

    # Accept bytes
    if isinstance(text, (bytes, bytearray)):
        try:
            text = text.decode("utf-8", "replace")
        except Exception:
            text = str(text)

    s = str(text).strip()

    if _is_missing_after_clean(s):
        return _MISSING_TOKEN

    # Unwrap list-like strings
    if _RE_LISTY_STR.match(s) and s.rstrip().endswith("]"):
        try:
            parts = ast.literal_eval(s)
            if isinstance(parts, (list, tuple)) and parts:
                first = str(parts[0]).strip()
                if len(first) <= 10 and len(parts) > 1:
                    s = str(parts[1]).strip()
                else:
                    s = first
        except Exception:
            pass

    # Decode HTML entities, remove tags
    s = html.unescape(s)
    s = BeautifulSoup(s, "html.parser").get_text(separator=" ")

    # Remove escapes, control chars, zero-width chars
    s = _RE_BSL_ESC.sub(" ", s)
    s = _RE_CTRL.sub(" ", s)
    s = _RE_ZWSP.sub("", s)

    # Remove light markdown artifacts
    s = _RE_MD_HEADINGS.sub("", s)
    s = _RE_MD_EMPH.sub("", s)

    # Collapse whitespace
    s = _RE_WS.sub(" ", s).strip()

    if _is_missing_after_clean(s):
        return _MISSING_TOKEN

    if max_len and len(s) > max_len:
        s = s[:max_len].rstrip() + "…"

    if s.lower() == "nan":
        return _MISSING_TOKEN

    return s

def clean_title(text) -> str:
    """Cleaning for 'Title'"""
    if text is None or pd.isna(text):
        return _MISSING_TOKEN

    if isinstance(text, (bytes, bytearray)):
        try:
            text = text.decode("utf-8", "replace")
        except Exception:
            text = str(text)

    s = str(text)

    # Replace line breaks/tabs with space
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")

    # Remove control and zero-width characters
    s = _RE_CTRL.sub(" ", s)
    s = _RE_ZWSP.sub("", s)

    # Collapse multiple spaces
    s = _RE_WS.sub(" ", s).strip()

    # Treat "-" as missing
    if _is_missing_after_clean(s):
        return _MISSING_TOKEN

    if s.lower() == "nan":
        return _MISSING_TOKEN

    return s

# Apply cleaning functions
metadata_df["Description"] = metadata_df["Description"].apply(clean_description)
metadata_df["Title"] = metadata_df["Title"].apply(clean_title)

# Ensure NA is consistent
metadata_df["Description"] = metadata_df["Description"].fillna(_MISSING_TOKEN)

# Save to CSV
metadata_df.to_csv('../data/unified_mannheim_metadata_cleaned.csv', index=False)
# Save your DataFrame to JSON
metadata_df.to_json("../data/unified_mannheim_metadata_cleaned.json", orient="records")

print("Cleaned unified metadata saved to ../data/unified_mannheim_metadata_cleaned.csv")
