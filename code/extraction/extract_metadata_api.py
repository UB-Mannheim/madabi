"""
Extract metadata from papers via an OpenAI-compatible chat API, using custom LLM
prompts per field (02–03, 05–07, 11–13, 15, 17).

Backends (``--backend``):
  maia   — Uni Mannheim MaIA (default if MAIA_API_KEY is set)
           base URL: https://maia.uni-mannheim.de/ollama/v1
  gwdg   — GWDG Academic Cloud / SAIA (rate-limited quota)
  ollama — local Ollama at http://localhost:11434/v1 (no API key required)

Environment variables (put in project-root ``.env``):
  MAIA_API_KEY              MaIA API key (sk-...)
  CHAT_AI_API_KEY           GWDG / fallback key
  CHAT_AI_BASE_URL          override base URL for any backend
  CHAT_AI_MODEL             override model id

  Run a subset: ``--limit N`` without naming files (default: first N PDFs by sorted filename).
  Add ``--shuffle`` to draw N random PDFs (optional ``--random-seed`` for reproducibility).
  Or use ``--pdf`` for specific files. Use ``--output-tag`` for separate output files.
  Use ``--resume`` to continue a prior run after rate limits (skips PDFs already fully extracted).
  Use ``--batch`` for 3 API calls per PDF (default: 11 calls, one field each).

Embeddings stay local (SentenceTransformer + FAISS) like the other extractors.

Outputs under ``data/from_papers/`` (default stem ``madata_results_api``; use ``--output-tag`` to change):
  madata_results_api.csv / .json / _flat.csv / _flat.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import pandas as pd
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ------------------ Config ------------------
DEFAULT_BASE_URL = "https://chat-ai.academiccloud.de/v1"
DEFAULT_MODEL = "llama-3.3-70b-instruct"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

BACKENDS: dict[str, dict[str, object]] = {
    "maia": {
        "base_url": "https://maia.uni-mannheim.de/ollama/v1",
        "model": "gemma4:31b",
        "api_key_envs": ("MAIA_API_KEY", "CHAT_AI_API_KEY"),
        "requires_key": True,
    },
    "gwdg": {
        "base_url": DEFAULT_BASE_URL,
        "model": DEFAULT_MODEL,
        "api_key_envs": ("CHAT_AI_API_KEY", "ACADEMIC_CLOUD_API_KEY"),
        "requires_key": True,
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "gemma3:12b",
        "api_key_envs": ("OLLAMA_API_KEY",),
        "requires_key": False,
    },
}
# SAIA sometimes sends Retry-After in the thousands of seconds; never sleep longer than this per retry.
MAX_RETRY_WAIT_SECONDS = 90.0
# Retry-After above this is treated as quota exhaustion (hours), not a transient blip.
LONG_RATE_LIMIT_THRESHOLD_SECONDS = 300.0

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"


def _first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _default_backend_name() -> str:
    if os.environ.get("MAIA_API_KEY"):
        return "maia"
    if os.environ.get("CHAT_AI_API_KEY") or os.environ.get("ACADEMIC_CLOUD_API_KEY"):
        return "gwdg"
    return "ollama"


def _resolve_backend_config(
    backend_name: str,
    base_url_override: str | None,
    model_override: str | None,
) -> tuple[str, str, str, bool]:
    if backend_name not in BACKENDS:
        raise SystemExit(f"Unknown backend {backend_name!r}. Choose from: {', '.join(BACKENDS)}")

    cfg = BACKENDS[backend_name]
    base_url = base_url_override or os.environ.get("CHAT_AI_BASE_URL") or str(cfg["base_url"])
    model = model_override or os.environ.get("CHAT_AI_MODEL") or str(cfg["model"])
    api_key = _first_env(*cfg["api_key_envs"])
    requires_key = bool(cfg["requires_key"])
    if requires_key and not api_key:
        env_names = ", ".join(cfg["api_key_envs"])
        raise SystemExit(f"Backend {backend_name!r} requires an API key ({env_names}).")
    return base_url, model, api_key, requires_key


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(BASE_DIR / ".env")


PDF_FOLDER = DATA_DIR / "pdf"
OUTPUT_FOLDER = DATA_DIR / "from_papers"

# ------------------ Queries: short retrieval text vs full LLM instructions ------------------
queries: dict[str, dict[str, str]] = {
    "02 DOI": {
        "retrieval": "article DOI journal DOI publication identifier title page",
        "instruction": (
            "Extract the DOI of the paper/article itself. Do not return dataset DOI, OSF DOI, Dataverse DOI, "
            "or software DOI here. If no article DOI is found, return an empty string."
        ),
    },
    "03 Citation": {
        "retrieval": "article title authors journal conference volume issue pages year DOI",
        "instruction": (
            "Provide the full citation of the paper/article in Chicago style. Use the article authors, year, title, "
            "journal/conference/book, volume, issue, pages, and DOI if available. Do not cite datasets or references."
        ),
    },
    "05 Data Type by Source": {
        "retrieval": "data collected dataset sample study data source methods participants",
        "instruction": (
            "Classify the study data as exactly one of: primary, secondary, mixed, no data, unclear. "
            "Primary means the authors collected/generated the data themselves for this study, for example survey, "
            "experiment, interviews, observations, scraping, manual coding, measurements, or simulations. "
            "Secondary means the authors reused an existing dataset, archive, database, corpus, registry, or data "
            "collected by others. Mixed means both primary and secondary data are central to the study. "
            "No data means the paper is theoretical, conceptual, mathematical, methodological, editorial, or a review "
            "without a dataset analyzed. Do not classify based only on cited literature or references."
        ),
    },
    "06 Mode of Collection": {
        "retrieval": (
            "data collection method survey interview experiment observation recording "
            "coding scraping simulation measurement"
        ),
        "instruction": (
            "Identify the main data collection mode used for the study data. Choose only from: Interview, "
            "Self-administered questionnaire, Focus group, Self-administered writings and/or diaries, Observation, "
            "Experiment, Recording, Automated data extraction, Content coding, Transcription, Compilation/Synthesis, "
            "Summary, Aggregation, Simulation, Measurements and tests, Other, not applicable, unclear. "
            "Return the collection mode, not the data source, instrument, software, or journal name."
        ),
    },
    "07 Type of Instrument": {
        "retrieval": (
            "data collection instrument questionnaire interview guide survey scale "
            "coding scheme task platform API device software tool"
        ),
        "instruction": (
            "Identify the concrete data collection instrument, tool, platform, questionnaire, scale, interview guide, "
            "coding scheme, API, device, software, or technical instrument used to collect/generate the data. "
            "Examples: Qualtrics, questionnaire, interview guide, Likert scale, Twitter API, E-Prime, eye tracker, "
            "coding scheme, Python script. Do not return statistical analysis software unless it was used for data collection."
        ),
    },
    "11 Data Collector": {
        "retrieval": (
            "data collected by authors research assistants survey company panel provider "
            "institution participants recruited"
        ),
        "instruction": (
            "If the study uses primary data, identify who collected or generated the data. Prefer the article authors "
            "if they conducted the survey, experiment, interviews, coding, scraping, measurements, or simulation themselves. "
            "If a survey company, panel provider, archive, institution, research assistants, or external organization collected it, "
            "name them. If the study uses only secondary data, return 'not applicable'. Do not return authors from the reference list "
            "unless they collected the dataset used in this study."
        ),
    },
    "12 Data Source": {
        "retrieval": (
            "data source dataset archive database corpus registry administrative records "
            "platform website secondary data"
        ),
        "instruction": (
            "If the study uses secondary data or mixed data, identify the original dataset, archive, database, corpus, registry, "
            "survey, platform, website, administrative source, or institution from which the data came. "
            "If the study uses only primary data, return 'not applicable'. Do not return analysis software, journal names, "
            "publisher names, or random references."
        ),
    },
    "13 Data DOI": {
        "retrieval": (
            "data availability dataset identifier persistent identifier repository "
            "replication package supplementary data"
        ),
        "instruction": (
            "Extract the DOI or persistent identifier of the dataset, replication package, "
            "or supplementary data used in the study. Prefer identifiers in formats such as "
            "'10.xxxx/...' (DOI), Handle, arXiv ID, or repository record identifiers. "
            "Do not return ordinary website URLs, project pages, journal URLs, or the article DOI "
            "unless it is explicitly also the dataset DOI. "
            "If no dataset identifier is explicitly provided, return an empty string."
        ),
    },
    "15 Data availability statement": {
        "retrieval": (
            "data availability statement data materials available replication package "
            "supplementary materials repository code available"
        ),
        "instruction": (
            "Extract the exact sentence or paragraph describing where the study data, replication materials, "
            "supplementary materials, analysis scripts, or code can be accessed. "
            "Prioritize statements containing repository names or links such as OSF, Dataverse, Zenodo, Figshare, "
            "GitHub, Dryad, ICPSR, supplementary materials, or 'data available'. "
            "Include the full repository URL or identifier if present. "
            "Do NOT extract funding statements, copyright notices, publisher notes, or ordinary references. "
            "If no data/materials availability statement exists, return an empty string."
        ),
    },
    "17 Code citation": {
        "retrieval": (
            "code availability analysis scripts replication scripts software repository "
            "source code materials available"
        ),
        "instruction": (
            "Extract ONLY shared code, analysis scripts, replication scripts, or software repositories "
            "made available by the article authors for this study. Prefer repository links such as OSF, "
            "GitHub, Zenodo, Dataverse, or explicit sentences saying analysis scripts/code are available. "
            "Do NOT return names of statistical software, R packages, Python packages, experiment software, "
            "or tools used for analysis or data collection unless they are part of a shared repository link. "
            "If no shared code/script repository is mentioned, return an empty string."
        ),
    },
}


def field_retrieval(field_name: str) -> str:
    return queries[field_name]["retrieval"]


def field_instruction(field_name: str) -> str:
    return queries[field_name]["instruction"]

# System message for every chat completion.
SYSTEM_PROMPT = (
    "You are a careful metadata extraction assistant for academic papers. "
    "Use only the provided context. Do not use outside knowledge. "
    "Do not guess from the reference list, bibliography, publisher notes, licenses, or unrelated citations. "
    "Extract metadata about the paper itself and the study data actually used in the paper. "
    "Return exactly one valid JSON object with exactly one key: value. "
    "The value should be short and directly answer the question. "
    "Use 'not applicable' when the question asks for something that does not apply to the study, "
    "for example a secondary-data source for a primary-data study, or a data collector for a secondary-data study. "
    "Use 'unclear' when the context suggests the field may exist but the provided context is insufficient to determine it. "
    "Use an empty string only when the requested information is not mentioned at all and neither 'not applicable' nor 'unclear' fits. "
    "Do not include explanations, markdown, evidence, confidence scores, or extra keys. "
    "Repository statements, OSF links, GitHub links, supplementary-material statements, "
    "and data availability sections are highly important and should be extracted preferentially when relevant. "
)

# Three requests per PDF (2 + 5 + 3 fields) when using --batch.
BATCH_GROUPS: dict[str, list[str]] = {
    "batch_1_bibliographic": ["02 DOI", "03 Citation"],
    "batch_2_study_data": [
        "05 Data Type by Source",
        "06 Mode of Collection",
        "07 Type of Instrument",
        "11 Data Collector",
        "12 Data Source",
    ],
    "batch_3_sharing": [
        "13 Data DOI",
        "15 Data availability statement",
        "17 Code citation",
    ],
}

SYSTEM_PROMPT_BATCH = (
    "You are a careful metadata extraction assistant for academic papers. "
    "Use only the provided context. Do not use outside knowledge. "
    "Do not guess from the reference list, bibliography, publisher notes, licenses, or unrelated citations. "
    "Return exactly one valid JSON object. "
    "Use exactly the field ids below as JSON keys (copy them character-for-character). "
    "Each value must be short and directly answer that field's question. "
    "If a question provides allowed labels, choose only from those labels. "
    "Use 'not applicable', 'unclear', or an empty string as instructed in each question. "
    "Do not include explanations, markdown, or extra keys."
)

REPOSITORY_FIELDS = {
    "15 Data availability statement",
    "17 Code citation",
}

REPOSITORY_KEYWORDS = [
    "data availability",
    "availability of data",
    "data and materials",
    "materials availability",
    "open science framework",
    "osf.io",
    "github",
    "dataverse",
    "zenodo",
    "figshare",
    "dryad",
    "icpsr",
    "supplementary material",
    "supplementary information",
    "analysis scripts",
    "replication package",
    "code availability",
]

def split_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)


def build_vector_index(chunks, model):
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks


def search_chunks(query, index, chunks, model, top_k: int = 3):
    query_vec = model.encode([query])
    D, I = index.search(query_vec, top_k)
    return [chunks[i] for i in I[0]]


def remove_sentences_containing(text: str, phrase: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    filtered_sentences = [s for s in sentences if phrase.lower() not in s.lower()]
    return " ".join(filtered_sentences)


def clean_extracted_text(text: str) -> str:
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def _load_pdf_pages(filepath: Path) -> list[str]:
    """Return one string per page. Prefer pdftotext; fall back to PyMuPDF (easier on Windows)."""
    try:
        import pdftotext

        with open(filepath, "rb") as f:
            pdf = pdftotext.PDF(f)
        return list(pdf)
    except ImportError:
        import fitz  # PyMuPDF

        doc = fitz.open(filepath)
        try:
            return [doc.load_page(i).get_text() for i in range(doc.page_count)]
        finally:
            doc.close()


@dataclass
class RateLimitState:
    """Tracks API quota cooldown across calls within one run."""

    blocked_until: float = 0.0
    last_retry_after: float = 0.0

    def wait_if_blocked(self) -> None:
        now = time.monotonic()
        if self.blocked_until > now:
            wait_s = self.blocked_until - now
            print(f"  [rate-limit] waiting {wait_s:.0f}s for quota reset...")
            time.sleep(wait_s)

    def note_retry_after(self, retry_after: float) -> None:
        self.last_retry_after = retry_after
        self.blocked_until = max(self.blocked_until, time.monotonic() + retry_after)


class RateLimitExceeded(Exception):
    def __init__(self, retry_after_seconds: float, message: str = "") -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message or f"API rate limit exceeded; retry after {retry_after_seconds:.0f}s")


def _parse_retry_after(response: requests.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _rate_limit_error_payload(retry_after: float | None, details: str) -> dict:
    payload: dict = {
        "error": "rate_limit_exceeded",
        "details": details[:4000],
    }
    if retry_after is not None:
        payload["retry_after_seconds"] = retry_after
    return payload


def _parse_chat_json_content(content: str) -> dict:
    content = (content or "").strip()
    if not content:
        return {"error": "empty_response"}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        return {"error": "Failed to parse JSON", "raw": content[:2000]}


def call_chat_api(
    session: requests.Session | None,
    base_url: str,
    api_key: str,
    model: str,
    user_prompt: str,
    system_prompt: str,
    timeout: float,
    use_json_object: bool,
    max_retries: int = 4,
    max_retry_wait: float = MAX_RETRY_WAIT_SECONDS,
    rate_limit_state: RateLimitState | None = None,
    long_rate_limit_threshold: float = LONG_RATE_LIMIT_THRESHOLD_SECONDS,
) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "top_p": 0.3,
        "max_tokens": 512,
    }
    if use_json_object:
        body["response_format"] = {"type": "json_object"}

    payload = json.dumps(body)
    post = session.post if session is not None else requests.post
    r = None
    for attempt in range(max_retries + 1):
        if rate_limit_state is not None:
            rate_limit_state.wait_if_blocked()

        try:
            r = post(url, headers=headers, data=payload, timeout=timeout)
        except requests.exceptions.Timeout:
            if attempt >= max_retries:
                return {
                    "error": "request_timeout",
                    "details": f"Read timed out after {timeout}s",
                }
            wait_s = min(max_retry_wait, 2.0**attempt + random.uniform(0, 1.0))
            print(
                f"  [retry] timeout after {timeout}s, sleeping {wait_s:.1f}s "
                f"(try {attempt + 1}/{max_retries + 1})"
            )
            time.sleep(wait_s)
            continue
        except requests.exceptions.RequestException as exc:
            if attempt >= max_retries:
                return {"error": "request_failed", "details": str(exc)[:4000]}
            wait_s = min(max_retry_wait, 2.0**attempt + random.uniform(0, 1.0))
            print(f"  [retry] {exc!r}, sleeping {wait_s:.1f}s (try {attempt + 1}/{max_retries + 1})")
            time.sleep(wait_s)
            continue

        if r.status_code == 200:
            break

        retriable = r.status_code in (429, 503)
        if not retriable or attempt >= max_retries:
            if r.status_code == 429:
                retry_after = _parse_retry_after(r)
                return _rate_limit_error_payload(retry_after, r.text)
            return {"error": f"Request failed: {r.status_code}", "details": r.text[:4000]}

        retry_after = _parse_retry_after(r)
        if retry_after is not None and rate_limit_state is not None:
            rate_limit_state.note_retry_after(retry_after)

        if retry_after is not None and retry_after >= long_rate_limit_threshold:
            hours = retry_after / 3600.0
            print(
                f"  [rate-limit] HTTP 429, quota exhausted for ~{hours:.1f}h "
                f"(Retry-After={retry_after:.0f}s). Stopping instead of retrying."
            )
            raise RateLimitExceeded(
                retry_after,
                f"API quota exhausted; Retry-After={retry_after:.0f}s (~{hours:.1f} hours)",
            )

        if retry_after is not None:
            wait_s = min(max_retry_wait, retry_after)
            if retry_after > max_retry_wait:
                print(
                    f"  [retry] HTTP {r.status_code}, Retry-After={retry_after:.0f}s "
                    f"capped to {max_retry_wait:.0f}s (try {attempt + 1}/{max_retries + 1})"
                )
            else:
                print(
                    f"  [retry] HTTP {r.status_code}, sleeping {wait_s:.1f}s "
                    f"(try {attempt + 1}/{max_retries + 1})"
                )
        else:
            wait_s = min(max_retry_wait, 2.0**attempt + random.uniform(0, 1.0))
            print(
                f"  [retry] HTTP {r.status_code}, sleeping {wait_s:.1f}s "
                f"(try {attempt + 1}/{max_retries + 1})"
            )
        time.sleep(wait_s)

    try:
        data = r.json()
    except json.JSONDecodeError:
        return {"error": "invalid_json_response", "details": r.text[:2000]}

    choices = data.get("choices") or []
    if not choices:
        return {"error": "no_choices", "details": str(data)[:2000]}

    msg = choices[0].get("message") or {}
    content = msg.get("content", "")
    return _parse_chat_json_content(content)


def _is_rate_limit_result(result: dict) -> bool:
    return result.get("error") in ("rate_limit_exceeded",) or (
        result.get("error") == "Request failed: 429"
    )


def _keyword_context(
    pdf_pages: list[str],
    text: str,
    keywords: list[str] | None = None,
    *,
    max_chars: int = 20000,
    max_hits: int = 12,
) -> str:
    kw = keywords if keywords is not None else REPOSITORY_KEYWORDS
    chunks = split_text(text, chunk_size=2000, overlap=100)
    hits = [c for c in chunks if any(k in c.lower() for k in kw)]
    hits = list(dict.fromkeys(hits))

    first_pages = "\n".join(pdf_pages[:2])
    last_pages = "\n".join(pdf_pages[-4:])
    sections = [first_pages, *hits[:max_hits], last_pages]
    return "\n\n".join(sections)[:max_chars]


def _field_context(
    field_name: str,
    pdf: list[str],
    text: str,
    index,
    chunk_list,
    embed_model,
) -> str:
    qn = int(field_name[0:2])

    if qn < 4:
        return "\n".join(pdf[:2])

    if field_name in REPOSITORY_FIELDS:
        return _keyword_context(pdf, text)

    return "\n".join(
        search_chunks(field_retrieval(field_name), index, chunk_list, embed_model, top_k=5)
    )


def _merged_batch_context(
    field_names: list[str],
    pdf: list[str],
    full_text: str,
    index,
    chunk_list,
    embed_model,
) -> str:
    if all(int(name[0:2]) < 4 for name in field_names):
        return "\n".join(pdf[:2])
    if field_names == BATCH_GROUPS["batch_3_sharing"]:
        return _keyword_context(pdf, full_text)
    seen: set[str] = set()
    parts: list[str] = []
    for field_name in field_names:
        for chunk in search_chunks(
            field_retrieval(field_name), index, chunk_list, embed_model, top_k=5
        ):
            if chunk not in seen:
                seen.add(chunk)
                parts.append(chunk)
    return "\n\n---\n\n".join(parts) if parts else "\n".join(pdf[:2])


def _batch_user_prompt(field_names: list[str], context: str) -> str:
    lines = [
        f"Context:\n{context}\n",
        "Answer all of the following. Return one JSON object whose keys are exactly these field ids:\n",
    ]
    for field_name in field_names:
        lines.append(f'- "{field_name}": {field_instruction(field_name)}')
    lines.append("\nReturn only JSON.")
    return "\n".join(lines)


def _parse_batch_response(raw: dict, field_names: list[str]) -> dict[str, dict]:
    if raw.get("error"):
        err = dict(raw)
        return {fn: err.copy() for fn in field_names}

    out: dict[str, dict] = {}
    for field_name in field_names:
        if field_name in raw:
            val = raw[field_name]
            out[field_name] = val if isinstance(val, dict) else {"value": val}
        else:
            alt = field_name.lstrip("0")
            if alt in raw:
                val = raw[alt]
                out[field_name] = val if isinstance(val, dict) else {"value": val}
            else:
                out[field_name] = {
                    "error": "missing_in_batch_response",
                    "raw_keys": list(raw.keys())[:20],
                }
    return out


def _mark_batch_rate_limited(metadata: dict, field_names: list[str]) -> None:
    for fn in field_names:
        if fn not in metadata:
            metadata[fn] = {
                "error": "skipped_due_to_rate_limit",
                "details": "Earlier batch hit API rate limit",
            }


def process_pdf_batch(
    filepath: Path,
    embed_model,
    session: requests.Session | None,
    base_url: str,
    api_key: str,
    chat_model: str,
    timeout: float,
    use_json_object: bool,
    max_retries: int,
    max_retry_wait: float,
    rate_limit_state: RateLimitState,
    long_rate_limit_threshold: float,
) -> tuple[dict, bool]:
    pdf = _load_pdf_pages(filepath)

    full_text = "\n".join(pdf)
    full_text = remove_sentences_containing(full_text, "downloaded from")
    full_text = clean_extracted_text(full_text)

    text_no_refs = re.split(
        r"\n\s*References\s*\n|\n\s*REFERENCES\s*\n",
        full_text,
        maxsplit=1,
    )[0]

    chunks = split_text(text_no_refs)
    index, chunk_list = build_vector_index(chunks, embed_model)

    metadata: dict = {"filename": os.path.basename(filepath)}

    for batch_name, field_names in BATCH_GROUPS.items():
        print(f"  -> Batch {batch_name} ({len(field_names)} fields)...")
        context = _merged_batch_context(
            field_names, pdf, full_text, index, chunk_list, embed_model
        )
        user_prompt = _batch_user_prompt(field_names, context)
        try:
            raw = call_chat_api(
                session,
                base_url,
                api_key,
                chat_model,
                user_prompt,
                SYSTEM_PROMPT_BATCH,
                timeout,
                use_json_object,
                max_retries=max_retries,
                max_retry_wait=max_retry_wait,
                rate_limit_state=rate_limit_state,
                long_rate_limit_threshold=long_rate_limit_threshold,
            )
        except RateLimitExceeded:
            _mark_batch_rate_limited(metadata, field_names)
            skip_upcoming = False
            for _bn, fns in BATCH_GROUPS.items():
                if _bn == batch_name:
                    skip_upcoming = True
                    continue
                if skip_upcoming:
                    _mark_batch_rate_limited(metadata, fns)
            print(f"  [rate-limit] stopping remaining batches for {filepath.name}")
            return metadata, True

        metadata.update(_parse_batch_response(raw, field_names))
        if any(_is_rate_limit_result(metadata.get(fn, {})) for fn in field_names):
            print(f"  [rate-limit] stopping remaining batches for {filepath.name}")
            skip_upcoming = False
            for _bn, fns in BATCH_GROUPS.items():
                if _bn == batch_name:
                    skip_upcoming = True
                    continue
                if skip_upcoming:
                    _mark_batch_rate_limited(metadata, fns)
            return metadata, True

    return metadata, False


def process_pdf(
    filepath: Path,
    embed_model,
    session: requests.Session | None,
    base_url: str,
    api_key: str,
    chat_model: str,
    timeout: float,
    delay: float,
    use_json_object: bool,
    max_retries: int,
    max_retry_wait: float,
    rate_limit_state: RateLimitState,
    long_rate_limit_threshold: float,
):
    pdf = _load_pdf_pages(filepath)

    full_text = "\n".join(pdf)
    full_text = remove_sentences_containing(full_text, "downloaded from")
    full_text = clean_extracted_text(full_text)

    text_no_refs = re.split(
        r"\n\s*References\s*\n|\n\s*REFERENCES\s*\n",
        full_text,
        maxsplit=1,
    )[0]

    chunks = split_text(text_no_refs)
    index, chunk_list = build_vector_index(chunks, embed_model)

    metadata = {"filename": os.path.basename(filepath)}

    for field_name in queries:
        context = _field_context(
            field_name, pdf, full_text, index, chunk_list, embed_model
        )
        instruction = field_instruction(field_name)
        user_prompt = f"""Context:\n{context}\n\nQuestion: {instruction}\n\nReturn only JSON."""
        print(f"  -> Extracting {field_name}...")
        try:
            result = call_chat_api(
                session,
                base_url,
                api_key,
                chat_model,
                user_prompt,
                SYSTEM_PROMPT,
                timeout,
                use_json_object,
                max_retries=max_retries,
                max_retry_wait=max_retry_wait,
                rate_limit_state=rate_limit_state,
                long_rate_limit_threshold=long_rate_limit_threshold,
            )
        except RateLimitExceeded:
            metadata[field_name] = {
                "error": "skipped_due_to_rate_limit",
                "details": "API quota exhausted (Retry-After exceeded threshold)",
            }
            print(f"  [rate-limit] stopping remaining fields for {filepath.name}")
            for remaining_field in queries:
                if remaining_field not in metadata:
                    metadata[remaining_field] = {
                        "error": "skipped_due_to_rate_limit",
                        "details": "Earlier field hit API rate limit",
                    }
            return metadata, True

        metadata[field_name] = (
            result if isinstance(result, dict) else {"raw": result}
        )
        if _is_rate_limit_result(metadata[field_name]):
            print(f"  [rate-limit] stopping remaining fields for {filepath.name}")
            for remaining_field in queries:
                if remaining_field not in metadata:
                    metadata[remaining_field] = {
                        "error": "skipped_due_to_rate_limit",
                        "details": "Earlier field hit API rate limit",
                    }
            return metadata, True
        if delay > 0:
            time.sleep(delay)

    return metadata, False


def _load_existing_results(json_path: Path) -> dict[str, dict]:
    if not json_path.is_file():
        return {}
    try:
        rows = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(rows, list):
        return {}
    return {row["filename"]: row for row in rows if isinstance(row, dict) and row.get("filename")}


def _pdf_fully_extracted(row: dict) -> bool:
    for field_name in queries:
        val = row.get(field_name)
        if not isinstance(val, dict):
            return False
        if val.get("error"):
            return False
    return True


def _save_results(stem: str, all_metadata: list[dict]) -> tuple[Path, Path, Path, Path]:
    df = pd.DataFrame(all_metadata)
    results_csv = OUTPUT_FOLDER / f"{stem}.csv"
    results_json = OUTPUT_FOLDER / f"{stem}.json"
    df.to_csv(results_csv, index=False)
    df.to_json(results_json, orient="records", indent=2)

    flattened_rows = []
    for _, row in df.iterrows():
        flat_row = {"filename": row["filename"]}
        for col in df.columns:
            if col == "filename":
                continue
            val = row[col]
            if isinstance(val, dict):
                for k, v in val.items():
                    if v:
                        flat_row[col] = v
                        break
                else:
                    flat_row[col] = ""
            else:
                flat_row[col] = val
        flattened_rows.append(flat_row)

    df_flat = pd.DataFrame(flattened_rows)
    results_flat_csv = OUTPUT_FOLDER / f"{stem}_flat.csv"
    results_flat_json = OUTPUT_FOLDER / f"{stem}_flat.json"
    df_flat.to_csv(results_flat_csv, index=False)
    df_flat.to_json(results_flat_json, orient="records", indent=2)
    return results_csv, results_json, results_flat_csv, results_flat_json


def main() -> None:
    _load_dotenv_if_available()

    p = argparse.ArgumentParser(description="Extract metadata via OpenAI-compatible chat API.")
    p.add_argument(
        "--backend",
        choices=tuple(BACKENDS),
        default=None,
        help="API backend preset (default: maia if MAIA_API_KEY is set, else gwdg, else ollama)",
    )
    p.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL (overrides backend preset and CHAT_AI_BASE_URL)",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Chat model id (overrides backend preset and CHAT_AI_MODEL)",
    )
    p.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout per request")
    p.add_argument(
        "--batch",
        action="store_true",
        help="Use 3 batched API calls per PDF (2+5+3 fields) instead of 10 single-field calls",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to sleep after each field (single-field mode only; ignored with --batch)",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=4,
        metavar="N",
        help="On HTTP 429/503, retry up to N times with backoff (Retry-After capped; see --max-retry-wait)",
    )
    p.add_argument(
        "--max-retry-wait",
        type=float,
        default=MAX_RETRY_WAIT_SECONDS,
        help="Max seconds to wait on one 429/503 retry (server Retry-After can be thousands of seconds)",
    )
    p.add_argument(
        "--long-rate-limit-threshold",
        type=float,
        default=LONG_RATE_LIMIT_THRESHOLD_SECONDS,
        help="Retry-After above this (seconds) is treated as quota exhaustion and stops the run",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip PDFs already present in the output JSON for this --output-tag",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N PDFs (0 = all). Without --shuffle: first N in sorted filename order.",
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomize PDF order before applying --limit (ignored for effect if you pass only one --pdf).",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=None,
        metavar="INT",
        help="Optional seed for --shuffle so the same N files are chosen on repeat runs.",
    )
    p.add_argument(
        "--output-tag",
        default="",
        metavar="TAG",
        help="If set, write madata_results_api_TAG.csv (etc.) so a small test does not overwrite a full run",
    )
    p.add_argument(
        "--pdf",
        action="append",
        default=None,
        metavar="FILENAME",
        help="Process only this file under data/pdf (basename or path; repeat for multiple). "
        "Example: --pdf paper1.pdf --pdf paper2.pdf. With --pdf, --limit caps how many of those are run.",
    )
    p.add_argument(
        "--pdf-list",
        type=Path,
        default=None,
        metavar="FILE",
        help="Text file with one PDF basename per line (from audit_gold_pdf_coverage.py). "
        "Combined with --pdf if both are set.",
    )
    p.add_argument(
        "--no-json-object-mode",
        action="store_true",
        help="Do not send response_format json_object (use if API returns 400)",
    )
    args = p.parse_args()

    def _output_stem() -> str:
        base = "madata_results_api"
        tag = (args.output_tag or "").strip()
        if not tag:
            return base
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", tag).strip("_")
        return f"{base}_{safe}" if safe else base

    stem = _output_stem()

    backend_name = args.backend or _default_backend_name()
    base_url, chat_model, api_key, _ = _resolve_backend_config(
        backend_name,
        args.base_url,
        args.model,
    )
    mode = "batch (3 calls/PDF)" if args.batch else f"single ({len(queries)} calls/PDF)"
    print(f"[backend] {backend_name} -> {base_url} (model={chat_model}, {mode})")

    PDF_FOLDER.mkdir(parents=True, exist_ok=True)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    use_json = not args.no_json_object_mode
    session = requests.Session()

    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    all_pdfs = sorted(PDF_FOLDER.glob("*.pdf"))
    by_name = {p.name: p for p in all_pdfs}

    pdf_names: list[str] = list(args.pdf or [])
    if args.pdf_list:
        if not args.pdf_list.is_file():
            raise SystemExit(f"--pdf-list not found: {args.pdf_list}")
        for line in args.pdf_list.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if name and not name.startswith("#"):
                pdf_names.append(name)

    if pdf_names:
        paths = []
        seen: set[str] = set()
        for raw in pdf_names:
            name = Path(raw).name
            if name in seen:
                continue
            if name not in by_name:
                print(f"[WARN] PDF not found under {PDF_FOLDER}, skipping: {name}")
                continue
            seen.add(name)
            paths.append(by_name[name])
    else:
        paths = list(all_pdfs)

    if args.shuffle:
        rng = random.Random(args.random_seed)
        paths = list(paths)
        rng.shuffle(paths)

    if args.limit and args.limit > 0:
        paths = paths[: args.limit]

    if not paths:
        raise SystemExit("No PDFs to process (check data/pdf/, --pdf names, or --limit).")

    results_json = OUTPUT_FOLDER / f"{stem}.json"
    existing_by_name = _load_existing_results(results_json) if args.resume else {}
    if args.resume and existing_by_name:
        complete = sum(1 for row in existing_by_name.values() if _pdf_fully_extracted(row))
        print(
            f"[resume] Loaded {len(existing_by_name)} row(s) from {results_json.name} "
            f"({complete} complete, {len(existing_by_name) - complete} will be retried)"
        )

    all_metadata: list[dict] = []
    if args.resume:
        all_metadata = [
            row for row in existing_by_name.values() if _pdf_fully_extracted(row)
        ]
    rate_limit_state = RateLimitState()
    rate_limited = False

    for filepath in tqdm(paths, desc="PDFs"):
        existing = existing_by_name.get(filepath.name)
        if args.resume and existing and _pdf_fully_extracted(existing):
            print(f"\n[skip] Already complete: {filepath.name}")
            continue

        print(f"\n[PDF] Processing {filepath.name}")
        try:
            if args.batch:
                result, pdf_rate_limited = process_pdf_batch(
                    filepath,
                    embed_model,
                    session,
                    base_url,
                    api_key,
                    chat_model,
                    args.timeout,
                    use_json,
                    args.max_retries,
                    args.max_retry_wait,
                    rate_limit_state,
                    args.long_rate_limit_threshold,
                )
            else:
                result, pdf_rate_limited = process_pdf(
                    filepath,
                    embed_model,
                    session,
                    base_url,
                    api_key,
                    chat_model,
                    args.timeout,
                    args.delay,
                    use_json,
                    args.max_retries,
                    args.max_retry_wait,
                    rate_limit_state,
                    args.long_rate_limit_threshold,
                )
            all_metadata = [row for row in all_metadata if row.get("filename") != result["filename"]]
            all_metadata.append(result)
            _save_results(stem, all_metadata)
            if pdf_rate_limited:
                rate_limited = True
                retry_after = rate_limit_state.last_retry_after
                if retry_after > 0:
                    hours = retry_after / 3600.0
                    print(
                        f"\n[rate-limit] Stopping run. "
                        f"Try again in ~{hours:.1f}h with the same --output-tag and --resume."
                    )
                else:
                    print(
                        f"\n[rate-limit] Stopping run. "
                        f"Try again later with the same --output-tag and --resume."
                    )
                break
        except Exception as e:
            print(f"[ERROR] processing {filepath.name}: {e}")

    if not all_metadata:
        raise SystemExit("No metadata collected.")

    saved = _save_results(stem, all_metadata)
    if rate_limited:
        print("\n[PARTIAL] Saved progress before rate limit:")
    else:
        print("\n[OK] Metadata extraction (API) completed. Saved to:")
    for path in saved:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
