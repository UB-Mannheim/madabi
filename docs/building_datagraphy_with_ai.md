# Building the Datagraphy with AI via Metadata Extraction

This document extends Section 3 of the MaDaBi paper (*Building the datagraphy with AI via
metadata extraction*) with the concrete, reproducible implementation that lives in this
repository. It describes how we turn University of Mannheim publications into a **datagraphy** —
a registry of the research data (datasets, software, notebooks, replication packages, …)
that those publications produced or reused but that does **not** live in MADATA.

## Motivation

MADATA holds only a fraction of the data created by University of Mannheim employees. The rest
is scattered across external repositories (OSF, Figshare, GitHub, Zenodo, Dataverse, …) and is
usually only *discoverable from inside the papers themselves* — in data-availability statements,
"supplementary materials", citations, and footnotes. Our solution mines those full texts with an
LLM-based extraction pipeline, recovers the data/code links, harvests metadata for each link,
attributes it back to Mannheim authors, and assembles the result into a unified datagraphy.

## Pipeline at a glance

```
MADOC (OAI-PMH)                                     [code/harvester/madoc.py]
  └─ madoc.csv
     └─ + UB EPrints flags                          [code/extraction/fetch_madoc_ubma_flags.py]
        └─ madoc_with_ubma.csv
           └─ download open-access PDFs             [code/extraction/download_madoc_papers.py]
              (+ long-name fallback)                [code/extraction/download_missing_pdfs.py]
              └─ data/pdf/
                 └─ ① AI METADATA EXTRACTION (RAG + LLM)
                    [extract_metadata.py | extract_metadata_reduced.py | extract_metadata_api.py]
                    └─ data/from_papers/madata_results_api*.csv/json
                       └─ ② REPOSITORY-LINK DISCOVERY
                          [notebooks/analyze_repository_links_from_papers.ipynb]
                          └─ extracted_repositories_2026-06.csv  (repository, url, …)
                             └─ ③ PER-SOURCE METADATA HARVEST
                                [run_osf_metadata.py | run_figshare_metadata.py
                                 | run_somef_github.py + run_github_contributors.py]
                                └─ osf_metadata/  figshare_metadata/  github_*/
                                   └─ ④ PROVENANCE + SCORING + UNIFICATION
                                      [build_paper_lookup_from_madoc.py → scoring.py
                                       (interactive: explore_mannheim_contributors.ipynb)
                                       → build_unified_metadata.py]
                                      └─ unified_osf_figshare_v5.csv  (the datagraphy)
```

---

## ① AI metadata extraction (the core)

We extract structured metadata from each publication's full text using **retrieval-augmented
generation (RAG)**: rather than feeding a whole PDF to the LLM, we retrieve only the passages
relevant to each field and ask a targeted question.

**How a single PDF is processed:**

1. **Text** — the PDF is converted to plain text (`pdftotext` / poppler).
2. **Chunk** — text is split with LangChain's `RecursiveCharacterTextSplitter`.
3. **Embed + index** — chunks are embedded with the `all-MiniLM-L6-v2` sentence-transformer and
   stored in a local **FAISS** vector index. Embeddings always run locally, regardless of which
   LLM backend is used.
4. **Per-field retrieval + answer** — for each metadata field we retrieve the most similar chunks
   and prompt the LLM with a field-specific question.

### Field templates

There are two question sets, chosen by which extractor you run:

- **Full template** (`extract_metadata.py`) — bibliographic + methodological + data fields:
  `URL, DOI, Citation, Objective, Data Type, Mode of Collection, Type of Instrument,
  Analysis Unit, Sample Size, Method Details, Data Collector, Data Source, Data DOI,
  Data citation, Data availability statement, Funding, Code citation`.
- **Reduced template** (`extract_metadata_reduced.py`) — a lean, data-availability-focused subset
  (data type by source/collection method, instrument, collector/source, data DOI + citation,
  data-availability statement, funding, code citation). Same RAG pipeline and directories.

### Two extractor implementations

| Script | LLM | Use when |
|---|---|---|
| `extract_metadata.py` / `extract_metadata_reduced.py` | Local **Ollama** (`gemma3:12b`) via `localhost:11434` | Fully local / offline runs |
| `extract_metadata_api.py` | OpenAI-compatible chat API | Higher-quality / hosted models, large batches |

`extract_metadata_api.py` is the current production extractor and supports three backends
(selected with `--backend`, auto-detected from which key is set):

- **`maia`** — Uni Mannheim MaIA (`https://maia.uni-mannheim.de/ollama/v1`, default `gemma4:31b`);
  used when `MAIA_API_KEY` is set.
- **`gwdg`** — GWDG Academic Cloud / SAIA (rate-limited quota); used when `CHAT_AI_API_KEY` is set.
- **`ollama`** — local Ollama (`gemma3:12b`).

It adds operational robustness: `--batch` (3 API calls per PDF instead of one per field),
`--resume` (skip PDFs already fully extracted after a rate-limit interruption), `--limit`/`--shuffle`
for sampling, and rate-limit backoff.

**Outputs** (under `data/from_papers/`, default stem `madata_results_api`):
`*.csv`, `*.json`, and flattened `*_flat.csv` / `*_flat.json`. The flattened CSV is the input to
link discovery. `2026-06` is the **label of the current production run** (`--output-tag`), not a
paper count — it covers the open-access PDFs that were extracted, not the full corpus (see *Coverage*
below).

### Coverage

The extraction does not run over every MADOC record — it funnels down to the publications whose
full text we can actually obtain and process:

| Stage | Count |
|---|---|
| MADOC records harvested (`madoc_with_ubma.csv`) | ~53,000 |
| Open-access PDFs downloaded (`data/pdf/`) | ~2,070 |
| Papers extracted (current `2026-06` run) | ~2,060 |

The download step keeps only open-access journal articles, which is why the extracted set is a small
fraction of the bibliography. Extending coverage means widening that filter (more resource types) or
adding external full-text sources for the DOIs that MADOC links out to.

---

## ② Repository-link discovery

[`notebooks/analyze_repository_links_from_papers.ipynb`](../notebooks/analyze_repository_links_from_papers.ipynb)
reads the flattened extraction output (`madata_results_api_2026-06_flat.csv`) and pulls out
**dataset/code repository links**. It scans only link-relevant columns (strict schema) to avoid
boilerplate (publisher pages, Creative Commons URLs, …), normalises URLs and DOIs, and classifies
each by host. The result is:

- `extracted_repositories_2026-06.csv` — one row per unique link (`repository, url,
  distinct_papers, mention_rows`),
- `extracted_repositories_with_papers_2026-06.csv` — link ↔ source-paper (filename) mapping.

A typical run finds links across OSF, Dataverse, GitHub, Zenodo, Figshare, MADATA, GESIS and GitLab.

---

## ③ Per-source metadata harvesting

For the repository types we can resolve programmatically, dedicated harvesters fetch compact,
de-cluttered metadata per item:

- **OSF** — `code/run_osf_metadata.py`: one pass that resolves each OSF GUID and collects
  resource metadata + file inventory + contributors → `data/from_papers/osf_metadata/osf_<id>.json`
  + `manifest.csv` (drops the bulky OSF JSON:API relationship graph).
- **Figshare** — `code/run_figshare_metadata.py`: Figshare API v2 article details + files →
  `data/from_papers/figshare_metadata/`.
- **GitHub** — `code/run_somef_github.py` (SOMEF software metadata) and
  `code/run_github_contributors.py` (contributor identities for affiliation heuristics). Both need
  a valid `GITHUB_TOKEN`; SOMEF additionally requires the `somef` CLI.

All harvesters take `--input data/from_papers/extracted_repositories_2026-06.csv`.

---

## ④ Provenance, scoring, and unification

> **When to run:** this whole stage runs **after stages ①–③ are complete**, and its three steps are
> **sequential** — each consumes the previous one's output. In short: harvest first (③), then look up
> provenance, then score, then unify.

1. **Provenance lookup** — `code/from_papers/build_paper_lookup_from_madoc.py` maps each downloaded
   PDF filename back to its MADOC record (title, authors, year, MADOC link, citation) **without**
   re-parsing PDFs, using `pdf_urls.txt` (+ the `missing_pdf_url_to_filename.csv` override for
   long filenames). *Depends only on the downloaded PDFs (stage ①)*, so it can technically run any
   time after the download step. Output: `paper_lookup_madoc.csv`.
2. **Mannheim scoring** — `code/from_papers/scoring.py` matches each item's people (OSF contributors
   / Figshare authors) against the Uni Mannheim employee list (ORCID first, name match as fallback)
   and checks whether a citing MADOC paper falls within that person's Mannheim employment (the
   *tenure* check). It emits a verdict + confidence per item →
   `osf_unimannheim_scored_potential.csv`, `figshare_unimannheim_scored_potential.csv`.
   *Depends on the harvested metadata (`osf_metadata/`, `figshare_metadata/` from stage ③), the
   employee list, and `paper_lookup_madoc.csv` (step 1).*
   **Interactive alternative:** [`notebooks/explore_mannheim_contributors.ipynb`](../notebooks/explore_mannheim_contributors.ipynb)
   drives the same logic (`from scoring import run_all`) and adds exploratory views — Mannheim-mention
   detection, ORCID/name matches, and project-level affiliation rules. Use the notebook to inspect and
   sanity-check the matching; use `scoring.py` for a clean headless run. Both write the same scored files.
3. **Unification** — `code/from_papers/build_unified_metadata.py` formats both sources into one
   schema, grafts in the scores and the citation/MADOC provenance, and writes the datagraphy:
   `unified_osf_figshare_v5.csv`. *Depends on the scored files (step 2), the harvested metadata
   (stage ③), and `paper_lookup_madoc.csv` (step 1).*

---

## Reproducing the pipeline

```bash
# 0. Harvest MADOC + enrich, then acquire full texts
python code/harvester/madoc.py
python code/extraction/fetch_madoc_ubma_flags.py
python code/extraction/download_madoc_papers.py
python code/extraction/download_missing_pdfs.py          # long-name PDF fallback

# 1. AI metadata extraction (API backend; needs a key in .env)
python code/extraction/extract_metadata_api.py --backend maia --batch --resume

# 2. Repository-link discovery
#    run notebooks/analyze_repository_links_from_papers.ipynb

# 3. Harvest per source (over the discovered links)
python code/run_osf_metadata.py      --input data/from_papers/extracted_repositories_2026-06.csv
python code/run_figshare_metadata.py --input data/from_papers/extracted_repositories_2026-06.csv
python code/run_github_contributors.py --input data/from_papers/extracted_repositories_2026-06.csv
python code/run_somef_github.py      --input data/from_papers/extracted_repositories_2026-06.csv

# 4. Provenance → scoring → unification
#    Run AFTER steps 0–3. The three lines are sequential (each needs the previous one's output):
#      - build_paper_lookup_from_madoc.py : needs the PDFs + pdf_urls.txt (from step 0)
#      - scoring.py                       : needs the harvested metadata (step 3) + employee list + the lookup
#      - build_unified_metadata.py        : needs the scored files + metadata + the lookup
python code/from_papers/build_paper_lookup_from_madoc.py
python code/from_papers/scoring.py                 # or explore interactively:
                                                   # notebooks/explore_mannheim_contributors.ipynb
python code/from_papers/build_unified_metadata.py
```

## Data & privacy

The full texts, the employee list, and every output that embeds personal data (names, emails,
ORCID matches, affiliation verdicts) are **git-ignored** — see the repository `.gitignore`. The
datagraphy can be regenerated from these local inputs, but the personal-data files themselves are
never committed; a publishable version would require anonymisation/aggregation.
