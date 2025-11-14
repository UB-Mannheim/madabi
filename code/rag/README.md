# RAG pipeline for MADATA metadata

The `rag_madata.py` script builds a retrieval‑augmented generation (RAG) stack on top of the Mannheim Data Bibliography (MADATA) OAI‑PMH endpoint. It can harvest metadata records, normalize them, build dense and TF‑IDF indices, stitch a knowledge graph of related records, and answer natural‑language questions by combining graph hops, vector search, and (optionally) a local Ollama model for response generation.

## Requirements

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The requirements cover harvesting (Sickle), embedding (sentence-transformers + faiss), graph reasoning (networkx), and CLI goodies (tqdm). PyTorch is pulled transitively by `sentence-transformers`. If you plan to call Ollama you also need the Ollama runtime with a local model (e.g. `ollama pull llama3.1:8b`).

## Workflow overview

1. **Harvest (`harvest` subcommand)** – Pulls MADATA (or any OAI-PMH endpoint) records and stores them as raw JSON.
2. **Normalize & index (`query` subcommand)** – Converts Dublin Core fields into the `Record` dataclass, caches them, creates TF‑IDF + sentence-transformer embeddings, and wires a graph where nodes are people, subjects, DOIs, landing pages, etc.
3. **Graph-aware retrieval** – Seeds results with semantic similarity, expands with graph neighbors, enforces author/subject/year filters, and boosts items that explicitly match quantity requests (“Give me three datasets …”).
4. **Answering (optional)** – If `--ollama` is enabled, the retrieved context is handed to a local LLM that follows a strict librarian-style system prompt.
5. **Utilities** – `grep`, `list-authors`, and `madoc` helpers make it easier to inspect harvested dumps or bring in MADOC OpenAIRE metadata to enrich the relationship graph.

Cached assets live in `.graphrag_cache/` (normalized records + FAISS index) so subsequent queries are instant.

## Typical usage

```bash
# 1) Harvest once (idempotent per day unless --force is passed)
python rag_madata.py harvest --out ../data/madata_harvest.json

# 2) Query with semantic + graph retrieval
python rag_madata.py query \
  --results ../data/madata_harvest.json \
  --question "Spatial datasets on Mannheim traffic after 2018" \
  --author "Müller" \
  --subject "transportation" \
  --final-k 10

# (Optional) Let Ollama summarize the retrieved records locally
python rag_madata.py query \
  --results ../data/madata_harvest.json \
  --question "Which Mannheim datasets include qualitative interviews?" \
  --ollama --ollama-model llama3.1:8b
```

Other handy entry points:

* `python rag_madata.py madoc --out ../data/madoc_harvest.json` – Harvest MADOC OpenAIRE metadata for blending with MADATA results.
* `python rag_madata.py grep --results ../data/madata_harvest.json --pattern "Universität Mannheim"` – Inspect raw Dublin Core payloads.
* `python rag_madata.py list-authors --results ../data/madata_harvest.json --top 25` – Surface prolific creators.
* `python rag_madata.py --quick --question "datasets on Mannheim housing"` – Harvest a small batch live (default 50 records) and immediately run a query.

For structured outputs, combine `--json` with `--json-fields Title,DOI,Source`, or emit citation-style snippets via `--cite`. Use `--seed-k`, `--expand-hops`, and `--final-k` to trade off recall vs. speed.

## Folder contents

* `rag_madata.py` – Single-file CLI that covers harvesting, normalization, graph/embedding index building, query orchestration, and optional answer generation.
* `requirements.txt` – Minimal pip dependencies required for the pipeline.

Everything (harvest, search, RAG, generation) can run locally on a laptop.
