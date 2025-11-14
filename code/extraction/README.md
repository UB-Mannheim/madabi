# PDF Extraction Utilities

The scripts in this folder automate the workflow from harvesting MADOC metadata, downloading open-access PDFs, and extracting structured information from those papers.

## Scripts

1. **`download_madoc_papers.py`**  
   Reads `data/madoc.csv`, filters for open-access journal articles, exports the corresponding full-text URLs, and downloads the PDFs with `aria2c` into `data/pdf/`.

2. **`extract_metadata.py`**  
   Builds a dense+FAISS index over each PDF and asks a local LLM (default: Ollama `gemma3:12b`) a wide set of questions to capture bibliographic, methodological, and data-sharing metadata. Results are written to CSV/JSON (both nested and flattened) under `data/from_papers/`.

3. **`extract_metadata_reduced.py`**  
   Variant focused on data availability fields (data type, collection method, DOI/citation, funding, etc.) with a condensed prompt set, but it reuses the same pipeline and directories as the full extractor.

All paths are derived from the repository root, so you can run the scripts from anywhere (e.g. `python code/extraction/download_madoc_papers.py`).

## Requirements

Install Python dependencies into a virtual environment:

```bash
cd code/extraction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Additional tooling:
- `aria2c` must be available on `PATH` for the downloader.
- `pdftotext` bindings require `poppler` (install via `brew install poppler` or your package manager).
- LLM extraction assumes an Ollama server at `http://localhost:11434`; adjust `MODEL_NAME`/`API_URL` in the scripts if you use another endpoint.

`data/pdf/` and `data/from_papers/` are created automatically if they do not exist. Ensure `data/madoc.csv` is up to date by running `python harvester/madoc.py` from `code/` before starting the download cycle.
