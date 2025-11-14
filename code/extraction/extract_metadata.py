import json
import os
import re
from pathlib import Path

import faiss
import pandas as pd
import pdftotext
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ------------------ Config ------------------
MODEL_NAME = "gemma3:12b"
API_URL = "http://localhost:11434/api/generate"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
PDF_FOLDER = DATA_DIR / "pdf"
OUTPUT_FOLDER = DATA_DIR / "from_papers"

PDF_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# ------------------ Queries ------------------
queries = {
    "01 URL": "What is the URL of this paper?",
    "02 DOI": "What is the DOI of this paper?",
    "03 Citation": "Provide the full citation for this paper in Chicago style.",
    "04 Objective": "What is the main objective of the study?",
    "05 Data Type": "What type of data is used in this study? Possible values are primary data or secondary data",
    "06 Mode of Collection": "How was the data collected in the study? Possible values: Interview, Self-administered questionnaire, Focus group, Self-administered writings and/or diaries, Observation, Experiment, Recording, Automated data extraction, Content coding, Transcription, Compilation/Synthesis, Summary, Aggregation, Simulation, Measurements and tests, Other",
    "07 Type of Instrument": "What instrument or tool was used for data collection? Questionnaire, Interview scheme and/or themes, Data collection guidelines, Participant tasks, Technical instrument(s), Programming script, Other",
    "08 Analysis Unit": "What is the unit of analysis in this study? Possible values: Individual, Organization/Institution, Family, Household, Housing unit, Event/Process/Activity, Geographic unit, Political-administrative area, Time unit, Media unit, Group, Object, Other",
    "09 Sample Size": "Sample size in this paper?",
    "10 Method Details": "Describe the methodology used in the study.",
    "11 Data Collector": "Who collected the data, if the used data is primary?",
    "12 Data Source": "Where does the data come from if the data is secondary?",
    "13 Data DOI": "DOI of the dataset used in this paper",
    "14 Data citation": "Provide citation to the dataset used in this paper in Chicago style. Or part of text in this paper where the dataset is cited.",
    "15 Data availability statement": "Extract data availability statement from this paper or information about replication package for this paper.",
    "16 Funding": "Extract information about funding and project number."
}

# ------------------ Helper Functions ------------------
def split_text(text, chunk_size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

def build_vector_index(chunks, model):
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks

def search_chunks(query, index, chunks, model, top_k=3):
    query_vec = model.encode([query])
    D, I = index.search(query_vec, top_k)
    return [chunks[i] for i in I[0]]

def remove_sentences_containing(text, phrase):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    filtered_sentences = [s for s in sentences if phrase.lower() not in s.lower()]
    return ' '.join(filtered_sentences)

def extract_sentences_with_word(text, word):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    filtered = [s for s in sentences if re.search(rf'\b{re.escape(word)}\b', s, re.IGNORECASE)]
    return ' '.join(filtered)

def call_llm(prompt):
    system = "You are a metadata extraction assistant. Return clean, minimal JSON."
    response = requests.post(
        API_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "model": MODEL_NAME,
            "prompt": prompt,
            "temperature": 0.01,
            "system": system,
            "top_p": 0.3,
            "stream": False,
            "format": "json"
        })
    )

    if response.status_code == 200:
        metadata_json = response.json().get("response", "")
        try:
            return json.loads(metadata_json)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', metadata_json, re.DOTALL)
            return json.loads(json_match.group(0)) if json_match else {"error": "Failed to parse JSON"}
    else:
        return {"error": f"Request failed: {response.status_code}", "details": response.text}

# ------------------ Main Processing ------------------
def process_pdf(filepath: Path, model):
    with open(filepath, "rb") as f:
        pdf = pdftotext.PDF(f)

    text = "\n".join(pdf)
    text = remove_sentences_containing(text, "downloaded from")
    text = text.split('References')[0].split('REFERENCES')[0]

    chunks = split_text(text)
    index, chunk_list = build_vector_index(chunks, model)

    metadata = {"filename": os.path.basename(filepath)}

    for field, question in queries.items():
        print(f"  ↪ Extracting {field}...")
        if int(field[0:2]) < 4:
            context = pdf[0]
        else:
            context = "\n".join(search_chunks(question, index, chunk_list, model))
        if field=="09 Sample Size":
            context = extract_sentences_with_word(text, "sample") + extract_sentences_with_word(text, "respondents") 

        prompt = f"""Context:\n{context}\n\nQuestion: {question}"""
        result = call_llm(prompt)
        metadata[field] = result if isinstance(result, dict) else {"raw": result}

    return metadata

# ------------------ Entry Point ------------------
if __name__ == "__main__":
    model = SentenceTransformer(EMBEDDING_MODEL)

    all_metadata = []

    for filepath in tqdm(sorted(PDF_FOLDER.glob("*.pdf"))):
        print(f"\n📄 Processing {filepath.name}")
        try:
            result = process_pdf(filepath, model)
            all_metadata.append(result)
        except Exception as e:
            print(f"❌ Error processing {filepath.name}: {e}")

    df = pd.DataFrame(all_metadata)
    results_csv = OUTPUT_FOLDER / "results.csv"
    results_json = OUTPUT_FOLDER / "results.json"
    df.to_csv(results_csv, index=False)
    df.to_json(results_json, orient="records", indent=2)

    flattened_rows = []
    
    for _, row in df.iterrows():
        flat_row = {'filename': row['filename']}
        for col in df.columns:
            if col == 'filename':
                continue
            val = row[col]
    
            if isinstance(val, dict):
                # Try to extract the first non-empty string in the dict
                for k, v in val.items():
                    if v:  # not None or empty
                        flat_row[col] = v
                        break
                else:
                    flat_row[col] = ""  # if all values were empty
            else:
                flat_row[col] = val  # already flat
        flattened_rows.append(flat_row)
    
    # Create flattened DataFrame
    df_flat = pd.DataFrame(flattened_rows)
    
    results_flat_csv = OUTPUT_FOLDER / "results_flat.csv"
    results_flat_json = OUTPUT_FOLDER / "results_flat.json"
    df_flat.to_csv(results_flat_csv, index=False)
    df_flat.to_json(results_flat_json, orient="records", indent=2)

    print("\n✅ Metadata extraction completed. Saved to:")
    for path in (results_csv, results_json, results_flat_csv, results_flat_json):
        print(f"  - {path}")
