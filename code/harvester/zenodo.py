import requests
import pandas as pd

base_url = "https://zenodo.org/api/records"

# Same logic as your working browser query, but without line breaks
query = (
    'type:(dataset OR software OR image OR video) '
    'AND creators.affiliation:("University of Mannheim" OR "Mannheim University" OR "Universität Mannheim")'
)

rows = 25  # IMPORTANT: Zenodo limit is 100
params = {"q": query, "size": rows}

all_records = []

# --- First request ---
response = requests.get(base_url, params=params)
print("Initial status:", response.status_code)
print("Initial URL   :", response.url)

if response.status_code != 200:
    print("Error body from Zenodo:\n", response.text)
    raise SystemExit("Zenodo returned a non-200 response on first request.")

result = response.json()
hits = result.get("hits", {}).get("hits", [])
all_records.extend(hits)

# --- Follow pagination using links.next ---
next_url = result.get("links", {}).get("next")

while next_url:
    print("Next URL:", next_url)
    response = requests.get(next_url)
    if response.status_code != 200:
        print("Error body from Zenodo on next page:\n", response.text)
        raise SystemExit("Zenodo returned a non-200 response on a subsequent page.")

    result = response.json()
    hits = result.get("hits", {}).get("hits", [])
    all_records.extend(hits)
    next_url = result.get("links", {}).get("next")

print(f"Total records fetched: {len(all_records)}")

# --- Extract detailed metadata ---
records_list = []
for rec in all_records:
    meta = rec.get("metadata", {})
    creators = ", ".join(c.get("name", "") for c in meta.get("creators", []))

    desc = (meta.get("description") or "").replace("\n", " ")
    desc = desc.replace("<p>", "").replace("</p>", "")

    record_data = {
        "DOI": rec.get("doi", ""),
        "Title": meta.get("title", ""),
        "Publication Date": meta.get("publication_date", ""),
        "Description": desc,
        "Creators": creators,
        "type": meta.get("resource_type", {}).get("type", ""),
        "License": meta.get("license", {}).get("id", ""),
        "URL": rec.get("links", {}).get("self_html", ""),
    }
    records_list.append(record_data)

df = pd.DataFrame(records_list)
df.to_csv("../data/zenodo.csv", index=False)
print("Saved to ../data/zenodo.csv")

