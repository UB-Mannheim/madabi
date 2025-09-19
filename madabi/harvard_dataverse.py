import requests
import pandas as pd
import time
from tqdm import tqdm

# Harvard Dataverse API endpoints
search_url = "https://dataverse.harvard.edu/api/search"
metadata_url = "https://dataverse.harvard.edu/api/datasets/:persistentId/metadata?persistentId={doi}"

# Query to find datasets affiliated with University of Mannheim
query = '(authorAffiliation:"University of Mannheim")'

# Search API parameters
params = {
    'q': query,
    'type': 'dataset',
    'fq': 'publicationStatus:Published',
    'per_page': 100,
    'start': 0
}

all_datasets = []
print("🔍 Starting metadata harvest from Harvard Dataverse...")

# Step 1: First request to get total count
first_response = requests.get(search_url, params=params)
first_response.raise_for_status()
first_data = first_response.json()['data']
total = first_data['total_count']
print(f"📦 Total datasets to harvest: {total}")

# Reset for iteration
params['start'] = 0

# Step 2: Iterate with tqdm progress bar
with tqdm(total=total, desc="⏳ Harvesting", unit="dataset") as pbar:
    while True:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        data = response.json()['data']
        items = data['items']

        if not items:
            break

        for item in items:
            doi = item.get('global_id')
            license_info = 'N/A'

            # Fetch license info from metadata API
            try:
                meta_response = requests.get(metadata_url.format(doi=doi))
                meta_response.raise_for_status()
                metadata_json = meta_response.json()
                license_info = metadata_json['data'].get('schema:license', 'N/A')
            except Exception as e:
                print(f"⚠️ License lookup failed for {doi}: {e}")

            all_datasets.append({
                'Title': item.get('name'),
                'DOI': doi,
                'URL': item.get('url'),
                'Description': item.get('description'),
                'Published At': item.get('published_at'),
                'Source': 'Harvard Dataverse',
                'Authors': '; '.join(item.get('authors', [])),
                'License': license_info,
                'type': item.get('type')
            })

            pbar.update(1)

        params['start'] += params['per_page']
        time.sleep(0.5)

# Save to CSV
df = pd.DataFrame(all_datasets)
df.to_csv('../metadata/harvard_dataverse.csv', index=False)
print(f"\n✅ Done! {len(df)} datasets saved to ../metadata/harvard_dataverse.csv")
