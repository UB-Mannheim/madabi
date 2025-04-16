import requests
import pandas as pd

base_url = "https://zenodo.org/api/records"
#query = """type:(dataset OR software OR image OR video OR physicalobject OR datamanagementplan OR softwaremanagementplan OR softwaredocumentation) 
#AND creators.affiliation:("University of Mannheim" OR "Mannheim University" OR "Universität Mannheim")"""
query = """type:(dataset OR software OR image OR video) 
AND creators.affiliation:("University of Mannheim" OR "Mannheim University" OR "Universität Mannheim")"""
rows = 1000
page = 1
all_records = []

# Initial request
params = {'q': query, 'size': rows, 'page': page}
response = requests.get(base_url, params=params)
result = response.json()
all_records.extend(result['hits']['hits'])

# Pagination
total_records = result['hits']['total']
pages = total_records // rows + (1 if total_records % rows else 0)

for page in range(2, pages + 1):
    params['page'] = page
    response = requests.get(base_url, params=params)
    result = response.json()
    all_records.extend(result['hits']['hits'])

# Extract detailed metadata
records_list = []
for rec in all_records:
    meta = rec['metadata']
    creators = ', '.join([creator['name'] for creator in meta.get('creators', [])])
    record_data = {
        'DOI': rec.get('doi', ''),
        'Title': meta.get('title', ''),
        'Publication Date': meta.get('publication_date', ''),
        'Description': meta.get('description', '').replace('\n', ' ').replace('<p>', '').replace('</p>', ''),
        'Creators': creators,
        'Type': meta.get('resource_type', {}).get('type', ''),
        'License': meta.get('license', {}).get('id', ''),
        'URL': rec['links'].get('self_html', '')
    }
    records_list.append(record_data)

# Save to CSV
df = pd.DataFrame(records_list)
df.to_csv('../metadata/zenodo.csv', index=False)

print(f"Total records retrieved and saved: {len(records_list)}")
