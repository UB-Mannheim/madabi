import requests
import json
import pandas as pd

# Define the base URL and query parameters
base_url = "https://zenodo.org/api/records"
query = """type:(dataset OR software OR image OR video OR physicalobject OR datamanagementplan OR softwaremanagementplan OR softwaredocumentation) 
AND creators.affiliation:("University of Mannheim" OR "Mannheim University" OR "Universität Mannheim")"""
# type:(dataset OR software
rows = 1000
page = 1

# Initialize a list to store all records
all_records = []

# Perform the initial API request
params = {
    'q': query,
    'size': rows,
    'page': page
}
response = requests.get(base_url, params=params)
result = response.json()

# Add the initial results to the records list
all_records.extend(result['hits']['hits'])

# Calculate the total number of pages
total_records = result['hits']['total']
pages = total_records // rows + (1 if total_records % rows else 0)

# Loop through the remaining pages (if any) and collect the records
for page in range(2, pages + 1):
    params['page'] = page
    response = requests.get(base_url, params=params)
    result = response.json()
    all_records.extend(result['hits']['hits'])

# Saving the results to a file
with open('zenodo_records.json', 'w') as file:
    json.dump(all_records, file, indent=4)

print(f"Total records retrieved: {len(all_records)}")

a = []
for v in all_records:
    a.append([v['doi'], v['metadata']['creators']])
adf = pd.DataFrame(a)

adf.to_csv('../metadata/zenodo.csv', index=False)
