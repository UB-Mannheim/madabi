import requests
import pandas as pd

def fetch_data(offset):
    # URL for the BASE API
    url = "https://api.base-search.net/cgi-bin/BaseHttpSearchInterface.fcgi"

    # Parameters for the API request
    params = {
        "func": "PerformSearch",
        "query": "(University of Mannheim AND dctypenorm:7 OR dctypenorm:6 OR dctypenorm:5 OR dctypenorm:4 OR dctypenorm:3 OR dctypenorm:2 )",
        "format": "json",
        "hits": 120,
        "offsett": offset
    }

    # Making the request to the BASE API
    response = requests.get(url, params=params)
    if response.status_code == 200:
        # Assuming the API returns JSON data, return it
        return response.json()
    else:
        print(f"Failed to retrieve data at offset {offset}. Status code:", response.status_code)
        return None

# Initialize an empty DataFrame to store all results
all_data = pd.DataFrame()

# Initialize the offset
offset = 0

# Initialize a list to store DataFrame objects
dataframes = []

# Retrieve the first batch of results
initial_data = fetch_data(0)
if initial_data:
    num_found = initial_data['response']['numFound']
    dataframes.append(pd.DataFrame(initial_data['response']['docs']))

    # Loop to fetch the rest of the data
    for offset in range(1, num_found // 120 + 1):
        next_data = fetch_data(offset)
        if next_data:
            dataframes.append(pd.DataFrame(next_data['response']['docs']))
        else:
            break

# Concatenate all DataFrame objects
all_data = pd.concat(dataframes, ignore_index=True)

# Reset DataFrame index
all_data.reset_index(drop=True, inplace=True)

all_data.to_csv('../metadata/base.csv', index=False)
