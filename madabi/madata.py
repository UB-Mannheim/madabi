from sickle import Sickle
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pandas as pd

# Initialize Sickle
sickle = Sickle('https://madata.bib.uni-mannheim.de/cgi/oai2')
metadata_prefix = 'oai_dc'

def process_record(record):
    try:
        root = ET.fromstring(record.raw)
    except Exception as e:
        print("Error parsing record:", e)
        return {}
    return xml_to_dict(root)

def get_tag_without_namespace(elem):
    return elem.tag.split('}', 1)[-1] if '}' in elem.tag else elem.tag

def xml_to_dict(element):
    if not isinstance(element, ET.Element):
        return element
    if len(element) == 0 and element.text:
        return element.text.strip()
    result = {}
    for child in element:
        child_tag = get_tag_without_namespace(child)
        child_dict = xml_to_dict(child)
        if child_tag not in result:
            result[child_tag] = child_dict
        else:
            if not isinstance(result[child_tag], list):
                result[child_tag] = [result[child_tag]]
            result[child_tag].append(child_dict)
    return result

# Sequentially process records
records = sickle.ListRecords(metadataPrefix=metadata_prefix)
results = [process_record(record) for record in tqdm(records)]

# Extract and process metadata fields from each record.
all_metadata = []

for entry in results:
    # Check if the record contains metadata under the 'dc' key.
    if entry and 'metadata' in entry and 'dc' in entry['metadata']:
        resource_data = entry['metadata']['dc']
        metadata_entry = {'raw_metadata': resource_data}  # store the raw metadata
        
        # Iterate over all fields in the dc metadata.
        for key, value in resource_data.items():
            # Process the creator field specially.
            if key == 'creator':
                creators = value
                # If creators is not a list, convert it to a list.
                if not isinstance(creators, list):
                    creators = [creators]
                creators_info = []
                for creator in creators:
                    # Here, the creator is expected to be a string.
                    creator_info = {
                        'creator_name': creator,
                        'name_identifier': ''  # No ORCID info provided in this format.
                    }
                    creators_info.append(creator_info)
                metadata_entry['creators'] = creators_info
            else:
                metadata_entry[key] = value
        all_metadata.append(metadata_entry)

# Create a DataFrame from the metadata list.
metadata_df = pd.DataFrame(all_metadata)

# Optional: Display DataFrame columns to verify.
print("DataFrame columns:", metadata_df.columns)

# Optional: Save the DataFrame to CSV.
metadata_df.to_csv('../metadata/madata.csv', index=False)

