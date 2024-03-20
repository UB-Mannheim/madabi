from sickle import Sickle
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pandas as pd

# Initialize Sickle
sickle = Sickle('https://madoc.bib.uni-mannheim.de/cgi/oai2')
metadata_prefix = 'oai_openaire'

# Function to process a single record
def process_record(record):
    root = ET.fromstring(record.raw)
    return xml_to_dict(root)

# Function to remove namespace
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


# Sequential processing of records
records = sickle.ListRecords(metadataPrefix=metadata_prefix)
results = [process_record(record) for record in tqdm(records)]

# Extract and process all metadata fields, including nameIdentifier
all_metadata = []

for entry in results:
    if entry and 'metadata' in entry and 'resource' in entry['metadata']:
        resource_data = entry['metadata']['resource']
        metadata_entry = {'raw_metadata': resource_data}  # Store the raw metadata
        for key, value in resource_data.items():
            # Process nested structures (like creators)
            if isinstance(value, dict) and 'creator' in value:
                creators = value['creator']
                if not isinstance(creators, list):  # Single creator case, convert to list
                    creators = [creators]
                creators_info = []
                for creator in creators:
                    creator_info = {'creator_name': creator.get('creatorName')}
                    name_identifier = creator.get('nameIdentifier')
                    # Check the type of nameIdentifier and extract appropriately
                    if isinstance(name_identifier, dict):
                        creator_info['name_identifier'] = name_identifier.get('#text', '')
                    elif isinstance(name_identifier, str):
                        creator_info['name_identifier'] = name_identifier
                    else:
                        creator_info['name_identifier'] = ''
                    creators_info.append(creator_info)
                value = creators_info
            metadata_entry[key] = value
        all_metadata.append(metadata_entry)

# Convert to DataFrame
metadata_df = pd.DataFrame(all_metadata)

# Authors with ORCIDs
creator_dict = {}
for creators in metadata_df.creators:
    if isinstance(creators, list):
        for creator in creators:
            creator_dict[creator['creator_name']] = creator['name_identifier']
    if isinstance(creators, dict):
        creator_dict[creator.get('creator_name')] = creator.get('name_identifier')


metadata_df.to_csv('../metadata/madoc.csv', index=False)


