from sickle import Sickle
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

concurrent = True # or True if sequential harvesting is preferred

# Initialize Sickle
sickle = Sickle('https://dbkapps.gesis.org/dbkoai')
metadata_prefix = 'oai_dc'

# Function to process a single record
def process_record(identifier_str):
    try:
        record = sickle.GetRecord(identifier=identifier_str, metadataPrefix=metadata_prefix)
        root = ET.fromstring(record.raw)
        return xml_to_dict(root), None
    except Exception:
        return None, identifier_str

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

if concurrent:
    # Concurrent processing of records
    results = []
    skipped = []
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for identifier in tqdm(sickle.ListIdentifiers(metadataPrefix=metadata_prefix)):
            futures.append(executor.submit(process_record, identifier.identifier))
    
        for f in tqdm(as_completed(futures), total=len(futures)):
            result, failed = f.result()
            if result:
                results.append(result)
            elif failed:
                skipped.append(failed)
else:
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

def mannheim_key(x):
    return ('Universität Mannheim' in x or
            'University of Mannheim' in x or
            'MZES' in x or
            'Mannheimer Zentrum für Europäische Sozialforschung' in x or
            'Universitäten Frankfurt und Mannheim'  in x or
            'Universitäten Kiel und Mannheim' in x)


def get_mannheim(inp):
    if type(inp)==list:
        return any([x for x in inp if mannheim_key(x)])
    if type(inp)==str:
        return mannheim_key(inp)
    if not inp:
        return False
    else:
        return False

metadata_df = pd.json_normalize(results)
metadata_df['from Mannheim'] = metadata_df['metadata.dc.creator'].apply(lambda x: get_mannheim(x))
gesis_metadata_df = metadata_df[metadata_df['from Mannheim']]

def extract_year_from_identifier(identifier):
    try:
        record = sickle.GetRecord(identifier=identifier, metadataPrefix='oai_ddi25')
        root = ET.fromstring(record.raw)
        for elem in root.iter():
            if elem.tag.endswith('version') and 'date' in elem.attrib:
                return elem.attrib['date']
        return None
    except Exception as e:
        print(f"Failed to process {identifier}: {e}")
        return None

tqdm.pandas()
gesis_metadata_df['Year'] = gesis_metadata_df['header.identifier'].progress_apply(extract_year_from_identifier)
gesis_metadata_df['Year'] = gesis_metadata_df['Year'].fillna(gesis_metadata_df['header.datestamp'])
gesis_metadata_df = gesis_metadata_df.rename(columns={"metadata.dc.type": "type"})
gesis_metadata_df.to_csv('../metadata/gesis.csv', index=False)