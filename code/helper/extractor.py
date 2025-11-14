import pandas as pd
from tqdm import tqdm
import requests
import pdftotext
import re
from urllib.parse import unquote


path2papers = '../madoc/all/'

def find_doi(text):
    # Split the text into paragraphs
    paragraphs = re.split('\n+', text)

    # Pattern to match a paragraph containing any "https" or "http" URL
    url_pattern = re.compile(r'\bhttps?://\S+')

    # Pattern to exclude URLs starting with "http://creativecommons"
    cc_exclude_pattern = re.compile(r'\bhttps?://(creativecommons|orcid\.org)\S*')

    # Pattern to match paragraphs containing specific keywords
    keywords_pattern = re.compile(r'\b(data|software|code|package)\b', re.IGNORECASE)

    # Pattern to match paragraphs containing "Supplemental material" or "supplemented material"
    supplemental_pattern = re.compile(r'supplement(al|ed|ary) material', re.IGNORECASE)

    # Pattern for "Data Availability Statement" or "Data Availability"
    data_availability_pattern = re.compile(r'data availability (statement|)', re.IGNORECASE)

    # List to store matched and merged paragraphs
    merged_paragraphs = []

    i = 0
    while i < len(paragraphs):
        #if "References" in paragraphs[i] or "references" in paragraphs[i]:
        #    break  # Stop searching once "References" or "reference" is encountered
        match_found = False
        merged_paragraph = ""

        # Check for URLs, but exclude "http://creativecommons" links
        if (url_pattern.search(paragraphs[i]) and not cc_exclude_pattern.search(paragraphs[i]) or 
            supplemental_pattern.search(paragraphs[i]) or 
            data_availability_pattern.search(paragraphs[i])):
            match_found = True
            # Merge with the previous paragraph if it contains one of the specific keywords
            if i > 0 and keywords_pattern.search(paragraphs[i-1]):
                merged_paragraph += paragraphs[i-1] + ""
            merged_paragraph += paragraphs[i]
        
        # Special handling for "Data Availability Statement" or "Data Availability"
        if (data_availability_pattern.search(paragraphs[i]) or supplemental_pattern.search(paragraphs[i])) and not url_pattern.search(paragraphs[i]):
            buffer_zone = 2  # Example buffer zone: Next two paragraphs
            for j in range(1, buffer_zone + 1):
                if i+j < len(paragraphs) and url_pattern.search(paragraphs[i+j]):
                    merged_paragraph += "" + paragraphs[i+j]
                    break  # Stop after finding the first URL in the buffer zone

        # Check and merge with the next paragraph for URL/supplemental condition
        if (i < len(paragraphs) - 1) and match_found:
            merged_paragraph += " " + paragraphs[i+1]
            i += 1  # Skip the next paragraph since it's already included
        
        if merged_paragraph:
            merged_paragraphs.append(merged_paragraph)
        i += 1

    return merged_paragraphs


mdf = pd.read_csv('./madoc_all_records.csv', low_memory=False)
mdf = mdf.fillna('')
mdf['data'] = ''

for index, row in tqdm(mdf.iterrows()):
    if row.file: # row.resourceType == "journal article" and 
       # if  str(int(row.date)) == '2024':
         filename = unquote(row.file.split('/')[-1]).replace('.pdf', '.txt').replace('.PDF', '.txt')
         try:
             with open(path2papers + filename, 'r') as file:
                 text = file.read()
             row.alternateIdentifiers = eval(row.alternateIdentifiers)
             if row.alternateIdentifiers:
                 ids = row.alternateIdentifiers['alternateIdentifier']
                 if isinstance(ids, str) and '/' in ids:
                     DOI = ids
                 if isinstance(ids, list):
                     DOI = [a for a in ids if '/' in a][0]
                 else:
                     DOI = ''
             data = find_doi(text)
             data = [d.replace('\u200b', '') for d in data if DOI not in d]
             if data==[]:
                 data = ''
             mdf.at[index, 'data'] = data
         except:
             print('cannot open ' + filename)

def url_extractor(strings):
    # Regular expression to match URLs
    url_pattern = r'https?://\S+'
    
    # Extracting URLs from the list of strings
    urls = []
    for string in strings:
        found_urls = re.findall(url_pattern, string)
        urls.extend(found_urls)
    return urls

a = mdf[mdf.data!='']
a['data_url'] = a['data'].apply(lambda x: url_extractor(x))
a.date = a.date.apply(lambda x: str(x))
