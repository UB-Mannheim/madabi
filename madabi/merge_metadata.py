import pandas as pd
import ast

# Load files
gesis_df = pd.read_csv('../metadata/gesis.csv')
zenodo_df = pd.read_csv('../metadata/zenodo.csv')
madata_df = pd.read_csv('../metadata/madata.csv')
harvard_df = pd.read_csv('../metadata/harvard_dataverse.csv')

# Standardize date columns explicitly
gesis_df['Year'] = gesis_df['Year'].apply(lambda x: pd.to_datetime(x, errors='coerce', dayfirst=True).year)
madata_df['raw_metadata'] = madata_df['raw_metadata'].apply(eval)
madata_df['Year'] = madata_df['raw_metadata'].apply(lambda x: x.get('date')[0:4])
zenodo_df['Year'] = pd.to_datetime(zenodo_df['Publication Date'], errors='coerce').dt.year
harvard_df['Year'] = pd.to_datetime(harvard_df['Published At'], errors='coerce').dt.year

def parse_creators_affiliations(creators):
    creators_list = []
    affiliations_list = []

    if pd.isna(creators):
        return pd.Series([None, None])

    try:
        # Parse stringified list if needed
        if isinstance(creators, str) and creators.strip().startswith('['):
            creators_parsed = ast.literal_eval(creators)
        else:
            creators_parsed = [creators]

        for entry in creators_parsed:
            parts = [part.strip() for part in entry.split(',')]
            if len(parts) >= 3:
                # First two parts as name, rest as affiliation
                name = ', '.join(parts[:2])
                affiliation = ', '.join(parts[2:])
            elif len(parts) == 2:
                name = parts[0]
                affiliation = parts[1]
            else:
                name = parts[0]
                affiliation = None

            creators_list.append(name)
            if affiliation:
                affiliations_list.append(affiliation)

        return pd.Series([
            '; '.join(creators_list),
            '; '.join(affiliations_list) if affiliations_list else None
        ])

    except Exception:
        return pd.Series([str(creators), None])


# Apply parsing to GESIS
gesis_df[['Creators_Clean', 'Affiliations']] = gesis_df['metadata.dc.creator'].apply(parse_creators_affiliations)

gesis_df_selected = pd.DataFrame({
    'Title': gesis_df['metadata.dc.title'],
    'Creators': gesis_df['Creators_Clean'],
    'Affiliations': gesis_df['Affiliations'],
    'Date': pd.to_datetime(gesis_df['metadata.dc.date'], errors='coerce', dayfirst=True),
    'Description': gesis_df['metadata.dc.description'],
    'DOI': gesis_df['metadata.dc.identifier'],
    'Source': 'GESIS',
    'Year': gesis_df['Year'],
    'Type': gesis_df['type']
})

zenodo_df_selected = pd.DataFrame({
    'Title': zenodo_df['Title'],
    'Creators': zenodo_df['Creators'],
    'Affiliations': pd.NA,
    'Date': pd.to_datetime(zenodo_df['Publication Date'], errors='coerce'),
    'Description': zenodo_df['Description'],
    'DOI': zenodo_df['DOI'],
    'Source': 'Zenodo',
    'Year': zenodo_df['Year'],
    'License': zenodo_df['License'],
    'Type': zenodo_df['type']
})

madata_df_selected = pd.DataFrame({
    'Title': madata_df['raw_metadata'].apply(lambda x: x.get('title')),
    'Creators': madata_df['raw_metadata'].apply(
                lambda x: (
                    '; '.join(x.get('creator')) if isinstance(x.get('creator'), list)
                    else x.get('creator', '')
                ) if isinstance(x, dict) else ''
            ),
    'Affiliations': "Universität Mannheim",
    'Date': pd.to_datetime(madata_df['raw_metadata'].apply(lambda x: x.get('date')), errors='coerce'),
    'Description': madata_df['raw_metadata'].apply(lambda x: x.get('description')),
    'DOI': madata_df['raw_metadata'].apply(lambda x: ', '.join(x.get('relation', []))),
    'Source': 'MADATA',
    'Year': madata_df['Year'],
    'License': madata_df['rights'],
    'Type': madata_df['type']
})

harvard_df_selected = pd.DataFrame({
    'Title': harvard_df['Title'],
    'Creators': harvard_df['Authors'],
    'Affiliations': 'University of Mannheim',
    'Date': pd.to_datetime(harvard_df['Published At'], errors='coerce'),
    'Description': harvard_df['Description'],
    'DOI': harvard_df['DOI'],
    'Source': 'Harvard Dataverse',
    'Year': harvard_df['Year'],
    'License': harvard_df['License'],
    'Type': harvard_df['type']
})

# Merge all into unified dataframe
unified_df = pd.concat([
    gesis_df_selected,
    zenodo_df_selected,
    madata_df_selected,
    harvard_df_selected
], ignore_index=True)

# Save to CSV
unified_df.to_csv('../metadata/unified_mannheim_metadata.csv', index=False)
# Save your DataFrame to JSON
unified_df.to_json("../metadata/unified_mannheim_metadata.json", orient="records")

print("Unified metadata saved to ../metadata/unified_mannheim_metadata.csv")
