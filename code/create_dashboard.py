import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
from datetime import datetime
import textwrap

# ---------- Configuration ----------
# Prefer the "metadate" variant for tracking updates.
DATA_PATH = '../data/unified_mannheim_metadata.csv'
TRACKING_PATH = '../data/madabi_tracking.csv'
OUTPUT_DIR = '../docs'
WORDCLOUD_IMG = os.path.join(OUTPUT_DIR, 'title_wordcloud.png')
HTML_OUTPUT = os.path.join(OUTPUT_DIR, 'index.html')
SOURCE_COLOR_MAP = {
    'MADATA': '#66c2a5',
    'GESIS': '#fc8d62',
    'Harvard Dataverse': '#8da0cb',
    'Zenodo': '#e78ac3',
}


# ---------- Data Loading ----------
def load_and_prepare_data(path):
    df = pd.read_csv(path)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    df['Source_clean'] = df['Source'].fillna('Unknown').astype(str).str.strip()
    df.loc[df['Source_clean'] == '', 'Source_clean'] = 'Unknown'
    df['Affiliations_clean'] = df['Affiliations'].fillna('').replace({
        'University of Mannheim': 'Universität Mannheim'
    }, regex=False)
    df['Affiliations_clean'] = df['Affiliations_clean'].str.replace(
        r'.*Universität Mannheim.*', 'Universität Mannheim', regex=True
    )
    df['Title_clean'] = df['Title'].fillna('').astype(str).str.strip().str.lower()
    df.loc[df['Title_clean'] == 'nan', 'Title_clean'] = ''
    return df


def compute_total_n(df):
    if 'Source_clean' not in df.columns:
        return len(df)
    return int(df['Source_clean'].notna().sum())


def update_tracking_csv(df, tracking_path):
    tracking_columns = ['source', 'month', 't', 'dataset', 'intersection', 'total', 'intersection_perc']
    if os.path.exists(tracking_path):
        tracking_df = pd.read_csv(tracking_path, sep=';')
    else:
        tracking_df = pd.DataFrame(columns=tracking_columns)

    tracking_df = tracking_df.drop_duplicates(subset=['source', 'month'], keep='last')
    tracking_df = tracking_df.sort_values(by=['t', 'source']).reset_index(drop=True)

    def normalize_month_str(value):
        if pd.isna(value):
            return None
        return str(value).strip().upper()

    def month_sort_key(month_value):
        # Expected format in existing tracking CSV: "FEB_2026"
        try:
            parts = str(month_value).strip().upper().split('_')
            if len(parts) == 2:
                month_abbr = parts[0].title()
                year = int(parts[1])
                return datetime.strptime(month_abbr, '%b').replace(year=year)
        except Exception:
            pass
        # Ensure a deterministic comparable return type for sorting.
        return datetime.max

    preferred_sources = ['MADATA', 'GESIS', 'Harvard Dataverse', 'Zenodo']
    data_sources = [s for s in sorted(df['Source_clean'].dropna().unique()) if str(s).strip() != '']
    sources = preferred_sources + [s for s in data_sources if s not in preferred_sources]
    existing_months = set(tracking_df['month'].astype(str)) if len(tracking_df) > 0 else set()

    month_col = None
    if 'month' in df.columns:
        month_col = 'month'
    elif 'Month' in df.columns:
        month_col = 'Month'

    # If the input contains month snapshots, compute tracking per month.
    # Otherwise, treat the entire input as the current snapshot.
    if month_col is not None:
        df = df.copy()
        df['__month_norm'] = df[month_col].apply(normalize_month_str)
        df = df.dropna(subset=['__month_norm'])
        unparseable_months = []
        for month_value in sorted(set(df['__month_norm'].astype(str))):
            parts = str(month_value).strip().upper().split('_')
            valid = False
            if len(parts) == 2:
                try:
                    datetime.strptime(parts[0].title(), '%b')
                    int(parts[1])
                    valid = True
                except Exception:
                    valid = False
            if not valid:
                unparseable_months.append(month_value)
        if len(unparseable_months) > 0:
            print(f"Unparseable tracking month values: {', '.join(unparseable_months[:5])}")
        derived_months = sorted(set(df['__month_norm'].astype(str)), key=month_sort_key)
    else:
        derived_months = [datetime.today().strftime('%b_%Y').upper()]

    months_to_update = set(derived_months)
    current_max_t = int(pd.to_numeric(tracking_df['t'], errors='coerce').max()) if len(tracking_df) > 0 else -1
    month_to_existing_t = {}
    if len(tracking_df) > 0:
        for m in months_to_update & existing_months:
            t_max = pd.to_numeric(tracking_df[tracking_df['month'].astype(str) == m]['t'], errors='coerce').max()
            month_to_existing_t[m] = int(t_max) if pd.notna(t_max) else current_max_t
    new_months = [m for m in derived_months if m not in existing_months]
    new_month_to_t = {}
    for i, m in enumerate(new_months):
        new_month_to_t[m] = current_max_t + 1 + i

    tracking_remaining = tracking_df[~tracking_df['month'].astype(str).isin(months_to_update)].copy() if len(tracking_df) > 0 else tracking_df
    new_rows = []

    for month_value in derived_months:
        if month_col is not None:
            month_df = df[df['__month_norm'].astype(str) == month_value]
        else:
            month_df = df

        madata_titles = set(get_valid_title_series(month_df[month_df['Source_clean'].str.lower() == 'madata']))
        other_titles = set(get_valid_title_series(month_df[month_df['Source_clean'].str.lower() != 'madata']))
        shared_titles = madata_titles & other_titles
        total = int(month_df['Source_clean'].isin(sources).sum())
        intersection = len(shared_titles)
        intersection_perc = round((intersection / total * 100), 1) if total > 0 else 0.0

        t_value = month_to_existing_t.get(month_value, new_month_to_t.get(month_value))
        for source in sources:
            dataset_count = int((month_df['Source_clean'] == source).sum())
            new_rows.append({
                'source': source,
                'month': month_value,
                't': t_value,
                'dataset': dataset_count,
                'intersection': intersection,
                'total': total,
                'intersection_perc': intersection_perc
            })

    new_df = pd.DataFrame(new_rows, columns=tracking_columns)
    tracking_df = pd.concat([tracking_remaining, new_df], ignore_index=True)
    tracking_df = tracking_df.drop_duplicates(subset=['source', 'month'], keep='last')
    tracking_df = tracking_df.sort_values(by=['t', 'source']).reset_index(drop=True)

    tracking_df.to_csv(tracking_path, sep=';', index=False)
    return tracking_df


# ---------- Plotting Functions ----------
def wrap_title(title, width=45):
    if title is None:
        return title
    title = str(title)
    if len(title) <= width:
        return title
    return '<br>'.join(textwrap.wrap(title, width=width))


def format_count_percentage(count, total):
    percentage = (count / total * 100) if total > 0 else 0.0
    return f"{count} ({percentage:.1f}%)"


def get_source_color_map(sources):
    color_map = dict(SOURCE_COLOR_MAP)
    fallback_colors = px.colors.qualitative.Pastel
    unknown_sources = [s for s in sorted(set(sources)) if s not in color_map]
    for i, source in enumerate(unknown_sources):
        color_map[source] = fallback_colors[i % len(fallback_colors)]
    return color_map


def get_valid_title_series(df):
    return df.loc[df['Title_clean'].notna() & (df['Title_clean'] != ''), 'Title_clean']


def log_data_quality(df):
    missing_source = int(df['Source_clean'].isna().sum() + (df['Source_clean'] == '').sum())
    missing_title = int(df['Title_clean'].isna().sum() + (df['Title_clean'] == '').sum())
    missing_year = int(df['Year'].isna().sum())
    unknown_sources = sorted([s for s in df['Source_clean'].dropna().unique() if s not in SOURCE_COLOR_MAP])

    if missing_source > 0:
        print(f"Missing source values: {missing_source}")
    if missing_title > 0:
        print(f"Missing title values: {missing_title}")
    if missing_year > 0:
        print(f"Missing/invalid year values: {missing_year}")
    if len(unknown_sources) > 0:
        print(f"Additional sources detected: {', '.join(unknown_sources)}")


def plot_madata_ratio_with_intersection(df, total_n):
    madata_df = df[df['Source_clean'] == 'MADATA']
    other_df = df[df['Source_clean'] != 'MADATA']

    madata_titles = set(get_valid_title_series(madata_df))
    other_titles = set(get_valid_title_series(other_df))
    shared_titles = madata_titles & other_titles

    madata_count = int((df['Source_clean'] == 'MADATA').sum())
    other_count = int((df['Source_clean'] != 'MADATA').sum())
    intersection_count = int(madata_df['Title_clean'].isin(shared_titles).sum())

    overlap_df = pd.DataFrame({
        'Category': ['MADATA', 'Other Repos', 'Intersection'],
        'Count': [madata_count, other_count, intersection_count],
    })

    total = madata_count + other_count
    overlap_df['Percentage'] = overlap_df['Count'] / total * 100 if total > 0 else 0.0
    overlap_df['Label'] = overlap_df.apply(
        lambda row: format_count_percentage(row['Count'], total), axis=1
    )

    fig = px.bar(
        overlap_df,
        x="Category",
        y="Count",
        color="Category",
        text="Label",
        title=wrap_title(f"Datasets in MADATA & Other Repositories (n={total_n})"),
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={'Count': 'Number of Datasets'}
    )
    fig.update_layout(
        width=600,
        height=400,
        showlegend=False,
        yaxis=dict(range=[0, overlap_df['Count'].max() * 1.2]),
        title_x=0
    )
    fig.update_traces(textposition='outside', cliponaxis=False)
    return fig


def plot_dataset_source_distribution(df, total_n):
    source_counts = df['Source_clean'].value_counts().reset_index()
    source_counts.columns = ['Source', 'Count']
    source_counts = source_counts.sort_values(by='Count', ascending=False)
    total = source_counts['Count'].sum()
    source_counts['Label'] = source_counts.apply(
        lambda row: format_count_percentage(row['Count'], total), axis=1
    )
    fig = px.bar(
        source_counts,
        x='Source',
        y='Count',
        color='Source',
        title=wrap_title(f'Datasets per Source (n={total_n})'),
        text='Label',
        color_discrete_map=get_source_color_map(source_counts['Source']),
        labels={'Count': 'Number of Datasets'}
    )
    fig.update_layout(
        width=600,
        height=450,
        showlegend=False,
        margin=dict(t=80, b=20, l=20, r=20),
        title_x=0
    )
    fig.update_traces(
        textposition='outside',
        cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>Number of Datasets: %{y}<extra></extra>"
    )
    return fig


def plot_yearly_dataset_bar(df, total_n):
    year_source = df.groupby(['Year', 'Source_clean']).size().reset_index(name='Count').dropna()
    year_source = year_source.rename(columns={'Source_clean': 'Source'})
    fig = px.bar(year_source, x='Year', y='Count', color='Source',
                 title=wrap_title(f'Datasets per Year by Source (n={total_n})'), text='Count',
                 color_discrete_map=get_source_color_map(year_source['Source']),
                 labels={'Count': 'Number of Datasets'})
    fig.update_layout(width=600, height=400, title_x=0)
    fig.update_traces(cliponaxis=False)
    return fig


def plot_top_creators(df):
    creators_series = df['Creators'].dropna().str.split(';').explode().value_counts().head(30).reset_index()
    creators_series.columns = ['Creator', 'Count']
    fig = px.bar(creators_series, y='Creator', x='Count', orientation='h',
                 title=wrap_title('Top 30 Creators'), color='Count',
                 color_continuous_scale='Tealgrn', text='Count',
                 labels={'Count': 'Number of Datasets'})
    fig.update_layout(width=1000, height=1000, yaxis=dict(autorange="reversed"), showlegend=False, title_x=0)
    fig.update_traces(textposition='inside')
    return fig


def plot_yearly_line_trend(df, total_n):
    year_source = df.groupby(['Year', 'Source_clean']).size().reset_index(name='Count').dropna()
    year_source = year_source.rename(columns={'Source_clean': 'Source'})
    fig = px.line(year_source, x='Year', y='Count', color='Source', markers=True,
                  title=wrap_title(f'Yearly Dataset Trend per Source (n={total_n})'),
                  color_discrete_map=get_source_color_map(year_source['Source']),
                  labels={'Count': 'Number of Datasets', 'Year': 'Year', 'Source': 'Source'})
    fig.update_layout(width=600, height=400, title_x=0)
    return fig


def plot_tracking_line_trend(tracking_df, total_n):
    tracking_df = tracking_df.sort_values(by=['t', 'source'])
    fig = px.line(
        tracking_df,
        x='month',
        y='dataset',
        color='source',
        markers=True,
        title=wrap_title(f'Datasets per Source and Intersection with MADATA over Time (n={total_n})'),
        color_discrete_map=get_source_color_map(tracking_df['source']),
        labels={'dataset': 'Number of Datasets', 'month': 'Month', 'source': 'Source'}
    )

    intersection_df = tracking_df.groupby('month', as_index=False).agg({
        't': 'max',
        'intersection': 'max',
        'intersection_perc': 'max'
    })
    intersection_df = intersection_df.sort_values(by='t')
    fig.add_trace(go.Scatter(
        x=intersection_df['month'],
        y=intersection_df['intersection'],
        mode='lines+markers',
        name='Intersection with MADATA',
        line=dict(color='black', dash='dash'),
        marker=dict(size=8),
        customdata=intersection_df['intersection_perc'],
        hovertemplate="<b>%{x}</b><br>Intersection with MADATA: %{y}<br>Intersection (%): %{customdata[0]:.1f}%<extra></extra>"
    ))
    fig.update_layout(width=600, height=400, title_x=0)
    return fig


# ---------- Word Cloud Generation ----------
def generate_wordcloud(df, output_path):
    text = ' '.join(df['Title'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# ---------- HTML Export ----------
def generate_html(figures, wordcloud_path, output_path):
    main_plots_html = '\n'.join(
        f"<div class='plot'>{pio.to_html(fig, full_html=False, include_plotlyjs=('cdn' if i == 0 else False))}</div>"
        for i, fig in enumerate(figures[:5])
    )
    top_creators_plot_html = ""
    if len(figures) > 5:
        top_creators_plot_html = f"<div class='plot plot-full'>{pio.to_html(figures[5], full_html=False, include_plotlyjs=False)}</div>"
    html = f"""
    <html>
    <head>
      <title>Mannheim Data Bibliography Dashboard</title>
      <style>
        :root {{ --plot-gap: 20px; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; padding: 20px; color: #343a40; }}
        h1 {{ text-align: center; margin-bottom: 40px; }}
        .plot-container {{ display: flex; flex-wrap: wrap; justify-content: center; gap: var(--plot-gap); }}
        .plot {{ box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-radius: 10px; background-color: white; padding: 10px; }}
        .plot-full {{ width: 100%; max-width: 1100px; margin: var(--plot-gap) auto 0 auto; }}
        .image-plot {{ width: 800px; margin: 0 auto; display: block; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
      </style>
    </head>
    <body>
      <h1>Mannheim Data Bibliography Dashboard</h1>
      <div class='plot-container'>
        {main_plots_html}
      </div>
      {top_creators_plot_html}
      <h2 style='text-align: center;'>Most Frequent Terms in Titles</h2>
      <img src='title_wordcloud.png' alt='Title WordCloud' class='image-plot'/>
    </body>
    </html>
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


# ---------- Main ----------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    df = load_and_prepare_data(DATA_PATH)
    log_data_quality(df)
    tracking_df = update_tracking_csv(df, TRACKING_PATH)
    total_n = compute_total_n(df)

    # Generate visualizations
    figures = [
        plot_madata_ratio_with_intersection(df, total_n),
        plot_dataset_source_distribution(df, total_n),
        plot_yearly_dataset_bar(df, total_n),
        plot_yearly_line_trend(df, total_n),
        plot_tracking_line_trend(tracking_df, total_n),
        plot_top_creators(df),
    ]

    generate_wordcloud(df, WORDCLOUD_IMG)
    generate_html(figures, WORDCLOUD_IMG, HTML_OUTPUT)

    print(f"Dashboard saved to: {HTML_OUTPUT}")


if __name__ == "__main__":
    main()
