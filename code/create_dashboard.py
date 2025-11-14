import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

# ---------- Configuration ----------
DATA_PATH = '../data/unified_mannheim_metadata.csv'
OUTPUT_DIR = '../docs'
WORDCLOUD_IMG = os.path.join(OUTPUT_DIR, 'title_wordcloud.png')
HTML_OUTPUT = os.path.join(OUTPUT_DIR, 'index.html')


# ---------- Data Loading ----------
def load_and_prepare_data(path):
    df = pd.read_csv(path)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').dropna().astype('Int64')
    df['Source_clean'] = df['Source'].fillna('Unknown')
    df['Affiliations_clean'] = df['Affiliations'].fillna('').replace({
        'University of Mannheim': 'Universität Mannheim'
    }, regex=False)
    df['Affiliations_clean'] = df['Affiliations_clean'].str.replace(
        r'.*Universität Mannheim.*', 'Universität Mannheim', regex=True
    )
    df['Title_clean'] = df['Title'].astype(str).str.strip().str.lower()
    return df


# ---------- Plotting Functions ----------
def plot_madata_ratio_with_intersection(df):
    madata_df = df[df['Source_clean'].str.lower() == 'madata']
    other_df = df[df['Source_clean'].str.lower() != 'madata']

    madata_titles = set(madata_df['Title_clean'].dropna())
    other_titles = set(other_df['Title_clean'].dropna())
    shared_titles = madata_titles & other_titles

    madata_only = madata_titles
    other_only = other_titles

    overlap_df = pd.DataFrame({
        'Category': ['MADATA', 'Other Repos', 'Intersection'],
        'Count': [len(madata_only), len(other_only), len(shared_titles)],
    })

    total = overlap_df['Count'].sum()
    overlap_df['Percentage'] = overlap_df['Count'] / total * 100
    overlap_df['Label'] = overlap_df.apply(
        lambda row: f"<b>{row['Count']}</b><br>{row['Percentage']:.1f}%", axis=1
    )

    fig = px.bar(
        overlap_df,
        x="Category",
        y="Count",
        color="Category",
        text="Label",
        title="Datasets in MADATA & Other Repositories",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(
        width=600,
        height=400,
        showlegend=False,
        yaxis=dict(range=[0, overlap_df['Count'].max() * 1.2])
    )
    fig.update_traces(textposition='outside')
    return fig


def plot_dataset_source_distribution(df):
    source_counts = df['Source'].value_counts().reset_index()
    source_counts.columns = ['Source', 'Count']
    source_counts = source_counts.sort_values(by='Count', ascending=False)
    total = source_counts['Count'].sum()
    source_counts['Label'] = source_counts.apply(
        lambda row: f"{row['Count']} ({(row['Count'] / total * 100):.1f}%)", axis=1
    )
    fig = go.Figure(data=[go.Pie(
        labels=source_counts['Source'],
        values=source_counts['Count'],
        hole=0.5,
        marker=dict(colors=px.colors.qualitative.Pastel),
        text=source_counts['Label'],
        textinfo='text',
        textposition='inside',
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    )])
    fig.update_layout(
        annotations=[dict(
            text=f"<b>Total</b><br>{total}", 
            x=0.5, y=0.5, font_size=16, showarrow=False
        )],
        title='Datasets per Source',
        width=600,
        height=450,
        showlegend=True,
        legend_title_text='Source',
        margin=dict(t=80, b=20, l=20, r=20),
        title_x=0.5
    )
    return fig


def plot_yearly_dataset_bar(df):
    year_source = df.groupby(['Year', 'Source']).size().reset_index(name='Count').dropna()
    fig = px.bar(year_source, x='Year', y='Count', color='Source',
                 title='Datasets per Year by Source', text='Count',
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(width=600, height=400)
    return fig


def plot_top_creators(df):
    creators_series = df['Creators'].dropna().str.split(';').explode().value_counts().head(30).reset_index()
    creators_series.columns = ['Creator', 'Count']
    fig = px.bar(creators_series, y='Creator', x='Count', orientation='h',
                 title='Top 30 Creators', color='Count',
                 color_continuous_scale='Tealgrn', text='Count')
    fig.update_layout(width=1000, height=1000, yaxis=dict(autorange="reversed"), showlegend=False)
    fig.update_traces(textposition='inside')
    return fig


def plot_yearly_line_trend(df):
    year_source = df.groupby(['Year', 'Source']).size().reset_index(name='Count').dropna()
    fig = px.line(year_source, x='Year', y='Count', color='Source', markers=True,
                  title='Yearly Dataset Trend per Source')
    fig.update_layout(width=600, height=400)
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
    plots_html = '\n'.join(
        f"<div class='plot'>{pio.to_html(fig, full_html=False, include_plotlyjs=('cdn' if i == 0 else False))}</div>"
        for i, fig in enumerate(figures)
    )
    html = f"""
    <html>
    <head>
      <title>Mannheim Data Bibliography Dashboard</title>
      <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; padding: 20px; color: #343a40; }}
        h1 {{ text-align: center; margin-bottom: 40px; }}
        .plot-container {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; }}
        .plot {{ box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-radius: 10px; background-color: white; padding: 10px; }}
        .image-plot {{ width: 800px; margin: 0 auto; display: block; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
      </style>
    </head>
    <body>
      <h1>Mannheim Data Bibliography Dashboard</h1>
      <div class='plot-container'>
        {plots_html}
      </div>
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
    df = load_and_prepare_data(DATA_PATH)

    # Generate visualizations
    figures = [
        plot_madata_ratio_with_intersection(df),
        plot_dataset_source_distribution(df),
        plot_yearly_dataset_bar(df),
        plot_yearly_line_trend(df),
        plot_top_creators(df),
    ]

    generate_wordcloud(df, WORDCLOUD_IMG)
    generate_html(figures, WORDCLOUD_IMG, HTML_OUTPUT)

    print(f"✅ Dashboard saved to: {HTML_OUTPUT}")


if __name__ == "__main__":
    main()
