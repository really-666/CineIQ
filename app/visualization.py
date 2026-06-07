import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def extract_year(title):
    """Extracts year of release from movie title string like 'Toy Story (1995)'."""
    if not isinstance(title, str):
        return None
    match = re.search(r'\((\d{4})\)', title)
    if match:
        return int(match.group(1))
    return None

def create_genre_radar_chart(history_df):
    """Generates a Plotly radar chart displaying movie count and average ratings by genre."""
    if history_df.empty:
        return None
        
    # Split genres and explode to get individual rows per genre
    # E.g. "Action|Adventure" -> 2 rows
    df = history_df.copy()
    df['genre_split'] = df['genres'].str.split('|')
    df = df.explode('genre_split')
    
    # Aggregate counts and average ratings
    genre_stats = df.groupby('genre_split').agg(
        count=('movieId', 'count'),
        avg_rating=('user_rating', 'mean')
    ).reset_index()
    
    # Sort and take top 10 for clean display
    genre_stats = genre_stats.sort_values(by='count', ascending=False).head(10)
    
    # Normalize count to a 0-5 scale so it plots nicely alongside ratings
    max_count = genre_stats['count'].max() if genre_stats['count'].max() > 0 else 1
    genre_stats['norm_count'] = (genre_stats['count'] / max_count) * 5.0
    
    # Build chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=genre_stats['norm_count'],
        theta=genre_stats['genre_split'],
        fill='toself',
        name='Relative Frequency',
        line_color='#636EFA',
        opacity=0.6
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=genre_stats['avg_rating'],
        theta=genre_stats['genre_split'],
        fill='toself',
        name='Avg Rating',
        line_color='#EF553B',
        opacity=0.5
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )
        ),
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),
        margin=dict(l=40, r=40, t=20, b=20),
        title={
            'text': "Genre Distribution & Ratings",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16}
        }
    )
    
    return fig

def create_decade_preference_chart(history_df):
    """Generates a bar chart mapping the user's favorite movie decades."""
    if history_df.empty:
        return None
        
    df = history_df.copy()
    df['year'] = df['title'].apply(extract_year)
    df = df.dropna(subset=['year'])
    
    if df.empty:
        return None
        
    # Calculate decade
    df['decade'] = (df['year'] // 10 * 10).astype(str) + "s"
    
    decade_stats = df.groupby('decade').agg(
        count=('movieId', 'count'),
        avg_rating=('user_rating', 'mean')
    ).reset_index().sort_values(by='decade')
    
    fig = px.bar(
        decade_stats, 
        x='decade', 
        y='count',
        color='avg_rating',
        color_continuous_scale='Plasma',
        labels={'count': 'Movies Watched', 'avg_rating': 'Avg Rating', 'decade': 'Era'},
        text='count'
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),
        xaxis=dict(showgrid=False, title_font=dict(size=12)),
        yaxis=dict(showgrid=True, gridcolor='#333333', title_font=dict(size=12)),
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_colorbar=dict(title="Rating", tickvals=[1, 2, 3, 4, 5])
    )
    
    fig.update_traces(textposition='outside')
    
    return fig

def create_affinity_charts(history_df, type_='director'):
    """Generates bar charts of director or actor affinities based on user history."""
    if history_df.empty:
        return None
        
    # We must load movies metadata to get director/top_cast details for the history movies
    # Wait, history_df passed here could already contain those if pre-merged, 
    # but to be safe, we merge here if columns don't exist
    df = history_df.copy()
    
    if type_ == 'director':
        if 'director' not in df.columns:
            return None
        df = df.dropna(subset=['director'])
        df = df[df['director'] != ""]
        group_col = 'director'
    else:
        # actor
        if 'top_cast' not in df.columns:
            return None
        # top_cast is a list or string representation of list
        import ast
        def parse_cast(val):
            if isinstance(val, list):
                return val
            try:
                return ast.literal_eval(val)
            except:
                return []
        df['cast_parsed'] = df['top_cast'].apply(parse_cast)
        df = df.explode('cast_parsed')
        df = df.dropna(subset=['cast_parsed'])
        df = df[df['cast_parsed'] != ""]
        group_col = 'cast_parsed'
        
    if df.empty:
        return None
        
    stats = df.groupby(group_col).agg(
        count=('movieId', 'count'),
        avg_rating=('user_rating', 'mean')
    ).reset_index()
    
    # Filter for items that have been watched at least once and sort
    stats = stats.sort_values(by=['count', 'avg_rating'], ascending=[False, False]).head(8)
    
    title_label = "Director Affinities" if type_ == 'director' else "Actor Affinities"
    
    fig = px.bar(
        stats,
        x='avg_rating',
        y=group_col,
        orientation='h',
        color='count',
        color_continuous_scale='Viridis',
        labels={'avg_rating': 'Average Rating Given', group_col: 'Name', 'count': 'Watches'},
        text_auto='.1f'
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),
        xaxis=dict(showgrid=True, gridcolor='#333333', range=[0, 5.2]),
        yaxis=dict(showgrid=False, categoryorder="total ascending"),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig
