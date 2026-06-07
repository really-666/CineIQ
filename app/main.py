import streamlit as st
import pandas as pd
import requests
import sys
import os
import re

# Add parent directory to path so we can import modules locally
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from app.visualization import (
    create_genre_radar_chart, 
    create_decade_preference_chart, 
    create_affinity_charts
)

# Page configuration
st.set_page_config(
    page_title="CineIQ | Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark CSS Styling
st.markdown("""
<style>
    /* Global Background and Fonts */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em;
    }
    
    .logo-container {
        display: flex;
        align-items: center;
        margin-bottom: 25px;
    }
    
    .logo-text {
        font-size: 42px;
        font-weight: 900;
        background: linear-gradient(90deg, #ff7e5f 0%, #feb47b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 2px;
    }
    
    /* Cards and Glassmorphism */
    .movie-card {
        background: rgba(22, 27, 34, 0.6);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: all 0.2s ease;
    }
    
    .movie-card:hover {
        transform: translateY(-2px);
        border-color: #58a6ff;
        box-shadow: 0 6px 16px rgba(88, 166, 255, 0.1);
    }
    
    .movie-title {
        font-size: 20px;
        font-weight: 700;
        color: #58a6ff;
        margin-bottom: 5px;
    }
    
    .movie-meta {
        font-size: 13px;
        color: #8b949e;
        margin-bottom: 12px;
    }
    
    .movie-explanation {
        background-color: rgba(33, 38, 45, 0.8);
        border-left: 4px solid #f9826c;
        padding: 10px 15px;
        border-radius: 4px;
        font-size: 14px;
        color: #e6edf3;
        margin-top: 10px;
    }
    
    /* Genre tags */
    .genre-tag {
        display: inline-block;
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 11px;
        color: #8b949e;
        margin-right: 5px;
        margin-bottom: 5px;
    }
    
    /* Sentiment badges */
    .sentiment-positive {
        background-color: rgba(46, 160, 67, 0.15);
        color: #3fb950;
        border: 1px solid rgba(46, 160, 67, 0.4);
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# API service configuration
API_URL = "http://127.0.0.1:8000"

# Check API availability and toggle local mode
@st.cache_resource
def check_api():
    try:
        response = requests.get(f"{API_URL}/health", timeout=1)
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException:
        pass
    return False

# Initialize local recommender if API is unavailable
@st.cache_resource
def get_local_recommender():
    try:
        from src.recommender import CineIQRecommender
        return CineIQRecommender()
    except Exception as e:
        st.error(f"Failed to load local Recommender Engine: {e}")
        return None

API_AVAILABLE = check_api()

# Sidebar Setup
with st.sidebar:
    st.markdown('<div class="logo-container"><span class="logo-text">CINEIQ</span></div>', unsafe_allow_html=True)
    st.markdown("### Controls & Settings")
    
    # Mode selection
    mode = st.radio(
        "Navigation",
        ["🎯 Recommendations", "🔍 Find Similar Movies", "📊 Taste Dashboard"]
    )
    
    st.markdown("---")
    
    # Mode configurations
    # Read user list from dataset to make it valid
    if config.RATINGS_FILE.exists():
        ratings = pd.read_csv(config.RATINGS_FILE)
        available_users = sorted(ratings['userId'].unique())
        # Provide some default user selections
        user_id = st.selectbox("Select User Profile", available_users, index=0)
    else:
        user_id = st.number_input("User ID", min_value=1, value=1, step=1)
        
    st.markdown("---")
    
    # API / Local Status Indicator
    if API_AVAILABLE:
        st.success("🟢 Connected to FastAPI Server")
    else:
        st.info("🟡 Server offline. Serving locally.")
        local_rec = get_local_recommender()
        if local_rec is None:
            st.warning("⚠️ Action required: Run preprocessing/model training.")

# Main Application Layout
st.markdown('<div class="logo-container"><span class="logo-text">CINEIQ</span></div>', unsafe_allow_html=True)
st.markdown("*An open, explainable movie recommendation engine combining SVD & Content Filtering*")

# Helper function to fetch recommendations
def fetch_recommendations(uid, limit=10):
    if API_AVAILABLE:
        try:
            res = requests.get(f"{API_URL}/recommend", params={"user_id": uid, "n": limit}, timeout=5)
            if res.status_code == 200:
                return res.json()
        except Exception as e:
            st.error(f"API request failed: {e}")
            
    # Fallback to local execution
    if not API_AVAILABLE and 'local_rec' in globals() and local_rec is not None:
        try:
            recs_df = local_rec.get_recommendations(uid, n=limit)
            
            # Format dataframe matching API JSON layout
            recs = []
            for _, row in recs_df.iterrows():
                cast_list = row['top_cast']
                if isinstance(cast_list, str):
                    import ast
                    try:
                        cast_list = ast.literal_eval(cast_list)
                    except:
                        cast_list = [cast_list]
                        
                recs.append({
                    "movieId": int(row['movieId']),
                    "title": row['title'],
                    "genres": row['genres'],
                    "director": str(row['director']) if not pd.isna(row['director']) else "Unknown",
                    "top_cast": cast_list,
                    "score": float(row.get('final_score', row.get('hybrid_score', 0.0))),
                    "explanation": row['explanation'],
                    "audience_summary": row.get('audience_summary', "Good reviews")
                })
            return recs
        except Exception as e:
            st.error(f"Local recommendation generation failed: {e}")
            
    return None

# Helper to fetch similar movies
def fetch_similar_movies(title, limit=10):
    if API_AVAILABLE:
        try:
            res = requests.get(f"{API_URL}/similar", params={"movie_title": title, "n": limit}, timeout=5)
            if res.status_code == 200:
                return res.json()
        except Exception as e:
            st.error(f"API request failed: {e}")
            
    # Local fallback
    if not API_AVAILABLE and 'local_rec' in globals() and local_rec is not None:
        try:
            sims_df = local_rec.get_similar_movies(title, n=limit)
            if sims_df.empty:
                return []
                
            sims = []
            for _, row in sims_df.iterrows():
                cast_list = row['top_cast']
                if isinstance(cast_list, str):
                    import ast
                    try:
                        cast_list = ast.literal_eval(cast_list)
                    except:
                        cast_list = [cast_list]
                        
                sims.append({
                    "movieId": int(row['movieId']),
                    "title": row['title'],
                    "genres": row['genres'],
                    "director": str(row['director']) if not pd.isna(row['director']) else "Unknown",
                    "top_cast": cast_list,
                    "similarity": float(row['similarity'])
                })
            return sims
        except Exception as e:
            st.error(f"Local similarity search failed: {e}")
            
    return None

# Helper to fetch user history
def fetch_user_history(uid):
    if API_AVAILABLE:
        try:
            res = requests.get(f"{API_URL}/user-history", params={"user_id": uid}, timeout=5)
            if res.status_code == 200:
                return pd.DataFrame(res.json())
        except Exception as e:
            pass
            
    # Local fallback
    if not API_AVAILABLE and 'local_rec' in globals() and local_rec is not None:
        try:
            hist_df = local_rec.get_user_liked_movies(uid)
            if not hist_df.empty:
                return hist_df.rename(columns={'rating': 'user_rating'})
        except Exception as e:
            pass
            
    return pd.DataFrame()

# Navigation routing
if mode == "🎯 Recommendations":
    st.header(f"Personalized Recommendations for User #{user_id}")
    
    # Fetch user history to display context
    history_df = fetch_user_history(user_id)
    
    if not history_df.empty:
        with st.expander("👁️ View Your Rating History (Movies rated 4.0+ stars)"):
            cols = st.columns(3)
            # Display history in columns
            for idx, row in history_df.iterrows():
                col_idx = idx % 3
                with cols[col_idx]:
                    st.markdown(f"**{row['title']}**")
                    genres_split = row['genres'].split('|')
                    for g in genres_split:
                        st.markdown(f'<span class="genre-tag">{g}</span>', unsafe_allow_html=True)
                    st.markdown(f"⭐ Rating: {row['user_rating']:.1f}")
                    st.markdown("---")
    else:
        st.info("No rating history found. Displaying standard cold-start fallback suggestions.")

    # Recommendations panel
    num_recs = st.slider("Number of recommendations", 5, 20, 10)
    
    if st.button("✨ Get Recommendations", type="primary"):
        with st.spinner("Generating explainable recommendations..."):
            recs = fetch_recommendations(user_id, num_recs)
            
            if recs:
                st.markdown("### Top Recommendations Found:")
                for r in recs:
                    genres_html = "".join([f'<span class="genre-tag">{g}</span>' for g in r['genres'].split('|')])
                    cast_str = ", ".join(r['top_cast']) if r['top_cast'] else "Unknown"
                    
                    st.markdown(f"""
                    <div class="movie-card">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <span class="movie-title">{r['title']}</span>
                            <span class="sentiment-positive">💬 {r['audience_summary']}</span>
                        </div>
                        <div class="movie-meta">
                            Director: {r['director']} | Starring: {cast_str}
                        </div>
                        <div style="margin-bottom: 15px;">
                            {genres_html}
                        </div>
                        <div class="movie-explanation">
                            💡 {r['explanation']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Could not fetch recommendations. Please ensure models are trained and pipeline has run.")

elif mode == "🔍 Find Similar Movies":
    st.header("Search Content-Similar Movies")
    
    # Select from existing movie titles if metadata loaded
    movie_title_query = ""
    if API_AVAILABLE or ('local_rec' in globals() and local_rec is not None):
        movies_df = local_rec.movies_df if not API_AVAILABLE else None
        
        # If API available, we can fetch metadata or use a text query
        # Let's provide a text input with autocompletes if we can load it locally
        if not API_AVAILABLE and movies_df is not None:
            movie_titles = sorted(movies_df['title'].tolist())
            default_idx = movie_titles.index("Rocky (1976)") if "Rocky (1976)" in movie_titles else 0
            movie_title_query = st.selectbox("Search for a movie you like:", movie_titles, index=default_idx)
        else:
            movie_title_query = st.text_input("Enter movie title (e.g. Rocky, Matrix):", "Rocky")
    else:
        movie_title_query = st.text_input("Enter movie title (e.g. Rocky, Matrix):", "Rocky")
        
    num_sims = st.slider("Number of similar movies", 5, 20, 10)
    
    if st.button("🔍 Find Similar", type="primary"):
        with st.spinner("Analyzing similarity vectors..."):
            sims = fetch_similar_movies(movie_title_query, num_sims)
            
            if sims:
                st.markdown(f"### Movies Similar to '{movie_title_query}':")
                for r in sims:
                    genres_html = "".join([f'<span class="genre-tag">{g}</span>' for g in r['genres'].split('|')])
                    cast_str = ", ".join(r['top_cast']) if r['top_cast'] else "Unknown"
                    sim_pct = int(r['similarity'] * 100)
                    
                    st.markdown(f"""
                    <div class="movie-card">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <span class="movie-title">{r['title']}</span>
                            <span class="sentiment-positive">🎯 {sim_pct}% Similarity</span>
                        </div>
                        <div class="movie-meta">
                            Director: {r['director']} | Starring: {cast_str}
                        </div>
                        <div>
                            {genres_html}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error(f"No similarity results found. Double check the spelling of '{movie_title_query}'.")

elif mode == "📊 Taste Dashboard":
    st.header(f"Taste Analytics Dashboard | User #{user_id}")
    
    # Load history
    history_df = fetch_user_history(user_id)
    
    if history_df.empty:
        st.warning(f"No viewing history found for User #{user_id}. Go back and rate some movies to generate your taste profile!")
    else:
        st.markdown("This dashboard maps out the taste profile of the selected user, analyzing their history of highly-rated films.")
        
        # Grid layout for charts
        col1, col2 = st.columns(2)
        
        # Radar chart (genres)
        with col1:
            st.subheader("Genre Preferences")
            radar_fig = create_genre_radar_chart(history_df)
            if radar_fig:
                st.plotly_chart(radar_fig, use_container_width=True)
            else:
                st.info("Not enough genre data to render radar chart.")
                
        # Decade distribution
        with col2:
            st.subheader("Favorite Movie Eras")
            decade_fig = create_decade_preference_chart(history_df)
            if decade_fig:
                st.plotly_chart(decade_fig, use_container_width=True)
            else:
                st.info("Year metadata missing or not parsable in rating history.")
                
        col3, col4 = st.columns(2)
        
        # Directors
        with col3:
            st.subheader("Filmmaker Affinities (Directors)")
            # Need to ensure we merge history with movies metadata to get director name
            if not API_AVAILABLE and local_rec is not None:
                # Merge local details
                history_detail = pd.merge(history_df[['movieId', 'user_rating']], local_rec.movies_df, on='movieId', how='inner')
                dir_fig = create_affinity_charts(history_detail, type_='director')
                if dir_fig:
                    st.plotly_chart(dir_fig, use_container_width=True)
                else:
                    st.info("No director affinities data available.")
            else:
                st.info("Director affinity mapping requires local database parsing.")
                
        # Actors
        with col4:
            st.subheader("Star Affinities (Actors)")
            if not API_AVAILABLE and local_rec is not None:
                # Merge local details
                history_detail = pd.merge(history_df[['movieId', 'user_rating']], local_rec.movies_df, on='movieId', how='inner')
                actor_fig = create_affinity_charts(history_detail, type_='actor')
                if actor_fig:
                    st.plotly_chart(actor_fig, use_container_width=True)
                else:
                    st.info("No actor affinities data available.")
            else:
                st.info("Actor affinity mapping requires local database parsing.")
