import os
import zipfile
import pandas as pd
import numpy as np
import pickle
import requests
from src import config

def download_movielens_small():
    """Downloads and extracts the MovieLens latest-small dataset if not present."""
    if config.RATINGS_FILE.exists() and config.MOVIES_FILE.exists():
        print("MovieLens datasets already exist.")
        return

    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    zip_path = config.RAW_DATA_DIR / "ml-latest-small.zip"
    
    print(f"Downloading MovieLens Small from {url}...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # The zip file extracts into a subfolder ml-latest-small/
        zip_ref.extractall(config.RAW_DATA_DIR)
        
    # Move files from ml-latest-small to raw/
    extracted_dir = config.RAW_DATA_DIR / "ml-latest-small"
    for file_name in ["ratings.csv", "movies.csv", "links.csv"]:
        src = extracted_dir / file_name
        dest = config.RAW_DATA_DIR / file_name
        if src.exists():
            if dest.exists():
                os.remove(dest)
            src.rename(dest)
            
    # Clean up zip and temporary directory
    if zip_path.exists():
        os.remove(zip_path)
    if extracted_dir.exists():
        # Remove empty dir or contents
        for item in extracted_dir.iterdir():
            os.remove(item)
        extracted_dir.rmdir()
        
    print("MovieLens dataset downloaded and extracted successfully.")

def generate_mock_tmdb(movies_df):
    """Generates a mock TMDB dataset based on MovieLens movies to ensure content-based similarity works out of the box."""
    print("Kaggle TMDB dataset not found. Generating mock metadata for TMDB integration...")
    
    # Cast, Directors, Keywords pool
    actors = ["Tom Hanks", "Leonardo DiCaprio", "Scarlett Johansson", "Brad Pitt", "Meryl Streep", 
              "Morgan Freeman", "Robert Downey Jr.", "Natalie Portman", "Denzel Washington", "Al Pacino"]
    directors = ["Steven Spielberg", "Christopher Nolan", "Martin Scorsese", "Quentin Tarantino", 
                 "Alfred Hitchcock", "Stanley Kubrick", "Ridley Scott", "James Cameron", "David Fincher"]
    keywords_pool = ["space", "crime", "love", "war", "explosion", "journey", "future", "magic", 
                     "superhero", "survival", "revenge", "family", "conspiracy", "ai", "mystery"]

    # Generate mock metadata for each movie
    records = []
    np.random.seed(42)
    for idx, row in movies_df.iterrows():
        title = row['title']
        genres_list = row['genres'].split('|')
        
        # Select random elements
        movie_actors = list(np.random.choice(actors, size=3, replace=False))
        movie_director = np.random.choice(directors)
        movie_keywords = list(np.random.choice(keywords_pool, size=3, replace=False))
        
        # Create an overview text
        genres_str = ", ".join(genres_list)
        overview = f"A thrilling {genres_str} movie featuring {movie_actors[0]} and {movie_actors[1]}, directed by {movie_director}. A masterpiece dealing with themes of {', '.join(movie_keywords)}."
        
        records.append({
            'budget': int(np.random.choice([10, 50, 100, 200]) * 1000000),
            'genres': [{"id": i, "name": g} for i, g in enumerate(genres_list)],
            'homepage': "",
            'id': int(row['movieId']), # Use movieId as TMDB id for mapping simplicity in fallback
            'keywords': [{"id": i, "name": k} for i, k in enumerate(movie_keywords)],
            'original_language': 'en',
            'original_title': title,
            'overview': overview,
            'popularity': np.random.uniform(1.0, 100.0),
            'production_companies': [],
            'production_countries': [],
            'release_date': "2000-01-01",
            'revenue': int(np.random.choice([20, 80, 150, 400]) * 1000000),
            'runtime': int(np.random.randint(90, 160)),
            'spoken_languages': [],
            'status': "Released",
            'tagline': "An epic cinematic experience.",
            'title': title,
            'vote_average': np.random.uniform(5.0, 9.0),
            'vote_count': int(np.random.randint(100, 5000))
        })
        
    tmdb_df = pd.DataFrame(records)
    tmdb_df.to_csv(config.TMDB_METADATA_FILE, index=False)
    
    # Generate mock credits
    credits_records = []
    for idx, row in movies_df.iterrows():
        movie_id = int(row['movieId'])
        movie_actors = list(np.random.choice(actors, size=3, replace=False))
        movie_director = np.random.choice(directors)
        
        cast = [{"cast_id": i, "character": f"Character {i}", "credit_id": f"c_{i}", "id": i*10, "name": name, "order": i} 
                for i, name in enumerate(movie_actors)]
        crew = [{"credit_id": "dir_1", "department": "Directing", "id": 999, "job": "Director", "name": movie_director}]
        
        credits_records.append({
            'movie_id': movie_id,
            'title': row['title'],
            'cast': str(cast),
            'crew': str(crew)
        })
    credits_df = pd.DataFrame(credits_records)
    credits_df.to_csv(config.TMDB_CREDITS_FILE, index=False)
    print("Generated mock TMDB movies & credits datasets.")

def generate_mock_imdb_reviews():
    """Generates mock reviews for sentiment re-ranking training/validation if missing."""
    if config.IMDB_REVIEWS_FILE.exists():
        return
        
    print("Kaggle IMDB dataset not found. Generating mock reviews...")
    pos_reviews = [
        "Absolutely loved this movie! The characters were fantastic and the plot kept me hooked.",
        "One of the best films of the decade. Stupendous acting and visual storytelling.",
        "A true masterpiece. Heartwarming, emotional, and highly entertaining.",
        "I was pleasantly surprised. The directing was incredible and the soundtrack was stunning.",
        "Highly recommended! A great movie to watch with family."
    ] * 20
    
    neg_reviews = [
        "Waste of time. Extremely boring, slow paced, and poor character development.",
        "I hated this movie. The acting was wood-like and the script made no sense.",
        "Very disappointing. It had so much potential but failed on every level.",
        "Do not watch this. It is a total snooze fest with mediocre acting.",
        "The plot was filled with holes and the ending was terrible."
    ] * 20
    
    reviews = pos_reviews + neg_reviews
    sentiments = ["positive"] * len(pos_reviews) + ["negative"] * len(neg_reviews)
    
    df = pd.DataFrame({
        'review': reviews,
        'sentiment': sentiments
    })
    df.to_csv(config.IMDB_REVIEWS_FILE, index=False)
    print("Generated mock IMDB Reviews dataset.")

def load_and_preprocess_data():
    """Loads all datasets, cleans them, constructs features for TF-IDF and saves processed files."""
    download_movielens_small()
    
    # Load MovieLens movies
    movies_df = pd.read_csv(config.MOVIES_FILE)
    
    # Check TMDB datasets
    if not config.TMDB_METADATA_FILE.exists() or not config.TMDB_CREDITS_FILE.exists():
        generate_mock_tmdb(movies_df)
    
    # Check IMDB reviews
    generate_mock_imdb_reviews()
    
    print("Loading TMDB Metadata and Credits...")
    # Read files
    # TMDB metadata could be a string evaluation for json columns, handle carefully
    tmdb_df = pd.read_csv(config.TMDB_METADATA_FILE)
    credits_df = pd.read_csv(config.TMDB_CREDITS_FILE)
    
    # Clean/parse JSON columns if string (this applies to Kaggle TMDB dataset)
    import ast
    def safe_parse(val, key="name"):
        if pd.isna(val):
            return []
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return [d[key] for d in parsed if isinstance(d, dict) and key in d]
            return []
        except Exception:
            return []

    # If it is raw TMDB metadata, parse genres, cast, crew
    # Let's check if the columns are strings
    if isinstance(tmdb_df['genres'].iloc[0], str) and tmdb_df['genres'].iloc[0].startswith('['):
        tmdb_df['genres_list'] = tmdb_df['genres'].apply(lambda x: safe_parse(x, "name"))
    else:
        # Custom mock or pre-parsed
        # If it's mock, it is a string of list of dict, eval it
        def eval_genres(val):
            try:
                return [g['name'] for g in ast.literal_eval(val)]
            except:
                return []
        tmdb_df['genres_list'] = tmdb_df['genres'].apply(eval_genres)
        
    # Parse keywords
    if isinstance(tmdb_df['keywords'].iloc[0], str) and tmdb_df['keywords'].iloc[0].startswith('['):
        tmdb_df['keywords_list'] = tmdb_df['keywords'].apply(lambda x: safe_parse(x, "name"))
    else:
        def eval_kw(val):
            try:
                return [k['name'] for k in ast.literal_eval(val)]
            except:
                return []
        tmdb_df['keywords_list'] = tmdb_df['keywords'].apply(eval_kw)

    # Process credits (cast & crew)
    # Merging credits and tmdb_df
    # In real TMDB, the ID is 'id', credits has 'movie_id'
    if 'movie_id' in credits_df.columns:
        credits_df = credits_df.rename(columns={'movie_id': 'id'})
    
    merged_tmdb = pd.merge(tmdb_df, credits_df[['id', 'cast', 'crew']], on='id', how='inner')
    
    # Extract director and top 3 actors
    def get_director(crew_str):
        try:
            crew = ast.literal_eval(crew_str)
            for member in crew:
                if member.get('job') == 'Director':
                    return member.get('name')
            return ""
        except:
            return ""

    def get_top_cast(cast_str):
        try:
            cast = ast.literal_eval(cast_str)
            return [member.get('name') for member in cast[:3]]
        except:
            return []
            
    merged_tmdb['director'] = merged_tmdb['crew'].apply(get_director)
    merged_tmdb['top_cast'] = merged_tmdb['cast'].apply(get_top_cast)
    
    # Build text soup for TF-IDF content extraction
    def create_soup(x):
        genres = " ".join(x['genres_list'])
        keywords = " ".join(x['keywords_list'])
        cast = " ".join(x['top_cast'])
        director = x['director']
        overview = str(x['overview']) if not pd.isna(x['overview']) else ""
        # Combine everything, making director twice as important by duplicating it
        return f"{genres} {keywords} {cast} {director} {director} {overview}".lower()
        
    merged_tmdb['soup'] = merged_tmdb.apply(create_soup, axis=1)
    
    # Match with MovieLens via links.csv
    # links.csv has movieId, imdbId, tmdbId
    links_df = pd.read_csv(config.LINKS_FILE)
    
    # Drop movies that don't have tmdbId
    links_df = links_df.dropna(subset=['tmdbId'])
    links_df['tmdbId'] = links_df['tmdbId'].astype(int)
    
    # Merge MovieLens movies with links
    movielens_merged = pd.merge(movies_df, links_df, on='movieId', how='inner')
    
    # Now merge with TMDB metadata on tmdbId
    # If tmdbId matches 'id' in TMDB metadata
    final_movies = pd.merge(movielens_merged, merged_tmdb, left_on='tmdbId', right_on='id', how='inner', suffixes=('_ml', '_tmdb'))
    
    # Clean final movies metadata dataframe
    columns_to_keep = [
        'movieId', 'title_ml', 'genres_ml', 'tmdbId', 'overview', 'soup', 
        'director', 'top_cast', 'genres_list', 'keywords_list', 'vote_average', 'vote_count'
    ]
    final_movies = final_movies[columns_to_keep].rename(columns={'title_ml': 'title', 'genres_ml': 'genres'})
    
    # Save processed movies
    processed_path = config.PROCESSED_DATA_DIR / "preprocessed_movies.csv"
    final_movies.to_csv(processed_path, index=False)
    print(f"Preprocessed metadata for {len(final_movies)} movies and saved to {processed_path}")
    
    # Save mapping helper
    with open(config.MOVIES_METADATA_PATH, "wb") as f:
        pickle.dump(final_movies, f)
        
    print("Data processing pipeline executed successfully.")

if __name__ == "__main__":
    load_and_preprocess_data()
