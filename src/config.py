import os
from pathlib import Path

# Base workspace directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model Storage Directory
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Datasets file paths
# MovieLens
RATINGS_FILE = RAW_DATA_DIR / "ratings.csv"
MOVIES_FILE = RAW_DATA_DIR / "movies.csv"
LINKS_FILE = RAW_DATA_DIR / "links.csv"

# TMDB Metadata
TMDB_METADATA_FILE = RAW_DATA_DIR / "tmdb_5000_movies.csv"
TMDB_CREDITS_FILE = RAW_DATA_DIR / "tmdb_5000_credits.csv"

# IMDB Reviews
IMDB_REVIEWS_FILE = RAW_DATA_DIR / "IMDB_Dataset.csv"

# Saved Model Paths
SVD_MODEL_PATH = MODELS_DIR / "svd_model.pkl"
TFIDF_MATRIX_PATH = MODELS_DIR / "tfidf_matrix.pkl"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
MOVIES_METADATA_PATH = MODELS_DIR / "movies_metadata.pkl"

# Model blending weights
HYBRID_WEIGHT_SVD = 0.6
HYBRID_WEIGHT_CONTENT = 0.4
SENTIMENT_WEIGHT = 0.2

# Development mode settings (to avoid running out of memory and slow training)
DEV_MODE = True
SUBSET_SIZE = 100000  # Number of ratings to load for development
