import os
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any

# Ensure parent directory is in Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.recommender import CineIQRecommender

app = FastAPI(
    title="CineIQ Movie Recommendation Service",
    description="A FastAPI serving layer providing SVD, Content-Based, and Sentiment Re-ranked movie suggestions with explainability.",
    version="1.0.0"
)

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Recommender instance
recommender = None

@app.on_event("startup")
def startup_event():
    """Initializes the recommendation engine by loading models on startup."""
    global recommender
    try:
        recommender = CineIQRecommender()
        print("CineIQ Recommender Engine loaded successfully on startup.")
    except Exception as e:
        print(f"Warning: Models could not be loaded at startup. Error: {e}")
        print("Please run the data preprocessing and training scripts first.")

# API Response Schemas
class MovieRecommendation(BaseModel):
    movieId: int
    title: str
    genres: str
    director: str
    top_cast: List[str]
    score: float
    explanation: str
    audience_summary: Optional[str] = None

class MovieSimilarity(BaseModel):
    movieId: int
    title: str
    genres: str
    director: str
    top_cast: List[str]
    similarity: float

class UserHistoryMovie(BaseModel):
    movieId: int
    title: str
    genres: str
    user_rating: float

class HealthResponse(BaseModel):
    status: str
    datasets_ready: bool
    models_loaded: bool
    error: Optional[str] = None

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Checks if data pipelines and models are ready to serve."""
    datasets_ready = config.RATINGS_FILE.exists() and config.MOVIES_FILE.exists()
    models_ready = (
        config.SVD_MODEL_PATH.exists() and 
        config.TFIDF_MATRIX_PATH.exists() and 
        config.MOVIES_METADATA_PATH.exists()
    )
    
    status = "healthy" if (datasets_ready and models_ready) else "degraded"
    
    error_msg = None
    if not datasets_ready:
        error_msg = "Datasets are missing. Run data pipeline."
    elif not models_ready:
        error_msg = "Model files are missing. Train SVD and Content models."
        
    return {
        "status": status,
        "datasets_ready": datasets_ready,
        "models_loaded": (recommender is not None and models_ready),
        "error": error_msg
    }

@app.get("/recommend", response_model=List[MovieRecommendation])
def get_recommendations(
    user_id: int = Query(..., description="The ID of the user to get recommendations for."),
    n: int = Query(10, ge=1, le=50, description="Number of recommendations to return.")
):
    """Generates personalized movie recommendations for a user using SVD, Content-Based, and Sentiment analysis."""
    global recommender
    if recommender is None:
        try:
            recommender = CineIQRecommender()
        except Exception as e:
            raise HTTPException(
                status_code=503, 
                detail=f"Recommendation engine is not initialized. Run training pipelines first. Error: {e}"
            )
            
    try:
        recs_df = recommender.get_recommendations(user_id, n=n)
        
        response = []
        for _, row in recs_df.iterrows():
            # Handle float conversions safely
            score_val = float(row.get('final_score', row.get('hybrid_score', 0.0)))
            
            # Extract top_cast (could be list or string representation of list)
            cast_list = row['top_cast']
            if isinstance(cast_list, str):
                import ast
                try:
                    cast_list = ast.literal_eval(cast_list)
                except:
                    cast_list = [cast_list]
            
            response.append({
                "movieId": int(row['movieId']),
                "title": row['title'],
                "genres": row['genres'],
                "director": str(row['director']) if not pd.isna(row['director']) else "Unknown",
                "top_cast": cast_list,
                "score": score_val,
                "explanation": row['explanation'],
                "audience_summary": row.get('audience_summary', "Good reviews")
            })
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

@app.get("/similar", response_model=List[MovieSimilarity])
def get_similar(
    movie_title: str = Query(..., description="The title of the movie to find similar films for."),
    n: int = Query(10, ge=1, le=50, description="Number of similar movies to return.")
):
    """Finds content-similar movies based on genre, cast, director, and keywords."""
    global recommender
    if recommender is None:
        try:
            recommender = CineIQRecommender()
        except Exception as e:
            raise HTTPException(
                status_code=503, 
                detail=f"Recommendation engine is not initialized. Run training pipelines first. Error: {e}"
            )
            
    try:
        sims_df = recommender.get_similar_movies(movie_title, n=n)
        if sims_df.empty:
            raise HTTPException(status_code=404, detail=f"Movie containing '{movie_title}' not found.")
            
        response = []
        for _, row in sims_df.iterrows():
            cast_list = row['top_cast']
            if isinstance(cast_list, str):
                import ast
                try:
                    cast_list = ast.literal_eval(cast_list)
                except:
                    cast_list = [cast_list]
                    
            response.append({
                "movieId": int(row['movieId']),
                "title": row['title'],
                "genres": row['genres'],
                "director": str(row['director']) if not pd.isna(row['director']) else "Unknown",
                "top_cast": cast_list,
                "similarity": float(row['similarity'])
            })
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

@app.get("/user-history", response_model=List[UserHistoryMovie])
def get_user_history(
    user_id: int = Query(..., description="The user ID to fetch rating history for.")
):
    """Retrieves highly rated movies from the user's history."""
    global recommender
    if recommender is None:
        try:
            recommender = CineIQRecommender()
        except Exception as e:
            raise HTTPException(
                status_code=503, 
                detail=f"Recommendation engine is not initialized. Error: {e}"
            )
            
    try:
        history_df = recommender.get_user_liked_movies(user_id)
        if history_df.empty:
            return []
            
        response = []
        for _, row in history_df.iterrows():
            response.append({
                "movieId": int(row['movieId']),
                "title": row['title'],
                "genres": row['genres'],
                "user_rating": float(row['rating'])
            })
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

if __name__ == "__main__":
    import uvicorn
    # Start FastAPI server on port 8000
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
