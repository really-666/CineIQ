import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from src import config
from src.sentiment import analyze_sentiment

class CineIQRecommender:
    def __init__(self):
        self.movies_df = None
        self.tfidf_matrix = None
        self.vectorizer = None
        self.svd_model = None
        self.ratings_df = None
        
        self.load_models()
        
    def load_models(self):
        """Loads serialized models, vectorizer, and preprocessed metadata from disk."""
        print("Loading models and metadata into Recommender Engine...")
        
        # Load movies metadata
        if not config.MOVIES_METADATA_PATH.exists():
            raise FileNotFoundError("Preprocessed movie metadata not found. Run training pipelines first.")
        with open(config.MOVIES_METADATA_PATH, "rb") as f:
            self.movies_df = pickle.load(f)
            
        # Load TF-IDF matrix
        if not config.TFIDF_MATRIX_PATH.exists():
            raise FileNotFoundError("TF-IDF matrix not found. Run train_content.py first.")
        with open(config.TFIDF_MATRIX_PATH, "rb") as f:
            self.tfidf_matrix = pickle.load(f)
            
        # Load TF-IDF Vectorizer
        if not config.TFIDF_VECTORIZER_PATH.exists():
            raise FileNotFoundError("TF-IDF vectorizer not found. Run train_content.py first.")
        with open(config.TFIDF_VECTORIZER_PATH, "rb") as f:
            self.vectorizer = pickle.load(f)
            
        # Load SVD Model
        if not config.SVD_MODEL_PATH.exists():
            raise FileNotFoundError("SVD model not found. Run train_svd.py first.")
        with open(config.SVD_MODEL_PATH, "rb") as f:
            self.svd_model = pickle.load(f)
            
        # Load ratings dataset for user history lookups
        if config.RATINGS_FILE.exists():
            self.ratings_df = pd.read_csv(config.RATINGS_FILE)
        else:
            self.ratings_df = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])
            
        print("Recommender Engine loaded all models successfully.")

    def get_user_liked_movies(self, user_id, threshold=4.0):
        """Retrieves movies that a user has rated highly (above or equal to threshold)."""
        if self.ratings_df is None or len(self.ratings_df) == 0:
            return pd.DataFrame()
            
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        liked_ratings = user_ratings[user_ratings['rating'] >= threshold]
        
        # Merge with movie metadata to get titles/genres
        user_liked_movies = pd.merge(liked_ratings, self.movies_df, on='movieId', how='inner')
        return user_liked_movies

    def generate_content_similarity_scores(self, user_liked_movies):
        """
        Creates a user taste profile vector from highly rated movies 
        and computes cosine similarity against all movies in the database.
        """
        if len(user_liked_movies) == 0:
            # Cold-start fallback: return zeros if no history
            return np.zeros(self.tfidf_matrix.shape[0])
            
        # Find indices of liked movies in self.movies_df
        liked_movie_ids = user_liked_movies['movieId'].tolist()
        matching_indices = self.movies_df[self.movies_df['movieId'].isin(liked_movie_ids)].index.tolist()
        
        if not matching_indices:
            return np.zeros(self.tfidf_matrix.shape[0])
            
        # Extract TF-IDF vectors of liked movies
        liked_vectors = self.tfidf_matrix[matching_indices]
        
        # Build user profile vector by averaging their liked movie TF-IDF vectors
        user_profile_vector = np.asarray(liked_vectors.mean(axis=0))
        
        # Compute cosine similarity between user profile vector and all movie vectors
        similarities = cosine_similarity(user_profile_vector, self.tfidf_matrix).flatten()
        return similarities

    def get_similar_movies(self, movie_title, n=10):
        """Returns the top n similar movies based strictly on content TF-IDF."""
        # Find movie by title
        match = self.movies_df[self.movies_df['title'].str.contains(movie_title, case=False, na=False)]
        if match.empty:
            return pd.DataFrame()
            
        movie_idx = match.index[0]
        movie_id = match['movieId'].iloc[0]
        
        # Compute similarities
        movie_vector = self.tfidf_matrix[movie_idx]
        similarities = cosine_similarity(movie_vector, self.tfidf_matrix).flatten()
        
        # Rank similarity indices, excluding the movie itself
        sim_indices = np.argsort(similarities)[::-1]
        sim_indices = [idx for idx in sim_indices if idx != movie_idx][:n]
        
        sim_movies = self.movies_df.iloc[sim_indices].copy()
        sim_movies['similarity'] = similarities[sim_indices]
        
        return sim_movies

    def get_mock_audience_reviews(self, movie_title, movie_genres, num_reviews=3):
        """
        Simulates fetching recent audience reviews for a movie. 
        Generates realistic reviews combining genres and keywords, to be parsed by VADER.
        """
        positive_templates = [
            "What a fantastic {genre} film! The director did a phenomenal job bringing this story to life.",
            "Absolutely stunning acting by the lead cast. A must-watch for fans of {genre}.",
            "Engaging, fast-paced, and highly emotional. I was hooked from the very first minute!"
        ]
        negative_templates = [
            "Very disappointed. A slow, boring {genre} movie that felt like a complete waste of time.",
            "The story was incoherent and the characters were highly unlikable. Wouldn't recommend.",
            "Felt generic and formulaic. Decent visuals, but lacking real depth or good writing."
        ]
        
        genres_list = movie_genres.split('|')
        genre = genres_list[0] if genres_list else "movie"
        
        # Check rating-based sentiment
        # Pull a rating if present, else default to 7.0
        movie_row = self.movies_df[self.movies_df['title'] == movie_title]
        rating = float(movie_row['vote_average'].iloc[0]) if not movie_row.empty and 'vote_average' in movie_row.columns else 7.0
        
        # Probability of generating positive reviews correlates with the movie rating
        pos_prob = max(0.1, min(0.9, (rating - 3.0) / 5.0)) # E.g., rating 8 -> prob 1.0, rating 4 -> prob 0.2
        
        reviews = []
        for _ in range(num_reviews):
            is_pos = np.random.random() < pos_prob
            template = np.random.choice(positive_templates if is_pos else negative_templates)
            reviews.append(template.format(genre=genre.lower()))
            
        return reviews

    def get_recommendations(self, user_id, n=10):
        """
        Generates hybrid ensemble recommendations, re-ranks them using review sentiment,
        and adds an explainability sentence to each.
        """
        # 1. Check if user exists. If not, fallback to popular movies (cold start)
        user_exists = (user_id in self.ratings_df['userId'].values) if self.ratings_df is not None else False
        
        if not user_exists:
            # Cold-start fallback: Recommend highly rated, popular movies
            print(f"User ID {user_id} not found. Recommending popular films (cold start).")
            popular_movies = self.movies_df.sort_values(by=['vote_average', 'vote_count'], ascending=False).head(n).copy()
            popular_movies['explanation'] = "Recommended because it is highly rated by the general community."
            popular_movies['score'] = popular_movies['vote_average'] / 10.0
            return popular_movies
            
        # 2. Get user liked movies list
        user_liked = self.get_user_liked_movies(user_id)
        liked_movie_ids = set(user_liked['movieId'].tolist())
        
        # 3. Calculate Content-Based Similarity scores
        content_scores = self.generate_content_similarity_scores(user_liked)
        
        # 4. Calculate SVD Collaborative Filtering predictions
        # Predict for all movies not already liked by the user
        candidate_indices = []
        svd_scores = []
        
        for idx, row in self.movies_df.iterrows():
            m_id = row['movieId']
            if m_id in liked_movie_ids:
                continue
                
            candidate_indices.append(idx)
            # SVD predicted rating
            pred = self.svd_model.predict(user_id, m_id).est
            # Normalize SVD prediction from [0.5, 5.0] to [0.0, 1.0]
            pred_norm = (pred - 0.5) / 4.5
            svd_scores.append(pred_norm)
            
        # Assemble scores for candidates
        candidates_df = self.movies_df.iloc[candidate_indices].copy()
        candidates_df['svd_raw'] = [self.svd_model.predict(user_id, m_id).est for m_id in candidates_df['movieId']]
        candidates_df['svd_score'] = svd_scores
        candidates_df['content_score'] = content_scores[candidate_indices]
        
        # 5. Hybrid blend
        candidates_df['hybrid_score'] = (
            config.HYBRID_WEIGHT_SVD * candidates_df['svd_score'] + 
            config.HYBRID_WEIGHT_CONTENT * candidates_df['content_score']
        )
        
        # 6. Retrieve top 20 candidates for Sentiment Re-ranking (to save CPU cycles)
        top_candidates = candidates_df.sort_values(by='hybrid_score', ascending=False).head(20).copy()
        
        # Compute review sentiment
        sentiment_scores = []
        audience_summaries = []
        
        for _, row in top_candidates.iterrows():
            reviews = self.get_mock_audience_reviews(row['title'], row['genres'])
            # Score each review and average
            review_scores = [analyze_sentiment(r) for r in reviews]
            avg_sentiment = np.mean(review_scores) if review_scores else 0.0
            
            sentiment_scores.append(avg_sentiment)
            
            # Format positive rating statement for explanation
            pos_rating_pct = int((sum(1 for s in review_scores if s > 0.05) / len(review_scores)) * 100)
            audience_summaries.append(f"{pos_rating_pct}% positive audience sentiment")
            
        top_candidates['sentiment_score'] = sentiment_scores
        top_candidates['audience_summary'] = audience_summaries
        
        # 7. Final Re-ranking score
        top_candidates['final_score'] = (
            top_candidates['hybrid_score'] + 
            config.SENTIMENT_WEIGHT * top_candidates['sentiment_score']
        )
        
        # Sort by final score and take top n
        final_recs = top_candidates.sort_values(by='final_score', ascending=False).head(n).copy()
        
        # 8. Generate Human-Readable Explanations (Explainability Layer)
        explanations = []
        for idx, row in final_recs.iterrows():
            # Describe why it fits the taste
            reasons = []
            
            # 1. Collaborative rating explanation
            if row['svd_raw'] >= 3.8:
                reasons.append(f"users with similar tastes rated it highly (predicted rating: {row['svd_raw']:.1f}/5.0)")
            
            # 2. Content similarity description
            if row['content_score'] > 0.25:
                # Find overlapping genres or attributes
                # Look up genres the user liked vs this movie
                user_fav_genres = user_liked['genres'].str.split('|').explode().value_counts().index[:2].tolist()
                movie_genres = row['genres'].split('|')
                common_genres = list(set(user_fav_genres) & set(movie_genres))
                
                if common_genres:
                    genre_str = " & ".join(common_genres)
                    reasons.append(f"it aligns with your love for {genre_str} movies")
                else:
                    reasons.append("it matches the style, keywords, and tone of your favorite movies")
                    
            # 3. Sentiment description
            if row['sentiment_score'] > 0.3:
                reasons.append("and recent audience reviews are highly positive")
            elif row['sentiment_score'] < -0.1:
                reasons.append("though audience reviews are mixed")
                
            # Construct sentence
            if len(reasons) >= 2:
                explanation = "Recommended because " + ", ".join(reasons[:-1]) + ", " + reasons[-1] + "."
            elif len(reasons) == 1:
                explanation = "Recommended because " + reasons[0] + "."
            else:
                explanation = "Recommended based on a balanced blend of your genre history and SVD ratings."
                
            explanations.append(explanation)
            
        final_recs['explanation'] = explanations
        
        return final_recs
