import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import json
import ast

def parse_json_field(field_str):
    """Safely parse JSON-like string field"""
    if pd.isna(field_str):
        return []
    
    try:
        if isinstance(field_str, str):
            return json.loads(field_str)
        elif isinstance(field_str, list):
            return field_str
        return []
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(field_str)
        except:
            return []
    except:
        return []

class MovieRecommenderInference:
    def __init__(self, model_dir='model'):
        """Initialize the inference model"""
        self.model_dir = Path(model_dir)
        
        # Load all model components
        print("Loading model components...")
        self.tfidf = joblib.load(self.model_dir / 'tfidf.joblib')
        self.movie_features = joblib.load(self.model_dir / 'movie_features.joblib')
        self.movie_titles = joblib.load(self.model_dir / 'movie_titles.joblib')
        self.title_to_idx = joblib.load(self.model_dir / 'title_to_idx.joblib')
        self.metadata = joblib.load(self.model_dir / 'metadata.joblib')
        self.genre_weights = joblib.load(self.model_dir / 'genre_weights.joblib')
        
        print("Model loaded successfully!")

    def _calculate_similarity_boost(self, base_movie, candidate_movie):
        """Calculate similarity boost based on metadata"""
        boost = 1.0
        
        # Year similarity
        try:
            year1 = int(str(base_movie['release_date'])[:4])
            year2 = int(str(candidate_movie['release_date'])[:4])
            year_diff = abs(year1 - year2)
            if year_diff <= 5:
                boost *= 1.3
            elif year_diff <= 10:
                boost *= 1.2
        except:
            pass
        
        # Genre similarity
        base_genres = set(g['name'] for g in parse_json_field(base_movie['genres']))
        candidate_genres = set(g['name'] for g in parse_json_field(candidate_movie['genres']))
        
        if base_genres and candidate_genres:
            genre_overlap = len(base_genres & candidate_genres) / len(base_genres)
            boost *= (1 + 0.4 * genre_overlap)
        
        # Rating similarity
        if (pd.notna(base_movie['vote_average']) and pd.notna(candidate_movie['vote_average'])):
            rating_diff = abs(float(base_movie['vote_average']) - float(candidate_movie['vote_average']))
            if rating_diff <= 0.5:
                boost *= 1.2
            elif rating_diff <= 1.0:
                boost *= 1.1
        
        return boost

    def get_recommendations(self, title, n_recommendations=5):
        """Get movie recommendations"""
        title = str(title).lower()
        
        # Find movie index
        if title not in self.title_to_idx:
            print(f"Movie '{title}' not found. Finding closest match...")
            closest_title = min(self.title_to_idx.keys(), 
                              key=lambda x: len(set(x.split()) & set(title.split())))
            movie_idx = self.title_to_idx[closest_title]
            print(f"Using '{closest_title}' as closest match")
        else:
            movie_idx = self.title_to_idx[title]
        
        # Get base movie
        base_movie = self.metadata.iloc[movie_idx]
        
        # Calculate base similarity
        movie_vector = self.movie_features[movie_idx:movie_idx+1]
        similarity_scores = cosine_similarity(movie_vector, self.movie_features)[0]
        
        # Calculate boosted scores
        boosted_scores = []
        for idx, score in enumerate(similarity_scores):
            if idx != movie_idx and score > 0.1:
                candidate_movie = self.metadata.iloc[idx]
                boosted_score = score * self._calculate_similarity_boost(base_movie, candidate_movie)
                boosted_scores.append((idx, boosted_score))
        
        # Sort and get top recommendations
        boosted_scores.sort(key=lambda x: x[1], reverse=True)
        recommendations = []
        
        for idx, score in boosted_scores[:n_recommendations]:
            movie = self.metadata.iloc[idx]
            genres = parse_json_field(movie['genres'])
            
            recommendations.append({
                'title': movie['title'],
                'similarity_score': float(score),
                'year': str(movie['release_date'])[:4] if pd.notna(movie['release_date']) else None,
                'genres': [g['name'] for g in genres],
                'vote_average': float(movie['vote_average']),
                'vote_count': int(float(movie['vote_count']))
            })
        
        return recommendations

# Example usage
if __name__ == "__main__":
    # Initialize the model
    recommender = MovieRecommenderInference()
    
    # Get recommendations
    movie_title = "The Dark Knight"
    recommendations = recommender.get_recommendations(movie_title)
    
    # Print recommendations
    print(f"\nRecommendations for '{movie_title}':")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['title']}")
        print(f"   Similarity Score: {rec['similarity_score']:.2f}")
        print(f"   Year: {rec['year']}")
        print(f"   Rating: {rec['vote_average']}/10 ({rec['vote_count']} votes)")
        print(f"   Genres: {', '.join(rec['genres'])}")
