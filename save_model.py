import pandas as pd
from recommender import EnhancedMovieRecommender
import joblib
import os

def save_model():
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('tmdb_movies.csv')
    print(f"Loaded {len(df)} movies")
    
    # Initialize and train recommender
    print("Training model...")
    recommender = EnhancedMovieRecommender(
        max_features=2000,
        min_votes=100,
        min_vote_avg=5.0,
        batch_size=1000
    )
    
    # Fit the model
    recommender.fit(df)
    
    # Create model directory
    os.makedirs('model', exist_ok=True)
    
    # Save all components
    print("Saving model components...")
    joblib.dump(recommender.tfidf, 'model/tfidf.joblib')
    joblib.dump(recommender.movie_features, 'model/movie_features.joblib')
    joblib.dump(recommender.movie_titles, 'model/movie_titles.joblib')
    joblib.dump(recommender.title_to_idx, 'model/title_to_idx.joblib')
    joblib.dump(recommender.metadata, 'model/metadata.joblib')
    joblib.dump(recommender.genre_weights, 'model/genre_weights.joblib')
    
    print("Model saved successfully!")

if __name__ == "__main__":
    save_model()
