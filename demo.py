import logging
import time
import pandas as pd
from recommender import EnhancedMovieRecommender

def print_recommendations(recommendations, logger):
    """Print formatted recommendations"""
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"\n{i}. {rec['title']}")
        logger.info(f"   Similarity Score: {rec['similarity_score']:.2f}")
        logger.info(f"   Rating: {rec['vote_average']}/10 ({rec['vote_count']} votes)")
        logger.info(f"   Year: {rec['year']}")
        logger.info(f"   Genres: {', '.join(rec['genres'])}")
        logger.info(f"   Language: {rec['language']}")
        logger.info(f"   Popularity: {rec['popularity']:.1f}")
        logger.info(f"   Overview: {rec['overview'][:200]}...")

def evaluate_recommendations(title, recommendations, logger):
    """Evaluate recommendation quality"""
    logger.info(f"\nRecommendation Analysis for '{title}':")
    
    # Genre diversity
    all_genres = set()
    for rec in recommendations:
        all_genres.update(rec['genres'])
    
    # Year spread
    years = [int(rec['year']) for rec in recommendations if rec['year']]
    year_spread = max(years) - min(years) if years else 0
    
    # Average rating
    avg_rating = sum(rec['vote_average'] for rec in recommendations) / len(recommendations)
    
    # Print analysis
    logger.info(f"Genre Diversity: {len(all_genres)} unique genres")
    logger.info(f"Year Spread: {year_spread} years")
    logger.info(f"Average Rating: {avg_rating:.1f}/10")

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Record start time
        total_start_time = time.time()
        
        # Load dataset
        logger.info("Loading dataset...")
        df = pd.read_csv('tmdb_movies.csv')
        logger.info(f"Loaded {len(df)} movies")
        
        # Initialize recommender
        recommender = EnhancedMovieRecommender(
            max_features=2000,    # Increased for better coverage
            min_votes=100,        # Minimum votes threshold
            min_vote_avg=5.0,     # Minimum rating threshold
            batch_size=1000       # Batch size for processing
        )
        
        # Fit the model
        recommender.fit(df)
        
        # Test different types of movies
        test_movies = [
            # Popular modern movies
            "The Dark Knight",
            "Avatar",
            "Inception",
            
            # Classic movies
            "The Godfather",
            "Pulp Fiction",
            
            # Sci-fi movies
            "The Matrix",
            "Interstellar",
            
            # Animation
            "Toy Story",
            "Spirited Away",
            
            # Different genres
            "The Shawshank Redemption",  # Drama
            "Jurassic Park",             # Adventure
            "The Silence of the Lambs"   # Thriller
        ]
        
        # Get recommendations for each movie
        for movie in test_movies:
            logger.info(f"\n{'='*80}")
            logger.info(f"Getting recommendations for '{movie}':")
            
            try:
                recommendations = recommender.get_recommendations(movie)
                print_recommendations(recommendations, logger)
                evaluate_recommendations(movie, recommendations, logger)
            except Exception as e:
                logger.error(f"Error getting recommendations for {movie}: {str(e)}")
        
        # Print performance statistics
        total_time = time.time() - total_start_time
        logger.info(f"\nTotal execution time: {total_time:.2f} seconds")
        
    except Exception as e:
        logger.error("Fatal error occurred:", exc_info=True)

if __name__ == "__main__":
    main()
