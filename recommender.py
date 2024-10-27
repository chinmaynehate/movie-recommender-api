import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import joblib
from scipy.sparse import save_npz, load_npz, vstack, csr_matrix
import json
import ast
import os
import time
from tqdm import tqdm
from pathlib import Path
import logging
from datetime import datetime
from utils import parse_json_field, calculate_weighted_rating, clean_text
import warnings
warnings.filterwarnings('ignore')

class EnhancedMovieRecommender:
    def __init__(self, 
                 max_features=2000,  # Increased for better coverage
                 cache_dir='cache',
                 min_votes=100,      # Minimum votes for reliable ratings
                 min_vote_avg=5.0,   # Minimum vote average for quality
                 batch_size=1000):   # Batch size for processing
        
        self.max_features = max_features
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_votes = min_votes
        self.min_vote_avg = min_vote_avg
        self.batch_size = batch_size
        
        # Initialize TF-IDF with improved parameters
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),      # Include bigrams
            min_df=3,                # Minimum document frequency
            max_df=0.95,             # Maximum document frequency
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w+',    # Include single letters
        )
        
        # Initialize scalers
        self.popularity_scaler = MinMaxScaler()
        self.vote_scaler = MinMaxScaler()
        self.year_scaler = MinMaxScaler()
        
        # Model components
        self.movie_features = None
        self.movie_titles = None
        self.title_to_idx = None
        self.metadata = None
        self.genre_weights = None
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.cache_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'recommender_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _get_cache_path(self, feature):
        """Get path for cached feature"""
        return self.cache_dir / f'{feature}_max{self.max_features}.joblib'

    def _calculate_genre_weights(self, df):
        """Calculate genre importance weights based on popularity and ratings"""
        genre_stats = {}
        
        for _, row in df.iterrows():
            genres = parse_json_field(row['genres'])
            for genre in genres:
                if 'name' in genre:
                    genre_name = genre['name']
                    if genre_name not in genre_stats:
                        genre_stats[genre_name] = {
                            'count': 0,
                            'total_rating': 0,
                            'total_popularity': 0
                        }
                    
                    genre_stats[genre_name]['count'] += 1
                    genre_stats[genre_name]['total_rating'] += float(row['vote_average'])
                    genre_stats[genre_name]['total_popularity'] += float(row['popularity'])
        
        # Calculate weights
        weights = {}
        total_movies = len(df)
        
        for genre, stats in genre_stats.items():
            frequency = stats['count'] / total_movies
            avg_rating = stats['total_rating'] / stats['count']
            avg_popularity = stats['total_popularity'] / stats['count']
            
            # Combine metrics into a single weight
            weights[genre] = (0.4 * avg_rating/10 + 
                            0.4 * frequency + 
                            0.2 * (avg_popularity / max(genre_stats[g]['total_popularity']/genre_stats[g]['count'] 
                                                      for g in genre_stats)))
        
        return weights

    def _prepare_text_features(self, row):
        """Prepare text features with enhanced weighting"""
        features = []
        
        # Process title (high weight)
        if pd.notna(row['title']):
            title_words = clean_text(row['title']).split()
            features.extend(title_words * 3)
        
        # Process genres with weighted importance
        genres = parse_json_field(row['genres'])
        for genre in genres:
            if 'name' in genre:
                genre_name = genre['name']
                weight = int(self.genre_weights.get(genre_name, 1) * 3)
                features.extend([genre_name.lower()] * weight)
        
        # Process keywords
        if pd.notna(row['keywords']):
            keywords = parse_json_field(row['keywords'])
            keyword_list = []
            for kw in keywords:
                if 'name' in kw:
                    keyword_list.append(clean_text(kw['name']))
            features.extend(keyword_list * 2)
        
        # Process overview
        if pd.notna(row['overview']):
            overview_words = clean_text(row['overview']).split()[:100]
            features.extend(overview_words)
        
        # Add year features
        if pd.notna(row['release_date']):
            try:
                year = str(row['release_date'])[:4]
                features.append(f"year_{year}")
                decade = f"{year[:3]}0s"
                features.append(f"decade_{decade}")
            except:
                pass
        
        # Add rating features
        if pd.notna(row['vote_average']):
            rating = float(row['vote_average'])
            if rating >= 7.5:
                features.extend(['high_rated'] * 3)
            elif rating >= 6.0:
                features.append('medium_rated')
        
        # Add language and popularity features
        if pd.notna(row['original_language']):
            features.append(f"lang_{row['original_language']}")
        
        if pd.notna(row['popularity']):
            if float(row['popularity']) > 20:
                features.append('popular')
        
        return ' '.join(features)

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
        
        # Language and popularity boost
        if base_movie['original_language'] == candidate_movie['original_language']:
            boost *= 1.1
        
        if float(candidate_movie['popularity']) > float(candidate_movie['popularity'].mean()):
            boost *= 1.1
        
        return boost

    def fit(self, df):
        """Train the recommender system"""
        start_time = time.time()
        
        self.logger.info("Starting model training...")
        self.logger.info(f"Initial dataset size: {len(df)} movies")
        
        # Calculate genre weights
        self.logger.info("Calculating genre weights...")
        self.genre_weights = self._calculate_genre_weights(df)
        
        # Filter and clean data
        df = df[
            (df['vote_count'].astype(float) >= self.min_votes) &
            (df['vote_average'].astype(float) >= self.min_vote_avg)
        ].copy()
        
        self.logger.info(f"Filtered dataset size: {len(df)} movies")
        
        # Store metadata
        self.metadata = df.copy()
        
        # Prepare features in batches
        self.logger.info("Preparing features...")
        all_features = []
        
        for start_idx in tqdm(range(0, len(df), self.batch_size), 
                            desc="Processing movies"):
            end_idx = min(start_idx + self.batch_size, len(df))
            batch = df.iloc[start_idx:end_idx]
            
            batch_features = []
            for _, row in batch.iterrows():
                try:
                    features = self._prepare_text_features(row)
                    batch_features.append(features)
                except Exception as e:
                    self.logger.error(f"Error processing movie: {row.get('title', 'Unknown')}")
                    batch_features.append("")
            
            all_features.extend(batch_features)
        
        # Fit TF-IDF
        self.logger.info("Fitting TF-IDF vectorizer...")
        self.movie_features = self.tfidf.fit_transform(all_features)
        
        # Store movie data
        self.movie_titles = df['title'].tolist()
        self.title_to_idx = {str(title).lower(): idx 
                            for idx, title in enumerate(self.movie_titles)}
        
        elapsed = time.time() - start_time
        self.logger.info(f"Training completed in {elapsed:.2f} seconds")
        self.logger.info(f"Model size: {self.movie_features.data.nbytes / 1024 / 1024:.2f} MB")
        
        return self

    def get_recommendations(self, title, n_recommendations=5):
        """Get movie recommendations"""
        start_time = time.time()
        
        # Find movie index
        title = str(title).lower()
        if title not in self.title_to_idx:
            self.logger.warning(f"Movie '{title}' not found. Finding closest match...")
            closest_title = min(self.title_to_idx.keys(), 
                              key=lambda x: len(set(x.split()) & set(title.split())))
            movie_idx = self.title_to_idx[closest_title]
            self.logger.info(f"Using '{closest_title}' as closest match")
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
            if idx != movie_idx and score > 0.1:  # Similarity threshold
                candidate_movie = self.metadata.iloc[idx]
                boosted_score = score * self._calculate_similarity_boost(base_movie, candidate_movie)
                boosted_scores.append((idx, boosted_score))
        
        # Sort and get top recommendations
        boosted_scores.sort(key=lambda x: x[1], reverse=True)
        recommendations = []
        
        for idx, score in boosted_scores[:n_recommendations]:
            movie = self.metadata.iloc[idx]
            recommendations.append({
                'title': movie['title'],
                'similarity_score': float(score),
                'vote_average': float(movie['vote_average']),
                'vote_count': int(float(movie['vote_count'])),
                'year': str(movie['release_date'])[:4] if pd.notna(movie['release_date']) else None,
                'genres': [g['name'] for g in parse_json_field(movie['genres'])],
                'popularity': float(movie['popularity']),
                'language': movie['original_language'],
                'overview': movie['overview']
            })
        
        elapsed = time.time() - start_time
        self.logger.info(f"Recommendations generated in {elapsed:.3f} seconds")
        
        return recommendations
