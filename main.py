from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from inference import MovieRecommenderInference  # Adjusted import path
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Movie Recommender API",
    description="API for getting movie recommendations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize the recommender model
model = MovieRecommenderInference()

class MovieRecommendation(BaseModel):
    title: str
    similarity_score: float
    year: Optional[str]
    genres: List[str]
    vote_average: float
    vote_count: int

class RecommendationResponse(BaseModel):
    recommendations: List[MovieRecommendation]

@app.get("/")
async def root():
    return {"message": "Movie Recommender API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/recommend/{movie_title}", response_model=RecommendationResponse)
async def get_recommendations(movie_title: str, n_recommendations: int = 5):
    try:
        recommendations = model.get_recommendations(movie_title, n_recommendations)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

