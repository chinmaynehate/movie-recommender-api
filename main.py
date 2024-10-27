from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from inference import MovieRecommenderInference  
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

# Define Pydantic models for the response
class MovieRecommendation(BaseModel):
    title: str
    similarity_score: float
    year: Optional[str] = None
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
        # Get full recommendations from the model
        recommendations = model.get_recommendations(movie_title, n_recommendations)

        # Ensure the response matches the Pydantic model
        formatted_recommendations = [
            MovieRecommendation(
                title=rec["title"],
                similarity_score=rec["similarity_score"],
                year=rec.get("year"),
                genres=rec.get("genres", []),
                vote_average=rec["vote_average"],
                vote_count=rec["vote_count"]
            )
            for rec in recommendations
        ]

        # Return recommendations in the expected format
        return {"recommendations": formatted_recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
