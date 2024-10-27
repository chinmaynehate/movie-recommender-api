# Movie Recommender API

A FastAPI-based movie recommendation system.

## Setup and Deployment

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run locally:
```bash
uvicorn app.main:app --reload
```

3. API Endpoints:
- GET `/`: Health check
- GET `/recommend/{movie_title}`: Get movie recommendations

## API Usage

```python
import requests

# Get recommendations
response = requests.get(
    "http://localhost:8000/recommend/The Dark Knight",
    params={"n_recommendations": 5}
)
recommendations = response.json()
```

## Model Files

The `model/` directory contains necessary model files for inference. These files are required for the API to function.
