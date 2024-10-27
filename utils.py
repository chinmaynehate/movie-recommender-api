import json
import ast
import pandas as pd

def parse_json_field(field_str):
    """Parse any JSON-like string field with robust error handling"""
    if pd.isna(field_str):
        return []
    
    try:
        if isinstance(field_str, str):
            data = json.loads(field_str)
            if isinstance(data, list):
                return data
            return []
        elif isinstance(field_str, list):
            return field_str
        return []
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(field_str)
            if isinstance(data, list):
                return data
            return []
        except (ValueError, SyntaxError):
            return []
    except Exception:
        return []

def calculate_weighted_rating(row, min_votes):
    """Calculate IMDB-style weighted rating"""
    v = float(row['vote_count'])
    R = float(row['vote_average'])
    m = min_votes
    C = 6.0  # Assume baseline rating of 6.0
    
    return (v / (v + m) * R) + (m / (v + m) * C)

def clean_text(text):
    """Clean text data"""
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower().strip()
    
    # Remove special characters but keep spaces
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    
    return text
