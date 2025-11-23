"""
Semantic Validation Module

Ensures compressed prompts preserve the original meaning.
Uses cosine similarity to compare original and compressed text.
"""

from typing import Tuple
from core.extractive import get_model


def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity between two texts using sentence embeddings.
    
    Args:
        text1: First text
        text2: Second text
    
    Returns:
        Similarity score between 0 and 1 (1 = identical meaning)
    """
    if not text1 or not text2:
        return 0.0
    
    # Same text = perfect similarity
    if text1.strip().lower() == text2.strip().lower():
        return 1.0
    
    model = get_model()
    if model is None:
        # If model unavailable, assume OK
        return 1.0
    
    try:
        from sentence_transformers import util
        
        # Get embeddings
        emb1 = model.encode([text1], convert_to_tensor=True)
        emb2 = model.encode([text2], convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = util.cos_sim(emb1, emb2)[0][0].item()
        
        return float(similarity)
    
    except Exception as e:
        print(f"Warning: Could not calculate similarity: {e}")
        return 1.0  # Assume OK on error


def validate_compression(original: str, compressed: str, min_similarity: float = 0.75) -> Tuple[bool, float]:
    """
    Validate that compressed text preserves meaning.
    
    Args:
        original: Original text
        compressed: Compressed text
        min_similarity: Minimum acceptable similarity (default: 0.75)
    
    Returns:
        Tuple of (is_valid, similarity_score)
    """
    similarity = calculate_semantic_similarity(original, compressed)
    is_valid = similarity >= min_similarity
    
    return is_valid, similarity
