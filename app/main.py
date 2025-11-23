from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sys
import os
import re

# Add prompt_compressor to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'prompt_compressor'))
from core.rule_based import optimize_rule_based, read_fillers
from core.smart_reduction import optimize_smart_reduction
from core.validator import validate_compression
from utils.co2_estimator import token_count, tokens_to_wh, per_wh
from utils.pricing import calculate_cost_savings, MODEL_PRICING, MODEL_CATEGORIES


class CompressRequest(BaseModel):
    prompt: str
    use_tiktoken: bool = False
    aggressive: bool = False
    remove_stopwords: bool = False
    remove_punctuation: bool = False
    light_stemming: bool = False
    super_aggressive: bool = False
    encode: bool = False
    model_id: str = None  # Optional: LLM model for cost calculation


app = FastAPI()

# Mount the `static` directory so the frontend can be served at /static/index.html
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/compress")
async def compress_endpoint(req: CompressRequest):
    original = req.prompt.strip()
    if not original:
        return {"error": "No prompt provided"}
    
    try:
        # Read filler patterns from config
        patterns, words, redundant_phrases, replacements, stopwords = read_fillers('prompt_compressor/config/fillers.json')
    except Exception:
        # Fallback if config not found
        patterns, words, redundant_phrases, replacements, stopwords = [], [], [], {}, []
    
    # Step 1: Enhanced rule-based compression
    layer1 = optimize_rule_based(
        original, 
        patterns=patterns, 
        words=words,
        redundant_phrases=redundant_phrases,
        verbose_replacements=replacements,
        stopwords=stopwords,
        remove_stopwords=False
    )
    
    # Step 2: Smart reduction using TF-IDF
    layer2 = optimize_smart_reduction(
        layer1,
        target_reduction=0.20,
        preserve_question_structure=True
    )
    
    # Validate compression quality
    is_valid, similarity = validate_compression(original, layer2)
    
    # Calculate token counts
    original_tokens = token_count(original)
    compressed_tokens = token_count(layer2)
    tokens_saved = max(0, original_tokens - compressed_tokens)
    
    # Calculate CO2 savings: tokens → Wh → g CO2
    wh_saved = tokens_to_wh(tokens_saved)
    co2_saved_g = per_wh(wh_saved)
    
    # Calculate cost savings if model_id is provided
    cost_savings = None
    if req.model_id and req.model_id in MODEL_PRICING:
        try:
            cost_info = calculate_cost_savings(tokens_saved, req.model_id)
            cost_savings = {
                "model": cost_info["model"],
                "cost_per_1m_tokens": cost_info["cost_per_1m_tokens"],
                "cost_saved_usd": cost_info["cost_saved_usd"],
            }
        except Exception:
            pass
    
    return {
        "original_prompt": original,
        "compressed_prompt": layer2,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "tokens_saved": tokens_saved,
        "co2_saved_g": co2_saved_g,
        "cost_savings": cost_savings,
        "semantic_similarity": similarity,
        "validation_passed": is_valid,
        "meta": {
            "layer1_tokens": token_count(layer1),
            "layer2_tokens": compressed_tokens,
        },
    }


@app.get("/")
async def root():
    # Redirect to the landing page
    return RedirectResponse(url="/static/landing.html")


@app.get("/models")
async def get_models():
    """Return list of available LLM models with pricing"""
    models = []
    for category, model_list in MODEL_CATEGORIES.items():
        for model_id, display_name in model_list:
            models.append({
                "id": model_id,
                "name": display_name,
                "category": category,
                "price_per_1m": MODEL_PRICING[model_id]
            })
    return {"models": models}
