"""
Model pricing calculator for LLM token costs.
Pricing is per 1M tokens (input) as of Nov 2024.
"""

# Pricing per 1M input tokens (USD)
MODEL_PRICING = {
    # OpenAI GPT Models
    "gpt-4o": 2.50,
    "gpt-4o-mini": 0.15,
    "gpt-4-turbo": 10.00,
    "gpt-4": 30.00,
    "gpt-3.5-turbo": 0.50,
    
    # Anthropic Claude Models
    "claude-3.5-sonnet": 3.00,
    "claude-3-opus": 15.00,
    "claude-3-sonnet": 3.00,
    "claude-3-haiku": 0.25,
    
    # Google Gemini Models
    "gemini-1.5-pro": 1.25,
    "gemini-1.5-flash": 0.075,
    "gemini-1.0-pro": 0.50,
}

# Model display names and categories
MODEL_CATEGORIES = {
    "OpenAI GPT": [
        ("gpt-4o", "GPT-4o (Latest, Multimodal)"),
        ("gpt-4o-mini", "GPT-4o Mini (Fast & Cheap)"),
        ("gpt-4-turbo", "GPT-4 Turbo"),
        ("gpt-4", "GPT-4"),
        ("gpt-3.5-turbo", "GPT-3.5 Turbo"),
    ],
    "Anthropic Claude": [
        ("claude-3.5-sonnet", "Claude 3.5 Sonnet (Latest)"),
        ("claude-3-opus", "Claude 3 Opus (Most Capable)"),
        ("claude-3-sonnet", "Claude 3 Sonnet"),
        ("claude-3-haiku", "Claude 3 Haiku (Fastest)"),
    ],
    "Google Gemini": [
        ("gemini-1.5-pro", "Gemini 1.5 Pro"),
        ("gemini-1.5-flash", "Gemini 1.5 Flash (Fastest)"),
        ("gemini-1.0-pro", "Gemini 1.0 Pro"),
    ]
}


def get_model_list():
    """Returns a formatted list of available models with their pricing."""
    models = []
    idx = 1
    for category, model_list in MODEL_CATEGORIES.items():
        models.append(f"\n{category}:")
        for model_id, display_name in model_list:
            price = MODEL_PRICING[model_id]
            models.append(f"  {idx}. {display_name} - ${price:.2f}/1M tokens")
            idx += 1
    return "\n".join(models)


def get_model_by_index(index: int):
    """Returns model ID based on numeric index."""
    idx = 1
    for category, model_list in MODEL_CATEGORIES.items():
        for model_id, display_name in model_list:
            if idx == index:
                return model_id
            idx += 1
    return None


def get_all_model_ids():
    """Returns list of all model IDs in order."""
    model_ids = []
    for category, model_list in MODEL_CATEGORIES.items():
        for model_id, _ in model_list:
            model_ids.append(model_id)
    return model_ids


def calculate_cost_savings(tokens_saved: int, model_id: str) -> dict:
    """
    Calculate cost savings based on tokens saved and model pricing.
    
    Args:
        tokens_saved: Number of tokens saved by compression
        model_id: Model identifier (e.g., 'gpt-4o', 'claude-3-opus')
    
    Returns:
        dict with cost_per_1m_tokens, cost_saved_usd, cost_saved_per_1k
    """
    if model_id not in MODEL_PRICING:
        raise ValueError(f"Unknown model: {model_id}. Available models: {list(MODEL_PRICING.keys())}")
    
    cost_per_1m = MODEL_PRICING[model_id]
    cost_saved_usd = (tokens_saved / 1_000_000) * cost_per_1m
    cost_saved_per_1k = (tokens_saved / 1_000) * (cost_per_1m / 1_000)
    
    return {
        "model": model_id,
        "cost_per_1m_tokens": cost_per_1m,
        "tokens_saved": tokens_saved,
        "cost_saved_usd": cost_saved_usd,
        "cost_saved_per_1k": cost_saved_per_1k,
    }


def format_cost_display(cost_info: dict) -> str:
    """Format cost savings for display."""
    lines = []
    lines.append(f"\nCost Savings ({cost_info['model']}):")
    lines.append(f"  Model pricing: ${cost_info['cost_per_1m_tokens']:.2f} per 1M tokens")
    lines.append(f"  Tokens saved: {cost_info['tokens_saved']}")
    lines.append(f"  💰 Cost saved: ${cost_info['cost_saved_usd']:.6f}")
    
    # Add per-1k estimate for easier comprehension
    if cost_info['cost_saved_per_1k'] > 0.0001:
        lines.append(f"  Per 1K tokens: ${cost_info['cost_saved_per_1k']:.4f}")
    
    return "\n".join(lines)
