import re
from typing import List, Optional


# Global model cache - load once, reuse forever
_CACHED_MODEL = None


def get_model():
    """
    Get or load the SentenceTransformer model.
    Caches the model globally to avoid reloading on every call.
    """
    global _CACHED_MODEL
    
    if _CACHED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _CACHED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Warning: Could not load SentenceTransformer model: {e}")
            return None
    
    return _CACHED_MODEL


def _split_sentences(text: str) -> List[str]:
    #sentence splitter: split on sentence-ending punctuation
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    # fallback: if no punctuation, split by lines
    if len(parts) == 1:
        parts = [p for line in text.splitlines() for p in line.split('\n') if p.strip()]
    return [p.strip() for p in parts if p.strip()]


def optimize_extractive(text: str, max_sentences: int = 2, model=None) -> str:
    if not text:
        return text

    # Use provided model or get from cache
    if model is None:
        model = get_model()
        if model is None:
            # Fallback: if model can't load, return text as-is
            return text
    
    try:
        from sentence_transformers import util
    except Exception:
        return text

    sentences = _split_sentences(text)
    if not sentences:
        return text
    
    # If only one sentence, return it
    if len(sentences) == 1:
        return sentences[0]

    sent_emb = model.encode(sentences, convert_to_tensor=True)
    doc_emb = model.encode([text], convert_to_tensor=True)
    sims = util.cos_sim(sent_emb, doc_emb).squeeze(-1).tolist()
    ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
    selected_idx = sorted(ranked[:max_sentences])
    return ' '.join([sentences[i] for i in selected_idx])
