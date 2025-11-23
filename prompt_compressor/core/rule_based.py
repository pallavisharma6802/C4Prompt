import re
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple


def read_fillers(path: str) -> Tuple[List[str], List[str], List[str], Dict[str, str], List[str]]:
    """
    Read filler patterns from JSON config.
    
    Returns:
        (patterns, words, redundant_phrases, verbose_replacements, stopwords)
    """
    p = Path(path)
    data = json.loads(p.read_text(encoding='utf-8'))
    
    patterns = data.get('patterns', [])
    words = data.get('words', [])
    redundant_phrases = data.get('redundant_phrases', [])
    verbose_replacements = data.get('verbose_replacements', {})
    stopwords = data.get('stopwords', [])
    
    return patterns, words, redundant_phrases, verbose_replacements, stopwords


def optimize_rule_based(
    text: str,
    patterns: Optional[List[str]] = None,
    words: Optional[List[str]] = None,
    redundant_phrases: Optional[List[str]] = None,
    verbose_replacements: Optional[Dict[str, str]] = None,
    stopwords: Optional[List[str]] = None,
    remove_stopwords: bool = False
) -> str:
    """
    Apply comprehensive rule-based compression.
    
    Args:
        text: Input text to compress
        patterns: Politeness/filler patterns to remove
        words: Filler words to remove
        redundant_phrases: Verbose phrases to remove
        verbose_replacements: Verbose constructions to replace with shorter versions
        stopwords: Common stopwords (a, an, the) with smart preservation
        remove_stopwords: Whether to remove stopwords (default: False for safety)
    
    Returns:
        Compressed text
    """
    if not text:
        return text

    # Set defaults
    if patterns is None:
        patterns = []
    if words is None:
        words = []
    if redundant_phrases is None:
        redundant_phrases = []
    if verbose_replacements is None:
        verbose_replacements = {}
    if stopwords is None:
        stopwords = []

    s = text.strip()

    # Step 1: Remove greeting/politeness openings
    s = re.sub(r'^(hello there[!,]?\s*)', '', s, flags=re.I)
    s = re.sub(r'^(hi there[!,]?\s*)', '', s, flags=re.I)
    s = re.sub(r'^(hey there[!,]?\s*)', '', s, flags=re.I)
    s = re.sub(r'^(greetings[!,]?\s*)', '', s, flags=re.I)
    s = re.sub(r'^\s*i hope (you\'re|you are|your) (doing )?(well|good|great)( today| this (morning|afternoon|evening))?[.,!]?\s*', '', s, flags=re.I)
    s = re.sub(r'^(can you|could you|would you|please|kindly)[:,\s]*', '', s, flags=re.I)

    # Step 2: Apply verbose replacements (before removal to avoid conflicts)
    for verbose, short in verbose_replacements.items():
        s = re.sub(re.escape(verbose), short, s, flags=re.I)

    # Step 3: Remove filler patterns
    for pat in patterns:
        try:
            s = re.sub(pat, '', s, flags=re.I)
        except re.error:
            continue

    # Step 4: Remove filler words
    for word in words:
        try:
            s = re.sub(word, '', s, flags=re.I)
        except re.error:
            continue

    # Step 5: Remove redundant phrases
    for phrase in redundant_phrases:
        try:
            s = re.sub(phrase, '', s, flags=re.I)
        except re.error:
            continue

    # Step 6: Optional stopword removal
    if remove_stopwords:
        for stopword in stopwords:
            try:
                s = re.sub(stopword, '', s, flags=re.I)
            except re.error:
                continue

    # Step 7: Clean up whitespace and punctuation
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\s+([.,!?;:])', r'\1', s)
    s = re.sub(r'([.,!?;:])([^\s.,!?:;])', r'\1 \2', s)
    
    # Step 8: Remove duplicate punctuation
    s = re.sub(r'([.,!?;:])\1+', r'\1', s)
    
    # Step 9: Clean leading/trailing
    s = s.strip(' \n\t\r"')
    
    return s
