import re
import json
from pathlib import Path
from typing import List, Optional


def read_fillers(path: str) -> (List[str], List[str]):
    p = Path(path)
    data = json.loads(p.read_text(encoding='utf-8'))
    patterns = data.get('patterns', [])
    words = data.get('words', [])
    return patterns, words


def optimize_rule_based(text: str, patterns: Optional[List[str]] = None, words: Optional[List[str]] = None) -> str:
    if not text:
        return text

    if patterns is None:
        patterns = []
    if words is None:
        words = []

    s = text.strip()

    s = re.sub(r'^(can you|could you|would you|please|kindly)[:,\s]*', '', s, flags=re.I)

    for pat in patterns + words:
        try:
            s = re.sub(pat, '', s, flags=re.I)
        except re.error:
            continue

    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\s+([.,!?;:])', r'\1', s)
    s = s.strip(' \n\t\r"')
    return s
