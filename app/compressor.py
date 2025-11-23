import re
from typing import Tuple, Dict, Any, List
import string

# Conservative filler words list — removing these is often safe for concision but may alter tone.
FILLER_WORDS = [
    "very",
    "actually",
    "basically",
    "just",
    "really",
    "literally",
    "obviously",
    "definitely",
    "extremely",
    "seriously",
    "completely",
]


def count_tokens(text: str, use_tiktoken: bool = False) -> int:
    """Return a token count. If tiktoken is available and requested, use it; otherwise fall back to word count."""
    if use_tiktoken:
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            # fallback to word count
            pass
    # simple fallback: number of non-space tokens
    return max(0, len(re.findall(r"\S+", text)))


def _mask_protected_spans(text: str) -> Tuple[str, Dict[str, str]]:
    """Mask code blocks, inline code, and quoted strings so compression won't touch them.

    Returns (masked_text, mapping) where mapping maps placeholder -> original span.
    """
    mapping: Dict[str, str] = {}
    masked = text

    patterns = [
        r"```[\s\S]*?```",  # fenced code blocks
        r"`[^`]+`",  # inline code
        r'"(?:\\.|[^"\\])*"',  # double quoted strings
        r"'(?:\\.|[^'\\])*'",  # single quoted strings
    ]

    i = 0
    for pat in patterns:
        for m in re.finditer(pat, masked):
            span = m.group(0)
            key = f"__MASK{i}__"
            mapping[key] = span
            masked = masked.replace(span, key, 1)
            i += 1
    return masked, mapping


def _unmask(text: str, mapping: Dict[str, str]) -> str:
    out = text
    for k, v in mapping.items():
        out = out.replace(k, v)
    return out


def _remove_fillers(words: List[str]) -> Tuple[List[str], int]:
    removed = 0
    new_words: List[str] = []
    filler_set = set(FILLER_WORDS)
    for w in words:
        lw = w.lower().strip(".,:;!?()[]{}\"')(")
        if lw in filler_set:
            removed += 1
            continue
        new_words.append(w)
    return new_words, removed


STOPWORDS = set([
    # conservative English stopwords (not exhaustive) — useful for compression but may remove nuance
    'the','a','an','and','or','but','if','while','of','at','by','for','with','about','against','between',
    'into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under',
    'again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most',
    'other','some','such','no','nor','not','only','own','same','so','than','too','very','can','will','just'
])

EXTRA_REMOVE = set([
    # extra adjectives/fillers to drop in super-aggressive mode
    'important','significant','detailed','thorough','helpful','useful','please','kindly','suggestions','suggest','example','examples'
])


def remove_stopwords(words: List[str]) -> Tuple[List[str], int]:
    removed = 0
    out: List[str] = []
    for w in words:
        lw = w.lower().strip(".,:;!?()[]{}\"')(")
        if lw in STOPWORDS:
            removed += 1
            continue
        out.append(w)
    return out, removed


def remove_punctuation_from_words(words: List[str]) -> Tuple[List[str], int]:
    removed = 0
    table = str.maketrans('', '', string.punctuation)
    out: List[str] = []
    for w in words:
        nw = w.translate(table)
        if nw != w:
            removed += 1
        if nw != '':
            out.append(nw)
    return out, removed


def light_stem(words: List[str]) -> Tuple[List[str], int]:
    # very small, safe heuristic stemmer: remove common suffixes
    removed = 0
    out: List[str] = []
    for w in words:
        base = w
        lw = w.lower()
        if len(lw) > 5 and lw.endswith('ing'):
            base = w[:-3]
            removed += 1
        elif len(lw) > 4 and lw.endswith('ed'):
            base = w[:-2]
            removed += 1
        elif len(lw) > 3 and lw.endswith('s') and not lw.endswith("ss"):
            base = w[:-1]
            removed += 1
        out.append(base)
    return out, removed


def _find_repeated_phrases(words: List[str], min_len=3, max_len=12) -> Dict[str, int]:
    """Find repeated contiguous word sequences. Return mapping phrase -> occurrences (>=2)."""
    n = len(words)
    counts: Dict[str, int] = {}
    max_l = min(max_len, n // 2)
    for L in range(max_l, min_len - 1, -1):
        seen: Dict[str, int] = {}
        for i in range(0, n - L + 1):
            phrase = " ".join(words[i : i + L])
            seen[phrase] = seen.get(phrase, 0) + 1
        for phrase, c in seen.items():
            if c > 1:
                counts[phrase] = c
        if counts:
            break
    return counts


def _find_repeated_substrings(text: str, min_len=4, max_len=20, min_count=2) -> Dict[str, int]:
    """Find repeated substrings (character-based) in text. Returns mapping substring -> occurrences."""
    n = len(text)
    counts: Dict[str, int] = {}
    # For reasonably sized inputs this naive approach is acceptable.
    max_l = min(max_len, n)
    for L in range(max_l, min_len - 1, -1):
        seen: Dict[str, int] = {}
        for i in range(0, n - L + 1):
            s = text[i : i + L]
            # skip masks and short whitespace-only strings
            if '__MASK' in s or s.strip() == '':
                continue
            seen[s] = seen.get(s, 0) + 1
        for s, c in seen.items():
            if c >= min_count:
                counts[s] = c
        if counts:
            break
    return counts


def _apply_substring_encoding(text: str, candidates: List[str]) -> Tuple[str, Dict[str, str]]:
    """Replace later occurrences of candidate substrings with short tokens [ENCn].

    Returns (new_text, mapping) where mapping maps ENCn -> substring.
    """
    mapping: Dict[str, str] = {}
    if not candidates:
        return text, mapping

    # We'll find occurrences and avoid overlapping replacements by tracking occupied ranges
    occupied = [False] * len(text)
    replacements: Dict[int, Tuple[int, str]] = {}
    enc_id = 1
    for substr in candidates:
        for m in re.finditer(re.escape(substr), text):
            start, end = m.start(), m.end()
            # skip if any occupied
            if any(occupied[start:end]):
                continue
            # record occurrence
            if substr not in mapping:
                # first occurrence: keep it
                mapping[substr] = 'KEEP'
                # mark range as occupied
                for i in range(start, end):
                    occupied[i] = True
            else:
                # subsequent occurrence: mark for replacement with ENC token
                token = f"[ENC{enc_id}]"
                replacements[start] = (end - start, token)
                # mark occupied
                for i in range(start, end):
                    occupied[i] = True
                mapping[token.strip('[]')] = substr
                enc_id += 1

    if not replacements:
        return text, {k: v for k, v in mapping.items() if v != 'KEEP'}

    # Build new text by walking through indices
    out = []
    i = 0
    n = len(text)
    while i < n:
        if i in replacements:
            L, token = replacements[i]
            out.append(token)
            i += L
        else:
            out.append(text[i])
            i += 1
    new_text = ''.join(out)
    # mapping: convert substr->encKey mapping for returned structure
    final_map: Dict[str, str] = {}
    for k, v in mapping.items():
        if v == 'KEEP':
            continue
        # k is token name in mapping? here we added token keys as mapping[tokenName]=substr
        # We want ENCn -> substr
        # mapping entries for ENC tokens were set with key like 'ENC1' -> substr
        pass
    # build final_map from new_text by scanning for [ENCn] tokens
    for m in re.finditer(r"\[ENC(\d+)\]", new_text):
        token = m.group(0).strip('[]')
        # mapping contains token name without brackets -> substr
        if token in mapping:
            final_map[token] = mapping[token]
    return new_text, final_map


def _apply_word_encoding(words: List[str], min_word_len: int = 6, min_count: int = 2) -> Tuple[List[str], Dict[str, str]]:
    """Replace later occurrences of long repeated words with [ENCn] tokens.

    Returns (new_words, mapping) where mapping maps ENCn -> word.
    """
    counts: Dict[str, int] = {}
    for w in words:
        wl = re.sub(r"[^A-Za-z0-9]", "", w).lower()
        if len(wl) >= min_word_len:
            counts[wl] = counts.get(wl, 0) + 1

    candidates = [w for w, c in counts.items() if c >= min_count]
    if not candidates:
        return words, {}

    mapping: Dict[str, str] = {}
    enc_id = 1
    seen_first: Dict[str, bool] = {}
    out: List[str] = []
    for w in words:
        key = re.sub(r"[^A-Za-z0-9]", "", w).lower()
        if key in candidates:
            if not seen_first.get(key, False):
                out.append(w)
                seen_first[key] = True
            else:
                token = f"[ENC{enc_id}]"
                mapping[f"ENC{enc_id}"] = w
                out.append(token)
                enc_id += 1
        else:
            out.append(w)
    return out, mapping


def _apply_phrase_encoding(words: List[str], min_len: int = 2, max_len: int = 12, min_count: int = 2) -> Tuple[List[str], Dict[str, str]]:
    """Replace later occurrences of repeated multi-word phrases with [ENCn] tokens.

    This prefers longer phrases first to maximize token savings under whitespace-based counts.
    Returns (new_words, mapping) where mapping maps ENCn -> phrase (string).
    """
    repeats = _find_repeated_phrases(words, min_len=min_len, max_len=max_len)
    if not repeats:
        return words, {}

    # Prefer longer phrases to maximize savings
    sorted_phrases = sorted(repeats.keys(), key=lambda p: -len(p.split()))
    n = len(words)
    occupied = [False] * n
    replace_starts: Dict[int, Tuple[int, str]] = {}
    mapping: Dict[str, str] = {}
    enc_id = 1

    for phrase in sorted_phrases:
        pw = phrase.split()
        L = len(pw)
        starts: List[int] = []
        for i in range(0, n - L + 1):
            if words[i : i + L] == pw:
                starts.append(i)
        if len(starts) <= 1:
            continue
        # keep first occurrence, replace subsequent ones
        for s in starts[1:]:
            # skip if any overlap with already reserved ranges
            if any(occupied[s : s + L]):
                continue
            for k in range(s, s + L):
                occupied[k] = True
            token = f"[ENC{enc_id}]"
            replace_starts[s] = (L, token)
            mapping[f"ENC{enc_id}"] = phrase
            enc_id += 1

    if not replace_starts:
        return words, {}

    out: List[str] = []
    i = 0
    while i < n:
        if i in replace_starts:
            L, token = replace_starts[i]
            out.append(token)
            i += L
        elif occupied[i]:
            # this index is part of an overlapped region that was skipped; skip it
            i += 1
        else:
            out.append(words[i])
            i += 1

    return out, mapping


def _apply_token_aware_phrase_encoding(words: List[str], use_tiktoken: bool = False, min_len: int = 2, max_len: int = 12) -> Tuple[List[str], Dict[str, str]]:
    """Greedy, token-aware phrase encoding.

    Finds repeated multi-word phrases and simulates replacing later occurrences with short tokens
    like [ENCn]. For each candidate phrase it computes the token-count savings (using
    `count_tokens(..., use_tiktoken=use_tiktoken)`) and greedily applies the best-positive
    saving candidate, repeating until no positive-saving candidates remain.
    """
    # Work on a mutable word list copy
    working = list(words)
    mapping: Dict[str, str] = {}
    enc_counter = 1

    def _join(wlist: List[str]) -> str:
        return " ".join(wlist)

    baseline_text = _join(working)
    baseline_tokens = count_tokens(baseline_text, use_tiktoken=use_tiktoken)

    # Iteratively pick best-saving phrase and apply
    while True:
        # find repeated phrases on current working list
        repeats = _find_repeated_phrases(working, min_len=min_len, max_len=max_len)
        if not repeats:
            break

        best_savings = 0
        best_phrase = None
        best_new_words: List[str] = []

        # evaluate each candidate phrase by simulating replacements
        for phrase in repeats.keys():
            pw = phrase.split()
            L = len(pw)
            # find indices of occurrences
            occs: List[int] = []
            for i in range(0, len(working) - L + 1):
                if working[i : i + L] == pw:
                    occs.append(i)
            if len(occs) <= 1:
                continue

            # build simulated replacement: keep first occ, replace others with token
            token = f"[ENC{enc_counter}]"
            sim: List[str] = []
            i = 0
            skip_set = set(occs[1:])
            while i < len(working):
                if i in skip_set:
                    # insert token and skip L words
                    sim.append(token)
                    i += L
                else:
                    sim.append(working[i])
                    i += 1

            sim_text = _join(sim)
            sim_tokens = count_tokens(sim_text, use_tiktoken=use_tiktoken)
            savings = baseline_tokens - sim_tokens
            if savings > best_savings:
                best_savings = savings
                best_phrase = phrase
                best_new_words = sim

        if not best_phrase or best_savings <= 0:
            break

        # Apply the best replacement to working words and record mapping
        # Find where to increment enc_counter appropriately (we used current enc_counter for sim)
        token_name = f"ENC{enc_counter}"
        mapping[token_name] = best_phrase
        enc_counter += 1
        working = best_new_words
        # update baseline tokens for the next round
        baseline_text = _join(working)
        baseline_tokens = count_tokens(baseline_text, use_tiktoken=use_tiktoken)

    return working, mapping


def compress(prompt: str, aggressive: bool = False, remove_stopwords_opt: bool = False, remove_punctuation_opt: bool = False, light_stemming_opt: bool = False, encode: bool = False, use_tiktoken: bool = False) -> Tuple[str, Dict[str, Any]]:
    """Compress prompt using conservative heuristics.

    Steps:
    - Mask protected spans (code, quotes)
    - Normalize whitespace
    - Remove filler words (outside protected spans)
    - Find repeated contiguous phrases and replace later occurrences with references

    Returns (compressed_prompt, meta) where meta includes replacements and counts.
    """
    original = prompt

    # Mask protected spans so we don't edit code or quoted content
    masked, mapping = _mask_protected_spans(prompt)

    # Extract parenthetical acronyms like "Trainable Steering Vector (TSV)" -> replace with TSV and remember mapping
    def _extract_acronyms(text: str) -> Tuple[str, Dict[str, str]]:
        acr = {}
        # pattern: long phrase followed by parentheses containing an uppercase acronym (2-6 letters)
        pat = re.compile(r"([A-Za-z][A-Za-z0-9 &\-]{2,80}?)\s*\(([A-Z]{2,6})\)")
        def repl(m):
            full = m.group(0)
            long = m.group(1).strip()
            a = m.group(2)
            # prefer to map when long phrase is multi-word and the acronym is shorter
            if len(long.split()) > 1 and len(a) < len(long):
                acr[a] = long
                return a
            return full
        new = pat.sub(repl, text)
        return new, acr

    masked, acronym_map = _extract_acronyms(masked)

    # Normalize whitespace in masked text
    text = " ".join(masked.split())

    words = re.findall(r"\S+", text)

    # Remove trivial filler words (safe outside protected spans because they are masked)
    removed_filler = 0
    if aggressive:
        # extend filler words locally for aggressive mode
        extra = ["please", "could", "would", "also", "then", "now"]
        global FILLER_WORDS
        filler_backup = list(FILLER_WORDS)
        FILLER_WORDS = list(set(FILLER_WORDS + extra))
        words, removed_filler = _remove_fillers(words)
        FILLER_WORDS = filler_backup
    else:
        words, removed_filler = _remove_fillers(words)

    # Optional pipeline steps inspired by prompt compression research
    removed_stopwords = 0
    removed_punct = 0
    removed_stems = 0
    if remove_stopwords_opt:
        words, removed_stopwords = remove_stopwords(words)
    if remove_punctuation_opt:
        words, removed_punct = remove_punctuation_from_words(words)
    if light_stemming_opt:
        words, removed_stems = light_stem(words)

    # In aggressive mode we may optionally remove additional common adjectives / filler nouns
    removed_extras = 0
    if aggressive:
        neww = []
        for w in words:
            lw = w.lower().strip(".,:;!?()[]{}\"')(")
            if lw in EXTRA_REMOVE:
                removed_extras += 1
                continue
            neww.append(w)
        words = neww

    # Optionally run encoding (BPE-like substring replacement) on the masked text
    encoded_map: Dict[str, str] = {}
    if encode:
        # Use a token-aware greedy phrase encoder that simulates token counts
        words, enc_map = _apply_token_aware_phrase_encoding(words, use_tiktoken=use_tiktoken, min_len=2, max_len=8)
        # If no phrase-level encoding candidates found, fall back to long-word encoding
        if not enc_map:
            words, enc_map = _apply_word_encoding(words, min_word_len=6, min_count=2)
        encoded_map = enc_map

    # Find repeated phrases and replace later occurrences with references.
    # Operate at the word-token level to avoid breaking boundaries or punctuation.
    min_phrase = 2 if aggressive else 3
    repeats = _find_repeated_phrases(words, min_len=min_phrase, max_len=12)
    replacements: Dict[str, str] = {}
    # include acronym_map in replacements so UI can display them
    for k, v in acronym_map.items():
        replacements[k] = v
    # include encoded map
    for k, v in encoded_map.items():
        replacements[k] = v
    if repeats:
        # Sort by phrase length (words) desc so longer phrases get priority
        sorted_phrases = sorted(repeats.keys(), key=lambda p: -len(p.split()))
        n = len(words)
        occupied = [False] * n  # tracks which word indices are already consumed by a replacement
        ref_id = 1
        # We'll build an index->replacement mapping for starts to replace
        replace_starts: Dict[int, Tuple[int, str]] = {}
        for phrase in sorted_phrases:
            pw = phrase.split()
            L = len(pw)
            starts: List[int] = []
            for i in range(0, n - L + 1):
                if occupied[i]:
                    continue
                if words[i : i + L] == pw:
                    starts.append(i)
            if len(starts) <= 1:
                continue
            # keep the first occurrence, mark others for replacement
            for s in starts[1:]:
                # mark indices occupied so other phrases don't overlap
                for k in range(s, s + L):
                    occupied[k] = True
                replace_starts[s] = (L, f"REF{ref_id}")
            replacements[f"REF{ref_id}"] = phrase
            ref_id += 1

        # Build final word list applying replacements
        out_words: List[str] = []
        i = 0
        while i < n:
            if i in replace_starts:
                L, refkey = replace_starts[i]
                out_words.append(f"[{refkey}]")
                i += L
            elif occupied[i]:
                # This index was occupied by overlapping replacement; skip it
                i += 1
            else:
                out_words.append(words[i])
                i += 1

        compressed_masked = " ".join(out_words)
    else:
        compressed_masked = " ".join(words)

    # Unmask protected spans back into the compressed content
    compressed = _unmask(compressed_masked, mapping)

    # Final cleanup
    compressed = re.sub(r"\s+", " ", compressed).strip()

    meta: Dict[str, Any] = {
        "original_length_chars": len(original),
        "compressed_length_chars": len(compressed),
        "removed_fillers": removed_filler,
        "removed_stopwords": removed_stopwords,
        "removed_punctuation_tokens": removed_punct,
        "removed_light_stems": removed_stems,
        "removed_extra_fillers": removed_extras,
        "replacements": replacements,
        "aggressive_used": aggressive,
        "opts": {
            "remove_stopwords": remove_stopwords_opt,
            "remove_punctuation": remove_punctuation_opt,
            "light_stemming": light_stemming_opt,
        }
    }
    return compressed, meta


if __name__ == "__main__":
    s = (
        "Write a long answer. For example write a step-by-step plan. "
        "Write a long answer. For example write a step-by-step plan. "
        "Make it thorough and detailed, very helpful and really useful."
    )
    c, m = compress(s)
    print("Original:\n", s)
    print("Compressed:\n", c)
    print("Meta:\n", m)
