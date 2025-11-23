import sys
import time
import psutil
import os
import re
from core.rule_based import optimize_rule_based, read_fillers
from core.extractive import optimize_extractive
from utils.co2_estimator import token_count, tokens_to_wh, per_wh
from core.validator import validate_compression


def run(prompt: str):
    original = prompt.strip()
    if not original:
        print('No prompt provided.')
        return

    try:
        patterns, words = read_fillers('config/fillers.json')
    except Exception:
        print('Missing or invalid config/fillers.json. Create config/fillers.json with {"patterns": [...], "words": [...]}')
        return

    # Start tracking compression energy
    process = psutil.Process(os.getpid())
    compression_start_time = time.time()
    compression_start_cpu = process.cpu_percent(interval=0.1)

    # Step 1: rule-based stripping using provided fillers
    cleaned = optimize_rule_based(original, patterns=patterns, words=words)

    def tidy_text(s: str) -> str:
        if not s:
            return s
        s = s.strip()
        s = re.sub(r"[\.!?]{2,}", lambda m: m.group(0)[0], s)
        s = re.sub(r"^[^\w]+", '', s)
        s = re.sub(r"\s+([\.,!?:;])", r"\1", s)
        s = re.sub(r"([\.,!?:;])([^\s\.,!?:;])", r"\1 \2", s)
        
        # Grammar cleanup
        s = re.sub(r"^you could ", "", s, flags=re.I)
        s = re.sub(r"^you would ", "", s, flags=re.I)
        s = re.sub(r"^you can ", "", s, flags=re.I)
        s = re.sub(r"^you should ", "", s, flags=re.I)
        s = re.sub(r"^I need to ", "", s, flags=re.I)
        s = re.sub(r"^I want to ", "", s, flags=re.I)
        
        # Capitalize first letter if it's lowercase
        if s and s[0].islower():
            s = s[0].upper() + s[1:]
        
        s = s.rstrip()
        return s

    cleaned = tidy_text(cleaned)

    # Step 2: extractive summarization on cleaned prompt
    orig_tokens = token_count(original)
    num_sentences = 2 if orig_tokens >= 60 else 1
    compressed = optimize_extractive(cleaned, max_sentences=num_sentences)
    compressed = tidy_text(compressed)

    cleaned_tokens = token_count(cleaned)
    compressed_tokens = token_count(compressed)
    if compressed_tokens >= cleaned_tokens:
        aggressive_patterns = []
        try:
            import json
            from pathlib import Path
            p = Path('config/fillers.json')
            if p.exists():
                data = json.loads(p.read_text(encoding='utf-8'))
                aggressive_patterns = data.get('aggressive', [])
        except Exception:
            aggressive_patterns = []

        aggressive_result = None
        for pat in aggressive_patterns:
            try:
                candidate = re.sub(pat, '', cleaned, flags=re.I)
                candidate = tidy_text(candidate)
                if token_count(candidate) < compressed_tokens:
                    aggressive_result = candidate
                    compressed = candidate
                    compressed_tokens = token_count(compressed)
                    break
            except re.error:
                continue

        if aggressive_result is None:
            candidate = re.sub(r"[,;:\-]\s*(and|for|that|which|please|include)\b.*$", '', cleaned, flags=re.I)
            candidate = tidy_text(candidate)
            if token_count(candidate) < compressed_tokens:
                compressed = candidate
                compressed_tokens = token_count(compressed)

    # Calculate compression energy cost
    compression_end_time = time.time()
    compression_end_cpu = process.cpu_percent(interval=0.1)
    
    compression_time_sec = compression_end_time - compression_start_time
    avg_cpu_percent = (compression_start_cpu + compression_end_cpu) / 2
    
    # Estimate compression energy: CPU power × time
    # Assuming ~15W CPU base power, scaled by usage
    cpu_power_watts = 15 * (avg_cpu_percent / 100)
    compression_energy_wh = (cpu_power_watts * compression_time_sec) / 3600

    # Semantic validation - ensure we didn't destroy meaning
    is_valid, similarity = validate_compression(original, compressed, min_similarity=0.75)

    # Token and CO2 calculations
    orig_tokens = token_count(original)
    cleaned_tokens = token_count(cleaned)
    compressed_tokens = token_count(compressed)
    tokens_saved = max(0, orig_tokens - compressed_tokens)
    
    # LLM energy saved
    llm_energy_saved_wh = tokens_to_wh(tokens_saved)
    
    # Net energy (accounting for compression cost)
    net_energy_saved_wh = llm_energy_saved_wh - compression_energy_wh
    
    # Get carbon intensity from API (with fallback)
    from utils.co2_estimator import get_carbon_intensity
    carbon_intensity = get_carbon_intensity()
    
    # CO2 calculations
    llm_co2_saved = per_wh(llm_energy_saved_wh)
    compression_co2_cost = per_wh(compression_energy_wh)
    net_co2_saved = llm_co2_saved - compression_co2_cost

    print('\nCAPO Metrics:')
    print(f'Original tokens: {orig_tokens}')
    print(f'After rule-based (step1) tokens: {cleaned_tokens}')
    print(f'After extractive (step2) tokens: {compressed_tokens}')
    print(f'Tokens saved: {tokens_saved}')
    print(f'Compression ratio: {(1 - compressed_tokens/orig_tokens)*100:.1f}%' if orig_tokens > 0 else 'N/A')
    print(f'Compression time: {compression_time_sec:.3f} seconds')
    print()
    print(f'Semantic Similarity: {similarity:.3f}')
    if is_valid:
        print(f'Quality Check: PASSED (similarity >= 0.75)')
    else:
        print(f'Quality Check: FAILED (similarity < 0.75 - meaning may be lost)')
    print()
    print(f'Energy Analysis:')
    print(f'  LLM energy saved:        {llm_energy_saved_wh:.8f} Wh')
    print(f'  Compression energy cost: {compression_energy_wh:.8f} Wh')
    print(f'  NET energy saved:        {net_energy_saved_wh:.8f} Wh')
    print()
    print(f'CO2 Analysis (Grid: {carbon_intensity:.1f} gCO2eq/kWh):')
    print(f'  LLM CO2 saved:        {llm_co2_saved:.8f} g')
    print(f'  Compression CO2 cost: {compression_co2_cost:.8f} g')
    print(f'  NET CO2 saved:        {net_co2_saved:.8f} g')

    print('\nCompressed Prompt:')
    print(compressed)
    
    if not is_valid:
        print('\nWARNING: Compressed prompt may have lost important information.')
        print('Consider using the original or a less aggressive compression.')


def _read_prompt_from_stdin_or_input() -> str:
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data.strip():
            return data
    try:
        return input('Enter your prompt (finish with Enter):\n')
    except EOFError:
        return ''


if __name__ == '__main__':
    prompt = _read_prompt_from_stdin_or_input()
    run(prompt)
