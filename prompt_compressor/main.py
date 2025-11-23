
import sys
import time
import psutil
import os
import re
from core.rule_based import optimize_rule_based, read_fillers
from core.smart_reduction import optimize_smart_reduction
from utils.co2_estimator import token_count, tokens_to_wh, per_wh
from core.validator import validate_compression
from utils.pricing import (
    get_model_list, 
    get_model_by_index, 
    calculate_cost_savings, 
    format_cost_display,
    get_all_model_ids,
    MODEL_CATEGORIES
)


def run(prompt: str, model_id: str = None):
    original = prompt.strip()
    if not original:
        print('No prompt provided.')
        return

    try:
        patterns, words, redundant_phrases, replacements, stopwords = read_fillers('config/fillers.json')
    except Exception:
        print('Missing or invalid config/fillers.json. Create config/fillers.json with patterns, words, redundant_phrases, verbose_replacements, and stopwords.')
        return

    # Start tracking compression energy
    process = psutil.Process(os.getpid())
    compression_start_time = time.time()
    compression_start_cpu = process.cpu_percent(interval=0.1)

    # Step 1: Enhanced rule-based compression
    cleaned = optimize_rule_based(
        original, 
        patterns=patterns, 
        words=words,
        redundant_phrases=redundant_phrases,
        verbose_replacements=replacements,
        stopwords=stopwords,
        remove_stopwords=False  # Can be made configurable
    )

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

    # Step 2: Smart reduction using TF-IDF
    compressed = optimize_smart_reduction(cleaned, target_reduction=0.20, preserve_question_structure=True)
    compressed = tidy_text(compressed)
    
    orig_tokens = token_count(original)
    compressed_tokens = token_count(compressed)
    
    # Fallback: if compression ratio is low, try aggressive patterns
    if compressed_tokens >= orig_tokens * 0.95:  # Less than 5% compression
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
    print(f'After 2-layer compression: {compressed_tokens}')
    print(f'  Layer 1 (rule-based): Pattern matching & phrase replacement')
    print(f'  Layer 2 (smart reduction): TF-IDF word importance')
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

    # Calculate and display cost savings if model was provided
    if model_id:
        try:
            cost_info = calculate_cost_savings(tokens_saved, model_id)
            print(format_cost_display(cost_info))
        except ValueError as e:
            print(f'\nCost calculation error: {e}')

    print('\nCompressed Prompt:')
    print(compressed)
    
    if not is_valid:
        print('\nWARNING: Compressed prompt may have lost important information.')
        print('Consider using the original or a less aggressive compression.')


def _read_prompt_from_stdin_or_input() -> str:
    if not sys.stdin.isatty():
        # When piped, the rest of stdin after model selection is the prompt
        data = sys.stdin.read()
        if data.strip():
            return data
    try:
        print('Enter your prompt (paste all at once, then press Ctrl+D or Ctrl+Z on Windows):\n')
        lines = []
        while True:
            try:
                line = input()
                lines.append(line)
            except EOFError:
                break
        return '\n'.join(lines)
    except KeyboardInterrupt:
        return ''


def _select_model() -> str:
    """Prompt user to select an LLM model for cost calculation."""
    print("\n" + "="*60)
    print("SELECT YOUR LLM MODEL")
    print("="*60)
    print(get_model_list())
    print("\nEnter the number of your model (or press Enter to skip cost calculation):")
    
    try:
        # Read from stdin if available, otherwise from terminal
        if not sys.stdin.isatty():
            # When piped, read from stdin
            choice = sys.stdin.readline().strip()
        else:
            # Interactive mode, read from terminal
            choice = input("> ").strip()
        
        if not choice:
            return None
        
        # Try to parse as number first
        try:
            index = int(choice)
            model_id = get_model_by_index(index)
            if model_id:
                return model_id
        except ValueError:
            # If not a number, try fuzzy matching with model names
            choice_lower = choice.lower()
            
            # Build list of all models with their display names
            for category, model_list in MODEL_CATEGORIES.items():
                for model_id, display_name in model_list:
                    # Check if user input matches model_id or display_name
                    if (choice_lower in model_id.lower() or 
                        choice_lower in display_name.lower() or
                        choice.replace(' ', '-').lower() in model_id.lower()):
                        print(f"✓ Selected: {display_name}")
                        return model_id
        
        print(f"Invalid selection '{choice}'. Skipping cost calculation.")
        return None
        
    except (FileNotFoundError, EOFError):
        print("Invalid input or no terminal available. Skipping cost calculation.")
        return None


if __name__ == '__main__':
    model_id = _select_model()
    prompt = _read_prompt_from_stdin_or_input()
    run(prompt, model_id)
