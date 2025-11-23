import pytest
from app import compressor


def test_remove_fillers_and_repeat():
    s = (
        "Write a long answer. For example write a step-by-step plan. "
        "Write a long answer. For example write a step-by-step plan. "
        "Make it thorough and detailed, very helpful and really useful."
    )
    compressed, meta = compressor.compress(s)
    # Should have replacements for the repeated sentence
    assert any(k.startswith("REF") for k in meta["replacements"].keys())
    # Should remove at least one filler word like "very" or "really"
    assert meta["removed_fillers"] >= 1
    # Compressed should be shorter than original
    assert len(compressed) < len(s)


def test_preserve_code_and_quotes():
    s = "Here is some code: `print('hello, very world')` and a quote: \"This is very important\""
    compressed, meta = compressor.compress(s)
    # masked content (code and quote) should remain present
    assert "print('hello, very world')" in compressed
    assert '"This is very important"' in compressed


def test_token_count_fallback():
    s = "one two three"
    assert compressor.count_tokens(s) == 3


def test_user_example_prompt():
    # Reproduction of the user's example to ensure the first occurrence of the phrase is preserved
    s = (
        "I have a problem. I sometimes feel nauseous when I have a lot to do, "
        "what do I do? Do I drink water, give me suggestions"
    )
    compressed, meta = compressor.compress(s)

    # First occurrence of the phrase should be preserved (not replaced by a ref token)
    assert "I have a problem" in compressed or "I have a problem." in compressed

    # There should be at least one replacement token for subsequent occurrence(s)
    assert any(k.startswith("REF") for k in meta["replacements"].keys())

    # The replacement mapping should contain the repeated phrase
    assert any("I have a" in v for v in meta["replacements"].values())

    # Compressed text should be no longer than original
    assert len(compressed) <= len(s)


def test_acronym_extraction():
    s = "Trainable Steering Vector (TSV) is a vector. TSV helps detect." 
    compressed, meta = compressor.compress(s)
    # should map TSV -> Trainable Steering Vector
    assert 'TSV' in meta['replacements']
    # compressed should contain TSV (shorter)
    assert 'TSV' in compressed
    # token count should be reduced or equal
    assert compressor.count_tokens(compressed) <= compressor.count_tokens(s)


def test_substring_encoding():
    # create text with repeated long substring
    sub = 'microarchitecture'
    s = f"{sub} {sub} {sub} plus extra words to pad the prompt. {sub}"
    compressed, meta = compressor.compress(s, encode=True)
    # expect encoding mapping present
    assert any(k.startswith('ENC') for k in meta['replacements'].keys())
    # token count using whitespace fallback may not reflect BPE-like benefits; check chars decreased
    assert len(compressed) < len(s)


def test_phrase_encoding_results_in_token_savings_whitespace():
    # Repeated multi-word phrase should be encoded and reduce whitespace-token count
    phrase = "trainable steering vector"
    s = f"{phrase} {phrase} {phrase} additional padding words {phrase}"
    compressed, meta = compressor.compress(s, encode=True)
    # ensure ENC mapping present
    assert any(k.startswith('ENC') for k in meta['replacements'].keys())
    # whitespace-token count should be lower after multi-word phrase encoding
    assert compressor.count_tokens(compressed) < compressor.count_tokens(s)
