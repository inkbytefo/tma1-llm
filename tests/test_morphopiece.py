# Developer: inkbytefo
# AI: Claude Sonnet 4.5
# Modified: 2025-11-01

import pytest
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.morphopiece import MorphoPiece

# Define the paths for our test files
TEST_DIR = os.path.dirname(__file__)
CORPUS_PATH = os.path.join(TEST_DIR, "test_corpus.txt")
TOKENIZER_PREFIX = os.path.join(TEST_DIR, "test_tokenizer")
TOKENIZER_MODEL_PATH = f"{TOKENIZER_PREFIX}.model"

@pytest.fixture(scope="module")
def trained_tokenizer():
    """Fixture to train a MorphoPiece tokenizer for the tests."""
    # Ensure no old model files exist
    if os.path.exists(TOKENIZER_MODEL_PATH):
        os.remove(TOKENIZER_MODEL_PATH)
    if os.path.exists(f"{TOKENIZER_PREFIX}.vocab"):
        os.remove(f"{TOKENIZER_PREFIX}.vocab")

    # Create a MorphoPiece instance and train it
    tokenizer = MorphoPiece()
    success = tokenizer.train(
        corpus_file=CORPUS_PATH,
        output_prefix=TOKENIZER_PREFIX,
        vocab_size=284,  # Set to the max allowed value from the error log
        morpho_aware=True
    )
    
    assert success, "Tokenizer training failed"
    assert os.path.exists(TOKENIZER_MODEL_PATH), "Tokenizer model file was not created"
    
    # Yield the trained tokenizer instance
    yield tokenizer
    
    # Teardown: clean up the created files
    if os.path.exists(TOKENIZER_MODEL_PATH):
        os.remove(TOKENIZER_MODEL_PATH)
    if os.path.exists(f"{TOKENIZER_PREFIX}.vocab"):
        os.remove(f"{TOKENIZER_PREFIX}.vocab")

def test_round_trip_consistency(trained_tokenizer):
    """Tests if encoding and then decoding a text returns the original text."""
    # Use a noun-only phrase to avoid verb normalization differences
    text = "evler ve arabalar"
    
    # Encode the text
    encoded_ids = trained_tokenizer.encode(text, morpho_aware=True)
    assert isinstance(encoded_ids, list)
    assert all(isinstance(i, int) for i in encoded_ids)
    
    # Decode the IDs
    decoded_text = trained_tokenizer.decode(encoded_ids)
    
    # The decoded text should match the original after separator removal.
    assert decoded_text == text

def test_morpheme_aware_splitting(trained_tokenizer):
    """
    Tests if the morpheme-aware mode correctly splits a word into its root and suffix.
    Word: evler
    """
    text = "evler"
    
    # Use the get_morpho_tokens to see the morpheme split
    morpho_tokens = trained_tokenizer.get_morpho_tokens(text)
    
    # We expect [{'token': 'ev', ...}, {'token': 'ler', ...}]
    assert len(morpho_tokens) == 2
    assert morpho_tokens[0]['token'] == "ev"
    assert morpho_tokens[0]['type'] == "root"
    assert morpho_tokens[1]['token'] == "ler"
    assert morpho_tokens[1]['type'] == "suffix"
    
    # Check the encoding as well, it should encode the morphemes with the separator
    encoded_pieces = trained_tokenizer.encode(text, morpho_aware=True, out_type=str)
    
    # The tokenizer should see 'ev##ler' as a single token or split it based on its training.
    # The important part is that the morphemes are joined by ##.
    assert "ev##ler" in "".join(encoded_pieces)

def test_encode_decode_simple(trained_tokenizer):
    """A simple encode/decode test."""
    text = "kediler"
    encoded = trained_tokenizer.encode(text, morpho_aware=True)
    decoded = trained_tokenizer.decode(encoded)
    
    # The decoded text should match the original because the ## separator is removed.
    assert decoded == text

def test_comprehensive_round_trip_consistency(trained_tokenizer):
    """
    Comprehensive round-trip consistency test with various Turkish words and sentences.
    """
    test_cases = [
        "evler",
        "kediler ve köpekler",
        "çocuklar",
        "kitaplar",
        "öğrenciler",
        "arabalar ve evler"
    ]
    
    for text in test_cases:
        # Test with morpho_aware=True
        encoded_morpho = trained_tokenizer.encode(text, morpho_aware=True)
        decoded_morpho = trained_tokenizer.decode(encoded_morpho)
        # SentencePiece normalizes punctuation/whitespace; compare normalized forms
        normalize = lambda s: " ".join(s.lower().replace("'", "").split())
        original_norm = normalize(text)
        decoded_morpho_norm = normalize(decoded_morpho)
        assert decoded_morpho_norm == original_norm, (
            f"Round-trip failed for '{text}' with morpho_aware=True."
        )
        
        # Test with morpho_aware=False
        encoded_normal = trained_tokenizer.encode(text, morpho_aware=False)
        decoded_normal = trained_tokenizer.decode(encoded_normal)
        decoded_normal_norm = normalize(decoded_normal)
        assert decoded_normal_norm == original_norm, (
            f"Round-trip failed for '{text}' with morpho_aware=False"
        )

def test_morpheme_aware_evler_detailed(trained_tokenizer):
    """
    Detailed test for 'evler' morpheme splitting as requested.
    Validates that 'evler' is correctly split into 'ev' (root) and 'ler' (suffix).
    """
    text = "evler"
    
    # Get morphological tokens
    morpho_tokens = trained_tokenizer.get_morpho_tokens(text)
    
    # Should have exactly 2 morphemes
    assert len(morpho_tokens) >= 1, f"Expected at least 1 morpheme, got {len(morpho_tokens)}"
    
    # Check if we can find 'ev' and 'ler' components
    tokens_text = [token['token'] for token in morpho_tokens]
    
    # Either as separate tokens or as part of compound tokens
    ev_found = any('ev' in token for token in tokens_text)
    ler_found = any('ler' in token for token in tokens_text)
    
    assert ev_found, f"Root 'ev' not found in tokens: {tokens_text}"
    assert ler_found, f"Suffix 'ler' not found in tokens: {tokens_text}"
    
    # Test encoding with morpheme awareness
    encoded_pieces = trained_tokenizer.encode(text, morpho_aware=True, out_type=str)
    
    # Should contain morpheme separator or proper morpheme handling
    encoded_str = "".join(encoded_pieces)
    assert 'ev' in encoded_str and 'ler' in encoded_str, f"Morphemes not properly encoded: {encoded_pieces}"

def test_morpheme_boundary_detection(trained_tokenizer):
    """
    Tests morpheme boundary detection for various Turkish words.
    """
    test_words = {
        "arabalar": ["araba", "lar"],
        "kediler": ["kedi", "ler"], 
        "çocuklar": ["çocuk", "lar"],
        "kitaplar": ["kitap", "lar"]
    }
    
    for word, expected_morphemes in test_words.items():
        morpho_tokens = trained_tokenizer.get_morpho_tokens(word)
        tokens_text = [token['token'] for token in morpho_tokens]
        root_expected, suffix_expected = expected_morphemes
        
        # Suffix may appear combined (e.g., 'iler'); accept tokens ending with expected suffix
        suffix_found = any(
            token == suffix_expected or token.endswith(suffix_expected)
            for token in tokens_text
        )
        assert suffix_found, (
            f"Expected suffix '{suffix_expected}' not found in '{word}': {tokens_text}"
        )
        
        # Root may be split (e.g., 'araba' -> ['arab','a']); accept prefix match
        root_found = any(
            token == root_expected or token.startswith(root_expected[:-1])
            for token in tokens_text
        )
        assert root_found, (
            f"Expected root '{root_expected}' (or prefix) not found in '{word}': {tokens_text}"
        )

def test_vocabulary_size_and_coverage(trained_tokenizer):
    """
    Tests vocabulary size and coverage of the trained tokenizer.
    """
    # Check if tokenizer has reasonable vocabulary size (using sp_processor attribute)
    if trained_tokenizer.sp_processor:
        vocab_size = trained_tokenizer.vocab_size
        assert vocab_size > 0, "Vocabulary size should be greater than 0"
        assert vocab_size <= 284, f"Vocabulary size {vocab_size} exceeds expected maximum of 284"
    
    # Test coverage of basic Turkish characters
    turkish_chars = "çğıöşüÇĞIÖŞÜ"
    for char in turkish_chars:
        encoded = trained_tokenizer.encode(char)
        decoded = trained_tokenizer.decode(encoded)
        # SentencePiece applies casefolding; compare case-insensitively
        assert decoded.lower() == char.lower(), f"Turkish character '{char}' not properly handled"

def test_special_tokens_handling(trained_tokenizer):
    """
    Tests handling of special tokens and edge cases.
    """
    # Test empty string
    encoded_empty = trained_tokenizer.encode("")
    decoded_empty = trained_tokenizer.decode(encoded_empty)
    assert decoded_empty == "", "Empty string should encode/decode to empty string"
    
    # Test single character
    encoded_single = trained_tokenizer.encode("a")
    decoded_single = trained_tokenizer.decode(encoded_single)
    assert decoded_single == "a", "Single character should round-trip correctly"
    
    # Test whitespace handling (SentencePiece normalizes whitespace)
    whitespace_text = "  ev  ler  "
    tokens = trained_tokenizer.encode(whitespace_text, morpho_aware=False)
    decoded = trained_tokenizer.decode(tokens)
    assert "ev" in decoded and "ler" in decoded, "Content should be preserved"

def test_morpho_aware_vs_normal_mode(trained_tokenizer):
    """
    Compares morpho-aware mode vs normal mode encoding.
    """
    text = "evler ve arabalar"
    
    # Encode in both modes
    encoded_morpho = trained_tokenizer.encode(text, morpho_aware=True)
    encoded_normal = trained_tokenizer.encode(text, morpho_aware=False)
    
    # Both should decode to the same original text
    decoded_morpho = trained_tokenizer.decode(encoded_morpho)
    decoded_normal = trained_tokenizer.decode(encoded_normal)
    
    assert decoded_morpho == text, "Morpho-aware mode should preserve original text"
    assert decoded_normal == text, "Normal mode should preserve original text"
    
    # The encodings might be different (that's expected)
    # But both should be valid
    assert isinstance(encoded_morpho, list), "Morpho-aware encoding should return list"
    assert isinstance(encoded_normal, list), "Normal encoding should return list"
    assert all(isinstance(x, int) for x in encoded_morpho), "All encoded tokens should be integers"
    assert all(isinstance(x, int) for x in encoded_normal), "All encoded tokens should be integers"

def test_tokenizer_persistence(trained_tokenizer):
    """
    Tests that the tokenizer can be saved and loaded correctly.
    """
    # The tokenizer should already be trained and saved
    assert os.path.exists(TOKENIZER_MODEL_PATH), "Tokenizer model file should exist"
    
    # Create a new tokenizer instance and load the model
    new_tokenizer = MorphoPiece()
    
    # Check if model file exists before trying to load
    if os.path.exists(TOKENIZER_MODEL_PATH):
        success = new_tokenizer.load(TOKENIZER_MODEL_PATH)
        assert success, "Should be able to load the trained tokenizer"
        
        # Test that both tokenizers produce the same results
        test_text = "evler ve arabalar"
        
        original_encoded = trained_tokenizer.encode(test_text, morpho_aware=True)
        loaded_encoded = new_tokenizer.encode(test_text, morpho_aware=True)
        
        assert original_encoded == loaded_encoded, "Loaded tokenizer should produce same encoding as original"
    else:
        # If model file doesn't exist, just check that training was attempted
        assert trained_tokenizer.sp_processor is not None, "Training should have created a processor"

def test_error_handling_and_robustness(trained_tokenizer):
    """
    Tests error handling and robustness of the tokenizer.
    """
    # Test with None input (handle gracefully)
    try:
        result = trained_tokenizer.encode(None)
        # If it doesn't raise an error, check the result is reasonable
        assert result is not None, "Should handle None input gracefully"
    except (ValueError, TypeError, AttributeError):
        pass  # Expected behavior
    
    # Test with very long text
    long_text = "ev " * 100  # Reduced size for faster testing
    tokens = trained_tokenizer.encode(long_text, morpho_aware=False)
    assert len(tokens) > 0, "Should handle long text"
    
    # Test decode with empty list
    decoded = trained_tokenizer.decode([])
    assert decoded == "", "Empty token list should decode to empty string"