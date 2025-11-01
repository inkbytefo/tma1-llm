#!/usr/bin/env python3
# Developer: inkbytefo
# AI: Claude Sonnet 4.5
# Modified: 2025-11-02

import os


def test_tiny_corpus_exists_and_utf8():
    """Verify tiny corpus file exists and contains Turkish UTF-8 characters."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'tiny_corpus.txt')
    assert os.path.exists(path), "tiny_corpus.txt should exist"

    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    assert len(text) > 100, "Corpus should be non-trivial length"
    # Check presence of common Turkish letters
    turkish_chars = set("ğüşiöçĞÜŞİÖÇ")
    assert any(ch in text for ch in turkish_chars), "Should contain Turkish-specific letters"
    # Ensure punctuation preserved
    assert "," in text and "." in text, "Basic punctuation should exist"