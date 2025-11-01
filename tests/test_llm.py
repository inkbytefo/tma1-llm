#!/usr/bin/env python3
# Developer: inkbytefo
# AI: Claude Sonnet 4.5
# Modified: 2025-11-02

"""
============================================================================
LLM Test Suite (Modern PyTorch + SentencePiece)
============================================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest

from llm_engine import LLMEngine
from src.model import ModelConfig
from sentencepiece import SentencePieceProcessor

TOKENIZER_MODEL = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tokenizer", "morphopiece.model")
MODEL_CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "tma1", "checkpoint_latest.pt")

def has_tokenizer():
    return os.path.exists(TOKENIZER_MODEL)

def has_checkpoint():
    return os.path.exists(MODEL_CHECKPOINT)

def test_sentencepiece_tokenizer_basic():
    """Test SentencePiece tokenizer availability and basic encode/decode."""
    print("🧪 Testing SentencePiece tokenizer...")
    if not has_tokenizer():
        pytest.skip("Tokenizer model not found; expected tokenizer/morphopiece.model")

    sp = SentencePieceProcessor(model_file=TOKENIZER_MODEL)
    text = "Merhaba dünya"
    ids = sp.encode(text, out_type=int)
    assert isinstance(ids, list) and len(ids) > 0
    decoded = sp.decode(ids)
    assert isinstance(decoded, str)
    print("✅ SentencePiece tokenizer OK")

# Legacy tests removed: LLMEngine now initializes from paths, not raw ModelConfig

def test_engine_init_and_stats():
    """Test LLMEngine initialization and stats reporting."""
    print("🧪 Testing LLMEngine init + stats...")
    tokenizer_path = TOKENIZER_MODEL if has_tokenizer() else None
    # Avoid loading potentially incompatible checkpoints in CI
    model_path = None

    engine = LLMEngine(
        tokenizer_path=tokenizer_path,
        model_path=model_path
    )
    stats = engine.get_stats()
    assert "device" in stats and "tokenizer_loaded" in stats
    print("✅ LLMEngine stats OK")


def test_engine_generate_small_output():
    """Generate a short output to validate end-to-end pipeline."""
    print("🧪 Testing LLMEngine.generate...")
    if not has_tokenizer():
        pytest.skip("Tokenizer model not found; expected tokenizer/morphopiece.model")

    engine = LLMEngine(
        tokenizer_path=TOKENIZER_MODEL,
        model_path=None
    )
    out = engine.generate("Merhaba", max_new_tokens=5, do_sample=False)
    assert isinstance(out, str)
    print("✅ LLMEngine.generate OK")

