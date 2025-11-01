#!/usr/bin/env python3
# Developer: inkbytefo
# AI: Claude Sonnet 4.5
# Modified: 2025-11-01

"""
Minimal compatibility test for SentencePiece training args.

Ensures that training without deprecated parameters succeeds on a tiny corpus.
"""

import os
import tempfile
import sentencepiece as spm


def test_sentencepiece_training_without_deprecated_args():
    # Create a tiny temporary corpus
    with tempfile.TemporaryDirectory() as tmp:
        corpus = os.path.join(tmp, "tiny.txt")
        with open(corpus, "w", encoding="utf-8") as f:
            f.write("Merhaba dünya\n")
            f.write("Bugün markete gittim\n")
            f.write("Evlerimiz çok güzeldi\n")

        model_prefix = os.path.join(tmp, "sp_tiny")

        training_args = {
            "input": corpus,
            "model_prefix": model_prefix,
            "vocab_size": 64,
            "model_type": "unigram",
            "character_coverage": 1.0,
            "input_sentence_size": 1000,
            "shuffle_input_sentence": True,
            "num_threads": 2,
            "max_sentence_length": 1024,
            "add_dummy_prefix": True,
            "remove_extra_whitespaces": True,
            "hard_vocab_limit": False,
            "byte_fallback": True,
        }

        # Should not raise; deprecated args are intentionally omitted
        spm.SentencePieceTrainer.train(**training_args)

        # Verify outputs exist
        assert os.path.exists(model_prefix + ".model")
        assert os.path.exists(model_prefix + ".vocab")