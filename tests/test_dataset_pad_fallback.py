#!/usr/bin/env python3
# Developer: inkbytefo
# AI: Claude Sonnet 4.5
# Modified: 2025-11-01

import os
import sys
from pathlib import Path
import io
import tempfile
import pytest
import torch
from sentencepiece import SentencePieceProcessor

# Ensure repository root is on sys.path for 'src' imports
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

from src.dataset import TurkishTextDataset


def test_padding_uses_unk_when_pad_disabled():
    """
    Ensure dataset pads with UNK ID when SentencePiece pad_id == -1.
    """
    model_path = os.path.join('tokenizer', 'morphopiece.model')
    if not os.path.exists(model_path):
        pytest.skip("morphopiece.model not present; skip pad fallback test")

    sp = SentencePieceProcessor(model_file=model_path)
    unk = sp.unk_id()

    # Require scenario where pad_id is disabled
    if sp.pad_id() >= 0:
        pytest.skip("pad_id is defined by model; fallback not applicable")

    # Create a tiny temporary corpus line to force padding
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.txt')
    os.close(tmp_fd)
    try:
        with io.open(tmp_path, 'w', encoding='utf-8') as f:
            f.write("evler ve arabalar\n")

        ds = TurkishTextDataset(
            corpus_file=tmp_path,
            tokenizer=sp,
            max_seq_len=32,
        )

        item = ds[0]
        input_ids = item['input_ids'].tolist()
        target_ids = item['target_ids'].tolist()

        # No negative IDs
        assert all(i >= 0 for i in input_ids), "input_ids contain negative IDs"
        assert all(t >= 0 for t in target_ids), "target_ids contain negative IDs"

        # Check that trailing padded positions equal UNK id
        # Find first zero attention (padded region)
        attn = item['attention_mask'].tolist()
        if 0 in attn:
            first_pad = attn.index(0)
            assert input_ids[first_pad:] == [unk] * (len(input_ids) - first_pad)
            assert target_ids[first_pad:] == [unk] * (len(target_ids) - first_pad)
    finally:
        os.remove(tmp_path)