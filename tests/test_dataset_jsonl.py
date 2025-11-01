#!/usr/bin/env python3
# Developer: inkbytefo
# AI: Claude Sonnet 4.5
# Modified: 2025-11-01

"""
============================================================================
Dataset JSONL Format Tests
Tests for TurkishTextDataset with JSONL format and morpho_types support
============================================================================
"""

import os
import sys
import json
import tempfile
import pytest
import torch
from pathlib import Path
from sentencepiece import SentencePieceProcessor

# Ensure repository root is on sys.path for 'src' imports
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

from src.dataset import TurkishTextDataset

def create_test_jsonl(jsonl_path, tokenizer_path):
    """Create a test JSONL file with morpho_types"""
    # Load tokenizer to get actual tokens
    sp = SentencePieceProcessor(model_file=tokenizer_path)
    
    # Sample Turkish text
    sample_text = "Evlerimdekiler gittim dün"
    
    # Tokenize to get tokens
    token_ids = sp.encode(sample_text, out_type=int)
    tokens = [sp.id_to_piece(token_id) for token_id in token_ids]
    
    # Create morpho_types (simplified: 0=root, 1=suffix, 2=verb, 3=other, 4=pad)
    # This is a mock - real preprocessing would use MorphoSplitter
    morpho_types = []
    for i, token in enumerate(tokens):
        if i == 0 or i == 3:  # Assume first and 4th tokens are roots
            morpho_types.append(0)  # root
        elif i == 4:  # Assume 5th token is verb
            morpho_types.append(2)  # verb
        elif len(token) <= 2:
            morpho_types.append(1)  # suffix (short tokens)
        else:
            morpho_types.append(3)  # other
    
    # Write JSONL
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        output = {
            "tokens": tokens,
            "morpho_types": morpho_types
        }
        f.write(json.dumps(output, ensure_ascii=False) + '\n')
        
        # Add empty line as second entry
        f.write(json.dumps({"tokens": [], "morpho_types": []}, ensure_ascii=False) + '\n')
        
        # Add another entry
        token_ids2 = sp.encode("yapay zeka", out_type=int)
        tokens2 = [sp.id_to_piece(tid) for tid in token_ids2]
        morpho_types2 = [0 if len(t) > 3 else 3 for t in tokens2]  # Simple heuristic
        f.write(json.dumps({"tokens": tokens2, "morpho_types": morpho_types2}, ensure_ascii=False) + '\n')

@pytest.fixture
def tokenizer_path():
    """Fixture to get tokenizer path"""
    path = os.path.join('tokenizer', 'morphopiece.model')
    if not os.path.exists(path):
        pytest.skip("morphopiece.model not found")
    return path

def test_jsonl_dataset_loading(tokenizer_path):
    """Test that JSONL dataset can be loaded"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
        jsonl_path = f.name
        create_test_jsonl(jsonl_path, tokenizer_path)
    
    try:
        sp = SentencePieceProcessor(model_file=tokenizer_path)
        ds = TurkishTextDataset(
            corpus_file=jsonl_path,
            tokenizer=sp,
            max_seq_len=32,
            is_jsonl=True
        )
        
        assert len(ds) > 0, "Dataset should have entries"
        
        # Get first item
        item = ds[0]
        
        assert 'input_ids' in item, "Should have input_ids"
        assert 'target_ids' in item, "Should have target_ids"
        assert 'attention_mask' in item, "Should have attention_mask"
        assert 'morpho_types' in item, "Should have morpho_types for JSONL format"
        
        assert isinstance(item['morpho_types'], torch.Tensor), "morpho_types should be tensor"
        assert item['morpho_types'].dtype == torch.long, "morpho_types should be long dtype"
        
    finally:
        os.unlink(jsonl_path)

def test_jsonl_morpho_types_shape(tokenizer_path):
    """Test that morpho_types tensor has correct shape"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
        jsonl_path = f.name
        create_test_jsonl(jsonl_path, tokenizer_path)
    
    try:
        sp = SentencePieceProcessor(model_file=tokenizer_path)
        ds = TurkishTextDataset(
            corpus_file=jsonl_path,
            tokenizer=sp,
            max_seq_len=32,
            is_jsonl=True
        )
        
        item = ds[0]
        input_ids = item['input_ids']
        morpho_types = item['morpho_types']
        
        # morpho_types should have same length as input_ids (minus 1 for shift)
        assert morpho_types.shape[0] == input_ids.shape[0], \
            f"morpho_types shape {morpho_types.shape} should match input_ids shape {input_ids.shape}"
        
    finally:
        os.unlink(jsonl_path)

def test_jsonl_morpho_types_values(tokenizer_path):
    """Test that morpho_types contain valid values (0-4)"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
        jsonl_path = f.name
        create_test_jsonl(jsonl_path, tokenizer_path)
    
    try:
        sp = SentencePieceProcessor(model_file=tokenizer_path)
        ds = TurkishTextDataset(
            corpus_file=jsonl_path,
            tokenizer=sp,
            max_seq_len=32,
            is_jsonl=True
        )
        
        item = ds[0]
        morpho_types = item['morpho_types']
        
        # All values should be between 0 and 4
        assert torch.all((morpho_types >= 0) & (morpho_types <= 4)), \
            "morpho_types should contain values between 0 and 4"
        
    finally:
        os.unlink(jsonl_path)

def test_jsonl_empty_entry(tokenizer_path):
    """Test that empty JSONL entries are handled correctly"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
        jsonl_path = f.name
        # Write empty entry
        f.write(json.dumps({"tokens": [], "morpho_types": []}, ensure_ascii=False) + '\n')
    
    try:
        sp = SentencePieceProcessor(model_file=tokenizer_path)
        ds = TurkishTextDataset(
            corpus_file=jsonl_path,
            tokenizer=sp,
            max_seq_len=32,
            is_jsonl=True
        )
        
        assert len(ds) == 1, "Should have one entry"
        
        item = ds[0]
        # Should still have valid tensors (all padding)
        assert 'input_ids' in item
        assert 'morpho_types' in item
        
    finally:
        os.unlink(jsonl_path)

def test_jsonl_vs_text_format(tokenizer_path):
    """Test that JSONL and text formats produce compatible outputs"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        text_path = f.name
        f.write("Evlerimdekiler gittim dün\n")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
        jsonl_path = f.name
        create_test_jsonl(jsonl_path, tokenizer_path)
    
    try:
        sp = SentencePieceProcessor(model_file=tokenizer_path)
        
        # Text format
        ds_text = TurkishTextDataset(
            corpus_file=text_path,
            tokenizer=sp,
            max_seq_len=32,
            is_jsonl=False
        )
        
        # JSONL format
        ds_jsonl = TurkishTextDataset(
            corpus_file=jsonl_path,
            tokenizer=sp,
            max_seq_len=32,
            is_jsonl=True
        )
        
        item_text = ds_text[0]
        item_jsonl = ds_jsonl[0]
        
        # Both should have same basic fields
        assert 'input_ids' in item_text
        assert 'input_ids' in item_jsonl
        assert 'target_ids' in item_text
        assert 'target_ids' in item_jsonl
        
        # JSONL should have morpho_types
        assert 'morpho_types' in item_jsonl
        # Text format should NOT have morpho_types (unless we add it later)
        # Currently it doesn't, which is expected
        
    finally:
        os.unlink(text_path)
        os.unlink(jsonl_path)

def test_jsonl_batch_compatibility(tokenizer_path):
    """Test that JSONL dataset works with DataLoader"""
    from torch.utils.data import DataLoader
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
        jsonl_path = f.name
        create_test_jsonl(jsonl_path, tokenizer_path)
    
    try:
        sp = SentencePieceProcessor(model_file=tokenizer_path)
        ds = TurkishTextDataset(
            corpus_file=jsonl_path,
            tokenizer=sp,
            max_seq_len=32,
            is_jsonl=True
        )
        
        dataloader = DataLoader(ds, batch_size=2, shuffle=False)
        
        # Get a batch
        batch = next(iter(dataloader))
        
        assert 'input_ids' in batch
        assert 'target_ids' in batch
        assert 'attention_mask' in batch
        assert 'morpho_types' in batch
        
        # Check shapes
        assert batch['input_ids'].shape[0] <= 2, "Batch size should be <= 2"
        assert batch['morpho_types'].shape == batch['input_ids'].shape, \
            "morpho_types should match input_ids shape"
        
    finally:
        os.unlink(jsonl_path)

