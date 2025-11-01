#!/usr/bin/env python3
# Developer: inkbytefo
# AI: Claude Sonnet 4.5
# Modified: 2025-11-01

"""
============================================================================
Dataset and DataLoader - Turkish Text Corpus
PyTorch Dataset for loading and batching Turkish text data
============================================================================
"""

import torch
from torch.utils.data import Dataset, DataLoader
from sentencepiece import SentencePieceProcessor
import os
import json
from typing import Optional
import random

class TurkishTextDataset(Dataset):
    """Dataset for Turkish text corpus with optional morphological preprocessing"""
    
    def __init__(
        self,
        corpus_file: str,
        tokenizer: SentencePieceProcessor,
        max_seq_len: int = 512,
        buffer_size: int = 10000,
        is_jsonl: bool = False
    ):
        """
        Args:
            corpus_file: Path to corpus file (text or JSONL)
            tokenizer: SentencePiece tokenizer instance
            max_seq_len: Maximum sequence length
            buffer_size: Number of lines to buffer in memory
            is_jsonl: If True, corpus_file is JSONL with preprocessed morpho_types
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.buffer_size = buffer_size
        self.is_jsonl = is_jsonl
        
        if not os.path.exists(corpus_file):
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
        
        # Count lines in file
        print(f"📊 Counting lines in corpus: {corpus_file}")
        with open(corpus_file, 'r', encoding='utf-8') as f:
            self.num_lines = sum(1 for _ in f)
        
        print(f"✅ Found {self.num_lines:,} lines")
        if is_jsonl:
            print(f"   Format: JSONL (with preprocessed morpho_types)")
        else:
            print(f"   Format: Text (morpho_types will be computed at runtime)")
        
        self.corpus_file = corpus_file
        self._buffer = []
        self._buffer_start_idx = 0
    
    def _load_buffer(self, start_idx: int):
        """Load buffer of lines from file"""
        self._buffer = []
        self._buffer_start_idx = start_idx
        
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            # Skip to start position
            for _ in range(start_idx):
                try:
                    next(f)
                except StopIteration:
                    break
            
            # Load buffer
            for _ in range(self.buffer_size):
                line = f.readline()
                if not line:
                    break
                if self.is_jsonl:
                    # JSONL format: parse JSON
                    try:
                        data = json.loads(line.strip())
                        self._buffer.append(data)
                    except json.JSONDecodeError:
                        self._buffer.append({"tokens": [], "morpho_types": []})
                else:
                    # Text format: just store line
                    self._buffer.append(line.strip())
    
    def __len__(self) -> int:
        return self.num_lines
    
    def __getitem__(self, idx: int) -> dict:
        """Get a single training example"""
        # Check if idx is in current buffer
        if idx < self._buffer_start_idx or idx >= self._buffer_start_idx + len(self._buffer):
            # Load new buffer
            buffer_start = max(0, idx - self.buffer_size // 2)
            self._load_buffer(buffer_start)
        
        # Get line from buffer
        local_idx = idx - self._buffer_start_idx
        if local_idx >= len(self._buffer):
            # Fallback: read directly
            with open(self.corpus_file, 'r', encoding='utf-8') as f:
                for _ in range(idx):
                    f.readline()
                line = f.readline().strip()
                if self.is_jsonl:
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        data = {"tokens": [], "morpho_types": []}
                else:
                    data = line
        else:
            data = self._buffer[local_idx]
        
        # Process data based on format
        if self.is_jsonl:
            # JSONL format: data already has tokens and morpho_types
            tokens = data.get("tokens", [])
            morpho_types_list = data.get("morpho_types", [])
            
            # Encode tokens to IDs
            token_ids = []
            for token in tokens:
                # Encode single token (SentencePiece handles tokenization)
                # If token is a SentencePiece piece, encode it directly
                try:
                    # Try to encode as single token
                    ids = self.tokenizer.encode(token, out_type=int)
                    if ids:
                        token_ids.extend(ids)
                        # If token maps to multiple IDs, extend morpho_types accordingly
                        if len(ids) > 1:
                            morpho_types_list.extend([morpho_types_list[-1] if morpho_types_list else 3] * (len(ids) - 1))
                    else:
                        # Fallback: use UNK
                        token_ids.append(self.tokenizer.unk_id())
                        morpho_types_list.append(3)  # Other
                except:
                    # Fallback: use UNK
                    token_ids.append(self.tokenizer.unk_id())
                    morpho_types_list.append(3)  # Other
            
            # Ensure morpho_types_list matches token_ids length
            while len(morpho_types_list) < len(token_ids):
                morpho_types_list.append(3)  # Other
            morpho_types_list = morpho_types_list[:len(token_ids)]
        else:
            # Text format: tokenize normally
            text = data if isinstance(data, str) else ""
            token_ids = self.tokenizer.encode(text, out_type=int)
            morpho_types_list = [3] * len(token_ids)  # Default: other (will be ignored)
        
        # Truncate if too long
        if len(token_ids) > self.max_seq_len:
            # Randomly select a window
            start = random.randint(0, len(token_ids) - self.max_seq_len)
            token_ids = token_ids[start:start + self.max_seq_len]
            morpho_types_list = morpho_types_list[start:start + self.max_seq_len]
        
        # Create input and target (shifted by 1 for language modeling)
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]
        input_morpho_types = morpho_types_list[:-1]
        target_morpho_types = morpho_types_list[1:]

        # Pad to max_seq_len
        pad_id = self.tokenizer.pad_id()
        if pad_id < 0:
            pad_id = self.tokenizer.unk_id()
        pad_morpho_type = 4  # pad type
        
        padding_len = self.max_seq_len - len(input_ids) - 1
        if padding_len > 0:
            input_ids = input_ids + [pad_id] * padding_len
            target_ids = target_ids + [pad_id] * padding_len
            input_morpho_types = input_morpho_types + [pad_morpho_type] * padding_len
            target_morpho_types = target_morpho_types + [pad_morpho_type] * padding_len
        
        result = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'attention_mask': torch.tensor(
                [1] * (len(input_ids) - padding_len) + [0] * padding_len,
                dtype=torch.long
            ) if padding_len > 0 else torch.ones(len(input_ids), dtype=torch.long)
        }
        
        # Add morpho_types if available
        if self.is_jsonl:
            result['morpho_types'] = torch.tensor(input_morpho_types, dtype=torch.long)
        
        return result

def create_dataloader(
    corpus_file: str,
    tokenizer_path: str,
    batch_size: int = 8,
    max_seq_len: int = 512,
    num_workers: int = 4,
    shuffle: bool = True,
    is_jsonl: bool = False
) -> DataLoader:
    """
    Create DataLoader for training
    
    Args:
        corpus_file: Path to corpus file (text or JSONL)
        tokenizer_path: Path to SentencePiece model file
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        is_jsonl: If True, corpus_file is JSONL with preprocessed morpho_types
    
    Returns:
        DataLoader instance
    """
    # Load tokenizer
    tokenizer = SentencePieceProcessor(model_file=tokenizer_path)
    
    # Create dataset
    dataset = TurkishTextDataset(
        corpus_file=corpus_file,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        is_jsonl=is_jsonl
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True  # Drop last incomplete batch
    )
    
    return dataloader, tokenizer

