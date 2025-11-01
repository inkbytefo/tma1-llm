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
from typing import Optional
import random

class TurkishTextDataset(Dataset):
    """Dataset for Turkish text corpus"""
    
    def __init__(
        self,
        corpus_file: str,
        tokenizer: SentencePieceProcessor,
        max_seq_len: int = 512,
        buffer_size: int = 10000
    ):
        """
        Args:
            corpus_file: Path to corpus text file (one text per line)
            tokenizer: SentencePiece tokenizer instance
            max_seq_len: Maximum sequence length
            buffer_size: Number of lines to buffer in memory
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.buffer_size = buffer_size
        
        if not os.path.exists(corpus_file):
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
        
        # Count lines in file
        print(f"📊 Counting lines in corpus: {corpus_file}")
        with open(corpus_file, 'r', encoding='utf-8') as f:
            self.num_lines = sum(1 for _ in f)
        
        print(f"✅ Found {self.num_lines:,} lines")
        
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
                text = f.readline().strip()
        else:
            text = self._buffer[local_idx]
        
        # Tokenize
        # Note: TurkishTextDataset expects SentencePieceProcessor directly
        # For morpho-aware training, text should already be preprocessed
        # (morphemes separated by spaces from train_morphopiece.py)
        token_ids = self.tokenizer.encode(text, out_type=int)
        
        # Truncate if too long
        if len(token_ids) > self.max_seq_len:
            # Randomly select a window
            start = random.randint(0, len(token_ids) - self.max_seq_len)
            token_ids = token_ids[start:start + self.max_seq_len]
        
        # Create input and target (shifted by 1 for language modeling)
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]

        # Pad to max_seq_len
        padding_len = self.max_seq_len - len(input_ids) - 1
        if padding_len > 0:
            # Fallback: SentencePiece may have pad_id == -1 (disabled)
            pad_id = self.tokenizer.pad_id()
            if pad_id < 0:
                pad_id = self.tokenizer.unk_id()
            input_ids = input_ids + [pad_id] * padding_len
            target_ids = target_ids + [pad_id] * padding_len
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'attention_mask': torch.tensor(
                [1] * (len(input_ids) - padding_len) + [0] * padding_len,
                dtype=torch.long
            ) if padding_len > 0 else torch.ones(len(input_ids), dtype=torch.long)
        }

def create_dataloader(
    corpus_file: str,
    tokenizer_path: str,
    batch_size: int = 8,
    max_seq_len: int = 512,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """
    Create DataLoader for training
    
    Args:
        corpus_file: Path to corpus file
        tokenizer_path: Path to SentencePiece model file
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
    
    Returns:
        DataLoader instance
    """
    # Load tokenizer
    tokenizer = SentencePieceProcessor(model_file=tokenizer_path)
    
    # Create dataset
    dataset = TurkishTextDataset(
        corpus_file=corpus_file,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len
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

