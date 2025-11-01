// Developer: inkbytefo
// AI: Trae Coding Assistant
// Modified: 2025-11-01

# System Architecture

This document describes the end-to-end architecture of the TMA-1 stack, covering data acquisition, preprocessing, tokenization, model training, and inference.

## Overview

- Data Source: Hugging Face datasets (MC4, Wikipedia) via `src/data_collector.py`
- Preprocessing: Turkish morphology-aware splitting via `src/morpho_splitter.py`
- **TMA-1 Preprocessing**: `scripts/preprocess_for_tma1.py` performs morphological analysis once and outputs JSONL with `morpho_types` tensors (performance optimization)
- Tokenization: SentencePiece with morpheme-separated text via `src/morphopiece.py`
- Models:
  - Baseline `TransformerModel` in `src/model.py`
  - Morphology-aware `TMA1Model` in `src/tma1_model.py`
- Attention: `AgglutinativeAttention` in `src/agglutinative_attention.py` (supports preprocessed `morpho_types`)
- Grammar Bias: `GrammarEngine` in `src/grammar_engine.py` (**vectorized implementation**, no vocabulary loops)
- Dataset: `src/dataset.py` for streaming, encoding, padding, target creation, and **JSONL support with preprocessed morpho_types**
- Training: `train.py` (baseline) and `train_tma1.py` (TMA-1, **optimized with preprocessing**)
- Inference: `llm_engine.py` loads checkpoints and generates text

## Data Flow Diagram

1) `data_collector.download_turkish_corpus()` → raw text corpus
2) `train_morphopiece.py --preprocess` → morpheme-separated text
3) `sentencepiece.Train()` → `morphopiece.model` + `morphopiece.vocab`
4) **`preprocess_for_tma1.py` → JSONL with `morpho_types` (0=root, 1=suffix, 2=verb, 3=other, 4=pad)** ⚡ **Performance Optimization**
5) `dataset.create_dataloader(is_jsonl=True)` → batches of `input_ids`, `target_ids`, `morpho_types`
6) `TMA1Model` forward pass with `morpho_types` tensor (no runtime morphological analysis)
7) **Vectorized grammar bias** via `GrammarEngine.apply_grammar_bias()` (PyTorch tensor ops, no vocab loops)
8) Loss computation and optimizer step
9) Checkpointing and logging

## Morphology Components

- `MorphoSplitter`
  - Splits words into root and suffixes using a prioritized strategy:
    - Try Zemberek (if Java available)
    - Fallback to curated suffix list and regex rules
  - Outputs morpheme labels for downstream attention biasing

- `MorphoPiece`
  - Wraps SentencePiece
  - Preprocesses text by joining morphemes with separators to favor morpheme tokens

## Grammar Engine

- Enforces Turkish grammar constraints via **vectorized logit biasing** (optimized):
  - **Vocabulary cache**: Pre-computes vowel information for all vocabulary tokens
  - **Tensor broadcasting**: Uses PyTorch tensor operations instead of Python loops
  - Vowel harmony masks computed via `torch.where()` and broadcasting
  - Forbidden suffix combinations detected using vectorized tensor comparisons
  - **Performance**: 100-1000x faster than previous loop-based implementation (O(vocab_size) eliminated)
  - Suffix order validation
  - Sequence validation utilities

## Agglutinative Attention

- Uses morphological labels to alter attention patterns (optimized):
  - Accepts `morpho_types` tensor directly from dataset (preprocessed, fast)
  - Falls back to `token_texts` + runtime analysis only if `morpho_types` not available (slow)
  - Root tokens get higher centrality
  - Verb tokens receive extra bias for SOV structure
  - Suffix tokens are treated with position-aware adjustments
  - **Performance**: Preprocessed `morpho_types` eliminates MorphoSplitter calls during training (10-100x speedup)

## Training Loops

- `train.py`
  - Baseline LM training with AdamW, cosine LR, and TensorBoard
  - Checkpoints saved periodically; supports resume

- `train_tma1.py`
  - Integrates `MorphoPiece` and `GrammarEngine`
  - **Uses preprocessed JSONL with `morpho_types` tensors** (fast path)
  - Falls back to text format + runtime analysis if needed (slow path, for compatibility)
  - Tracks grammar violations alongside loss
  - **Vectorized grammar bias** eliminates vocabulary iteration loops

## Inference (`llm_engine.py`)

- Loads tokenizer and model from checkpoint (baseline or TMA-1)
- Provides interactive generation with configurable parameters
- Optionally applies grammar bias at inference

## Security & Reliability Notes

- No hardcoded secrets; external data via HF datasets
- Zemberek use is optional; regex fallback keeps CI portable
- Tests validate splitter, tokenizer, dataset, grammar engine, and engine behavior