// Developer: inkbytefo
// AI: Trae Coding Assistant
// Modified: 2025-11-01

# System Architecture

This document describes the end-to-end architecture of the TMA-1 stack, covering data acquisition, preprocessing, tokenization, model training, and inference.

## Overview

- Data Source: Hugging Face datasets (MC4, Wikipedia) via `src/data_collector.py`
- Preprocessing: Turkish morphology-aware splitting via `src/morpho_splitter.py`
- Tokenization: SentencePiece with morpheme-separated text via `src/morphopiece.py`
- Models:
  - Baseline `TransformerModel` in `src/model.py`
  - Morphology-aware `TMA1Model` in `src/tma1_model.py`
- Attention: `AgglutinativeAttention` in `src/agglutinative_attention.py`
- Grammar Bias: `GrammarEngine` in `src/grammar_engine.py`
- Dataset: `src/dataset.py` for streaming, encoding, padding, and target creation
- Training: `train.py` (baseline) and `train_tma1.py` (TMA-1)
- Inference: `llm_engine.py` loads checkpoints and generates text

## Data Flow Diagram

1) `data_collector.download_turkish_corpus()` → raw text corpus
2) `train_morphopiece.py --preprocess` → morpheme-separated text
3) `sentencepiece.Train()` → `morphopiece.model` + `morphopiece.vocab`
4) `dataset.create_dataloader()` → batches of `input_ids`, `targets`
5) `TransformerModel` or `TMA1Model` forward pass
6) Loss computation and optimizer step
7) Checkpointing and logging

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

- Enforces Turkish grammar constraints via logit biasing:
  - Vowel harmony masks
  - Forbidden suffix combinations
  - Suffix order validation
  - Sequence validation utilities

## Agglutinative Attention

- Uses morphological labels to alter attention patterns:
  - Root tokens get higher centrality
  - Verb tokens receive extra bias for SOV structure
  - Suffix tokens are treated with position-aware adjustments

## Training Loops

- `train.py`
  - Baseline LM training with AdamW, cosine LR, and TensorBoard
  - Checkpoints saved periodically; supports resume

- `train_tma1.py`
  - Integrates `MorphoPiece` and `GrammarEngine`
  - Tracks grammar violations alongside loss
  - Converts token IDs back to text for morphology validation

## Inference (`llm_engine.py`)

- Loads tokenizer and model from checkpoint (baseline or TMA-1)
- Provides interactive generation with configurable parameters
- Optionally applies grammar bias at inference

## Security & Reliability Notes

- No hardcoded secrets; external data via HF datasets
- Zemberek use is optional; regex fallback keeps CI portable
- Tests validate splitter, tokenizer, dataset, grammar engine, and engine behavior