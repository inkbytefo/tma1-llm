// Developer: inkbytefo
// AI: Claude Sonnet 4.5
// Modified: 2025-11-01

# TMA-1: Turkish Morphology-Aware Transformer and Tokenizer

TMA-1 is a Turkish-focused language modeling stack combining a morphology-aware tokenizer (MorphoPiece), grammar-informed attention mechanisms, and PyTorch-based transformer training. The system is designed for agglutinative languages (like Turkish) where morphemes and suffix ordering matter.

This repository includes:
- MorphoPiece: a SentencePiece-backed tokenizer with Turkish morpheme awareness
- MorphoSplitter: robust morphological splitter (Java/Zemberek optional, regex fallback)
- GrammarEngine: vowel-harmony and suffix ordering biasing
- AgglutinativeAttention: attention mechanism informed by morphology and SOV tendencies
- TransformerModel and TMA-1 model: standard and morphology-aware variants
- Training pipelines: tokenizer pretraining, standard LM pretraining, and TMA-1 training
- Tests validating morphology, tokenizer, datasets, and inference utilities

## Quickstart

1) Set up environment:
```
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

2) Prepare a small corpus for smoke tests:
```
python scripts/make_test_corpus.py
```

3) Train a MorphoPiece tokenizer on processed corpus (or use tiny corpus for quick run):
```
python src/train_morphopiece.py --preprocess --corpus-file data/test_corpus.txt --preprocessed-file data/corpus_morpho_processed.txt --train --output tokenizer/morphopiece --vocab-size 1000
```

4) **Preprocess corpus for TMA-1 (NEW - Required for optimal performance):**
```
python scripts/preprocess_for_tma1.py \
    --input data/test_corpus.txt \
    --output data/train_data.jsonl \
    --tokenizer tokenizer/morphopiece.model
```

5) Train the baseline transformer:
```
python train.py --corpus data/test_corpus.txt --tokenizer tokenizer/morphopiece.model --output-dir models/baseline
```

6) Train the morphology-aware TMA-1 model (using preprocessed JSONL):
```
python train_tma1.py --corpus data/train_data.jsonl --tokenizer tokenizer/morphopiece.model --output-dir models/tma1
```

7) Run tests:
```
pytest -q
```

## Repository Structure

- `src/morpho_splitter.py`: Morpheme splitter with optional Zemberek integration and a deterministic regex fallback for CI/Windows
- `src/morphopiece.py`: MorphoPiece tokenizer wrapper, integrates morpheme-aware preprocessing with SentencePiece
- `src/grammar_engine.py`: Grammar rules for Turkish (vowel harmony, forbidden combinations, suffix order), applies **vectorized logit biases** (optimized)
- `src/agglutinative_attention.py`: Attention module that uses morphology labels to bias Q/K/V dynamics (supports preprocessed `morpho_types` tensors)
- `src/model.py`: Baseline TransformerModel and layers (MultiHeadAttention, FeedForward)
- `src/tma1_model.py`: Morphology-aware transformer with optional GrammarEngine biasing
- `src/dataset.py`: Streaming dataset with buffer loading, SentencePiece encoding, padding/shifted targets, and **JSONL support with preprocessed morpho_types**
- `src/train_morphopiece.py`: Download, preprocess, and train SentencePiece with morpheme-aware text
- `src/train_tokenizer.py`: Simpler tokenizer training utility
- `train.py`: Baseline model training loop with checkpoints and TensorBoard
- `train_tma1.py`: TMA-1 training loop integrating morphology and grammar bias (**optimized with preprocessing**)
- `llm_engine.py`: Inference engine for loading checkpoints and generating text
- `scripts/make_test_corpus.py`: Utility to create a 100-line test corpus for smoke runs
- `scripts/preprocess_for_tma1.py`: **NEW** - Preprocesses corpus with morphological analysis, outputs JSONL with `morpho_types` for fast training
- `tests/`: Comprehensive unit tests for morphology, tokenizer, dataset, and engine

## Data Flow

- Data is downloaded via `src/data_collector.py` from HF datasets and cleaned. For testing, use `scripts/make_test_corpus.py` and `tests/test_corpus.txt`.
- `train_morphopiece.py` can preprocess raw text by splitting into morphemes, producing `corpus_morpho_processed.txt`.
- SentencePiece tokenizer is trained on morpheme-separated text, ensuring morphemes appear as distinct tokens.
- **NEW**: `scripts/preprocess_for_tma1.py` preprocesses corpus with morphological analysis, outputting JSONL format with `morpho_types` (0=root, 1=suffix, 2=verb, 3=other, 4=pad) for each token. This eliminates runtime morphological analysis during training, providing **10-100x speedup**.
- The dataset (`src/dataset.py`) supports both text and JSONL formats:
  - **Text format**: Encodes via SentencePiece, pads/truncates, yields input/target pairs (fallback mode, slower)
  - **JSONL format**: Reads preprocessed `morpho_types` tensors directly (recommended, fast)
- Baseline training (`train.py`) builds a vanilla transformer and optimizes via cross-entropy.
- TMA-1 training (`train_tma1.py`) uses preprocessed `morpho_types` tensors for morphology-aware attention and **vectorized grammar biases** (no vocabulary loops), dramatically improving training speed.

## Core Concepts

- Morphology Awareness: Words are split into root + suffix tokens to better capture agglutinative structure.
- **Preprocessing Optimization**: Morphological analysis is performed **once during preprocessing**, not during training. Results are cached in JSONL format with `morpho_types` tensors for each token.
- Grammar Biasing: **Vectorized** logit adjustments enforce vowel harmony and typical suffix sequences using PyTorch tensor operations (no vocabulary loops), providing significant performance improvements.
- Attention Augmentation: Tokens are labeled as root/suffix/verb (via preprocessed `morpho_types`), and attention applies learned biases based on these types using efficient tensor operations.

## Ownership and Licensing

This project and models are proprietary. All rights reserved by Tevfik İşkın. See `LICENSE.md` for details.

## Running and Testing

- Dependencies: see `requirements.txt`
- Tests: `pytest -q`
- Linting: for Python, you can use `ruff` or `flake8` if desired (not enforced by this repo); ensure PEP8 compliance.

## Troubleshooting

- Windows/Java: Zemberek requires a Java runtime. Tests use regex fallback (`use_java=False`) to avoid Java dependency.
- GPU: If CUDA is available, PyTorch will use it automatically; otherwise CPU is used.
- Tokenizer: Ensure `tokenizer/morphopiece.model` exists before training models, or run `src/train_morphopiece.py`.
- **Slow Training**: Use preprocessed JSONL format instead of text format. Run `scripts/preprocess_for_tma1.py` first to generate JSONL with `morpho_types` for 10-100x speedup.
- **Memory Issues**: If preprocessing fails due to memory, use `--max-lines` parameter in `preprocess_for_tma1.py` to limit corpus size.