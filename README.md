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

4) Train the baseline transformer:
```
python train.py --corpus data/test_corpus.txt --tokenizer tokenizer/morphopiece.model --output-dir models/baseline
```

5) Train the morphology-aware TMA-1 model:
```
python train_tma1.py --corpus data/test_corpus.txt --tokenizer tokenizer/morphopiece.model --output-dir models/tma1
```

6) Run tests:
```
pytest -q
```

## Repository Structure

- `src/morpho_splitter.py`: Morpheme splitter with optional Zemberek integration and a deterministic regex fallback for CI/Windows
- `src/morphopiece.py`: MorphoPiece tokenizer wrapper, integrates morpheme-aware preprocessing with SentencePiece
- `src/grammar_engine.py`: Grammar rules for Turkish (vowel harmony, forbidden combinations, suffix order), applies logit biases
- `src/agglutinative_attention.py`: Attention module that uses morphology labels to bias Q/K/V dynamics
- `src/model.py`: Baseline TransformerModel and layers (MultiHeadAttention, FeedForward)
- `src/tma1_model.py`: Morphology-aware transformer with optional GrammarEngine biasing
- `src/dataset.py`: Streaming dataset with buffer loading, SentencePiece encoding, and padding/shifted targets
- `src/train_morphopiece.py`: Download, preprocess, and train SentencePiece with morpheme-aware text
- `src/train_tokenizer.py`: Simpler tokenizer training utility
- `train.py`: Baseline model training loop with checkpoints and TensorBoard
- `train_tma1.py`: TMA-1 training loop integrating morphology and grammar bias
- `llm_engine.py`: Inference engine for loading checkpoints and generating text
- `scripts/make_test_corpus.py`: Utility to create a 100-line test corpus for smoke runs
- `tests/`: Comprehensive unit tests for morphology, tokenizer, dataset, and engine

## Data Flow

- Data is downloaded via `src/data_collector.py` from HF datasets and cleaned. For testing, use `scripts/make_test_corpus.py` and `tests/test_corpus.txt`.
- `train_morphopiece.py` can preprocess raw text by splitting into morphemes, producing `corpus_morpho_processed.txt`.
- SentencePiece tokenizer is trained on morpheme-separated text, ensuring morphemes appear as distinct tokens.
- The dataset (`src/dataset.py`) encodes text via SentencePiece, pads/truncates, and yields input/target pairs for language modeling.
- Baseline training (`train.py`) builds a vanilla transformer and optimizes via cross-entropy.
- TMA-1 training (`train_tma1.py`) adds morphology-aware attention and grammar biases to improve Turkish-specific generation.

## Core Concepts

- Morphology Awareness: Words are split into root + suffix tokens to better capture agglutinative structure.
- Grammar Biasing: Logit adjustments enforce vowel harmony and typical suffix sequences, discouraging invalid token transitions.
- Attention Augmentation: Tokens are labeled as root/suffix/verb, and attention applies learned biases based on these types.

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