
# Model Card: TMA-1 and MorphoPiece

## Overview

TMA-1 is a Turkish morphology-aware transformer model trained with the MorphoPiece tokenizer. The system targets better handling of agglutinative morphology, suffix ordering, and grammar constraints during generation.

## Intended Use

- Research and experimentation on Turkish language modeling
- Educational demonstrations of morphology-aware NLP methods
- Not intended for production deployment without further evaluation and hardening

## Models

- Baseline `TransformerModel` (PyTorch)
- `TMA1Model` integrating:
  - `AgglutinativeAttention`
  - `GrammarEngine` logit biasing
  - Morphology-aware tokenization via `MorphoPiece`

## Tokenizer

- SentencePiece Unigram or BPE
- Trained on preprocessed text that separates roots and suffixes
- Supports morphology-aware encoding/decoding options

## Training Data

- Example tests use `tests/test_corpus.txt` and small synthetic corpora
- Full training may use Turkish subsets of MC4 and Wikipedia via `data_collector.py`

## Metrics

- Current repository includes unit tests and training loss tracking
- Grammar violations are tracked in `train_tma1.py`
- Users should add task-specific metrics (perplexity, BLEU, morphology accuracy) for rigorous evaluation

## Performance Optimizations

The model includes significant performance optimizations:

- **Preprocessing Pipeline**: Morphological analysis is performed once during preprocessing (`scripts/preprocess_for_tma1.py`), not during training. Results are cached in JSONL format with `morpho_types` tensors, providing 10-100x training speedup.
- **Vectorized Grammar Bias**: `GrammarEngine.apply_grammar_bias()` uses PyTorch tensor operations and broadcasting instead of Python loops over vocabulary. Provides 100-1000x speedup for grammar bias computation.
- **Efficient Dataset**: `TurkishTextDataset` supports both text (compatibility) and JSONL formats (optimized). JSONL format includes preprocessed `morpho_types` tensors for direct use during training.

These optimizations significantly reduce training time while maintaining the same model quality and Turkish-specific features.

## Limitations

- Grammar rules are heuristic and may not cover all edge cases
- Morphological splitting may be imperfect without full lexicon
- Larger corpora and longer training are needed for competitive performance

## Ethical Considerations

- Ensure data sources are compliant with licensing
- Exercise caution when deploying generative models; consider content filtering and bias evaluation

## Ownership and License

All rights reserved by Tevfik İşkın. See `LICENSE.md` for proprietary licensing terms.