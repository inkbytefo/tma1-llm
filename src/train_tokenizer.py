#!/usr/bin/env python3
# Developer: inkbytefo
# AI: QuantumSoul-Engineer-v1
# Modified: 2024-12-19

"""
============================================================================
Tokenizer Training - SentencePiece for Turkish
Trains a specialized tokenizer for Turkish language
============================================================================
"""

import sentencepiece as spm
import argparse
import os
from pathlib import Path

def train_sentencepiece_tokenizer(
    input_file: str,
    model_prefix: str = "tokenizer/tr_tokenizer",
    vocab_size: int = 32000,
    model_type: str = "unigram",
    character_coverage: float = 0.9995
):
    """
    Train SentencePiece tokenizer on Turkish corpus
    
    Args:
        input_file: Path to training corpus
        model_prefix: Output file prefix (will create .model and .vocab)
        vocab_size: Vocabulary size
        model_type: "unigram", "bpe", "char", or "word"
        character_coverage: Character coverage (0.9995 for languages with large vocab)
    """
    print(f"🔤 Training SentencePiece tokenizer for Turkish...")
    print(f"   Input: {input_file}")
    print(f"   Output prefix: {model_prefix}")
    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Model type: {model_type}")
    print("")
    
    # Create output directory
    os.makedirs(os.path.dirname(model_prefix) if os.path.dirname(model_prefix) else ".", exist_ok=True)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        print(f"💡 Run 'src/data_collector.py' first to create corpus")
        return False
    
    try:
        # SentencePiece training parameters
        training_args = {
            'input': input_file,
            'model_prefix': model_prefix,
            'vocab_size': vocab_size,
            'model_type': model_type,
            'character_coverage': character_coverage,
            'input_sentence_size': 1000000,  # Limit for faster training
            'shuffle_input_sentence': True,
            'seed_sentencepiece_size': 1000000,
            'shrinking_factor': 0.75,
            'num_threads': 4,
            'max_sentence_length': 4192,
            'split_by_unicode_script': True,
            'split_by_whitespace': True,
            'split_by_number': True,
            'normalization_rule_name': 'nmt_nfkc_cf',  # Normalization
            'add_dummy_prefix': True,
            'remove_extra_whitespaces': True,
            'hard_vocab_limit': False,
            'use_all_vocab': False,
            'byte_fallback': True,  # Handle OOV
            'vocab_output_piece_score': True,
            'train_extremely_large_corpus': False,
        }
        
        print("⏳ Training tokenizer (this may take several minutes)...")
        print("")
        
        # Train tokenizer
        spm.SentencePieceTrainer.train(**training_args)
        
        # Verify output files
        model_file = f"{model_prefix}.model"
        vocab_file = f"{model_prefix}.vocab"
        
        if os.path.exists(model_file) and os.path.exists(vocab_file):
            model_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
            vocab_size_actual = sum(1 for _ in open(vocab_file, 'r', encoding='utf-8'))
            
            print("")
            print(f"✅ Tokenizer training completed!")
            print(f"   Model file: {model_file} ({model_size:.2f} MB)")
            print(f"   Vocab file: {vocab_file} ({vocab_size_actual:,} tokens)")
            print("")
            print("💡 You can now use this tokenizer in training:")
            print(f"   from sentencepiece import SentencePieceProcessor")
            print(f"   sp = SentencePieceProcessor(model_file='{model_file}')")
            
            return True
        else:
            print(f"❌ Output files not created properly")
            return False
    
    except Exception as e:
        print(f"❌ Error training tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tokenizer(model_file: str, test_texts: list = None):
    """Test the trained tokenizer"""
    print(f"\n🧪 Testing tokenizer: {model_file}")
    
    if not os.path.exists(model_file):
        print(f"❌ Model file not found: {model_file}")
        return
    
    sp = spm.SentencePieceProcessor(model_file=model_file)
    
    if test_texts is None:
        test_texts = [
            "Merhaba dünya! Bu bir test cümlesi.",
            "Türkçe, eklemeli bir dil yapısına sahiptir.",
            "Yapay zeka ve makine öğrenmesi geleceğin teknolojileridir."
        ]
    
    print("\n" + "=" * 60)
    for text in test_texts:
        tokens = sp.encode(text, out_type=str)
        token_ids = sp.encode(text, out_type=int)
        decoded = sp.decode(token_ids)
        
        print(f"Text:     {text}")
        print(f"Tokens:   {tokens[:20]}..." if len(tokens) > 20 else f"Tokens:   {tokens}")
        print(f"IDs:      {token_ids[:20]}..." if len(token_ids) > 20 else f"IDs:      {token_ids}")
        print(f"Decoded:  {decoded}")
        print("-" * 60)

def main():
    parser = argparse.ArgumentParser(description='Train SentencePiece tokenizer for Turkish')
    parser.add_argument('--input', type=str, default='data/corpus.txt',
                       help='Input corpus file')
    parser.add_argument('--output', type=str, default='tokenizer/tr_tokenizer',
                       help='Output model prefix')
    parser.add_argument('--vocab-size', type=int, default=32000,
                       help='Vocabulary size')
    parser.add_argument('--model-type', type=str, default='unigram',
                       choices=['unigram', 'bpe', 'char', 'word'],
                       help='SentencePiece model type')
    parser.add_argument('--test', action='store_true',
                       help='Test the tokenizer after training')
    
    args = parser.parse_args()
    
    # Train tokenizer
    success = train_sentencepiece_tokenizer(
        input_file=args.input,
        model_prefix=args.output,
        vocab_size=args.vocab_size,
        model_type=args.model_type
    )
    
    # Test if requested
    if success and args.test:
        test_tokenizer(f"{args.output}.model")

if __name__ == "__main__":
    main()

