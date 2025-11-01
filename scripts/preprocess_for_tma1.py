#!/usr/bin/env python3
# Developer: inkbytefo
# AI: Claude Sonnet 4.5
# Modified: 2025-11-01

"""
============================================================================
TMA-1 Preprocessing Script - Morfolojik Analizi Ön İşleme
Ham corpus'u okuyup her token için morfo-tip bilgisi ekler
============================================================================
"""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
from sentencepiece import SentencePieceProcessor
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.morpho_splitter import MorphoSplitter

def get_morpho_type(token: str, morpho_splitter: MorphoSplitter) -> int:
    """
    Token'ın morfolojik tipini belirle
    
    Args:
        token: Token string
        morpho_splitter: MorphoSplitter instance
    
    Returns:
        Morfolojik tip (0=root, 1=suffix, 2=verb, 3=other, 4=pad)
    """
    # Özel token'lar için
    if not token or token.strip() == "":
        return 4  # pad
    if token.startswith("<") and token.endswith(">"):
        return 3  # special token (other)
    
    # Morfolojik analiz yap
    analysis = morpho_splitter.split_word(token.strip())
    
    # Yüklem kontrolü (zaman eki varsa)
    if any('zaman' in morf.get('tür', '').lower() for morf in analysis['morfemler']):
        return 2  # Verb
    # Kök kontrolü (ek yoksa ve kelime kök olarak kalıyorsa)
    elif not analysis['ekler'] or analysis['kök'] == token.strip():
        return 0  # Root
    # Ek var ama yüklem değil
    elif analysis['ekler']:
        return 1  # Suffix
    else:
        return 3  # Other

def preprocess_corpus(
    input_file: str,
    output_file: str,
    tokenizer_path: str,
    morpho_splitter: MorphoSplitter,
    max_lines: int = None
) -> bool:
    """
    Corpus'u ön işleme tabi tutup JSONL formatında kaydet
    
    Args:
        input_file: Ham corpus dosyası
        output_file: Çıktı JSONL dosyası
        tokenizer_path: SentencePiece tokenizer model path
        morpho_splitter: MorphoSplitter instance
        max_lines: Maksimum işlenecek satır sayısı (None = hepsi)
    
    Returns:
        Başarılı ise True
    """
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return False
    
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        print(f"💡 Train tokenizer first: python src/train_morphopiece.py --train")
        return False
    
    # Load tokenizer
    print(f"\n📥 Loading tokenizer: {tokenizer_path}")
    tokenizer = SentencePieceProcessor(model_file=tokenizer_path)
    print(f"✅ Tokenizer loaded (vocab size: {tokenizer.vocab_size():,})")
    
    # Count lines in input file
    print(f"\n📊 Counting lines in corpus: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    if max_lines:
        total_lines = min(total_lines, max_lines)
    
    print(f"✅ Found {total_lines:,} lines to process")
    
    # Process corpus
    print(f"\n🔄 Processing corpus...")
    print(f"   Input: {input_file}")
    print(f"   Output: {output_file}")
    
    processed_count = 0
    error_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        progress_bar = tqdm(total=total_lines, desc="Processing", unit="lines")
        
        for line_idx, line in enumerate(f_in):
            if max_lines and line_idx >= max_lines:
                break
            
            line = line.strip()
            if not line:
                # Boş satır için boş output
                output = {
                    "tokens": [],
                    "morpho_types": []
                }
                f_out.write(json.dumps(output, ensure_ascii=False) + '\n')
                processed_count += 1
                progress_bar.update(1)
                continue
            
            try:
                # Tokenize
                token_ids = tokenizer.encode(line, out_type=int)
                
                # Decode token IDs to get token strings
                tokens = []
                for token_id in token_ids:
                    token = tokenizer.id_to_piece(token_id)
                    # Remove SentencePiece prefix (▁ for space)
                    if token.startswith('▁'):
                        token = token[1:]
                    if not token:
                        token = '<SPACE>'
                    tokens.append(token)
                
                # Get morphological types for each token
                morpho_types = []
                for token in tokens:
                    morpho_type = get_morpho_type(token, morpho_splitter)
                    morpho_types.append(morpho_type)
                
                # Write JSONL line
                output = {
                    "tokens": tokens,
                    "morpho_types": morpho_types
                }
                f_out.write(json.dumps(output, ensure_ascii=False) + '\n')
                processed_count += 1
                
            except Exception as e:
                error_count += 1
                print(f"\n⚠️  Error processing line {line_idx}: {e}")
                # Error durumunda boş output yaz
                output = {
                    "tokens": [],
                    "morpho_types": []
                }
                f_out.write(json.dumps(output, ensure_ascii=False) + '\n')
            
            progress_bar.update(1)
        
        progress_bar.close()
    
    print(f"\n✅ Preprocessing completed!")
    print(f"   Processed: {processed_count:,} lines")
    print(f"   Errors: {error_count:,} lines")
    print(f"   Output: {output_file}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Preprocess corpus for TMA-1 training')
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input corpus file (raw text, one line per sentence)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSONL file (preprocessed with morpho types)')
    parser.add_argument('--tokenizer', type=str, required=True,
                       help='Path to SentencePiece tokenizer model (.model file)')
    parser.add_argument('--max-lines', type=int, default=None,
                       help='Maximum number of lines to process (default: all)')
    parser.add_argument('--use-java', action='store_true', default=False,
                       help='Use Zemberek Java (if available) for morphological analysis')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize MorphoSplitter
    print(f"\n📝 Initializing MorphoSplitter...")
    morpho_splitter = MorphoSplitter(use_java=args.use_java)
    
    # Preprocess corpus
    success = preprocess_corpus(
        input_file=args.input,
        output_file=args.output,
        tokenizer_path=args.tokenizer,
        morpho_splitter=morpho_splitter,
        max_lines=args.max_lines
    )
    
    if success:
        print(f"\n✅ Preprocessing successful!")
        print(f"💡 Next step: Train TMA-1 model with:")
        print(f"   python train_tma1.py --corpus {args.output} --tokenizer {args.tokenizer}")
    else:
        print(f"\n❌ Preprocessing failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

