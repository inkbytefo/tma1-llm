#!/usr/bin/env python3
# Developer: inkbytefo
# AI: QuantumSoul-Engineer-v1
# Modified: 2024-12-19

"""
============================================================================
Data Collector - Turkish Corpus Preparation
Downloads and processes Turkish text data from Hugging Face datasets
============================================================================
"""

import os
from datasets import load_dataset
from tqdm import tqdm
import argparse

def download_turkish_corpus(output_path: str = "data/corpus.txt", max_size_gb: float = 1.0):
    """
    Download Turkish text from MC4 dataset and save to corpus file
    
    Args:
        output_path: Output file path for corpus
        max_size_gb: Maximum corpus size in GB
    """
    print("📥 Downloading Turkish corpus from MC4 dataset...")
    print(f"   Target size: {max_size_gb} GB")
    print("")
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    try:
        # Load MC4 Turkish dataset
        print("🔍 Loading MC4 Turkish dataset...")
        dataset = load_dataset("mc4", "tr", streaming=True)
        
        # Calculate target size in bytes
        max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        
        total_size = 0
        text_count = 0
        min_text_length = 100  # Minimum text length to include
        
        print("\n📝 Processing texts...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in tqdm(dataset['train'], desc="Processing"):
                text = item.get('text', '')
                
                # Filter short texts
                if len(text) < min_text_length:
                    continue
                
                # Clean text (basic)
                text = text.strip()
                text = ' '.join(text.split())  # Normalize whitespace
                
                if not text:
                    continue
                
                # Write text
                f.write(text + '\n')
                total_size += len(text.encode('utf-8'))
                text_count += 1
                
                # Check size limit
                if total_size >= max_size_bytes:
                    break
        
        # Calculate final size
        final_size_gb = total_size / (1024 * 1024 * 1024)
        
        print(f"\n✅ Corpus created successfully!")
        print(f"   File: {output_path}")
        print(f"   Size: {final_size_gb:.2f} GB")
        print(f"   Texts: {text_count:,}")
        
        return output_path
    
    except Exception as e:
        print(f"\n❌ Error downloading corpus: {e}")
        print("\n💡 Alternative: You can manually download Turkish texts and save to data/corpus.txt")
        return None

def clean_corpus(input_path: str, output_path: str):
    """
    Clean and normalize corpus text
    
    Args:
        input_path: Input corpus file
        output_path: Cleaned output file
    """
    print(f"🧹 Cleaning corpus: {input_path}")
    
    import re
    
    with open(input_path, 'r', encoding='utf-8') as f_in:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for line in tqdm(f_in, desc="Cleaning"):
                # Remove excessive whitespace
                line = re.sub(r'\s+', ' ', line)
                # Remove special characters (keep Turkish chars)
                line = re.sub(r'[^\w\sçğıöşüÇĞIİÖŞÜ.,!?;:()\-\'"]', '', line)
                line = line.strip()
                
                if len(line) > 50:  # Minimum length
                    f_out.write(line + '\n')
    
    print(f"✅ Cleaned corpus saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Download Turkish corpus')
    parser.add_argument('--output', type=str, default='data/corpus.txt',
                       help='Output corpus file path')
    parser.add_argument('--size', type=float, default=1.0,
                       help='Maximum corpus size in GB')
    parser.add_argument('--clean', action='store_true',
                       help='Clean the downloaded corpus')
    
    args = parser.parse_args()
    
    # Download corpus
    corpus_path = download_turkish_corpus(args.output, args.size)
    
    if corpus_path and args.clean:
        clean_path = corpus_path.replace('.txt', '_cleaned.txt')
        clean_corpus(corpus_path, clean_path)
        print(f"\n💡 Using cleaned corpus: {clean_path}")

if __name__ == "__main__":
    main()

