#!/usr/bin/env python3
"""
============================================================================
Baseline vs TMA-1 Comparison Script
Run both baseline and TMA-1 models with identical settings for comparison
============================================================================
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print("")
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(
        description='Compare Baseline Transformer vs TMA-1 Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python scripts/run_baseline_comparison.py \\
      --train-corpus data/train_data_train.jsonl \\
      --val-corpus data/train_data_val.jsonl \\
      --tokenizer tokenizer/morphopiece.model \\
      --hidden-size 512 \\
      --num-layers 6 \\
      --num-heads 8 \\
      --batch-size 4 \\
      --learning-rate 3e-4 \\
      --num-epochs 1 \\
      --max-steps 1000

This will train both models with identical settings and generate comparison logs.
        """
    )
    
    # Data paths
    parser.add_argument('--train-corpus', type=str, required=True,
                       help='Path to training corpus (JSONL or text)')
    parser.add_argument('--val-corpus', type=str, required=True,
                       help='Path to validation corpus (JSONL or text)')
    parser.add_argument('--tokenizer', type=str, required=True,
                       help='Path to SentencePiece tokenizer model')
    
    # Model config (same for both)
    parser.add_argument('--hidden-size', type=int, default=512,
                       help='Hidden size (default: 512 for tiny model)')
    parser.add_argument('--num-layers', type=int, default=6,
                       help='Number of transformer layers (default: 6 for tiny model)')
    parser.add_argument('--num-heads', type=int, default=8,
                       help='Number of attention heads (default: 8)')
    parser.add_argument('--ffn-size', type=int, default=2048,
                       help='Feed-forward network size (default: 2048)')
    parser.add_argument('--max-seq-len', type=int, default=512,
                       help='Maximum sequence length (default: 512)')
    
    # Training config (same for both)
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size (default: 4 for tiny model)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--num-epochs', type=int, default=1,
                       help='Number of epochs (default: 1)')
    parser.add_argument('--save-every', type=int, default=500,
                       help='Save checkpoint every N steps (default: 500)')
    parser.add_argument('--eval-every', type=int, default=250,
                       help='Evaluate every N steps (default: 250)')
    parser.add_argument('--warmup-steps', type=int, default=100,
                       help='Warmup steps for learning rate (default: 100)')
    parser.add_argument('--max-steps', type=int, default=None,
                       help='Maximum training steps (None = full epoch)')
    
    # Output paths
    parser.add_argument('--baseline-output-dir', type=str, default='models/baseline_comparison',
                       help='Output directory for baseline model (default: models/baseline_comparison)')
    parser.add_argument('--tma1-output-dir', type=str, default='models/tma1_comparison',
                       help='Output directory for TMA-1 model (default: models/tma1_comparison)')
    parser.add_argument('--baseline-log-dir', type=str, default='logs/baseline_comparison',
                       help='TensorBoard log directory for baseline (default: logs/baseline_comparison)')
    parser.add_argument('--tma1-log-dir', type=str, default='logs/tma1_comparison',
                       help='TensorBoard log directory for TMA-1 (default: logs/tma1_comparison)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.train_corpus):
        print(f"❌ Training corpus not found: {args.train_corpus}")
        return 1
    
    if not os.path.exists(args.val_corpus):
        print(f"❌ Validation corpus not found: {args.val_corpus}")
        return 1
    
    if not os.path.exists(args.tokenizer):
        print(f"❌ Tokenizer not found: {args.tokenizer}")
        return 1
    
    print("\n" + "="*60)
    print("  Baseline vs TMA-1 Comparison Experiment")
    print("="*60)
    print(f"\n📊 Experiment Configuration:")
    print(f"   Training corpus: {args.train_corpus}")
    print(f"   Validation corpus: {args.val_corpus}")
    print(f"   Tokenizer: {args.tokenizer}")
    print(f"\n   Model config:")
    print(f"     Hidden size: {args.hidden_size}")
    print(f"     Layers: {args.num_layers}")
    print(f"     Attention heads: {args.num_heads}")
    print(f"     FFN size: {args.ffn_size}")
    print(f"     Max sequence length: {args.max_seq_len}")
    print(f"\n   Training config:")
    print(f"     Batch size: {args.batch_size}")
    print(f"     Learning rate: {args.learning_rate}")
    print(f"     Epochs: {args.num_epochs}")
    print(f"     Eval every: {args.eval_every} steps")
    print(f"     Save every: {args.save_every} steps")
    
    # Common arguments for both models
    common_args = [
        '--corpus', args.train_corpus,
        '--val-corpus', args.val_corpus,
        '--tokenizer', args.tokenizer,
        '--hidden-size', str(args.hidden_size),
        '--num-layers', str(args.num_layers),
        '--num-heads', str(args.num_heads),
        '--ffn-size', str(args.ffn_size),
        '--max-seq-len', str(args.max_seq_len),
        '--batch-size', str(args.batch_size),
        '--learning-rate', str(args.learning_rate),
        '--num-epochs', str(args.num_epochs),
        '--save-every', str(args.save_every),
        '--eval-every', str(args.eval_every),
        '--warmup-steps', str(args.warmup_steps),
    ]
    
    # Train Baseline Model
    baseline_cmd = ['python', 'train.py'] + common_args + [
        '--output-dir', args.baseline_output_dir,
        '--log-dir', args.baseline_log_dir
    ]
    
    baseline_success = run_command(
        baseline_cmd,
        "Training Baseline Transformer Model"
    )
    
    if not baseline_success:
        print(f"\n❌ Baseline training failed!")
        return 1
    
    # Train TMA-1 Model
    tma1_cmd = ['python', 'train_tma1.py'] + common_args + [
        '--output-dir', args.tma1_output_dir,
        '--log-dir', args.tma1_log_dir
    ]
    
    tma1_success = run_command(
        tma1_cmd,
        "Training TMA-1 Model"
    )
    
    if not tma1_success:
        print(f"\n❌ TMA-1 training failed!")
        return 1
    
    # Summary
    print("\n" + "="*60)
    print("  Comparison Experiment Completed!")
    print("="*60)
    print(f"\n✅ Both models trained successfully!")
    print(f"\n📊 Comparison Results:")
    print(f"   Baseline logs: {args.baseline_log_dir}")
    print(f"   TMA-1 logs: {args.tma1_log_dir}")
    print(f"\n💡 Compare results using TensorBoard:")
    print(f"   tensorboard --logdir logs/")
    print(f"\n📈 To compare side-by-side:")
    print(f"   1. Open TensorBoard: tensorboard --logdir logs/")
    print(f"   2. Navigate to 'SCALARS' tab")
    print(f"   3. Compare 'Validation/Perplexity' and 'Validation/Loss' for both models")
    print(f"   4. Lower PPL and faster convergence = better model")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

