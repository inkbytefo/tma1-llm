#!/usr/bin/env python3
# Developer: inkbytefo
# AI: QuantumSoul-Engineer-v1
# Modified: 2024-12-19

"""
============================================================================
Training Script - Turkish LLM Pre-training
Main training loop for transformer language model
============================================================================
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from src.model import TransformerModel, ModelConfig
from src.dataset import create_dataloader

def save_checkpoint(
    model: TransformerModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    step: int,
    loss: float,
    checkpoint_dir: str,
    config: ModelConfig
):
    """Save training checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'config': {
            'vocab_size': config.vocab_size,
            'hidden_size': config.hidden_size,
            'num_layers': config.num_layers,
            'num_heads': config.num_heads,
            'ffn_size': config.ffn_size,
            'max_seq_len': config.max_seq_len,
            'head_dim': config.head_dim,
            'layer_norm_eps': config.layer_norm_eps,
            'dropout': config.dropout,
            'use_bias': config.use_bias,
            'pad_token_id': config.pad_token_id,
        }
    }
    
    # Save latest checkpoint
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)
    
    # Save periodic checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{step}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    print(f"💾 Checkpoint saved: {checkpoint_path}")

def load_checkpoint(
    checkpoint_path: str,
    model: TransformerModel,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None
):
    """Load training checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        return None
    
    print(f"📥 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint

def train(
    model: TransformerModel,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    writer: SummaryWriter,
    start_step: int = 0,
    save_every: int = 1000,
    eval_every: int = 500,
    checkpoint_dir: str = "checkpoints"
):
    """Main training loop"""
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    
    total_loss = 0.0
    current_step = start_step
    
    progress_bar = tqdm(dataloader, desc="Training", initial=current_step)
    
    for batch in progress_bar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass
        logits, _ = model(input_ids, attention_mask=attention_mask)
        
        # Reshape for loss calculation
        # logits: [batch_size, seq_len, vocab_size]
        # target_ids: [batch_size, seq_len]
        logits = logits.reshape(-1, logits.size(-1))
        target_ids = target_ids.reshape(-1)
        
        # Calculate loss
        loss = criterion(logits, target_ids)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        # Update metrics
        total_loss += loss.item()
        current_step += 1
        
        # Logging
        if current_step % 10 == 0:
            avg_loss = total_loss / 10
            total_loss = 0.0
            
            writer.add_scalar('Train/Loss', avg_loss, current_step)
            writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], current_step)
            
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                'step': current_step
            })
        
        # Evaluation (simple validation loss)
        if current_step % eval_every == 0:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            
            with torch.no_grad():
                # Simple validation on same data (in production, use separate validation set)
                for val_batch in dataloader:
                    if val_steps >= 10:  # Limit validation steps
                        break
                    
                    val_input_ids = val_batch['input_ids'].to(device)
                    val_target_ids = val_batch['target_ids'].to(device)
                    val_attention_mask = val_batch['attention_mask'].to(device)
                    
                    val_logits, _ = model(val_input_ids, attention_mask=val_attention_mask)
                    val_logits = val_logits.reshape(-1, val_logits.size(-1))
                    val_target_ids = val_target_ids.reshape(-1)
                    
                    val_loss += criterion(val_logits, val_target_ids).item()
                    val_steps += 1
            
            avg_val_loss = val_loss / val_steps
            writer.add_scalar('Validation/Loss', avg_val_loss, current_step)
            
            model.train()
            tqdm.write(f"Step {current_step}: Train Loss = {avg_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save checkpoint
        if current_step % save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler,
                epoch=0, step=current_step, loss=avg_loss if total_loss == 0 else total_loss / 10,
                checkpoint_dir=checkpoint_dir,
                config=model.config
            )
    
    return current_step

def main():
    parser = argparse.ArgumentParser(description='Train Turkish LLM')
    parser.add_argument('--corpus', type=str, default='data/corpus.txt',
                       help='Path to training corpus')
    parser.add_argument('--tokenizer', type=str, default='tokenizer/tr_tokenizer.model',
                       help='Path to SentencePiece tokenizer')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Output directory for checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='TensorBoard log directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint path')
    
    # Model config
    parser.add_argument('--vocab-size', type=int, default=32000,
                       help='Vocabulary size')
    parser.add_argument('--hidden-size', type=int, default=768,
                       help='Hidden size')
    parser.add_argument('--num-layers', type=int, default=12,
                       help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=12,
                       help='Number of attention heads')
    parser.add_argument('--ffn-size', type=int, default=3072,
                       help='Feed-forward network size')
    parser.add_argument('--max-seq-len', type=int, default=512,
                       help='Maximum sequence length')
    
    # Training config
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=1,
                       help='Number of epochs')
    parser.add_argument('--save-every', type=int, default=1000,
                       help='Save checkpoint every N steps')
    parser.add_argument('--eval-every', type=int, default=500,
                       help='Evaluate every N steps')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                       help='Warmup steps for learning rate')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Model configuration
    config = ModelConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_size=args.ffn_size,
        max_seq_len=args.max_seq_len,
        pad_token_id=0
    )
    
    print("\n" + "=" * 60)
    print("    QuantumSoul LLM Training")
    print("=" * 60)
    print(f"\n📊 Model Configuration:")
    print(f"   Vocabulary size: {config.vocab_size:,}")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Layers: {config.num_layers}")
    print(f"   Attention heads: {config.num_heads}")
    print(f"   FFN size: {config.ffn_size}")
    print(f"   Max sequence length: {config.max_seq_len}")
    
    # Create model
    model = TransformerModel(config)
    model = model.to(device)
    
    num_params = model.get_num_params()
    print(f"\n🧠 Model parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    
    # Create data loader
    print(f"\n📂 Loading dataset...")
    print(f"   Corpus: {args.corpus}")
    print(f"   Tokenizer: {args.tokenizer}")
    
    dataloader, tokenizer = create_dataloader(
        corpus_file=args.corpus,
        tokenizer_path=args.tokenizer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_workers=4,
        shuffle=True
    )
    
    print(f"   Batch size: {args.batch_size}")
    print(f"   Dataset size: {len(dataloader.dataset):,} examples")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    total_steps = len(dataloader) * args.num_epochs
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - args.warmup_steps,
        eta_min=args.learning_rate * 0.1
    )
    
    # TensorBoard writer
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Resume from checkpoint
    start_step = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
        if checkpoint:
            start_step = checkpoint.get('step', 0)
            print(f"✅ Resumed from step {start_step}")
    
    # Training
    print(f"\n🚀 Starting training...")
    print(f"   Total steps: {total_steps:,}")
    print(f"   Checkpoint dir: {args.output_dir}")
    print(f"   Log dir: {args.log_dir}")
    print("")
    
    try:
        final_step = train(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            writer=writer,
            start_step=start_step,
            save_every=args.save_every,
            eval_every=args.eval_every,
            checkpoint_dir=args.output_dir
        )
        
        # Final checkpoint
        print(f"\n💾 Saving final checkpoint...")
        save_checkpoint(
            model, optimizer, scheduler,
            epoch=args.num_epochs, step=final_step, loss=0.0,
            checkpoint_dir=args.output_dir,
            config=config
        )
        
        print(f"\n✅ Training completed!")
        print(f"   Final step: {final_step:,}")
        print(f"   Model saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted")
        print(f"💾 Saving checkpoint...")
        save_checkpoint(
            model, optimizer, scheduler,
            epoch=0, step=start_step, loss=0.0,
            checkpoint_dir=args.output_dir,
            config=config
        )
        print(f"✅ Checkpoint saved. Resume with: --resume {args.output_dir}/checkpoint_latest.pt")
    
    writer.close()

if __name__ == "__main__":
    main()

