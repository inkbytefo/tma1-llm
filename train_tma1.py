#!/usr/bin/env python3
# Developer: inkbytefo
# AI: Claude Sonnet 4.5
# Modified: 2025-11-01

"""
============================================================================
TMA-1 Training Script - Turkish Morphological Language Model
Türkçe Mantık Ağı (TMA-1) eğitim scripti
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

# TMA-1 imports
from src.tma1_model import TMA1Model
from src.model import ModelConfig
from src.morphopiece import MorphoPiece
from src.grammar_engine import GrammarEngine
from src.morpho_splitter import MorphoSplitter
from src.dataset import TurkishTextDataset
from torch.utils.data import DataLoader

def save_checkpoint(
    model: TMA1Model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    tokenizer: MorphoPiece,
    epoch: int,
    step: int,
    loss: float,
    grammar_violations: int,
    checkpoint_dir: str,
    config: ModelConfig,
    tokenizer_path: str | None = None
):
    """Save TMA-1 training checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'grammar_violations': grammar_violations,
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
        },
        'tokenizer_path': tokenizer_path
    }
    
    # Save latest checkpoint
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)
    
    # Save periodic checkpoint (every 500 steps)
    if step % 500 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{step}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Periodic checkpoint saved: {checkpoint_path}")
    else:
        print(f"💾 Latest checkpoint saved: {latest_path}")

def load_checkpoint(
    checkpoint_path: str,
    model: TMA1Model,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None
):
    """Load TMA-1 training checkpoint"""
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

def count_grammar_violations(
    token_ids: torch.Tensor,
    tokenizer: MorphoPiece,
    grammar_engine: GrammarEngine,
    vocab: list
) -> int:
    """
    Üretilen token'lardaki dilbilgisi ihlallerini say
    
    Args:
        token_ids: Token ID tensor [batch, seq_len]
        tokenizer: MorphoPiece tokenizer
        grammar_engine: Grammar engine
        vocab: Vocabulary listesi
    
    Returns:
        İhlal sayısı
    """
    violations = 0
    
    try:
        batch_size, seq_len = token_ids.shape
        
        for b in range(batch_size):
            # Token ID'leri decode et
            token_ids_list = token_ids[b].cpu().tolist()
            tokens_text = []
            
            for token_id in token_ids_list:
                if token_id < len(vocab):
                    tokens_text.append(vocab[token_id])
                else:
                    tokens_text.append("<UNK>")
            
            # Boş olmayan token'ları al
            tokens_clean = [t for t in tokens_text if t and t != "<UNK>" and t != "<PAD>"]
            
            if len(tokens_clean) < 2:
                continue
            
            # Grammar validation
            is_valid, errors = grammar_engine.validate_sequence(tokens_clean)
            if not is_valid:
                violations += len(errors)
    
    except Exception as e:
        # Hata durumunda 0 döndür
        pass
    
    return violations

def train_tma1(
    model: TMA1Model,
    dataloader: DataLoader,
    val_dataloader: DataLoader,
    tokenizer: MorphoPiece,
    grammar_engine: GrammarEngine,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    writer: SummaryWriter,
    vocab: list,
    start_step: int = 0,
    save_every: int = 500,
    eval_every: int = 250,
    checkpoint_dir: str = "checkpoints"
):
    """Main TMA-1 training loop"""
    model.train()
    # Ignore padding tokens per model config
    criterion = nn.CrossEntropyLoss(ignore_index=model.config.pad_token_id)
    
    total_loss = 0.0
    total_grammar_violations = 0
    current_step = start_step
    
    progress_bar = tqdm(dataloader, desc="Training TMA-1", initial=current_step)
    
    for batch in progress_bar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Get morpho_types from batch (if available - from preprocessing)
        morpho_types = None
        if 'morpho_types' in batch:
            morpho_types = batch['morpho_types'].to(device)
        
        # Fallback: Get token texts only if morpho_types not available (SLOW)
        token_texts = None
        if morpho_types is None:
            token_texts = []
            for b_idx in range(input_ids.shape[0]):
                batch_tokens = []
                for s_idx in range(input_ids.shape[1]):
                    token_id = input_ids[b_idx, s_idx].item()
                    if token_id < len(vocab):
                        batch_tokens.append(vocab[token_id])
                    else:
                        batch_tokens.append("")
                token_texts.append(batch_tokens)
        
        # Forward pass (TMA-1 with grammar bias)
        logits, _ = model(
            input_ids,
            attention_mask=attention_mask,
            morpho_types=morpho_types,
            token_texts=token_texts,
            vocab=vocab if morpho_types is None else None,
            use_cache=False
        )
        
        # Reshape for loss calculation
        logits = logits.reshape(-1, logits.size(-1))
        target_ids = target_ids.reshape(-1)
        
        # Calculate loss
        loss = criterion(logits, target_ids)
        
        # Count grammar violations (on predictions)
        with torch.no_grad():
            pred_ids = logits.argmax(dim=-1).view(input_ids.shape[0], -1)
            violations = count_grammar_violations(pred_ids, tokenizer, grammar_engine, vocab)
            total_grammar_violations += violations
        
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
        
        # Logging (every 10 steps)
        if current_step % 10 == 0:
            avg_loss = total_loss / 10
            avg_violations = total_grammar_violations / 10
            total_loss = 0.0
            total_grammar_violations = 0
            
            writer.add_scalar('Train/Loss', avg_loss, current_step)
            writer.add_scalar('Train/GrammarViolations', avg_violations, current_step)
            writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], current_step)
            
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'grammar': f'{avg_violations:.1f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                'step': current_step
            })
        
        # Evaluation (validation set)
        if current_step % eval_every == 0:
            model.eval()
            val_loss = 0.0
            val_violations = 0.0
            val_steps = 0
            
            with torch.no_grad():
                for val_batch in val_dataloader:
                    if val_steps >= 10:  # Limit validation steps
                        break
                    
                    val_input_ids = val_batch['input_ids'].to(device)
                    val_target_ids = val_batch['target_ids'].to(device)
                    val_attention_mask = val_batch['attention_mask'].to(device)
                    
                    # Get morpho_types from batch (if available)
                    val_morpho_types = None
                    if 'morpho_types' in val_batch:
                        val_morpho_types = val_batch['morpho_types'].to(device)
                    
                    # Fallback: Get token texts only if morpho_types not available
                    val_token_texts = None
                    if val_morpho_types is None:
                        val_token_texts = []
                        for b_idx in range(val_input_ids.shape[0]):
                            batch_tokens = []
                            for s_idx in range(val_input_ids.shape[1]):
                                token_id = val_input_ids[b_idx, s_idx].item()
                                if token_id < len(vocab):
                                    batch_tokens.append(vocab[token_id])
                                else:
                                    batch_tokens.append("")
                            val_token_texts.append(batch_tokens)
                    
                    val_logits, _ = model(
                        val_input_ids,
                        attention_mask=val_attention_mask,
                        morpho_types=val_morpho_types,
                        token_texts=val_token_texts,
                        vocab=vocab if val_morpho_types is None else None
                    )
                    
                    val_logits = val_logits.reshape(-1, val_logits.size(-1))
                    val_target_ids = val_target_ids.reshape(-1)
                    
                    val_loss += criterion(val_logits, val_target_ids).item()
                    
                    # Grammar violations
                    val_pred_ids = val_logits.argmax(dim=-1).view(val_input_ids.shape[0], -1)
                    violations = count_grammar_violations(val_pred_ids, tokenizer, grammar_engine, vocab)
                    val_violations += violations
                    
                    val_steps += 1
            
            avg_val_loss = val_loss / val_steps
            avg_val_violations = val_violations / val_steps
            
            # Calculate Perplexity (PPL) from validation loss
            # Perplexity = exp(loss), lower is better
            val_perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
            train_perplexity = torch.exp(torch.tensor(avg_loss)).item()
            
            # Log metrics to TensorBoard
            writer.add_scalar('Validation/Loss', avg_val_loss, current_step)
            writer.add_scalar('Validation/GrammarViolations', avg_val_violations, current_step)
            writer.add_scalar('Validation/Perplexity', val_perplexity, current_step)
            writer.add_scalar('Train/Perplexity', train_perplexity, current_step)
            
            model.train()
            tqdm.write(f"Step {current_step}: Train Loss = {avg_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            tqdm.write(f"  Train PPL = {train_perplexity:.2f}, Val PPL = {val_perplexity:.2f}")
            tqdm.write(f"  Grammar Violations - Train: {avg_violations:.1f}, Val: {avg_val_violations:.1f}")
        
        # Save checkpoint (every 500 steps)
        if current_step % save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, tokenizer,
                epoch=0, step=current_step, loss=avg_loss if total_loss == 0 else total_loss / 10,
                grammar_violations=int(avg_violations if total_grammar_violations == 0 else total_grammar_violations / 10),
                checkpoint_dir=checkpoint_dir,
                config=model.config,
                tokenizer_path=None
            )
    
    return current_step

def get_vocab_from_tokenizer(tokenizer: MorphoPiece) -> list:
    """Tokenizer'dan vocabulary listesi al"""
    vocab = []
    if tokenizer.sp_processor:
        for i in range(tokenizer.sp_processor.vocab_size()):
            vocab.append(tokenizer.sp_processor.id_to_piece(i))
    return vocab

def main():
    parser = argparse.ArgumentParser(description='Train TMA-1 (Turkish Morphological Language Model)')
    
    # Data
    parser.add_argument('--corpus', type=str, default='data/corpus_morpho_processed.txt',
                       help='Path to training corpus (morphologically processed)')
    parser.add_argument('--val-corpus', type=str, default=None,
                       help='Path to validation corpus (morphologically processed). If not provided, uses training corpus for validation (not recommended)')
    parser.add_argument('--tokenizer', type=str, default='tokenizer/morphopiece.model',
                       help='Path to MorphoPiece tokenizer model')
    
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
    parser.add_argument('--save-every', type=int, default=500,
                       help='Save checkpoint every N steps')
    parser.add_argument('--eval-every', type=int, default=250,
                       help='Evaluate every N steps')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                       help='Warmup steps for learning rate')
    
    # Paths
    parser.add_argument('--output-dir', type=str, default='models/tma1',
                       help='Output directory for checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs/tma1',
                       help='TensorBoard log directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint path')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print("\n" + "=" * 60)
    print("    TMA-1 Training (Turkish Morphological Language Model)")
    print("=" * 60)
    
    # Load MorphoPiece tokenizer
    print(f"\n📥 Loading MorphoPiece tokenizer...")
    print(f"   Path: {args.tokenizer}")
    
    tokenizer = MorphoPiece(tokenizer_path=args.tokenizer)
    if not tokenizer.sp_processor:
        print(f"❌ Failed to load tokenizer: {args.tokenizer}")
        print(f"💡 Train tokenizer first: python src/train_morphopiece.py --all")
        return
    
    vocab = get_vocab_from_tokenizer(tokenizer)
    actual_vocab_size = len(vocab)
    print(f"✅ Tokenizer loaded (vocab size: {actual_vocab_size:,})")
    
    # Model configuration
    # Determine a valid pad token id: fallback to UNK when pad is disabled (-1)
    sp = tokenizer.sp_processor
    pad_token_id = sp.pad_id() if sp else 0
    if sp and pad_token_id < 0:
        pad_token_id = sp.unk_id()

    config = ModelConfig(
        vocab_size=actual_vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_size=args.ffn_size,
        max_seq_len=args.max_seq_len,
        pad_token_id=pad_token_id
    )
    
    print(f"\n📊 Model Configuration:")
    print(f"   Vocabulary size: {config.vocab_size:,}")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Layers: {config.num_layers}")
    print(f"   Attention heads: {config.num_heads}")
    print(f"   FFN size: {config.ffn_size}")
    print(f"   Max sequence length: {config.max_seq_len}")
    
    # Create TMA-1 model
    print(f"\n🧠 Creating TMA-1 model...")
    morpho_splitter = MorphoSplitter()
    grammar_engine = GrammarEngine()
    
    model = TMA1Model(
        config,
        morpho_splitter=morpho_splitter,
        grammar_engine=grammar_engine,
        use_grammar_bias=True
    )
    model = model.to(device)
    
    num_params = model.get_num_params()
    print(f"✅ TMA-1 model created")
    print(f"   Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    
    # Create dataset and dataloader
    print(f"\n📂 Loading dataset...")
    print(f"   Corpus: {args.corpus}")
    
    if not os.path.exists(args.corpus):
        print(f"❌ Corpus file not found: {args.corpus}")
        print(f"💡 Preprocess corpus first: python scripts/preprocess_for_tma1.py")
        return
    
    # Check if corpus is JSONL (preprocessed) or text
    is_jsonl = args.corpus.endswith('.jsonl')
    if is_jsonl:
        print(f"   Format: JSONL (preprocessed with morpho_types)")
    else:
        print(f"   Format: Text (will use fallback token_texts)")
        print(f"   ⚠️  WARNING: Using text format is SLOW! Preprocess corpus first:")
        print(f"      python scripts/preprocess_for_tma1.py --input {args.corpus} --output {args.corpus}.jsonl --tokenizer {args.tokenizer}")
    
    # Create training dataset and dataloader
    print(f"\n📂 Loading training dataset...")
    train_dataset = TurkishTextDataset(
        corpus_file=args.corpus,
        tokenizer=tokenizer.sp_processor,  # Use SentencePiece processor directly
        max_seq_len=args.max_seq_len,
        is_jsonl=is_jsonl
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    print(f"   Training batch size: {args.batch_size}")
    print(f"   Training dataset size: {len(train_dataset):,} examples")
    
    # Create validation dataset and dataloader
    if args.val_corpus and os.path.exists(args.val_corpus):
        print(f"\n📂 Loading validation dataset...")
        print(f"   Validation corpus: {args.val_corpus}")
        val_is_jsonl = args.val_corpus.endswith('.jsonl')
        
        val_dataset = TurkishTextDataset(
            corpus_file=args.val_corpus,
            tokenizer=tokenizer.sp_processor,
            max_seq_len=args.max_seq_len,
            is_jsonl=val_is_jsonl
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,  # Don't shuffle validation set
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
            drop_last=False  # Don't drop last batch in validation
        )
        
        print(f"   Validation batch size: {args.batch_size}")
        print(f"   Validation dataset size: {len(val_dataset):,} examples")
    else:
        print(f"\n⚠️  WARNING: No validation corpus provided!")
        if args.val_corpus:
            print(f"   Validation corpus file not found: {args.val_corpus}")
        print(f"   Using training data for validation (NOT RECOMMENDED)")
        print(f"   💡 Use --split-dataset in preprocess_for_tma1.py to create validation set:")
        print(f"      python scripts/preprocess_for_tma1.py --split-dataset --input <corpus> --output <output> --tokenizer <tokenizer>")
        
        # Fallback: Use training dataloader for validation (not recommended)
        val_dataloader = train_dataloader
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    total_steps = len(train_dataloader) * args.num_epochs
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
    print(f"\n🚀 Starting TMA-1 training...")
    print(f"   Total steps: {total_steps:,}")
    print(f"   Checkpoint dir: {args.output_dir}")
    print(f"   Log dir: {args.log_dir}")
    print(f"   Save every: {args.save_every} steps")
    print(f"   Eval every: {args.eval_every} steps")
    print("")
    
    try:
        final_step = train_tma1(
            model=model,
            dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            tokenizer=tokenizer,
            grammar_engine=grammar_engine,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            writer=writer,
            vocab=vocab,
            start_step=start_step,
            save_every=args.save_every,
            eval_every=args.eval_every,
            checkpoint_dir=args.output_dir
        )
        
        # Final checkpoint
        print(f"\n💾 Saving final checkpoint...")
        save_checkpoint(
            model, optimizer, scheduler, tokenizer,
            epoch=args.num_epochs, step=final_step, loss=0.0, grammar_violations=0,
            checkpoint_dir=args.output_dir,
            config=config,
            tokenizer_path=args.tokenizer
        )
        
        print(f"\n✅ TMA-1 training completed!")
        print(f"   Final step: {final_step:,}")
        print(f"   Model saved to: {args.output_dir}")
        print(f"   TensorBoard logs: {args.log_dir}")
        print(f"\n💡 View training progress:")
        print(f"   tensorboard --logdir {args.log_dir}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted")
        print(f"💾 Saving checkpoint...")
        save_checkpoint(
            model, optimizer, scheduler, tokenizer,
            epoch=0, step=start_step, loss=0.0, grammar_violations=0,
            checkpoint_dir=args.output_dir,
            config=config,
            tokenizer_path=args.tokenizer
        )
        print(f"✅ Checkpoint saved. Resume with: --resume {args.output_dir}/checkpoint_latest.pt")
    
    writer.close()

if __name__ == "__main__":
    main()

