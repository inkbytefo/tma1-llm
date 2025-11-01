#!/usr/bin/env python3
# Developer: inkbytefo
# AI: Claude Sonnet 4.5
# Modified: 2025-11-02

"""
============================================================================
LLM Engine - PyTorch Inference Interface
High-level interface for trained Turkish transformer model
============================================================================
"""

import torch
import os
from typing import Optional, Dict, List
from sentencepiece import SentencePieceProcessor
from pathlib import Path

from src.model import TransformerModel, ModelConfig

class LLMEngine:
    """Main LLM inference engine using PyTorch model"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize LLM engine
        
        Args:
            model_path: Path to model checkpoint (.pt file)
            tokenizer_path: Path to SentencePiece tokenizer (.model file)
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = None
        self.tokenizer = None
        self.config = None
        self.model_loaded = False
        
        # Default paths
        self.model_path = model_path or "models/checkpoint_latest.pt"
        self.tokenizer_path = tokenizer_path or "tokenizer/tr_tokenizer.model"
        
        print(f"🖥️  Device: {self.device}")
        
        # Auto-load if files exist
        if os.path.exists(self.model_path) and os.path.exists(self.tokenizer_path):
            self.load_model(self.model_path, self.tokenizer_path)
    
    def load_model(self, model_path: Optional[str] = None, tokenizer_path: Optional[str] = None) -> bool:
        """
        Load trained model and tokenizer
        
        Args:
            model_path: Path to checkpoint file
            tokenizer_path: Path to tokenizer model file
        """
        model_path = model_path or self.model_path
        tokenizer_path = tokenizer_path or self.tokenizer_path
        
        # Load tokenizer
        if not os.path.exists(tokenizer_path):
            print(f"❌ Tokenizer not found: {tokenizer_path}")
            print(f"💡 Train tokenizer first: python src/train_tokenizer.py")
            return False
        
        print(f"📥 Loading tokenizer: {tokenizer_path}")
        self.tokenizer = SentencePieceProcessor(model_file=tokenizer_path)
        print(f"✅ Tokenizer loaded (vocab size: {self.tokenizer.vocab_size()})")
        
        # Load model checkpoint
        if not os.path.exists(model_path):
            print(f"⚠️  Model checkpoint not found: {model_path}")
            print(f"💡 Train model first: python train.py")
            print(f"💡 Using randomly initialized model for demo...")
            
            # Use default config
            self.config = ModelConfig(
                vocab_size=self.tokenizer.vocab_size(),
                hidden_size=768,
                num_layers=12,
                num_heads=12
            )
            self.model = TransformerModel(self.config).to(self.device)
            self.model_loaded = True
            return True
        
        print(f"📥 Loading model checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load config from checkpoint
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            self.config = ModelConfig(**config_dict)
        else:
            # Fallback to defaults
            self.config = ModelConfig(
                vocab_size=self.tokenizer.vocab_size(),
                hidden_size=768,
                num_layers=12,
                num_heads=12
            )
        
        # Create model
        self.model = TransformerModel(self.config).to(self.device)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model weights loaded")
        else:
            print(f"⚠️  No model weights in checkpoint, using random initialization")
        
        # Set to evaluation mode
        self.model.eval()
        self.model_loaded = True
        
        # Print model info
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"🧠 Model parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
        
        if 'step' in checkpoint:
            print(f"📊 Checkpoint step: {checkpoint['step']:,}")
        if 'loss' in checkpoint:
            print(f"📉 Training loss: {checkpoint['loss']:.4f}")
        
        return True
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
        do_sample: bool = True
    ) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            eos_token_id: End-of-sequence token ID (None = use tokenizer's)
            do_sample: Whether to sample (False = greedy decoding)
        
        Returns:
            Generated text
        """
        if not self.model_loaded:
            return "❌ Model not loaded. Please load a model first."
        
        if not self.tokenizer:
            return "❌ Tokenizer not loaded."
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, out_type=int)
        
        # Convert to tensor
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        # Get EOS token
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_id()
        if eos_token_id == -1:
            eos_token_id = self.tokenizer.pad_id()
        
        # Generate
        with torch.no_grad():
            if do_sample:
                # Use model's generate method
                generated_ids = self.model.generate(
                    input_tensor,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    eos_token_id=eos_token_id,
                    pad_token_id=self.tokenizer.pad_id()
                )
            else:
                # Greedy decoding
                generated_ids = input_tensor.clone()
                past_key_values = None
                
                for _ in range(max_new_tokens):
                    # Forward pass
                    logits, past_key_values = self.model(
                        generated_ids[:, -self.config.max_seq_len:],
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    
                    # Get next token (greedy)
                    next_token = logits[0, -1, :].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)
                    
                    # Check EOS
                    if next_token.item() == eos_token_id:
                        break
        
        # Decode
        generated_ids = generated_ids[0].cpu().tolist()
        generated_text = self.tokenizer.decode(generated_ids)
        
        # Remove input from output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def get_stats(self) -> Dict:
        """Get model statistics"""
        stats = {
            "device": str(self.device),
            "model_loaded": self.model_loaded,
            "tokenizer_loaded": self.tokenizer is not None,
        }
        
        if self.config:
            stats.update({
                "vocab_size": self.config.vocab_size,
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "num_heads": self.config.num_heads,
                "ffn_size": self.config.ffn_size,
                "max_seq_len": self.config.max_seq_len,
            })
        
        if self.tokenizer:
            stats["tokenizer_vocab_size"] = self.tokenizer.vocab_size()
        
        if self.model:
            num_params = sum(p.numel() for p in self.model.parameters())
            stats["model_parameters"] = f"{num_params:,} ({num_params / 1e6:.2f}M)"
        
        return stats

def main():
    """Interactive LLM interface"""
    print("\n" + "=" * 60)
    print("    TMA-1 - Turkish Transformer (PyTorch)")
    print("=" * 60)
    
    engine = LLMEngine()
    
    if not engine.model_loaded:
        print("\n⚠️  No trained model found.")
        print("\n💡 To train a model:")
        print("   1. Download corpus: python src/data_collector.py")
        print("   2. Train tokenizer: python src/train_tokenizer.py")
        print("   3. Train model: python train.py")
        print("\n💡 Using randomly initialized model for demo...")
    
    print("\n🤖 LLM Engine ready!")
    print("Commands: 'status', 'exit', or enter a prompt")
    print("=" * 60)
    
    while True:
        try:
            prompt = input("\nYou > ").strip()
            
            if prompt.lower() in ['exit', 'quit']:
                print("👋 Goodbye!")
                break
            
            if prompt.lower() == 'status':
                stats = engine.get_stats()
                print("\n" + "=" * 60)
                print("LLM Status Report")
                print("=" * 60)
                for key, value in stats.items():
                    print(f"{key:25s}: {value}")
                print("=" * 60)
                continue
            
            if not prompt:
                continue
            
            print("\n💭 Generating...")
            response = engine.generate(
                prompt,
                max_new_tokens=100,
                temperature=0.8,
                top_p=0.9,
                top_k=50
            )
            print(f"\nLLM > {response}")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
