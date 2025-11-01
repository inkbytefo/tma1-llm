#!/usr/bin/env python3
# Developer: inkbytefo
# AI: QuantumSoul-Engineer-v1
# Modified: 2024-12-19

"""
============================================================================
Transformer Model - PyTorch Implementation
Based on llm_core.h architecture with trainable parameters
============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ModelConfig:
    """Model configuration matching llm_core.h"""
    vocab_size: int = 50000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    ffn_size: int = 3072
    max_seq_len: int = 512
    head_dim: int = 64
    layer_norm_eps: float = 1e-5
    dropout: float = 0.1
    use_bias: bool = True
    pad_token_id: int = 0
    
    def __post_init__(self):
        """Validate and adjust configuration"""
        # Ensure head_dim matches
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_heads
        assert self.hidden_size % self.num_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention Mechanism"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Q, K, V projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] or [batch_size, seq_len, seq_len]
            past_key_value: Tuple of (past_key, past_value) for caching
            use_cache: Whether to return key/value cache
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Project to Q, K, V
        query = self.q_proj(hidden_states)  # [batch, seq_len, hidden_size]
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape for multi-head: [batch, num_heads, seq_len, head_dim]
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Use past key/value if provided (for generation)
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)
            kv_seq_len = key.shape[2]
        else:
            kv_seq_len = seq_len
        
        # Compute attention scores: Q @ K^T
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale  # [batch, num_heads, seq_len, kv_seq_len]
        
        # Apply causal mask if needed (for generation)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Expand to [batch, 1, 1, seq_len]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply to values
        attn_output = torch.matmul(attn_weights, value)  # [batch, num_heads, seq_len, head_dim]
        
        # Concatenate heads: [batch, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, hidden_size)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        # Prepare cache if needed
        present_key_value = None
        if use_cache:
            present_key_value = (key, value)
        
        return output, present_key_value

class FeedForward(nn.Module):
    """Feed-Forward Network with GLU activation"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # GLU-style feed-forward
        self.gate_proj = nn.Linear(config.hidden_size, config.ffn_size, bias=config.use_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.ffn_size, bias=config.use_bias)
        self.down_proj = nn.Linear(config.ffn_size, config.hidden_size, bias=config.use_bias)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """SiLU(Gate) * Up"""
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        
        # SiLU activation: x * sigmoid(x)
        gate = F.silu(gate)
        
        # Element-wise multiply and down projection
        output = self.down_proj(gate * up)
        output = self.dropout(output)
        
        return output

class TransformerLayer(nn.Module):
    """Single Transformer Layer"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Pre-attention layer norm
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(config)
        
        # Pre-FFN layer norm
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Feed-forward network
        self.ffn = FeedForward(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-attention layer norm + residual
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        attn_output, present_key_value = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        hidden_states = residual + attn_output
        
        # Pre-FFN layer norm + residual
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        ffn_output = self.ffn(hidden_states)
        hidden_states = residual + ffn_output
        
        return hidden_states, present_key_value

class TransformerModel(nn.Module):
    """Complete Transformer Model"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.ln_final = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Output projection (tied with input embedding or separate)
        self.output_proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Optionally tie weights with embedding
        # self.output_proj.weight = self.embedding.weight
        
        # Positional encoding (learned or sinusoidal)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.max_seq_len, config.hidden_size) * 0.02
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier uniform"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            past_key_values: List of past key/value tuples for each layer
            use_cache: Whether to return key/value cache
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            past_key_values: List of present key/value tuples (if use_cache)
        """
        batch_size, seq_len = input_ids.shape
        
        # Embedding lookup
        hidden_states = self.embedding(input_ids)  # [batch, seq_len, hidden_size]
        
        # Add positional encoding
        hidden_states = hidden_states + self.pos_encoding[:, :seq_len, :]
        
        # Process through transformer layers
        present_key_values = [] if use_cache else None
        
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values else None
            hidden_states, present_key_value = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache
            )
            if use_cache:
                present_key_values.append(present_key_value)
        
        # Final layer norm
        hidden_states = self.ln_final(hidden_states)
        
        # Output projection
        logits = self.output_proj(hidden_states)  # [batch, seq_len, vocab_size]
        
        return logits, present_key_values if use_cache else None
    
    def get_num_params(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int = 2,
        pad_token_id: int = 0
    ) -> torch.Tensor:
        """
        Generate text autoregressively
        
        Args:
            input_ids: [batch_size, seq_len] initial tokens
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            eos_token_id: End-of-sequence token ID
            pad_token_id: Padding token ID
        
        Returns:
            Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # Initialize past key values
        past_key_values = None
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass (only last token)
                logits, past_key_values = self(
                    generated[:, -self.config.max_seq_len:],
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                # Get logits for last token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for EOS
                if (next_token == eos_token_id).all():
                    break
        
        return generated

