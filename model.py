#!/usr/bin/env python3
"""
Implementation of a GPT-2 style model using PyTorch.
This module provides a transformer-based language model similar to GPT-2.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """
    Layer normalization module with optional bias.
    """
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention module.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Key, query, value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Flash attention support if available
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "mask", 
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # Causal self-attention
        if self.flash:
            # Flash attention implementation (much faster)
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # Manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    Multi-layer perceptron module used in transformer blocks.
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer block: communication followed by computation.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        # Pre-LayerNorm formulation
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    GPT-2 style transformer model.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying: tie the weights of the token embedding and the final language model head
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        # Report number of parameters
        print(f"Number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        # Forward the GPT model
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        
        # Token embeddings
        tok_emb = self.transformer.wte(idx) # (b, t, n_embd)
        # Position embeddings
        pos_emb = self.transformer.wpe(pos) # (1, t, n_embd)
        
        # Add token and position embeddings
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Apply transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        # Apply final layer norm
        x = self.transformer.ln_f(x)
        
        # Project to vocabulary
        logits = self.lm_head(x) # (b, t, vocab_size)
        
        # If we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text by sampling from the model.
        
        Args:
            idx: Context tokens as a tensor of shape (b, t)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (1.0 = no change, < 1.0 = less random, > 1.0 = more random)
            top_k: If specified, restricts sampling to the top k most likely tokens
            
        Returns:
            Tensor of shape (b, t+max_new_tokens) containing the generated tokens
        """
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            
            # Focus only on the last time step
            logits = logits[:, -1, :] # becomes (b, vocab_size)
            
            # Apply temperature
            logits = logits / temperature
            
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


class GPTConfig:
    """
    Configuration class for GPT model hyperparameters.
    """
    def __init__(
        self,
        vocab_size=50257,  # GPT-2 vocabulary size
        block_size=1024,   # Maximum sequence length
        n_layer=12,        # Number of transformer blocks
        n_head=12,         # Number of attention heads
        n_embd=768,        # Embedding dimension
        dropout=0.1,       # Dropout rate
        bias=True,         # Use bias in LayerNorm and Linear layers
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias


# Example configurations for different model sizes
def get_gpt2_config():
    """
    Returns the configuration for the original GPT-2 model (124M parameters).
    """
    return GPTConfig(
        vocab_size=50257,
        block_size=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
    )


def get_gpt2_medium_config():
    """
    Returns the configuration for the GPT-2 Medium model (355M parameters).
    """
    return GPTConfig(
        vocab_size=50257,
        block_size=1024,
        n_layer=24,
        n_head=16,
        n_embd=1024,
    )


def get_gpt2_mini_config():
    """
    Returns a configuration for a much smaller GPT-2 model for experimentation.
    """
    return GPTConfig(
        vocab_size=50257,
        block_size=1024,
        n_layer=6,
        n_head=6,
        n_embd=384,
    )


def get_gpt2_micro_config():
    """
    Returns a configuration for a tiny GPT-2 model for quick testing.
    """
    return GPTConfig(
        vocab_size=50257,
        block_size=1024,
        n_layer=4,
        n_head=4,
        n_embd=128,
    )


if __name__ == "__main__":
    # Simple test to verify the model works
    config = get_gpt2_micro_config()
    model = GPT(config)
    
    # Create a small batch of token indices
    x = torch.randint(0, config.vocab_size, (2, 10))
    
    # Forward pass
    logits, loss = model(x, targets=x)
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test generation
    generated = model.generate(x[:, :1], max_new_tokens=20, temperature=0.8, top_k=40)
    print(f"Generated shape: {generated.shape}")