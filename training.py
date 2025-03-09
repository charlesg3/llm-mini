#!/usr/bin/env python3
"""
Training module for the GPT model.
This module provides functionality for training and evaluating the GPT model on token data.
"""

import os
import time
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
import pyarrow.parquet as pq
from model import GPT, get_gpt2_micro_config, get_gpt2_mini_config
from config import get_config


class TokenDataset(Dataset):
    """
    Dataset for training the GPT model on token sequences.
    """
    def __init__(self, tokens_file, block_size):
        """
        Initialize the dataset.
        
        Args:
            tokens_file: Path to the parquet file containing tokenized data
            block_size: Size of the context window for the model
        """
        self.block_size = block_size
        
        # Load tokens from parquet file
        print(f"Loading tokens from {tokens_file}...")
        table = pq.read_table(tokens_file)
        df = table.to_pandas()
        
        # Check if the dataframe has the expected format
        if 'token_id' in df.columns:
            # Format from tokenizer.py - a single column of token IDs
            self.tokens = df['token_id'].values
        else:
            # Try to handle other formats or raise an error
            print(f"Warning: Expected 'token_id' column not found in {tokens_file}")
            print(f"Available columns: {df.columns.tolist()}")
            raise ValueError(f"Could not find token data in {tokens_file}")
        
        print(f"Loaded {len(self.tokens)} tokens")
    
    def __len__(self):
        # Return the number of possible starting positions for sequences
        return max(0, len(self.tokens) - self.block_size)
    
    def __getitem__(self, idx):
        # Grab a chunk of tokens starting at position idx
        chunk = self.tokens[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class Trainer:
    """
    Trainer class for the GPT model.
    """
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        grad_clip=1.0,
        warmup_tokens=512*20,
        final_tokens=None,
        batch_size=64,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir='checkpoints',
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The GPT model to train
            train_dataset: Dataset for training
            val_dataset: Optional dataset for validation
            lr: Learning rate
            betas: Adam optimizer betas
            weight_decay: Weight decay for AdamW optimizer
            grad_clip: Gradient clipping value
            warmup_tokens: Number of tokens for learning rate warmup
            final_tokens: Total number of tokens for learning rate decay
            batch_size: Batch size for training
            device: Device to train on ('cuda' or 'cpu')
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.warmup_tokens = warmup_tokens
        
        # Handle the case where train_dataset is a Subset
        if hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'block_size'):
            block_size = train_dataset.dataset.block_size
        elif hasattr(train_dataset, 'block_size'):
            block_size = train_dataset.block_size
        else:
            block_size = 1024  # Default block size
            print("Warning: Could not determine block_size from dataset, using default value of 1024")
        
        self.final_tokens = final_tokens if final_tokens is not None else 2*len(train_dataset)*block_size
        
        self.batch_size = batch_size
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Move model to device
        self.model.to(device)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
        
        # Training state
        self.tokens = 0  # Counter for number of tokens processed
        self.best_val_loss = float('inf')
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
            )
        else:
            self.val_loader = None
    
    def get_lr(self):
        """
        Get the current learning rate based on warmup and decay schedule.
        """
        if self.tokens < self.warmup_tokens:
            # Linear warmup
            return self.lr * self.tokens / self.warmup_tokens
        else:
            # Cosine decay
            decay_ratio = (self.tokens - self.warmup_tokens) / (self.final_tokens - self.warmup_tokens)
            decay_ratio = min(decay_ratio, 1.0)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return self.lr * coeff
    
    def train_epoch(self):
        """
        Train the model for one epoch.
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        losses = []
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")
        
        for it, (x, y) in pbar:
            # Move batch to device
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward pass
            logits, loss = self.model(x, y)
            losses.append(loss.item())
            
            # Update learning rate
            self.tokens += x.numel()
            lr = self.get_lr()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            # Update weights
            self.optimizer.step()
            
            # Update progress bar
            pbar.set_description(f"Training (loss: {loss.item():.4f}, lr: {lr:.6f})")
        
        return sum(losses) / len(losses)
    
    def validate(self):
        """
        Validate the model on the validation set.
        
        Returns:
            Average validation loss
        """
        if not self.val_loader:
            return None
        
        self.model.eval()
        losses = []
        
        with torch.no_grad():
            for x, y in tqdm(self.val_loader, desc="Validating"):
                # Move batch to device
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward pass
                logits, loss = self.model(x, y)
                losses.append(loss.item())
        
        return sum(losses) / len(losses)
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """
        Save a checkpoint of the model.
        
        Args:
            epoch: Current epoch number
            loss: Current loss value
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'tokens': self.tokens,
            'config': self.model.config,
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model if this is the best so far
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        
        Returns:
            Epoch number of the loaded checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.tokens = checkpoint.get('tokens', 0)
        epoch = checkpoint.get('epoch', 0)
        
        print(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch})")
        return epoch
    
    def train(self, num_epochs, save_every=1, eval_every=1):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train for
            save_every: Save a checkpoint every N epochs
            eval_every: Evaluate on validation set every N epochs
        """
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train for one epoch
            train_loss = self.train_epoch()
            print(f"Train loss: {train_loss:.4f}")
            
            # Evaluate on validation set
            if self.val_loader and (epoch + 1) % eval_every == 0:
                val_loss = self.validate()
                print(f"Validation loss: {val_loss:.4f}")
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
            else:
                is_best = False
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch + 1, train_loss, is_best)
        
        # Calculate total training time
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


def sample_from_model(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=40):
    """
    Sample text from the model given a prompt.
    
    Args:
        model: The GPT model
        tokenizer: Tokenizer to use for encoding/decoding
        prompt: Text prompt to start generation
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        top_k: Restrict sampling to top k most likely tokens
    
    Returns:
        Generated text
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Encode the prompt
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    
    # Decode the output
    output_text = tokenizer.decode(output_ids[0].tolist())
    return output_text


def train_model(args):
    """
    Train a model on the token dataset.
    
    Args:
        args: Command line arguments
    
    Returns:
        True if training was successful, False otherwise
    """
    import tiktoken
    
    # Load tokenizer using encoding from config
    encoding_name = get_config('tokenizer/encoding')
    if not encoding_name:
        print("Error: 'tokenizer/encoding' not found in configuration.")
        print("Please set this value in config/config.json under the 'tokenizer' section.")
        return False
    
    print(f"Using tokenizer encoding: {encoding_name}")
    tokenizer = tiktoken.get_encoding(encoding_name)
    
    # Get model size from config or args
    model_size = args.model_size if args.model_size else get_config('training/model_size')
    if not model_size:
        print("Error: Model size not specified in args or config.")
        print("Please set 'training/model_size' in config/config.json or use --model-size.")
        return False
    
    # Create a model for training
    if model_size == "micro":
        config = get_gpt2_micro_config()
    elif model_size == "mini":
        config = get_gpt2_mini_config()
    else:
        print(f"Unknown model size: {model_size}")
        return False
    
    config.vocab_size = tokenizer.n_vocab
    model = GPT(config)
    
    # Create dataset from token data
    tokens_file = args.input
    if not os.path.exists(tokens_file):
        print(f"Error: Tokens file not found at {tokens_file}")
        print("Please run 'python main.py tokenize_data' first.")
        return False
    
    dataset = TokenDataset(tokens_file, block_size=config.block_size)
    
    # Split into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Get batch size from config or args
    batch_size = args.batch_size if args.batch_size else get_config('training/batch_size')
    if not batch_size:
        print("Error: Batch size not specified in args or config.")
        print("Please set 'training/batch_size' in config/config.json or use --batch-size.")
        return False
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        checkpoint_dir=args.output_dir,
    )
    
    # Load checkpoint if specified
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            trainer.load_checkpoint(args.checkpoint)
        else:
            print(f"Warning: Checkpoint file {args.checkpoint} not found. Starting from scratch.")
    
    # Get epochs from config or args
    epochs = args.epochs if args.epochs else get_config('training/epochs')
    if not epochs:
        print("Error: Number of epochs not specified in args or config.")
        print("Please set 'training/epochs' in config/config.json or use --epochs.")
        return False
    
    # Get save_every from config or args
    save_every = args.save_every if args.save_every else get_config('training/save_every')
    if not save_every:
        print("Error: save_every not specified in args or config.")
        print("Please set 'training/save_every' in config/config.json or use --save-every.")
        return False
    
    # Get eval_every from config or args
    eval_every = args.eval_every if args.eval_every else get_config('training/eval_every')
    if not eval_every:
        print("Error: eval_every not specified in args or config.")
        print("Please set 'training/eval_every' in config/config.json or use --eval-every.")
        return False
    
    # Train for specified number of epochs
    trainer.train(num_epochs=epochs, save_every=save_every, eval_every=eval_every)
    
    # Sample from the model if requested
    if args.sample:
        # Get prompt from config or args
        prompt = args.prompt if args.prompt else get_config('training/prompt')
        if not prompt:
            print("Error: Prompt not specified in args or config.")
            print("Please set 'training/prompt' in config/config.json or use --prompt.")
            return False
        
        # Get max_tokens from config or args
        max_tokens = args.max_tokens if args.max_tokens else get_config('training/max_tokens')
        if not max_tokens:
            print("Error: max_tokens not specified in args or config.")
            print("Please set 'training/max_tokens' in config/config.json or use --max-tokens.")
            return False
        
        # Get temperature from config or args
        temperature = args.temperature if args.temperature else get_config('training/temperature')
        if not temperature:
            print("Error: temperature not specified in args or config.")
            print("Please set 'training/temperature' in config/config.json or use --temperature.")
            return False
        
        sample_text = sample_from_model(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        print("\nSample text:")
        print(sample_text)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a GPT model on token data")
    parser.add_argument("--input", type=str, default="data/tokens.parquet", help="Input tokens file")
    parser.add_argument("--output-dir", type=str, default="checkpoints/web", help="Directory to save checkpoints")
    parser.add_argument("--model-size", type=str, default="micro", choices=["micro", "mini"], help="Model size")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--eval-every", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--sample", action="store_true", help="Sample from the model after training")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt for sampling")
    parser.add_argument("--max-tokens", type=int, default=50, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    
    args = parser.parse_args()
    train_model(args)