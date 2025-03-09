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
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
from model import GPT, get_gpt2_micro_config, get_gpt2_mini_config
from config import get_config
from token_dataset import TokenDataset


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
        use_multi_gpu=True,
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
            use_multi_gpu: Whether to use multiple GPUs if available
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
        self.use_multi_gpu = use_multi_gpu
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Check for multiple GPUs and use DataParallel if available and requested
        if self.use_multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training")
            self.model = torch.nn.DataParallel(model)
        else:
            self.model = model
            
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
        
        # Create data loaders with multi-process loading
        # Use 4 workers by default, can be adjusted based on CPU cores
        num_workers = 4
        print(f"Using {num_workers} worker processes for data loading")
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=True,  # Keep workers alive between iterations
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=num_workers,
                persistent_workers=True,
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
        total_batches = len(self.train_loader)
        pbar = tqdm(enumerate(self.train_loader), total=total_batches, desc="Training")
        
        for it, (x, y) in pbar:
            # Add debug print to track progress
            if it == 0:
                print(f"Starting first batch processing at {time.strftime('%H:%M:%S')}")
                
            # Move batch to device with non-blocking transfer
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            
            # Forward pass
            logits, loss = self.model(x, y)
            # Handle multi-GPU case where loss is a tensor array
            if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                loss = loss.mean()
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
            
            # Calculate running average loss
            avg_loss = sum(losses[-100:]) / min(len(losses), 100)  # Moving average of last 100 batches
            
            # Update progress bar with detailed information
            progress_percent = (it + 1) / total_batches * 100
            pbar.set_description(f"Training [{it+1}/{total_batches} ({progress_percent:.1f}%)] - loss: {loss.item():.4f}, avg_loss: {avg_loss:.4f}, lr: {lr:.6f}")
            
            # Print progress every 10% of batches or at least every 10 batches
            if (it + 1) % max(1, min(total_batches // 10, 10)) == 0:
                print(f"Progress: {progress_percent:.1f}% - Batch {it+1}/{total_batches}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}, LR: {lr:.6f}")
            
            # Add debug print for first few batches
            if it < 3:
                print(f"Completed batch {it+1} at {time.strftime('%H:%M:%S')}")
        
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
        total_batches = len(self.val_loader)
        
        with torch.no_grad():
            pbar = tqdm(enumerate(self.val_loader), total=total_batches, desc="Validating")
            
            for i, (x, y) in pbar:
                # Add debug print for first batch
                if i == 0:
                    print(f"Starting first validation batch at {time.strftime('%H:%M:%S')}")
                    
                # Move batch to device with non-blocking transfer
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                
                # Forward pass
                logits, loss = self.model(x, y)
                # Handle multi-GPU case where loss is a tensor array
                if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                    loss = loss.mean()
                losses.append(loss.item())
                
                # Update progress bar
                avg_loss = sum(losses) / len(losses)
                progress_percent = (i + 1) / total_batches * 100
                pbar.set_description(f"Validating [{i+1}/{total_batches} ({progress_percent:.1f}%)] - loss: {loss.item():.4f}, avg_loss: {avg_loss:.4f}")
                
                # Print progress more frequently
                if (i + 1) % max(1, min(total_batches // 5, 10)) == 0:
                    print(f"Validation progress: {progress_percent:.1f}% - Batch {i+1}/{total_batches}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")
                
                # Add debug print for first few batches
                if i < 3:
                    print(f"Completed validation batch {i+1} at {time.strftime('%H:%M:%S')}")
        
        return sum(losses) / len(losses)
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """
        Save a checkpoint of the model.
        
        Args:
            epoch: Current epoch number
            loss: Current loss value
            is_best: Whether this is the best model so far
        """
        # If using DataParallel, save the underlying model
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'tokens': self.tokens,
            'config': model_to_save.config,
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
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch+1}/{num_epochs} starting at {time.strftime('%H:%M:%S')}")
            
            # Train for one epoch
            print(f"Starting training epoch {epoch+1} at {time.strftime('%H:%M:%S')}")
            train_loss = self.train_epoch()
            epoch_duration = time.time() - epoch_start_time
            print(f"Train loss: {train_loss:.4f} (took {epoch_duration:.2f}s)")
            
            # Evaluate on validation set
            if self.val_loader and (epoch + 1) % eval_every == 0:
                val_start_time = time.time()
                print(f"Starting validation at {time.strftime('%H:%M:%S')}")
                val_loss = self.validate()
                val_duration = time.time() - val_start_time
                print(f"Validation loss: {val_loss:.4f} (took {val_duration:.2f}s)")
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    print(f"New best validation loss: {val_loss:.4f}")
            else:
                is_best = False
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_start_time = time.time()
                print(f"Saving checkpoint at {time.strftime('%H:%M:%S')}")
                self.save_checkpoint(epoch + 1, train_loss, is_best)
                checkpoint_duration = time.time() - checkpoint_start_time
                print(f"Checkpoint saved (took {checkpoint_duration:.2f}s)")
            
            # Calculate ETA
            elapsed_time = time.time() - start_time
            avg_epoch_time = elapsed_time / (epoch + 1)
            remaining_epochs = num_epochs - (epoch + 1)
            eta_seconds = avg_epoch_time * remaining_epochs
            eta_minutes = eta_seconds / 60
            eta_hours = eta_minutes / 60
            
            if remaining_epochs > 0:
                if eta_hours >= 1:
                    print(f"Progress: {epoch+1}/{num_epochs} epochs ({(epoch+1)/num_epochs*100:.1f}%) - ETA: {eta_hours:.1f} hours")
                elif eta_minutes >= 1:
                    print(f"Progress: {epoch+1}/{num_epochs} epochs ({(epoch+1)/num_epochs*100:.1f}%) - ETA: {eta_minutes:.1f} minutes")
                else:
                    print(f"Progress: {epoch+1}/{num_epochs} epochs ({(epoch+1)/num_epochs*100:.1f}%) - ETA: {eta_seconds:.1f} seconds")
        
        # Calculate total training time
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
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
    # If using DataParallel, use the underlying model for generation
    if hasattr(model, 'module'):
        generation_model = model.module
    else:
        generation_model = model
        
    generation_model.eval()
    device = next(generation_model.parameters()).device
    
    # Encode the prompt
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    
    # Generate
    with torch.no_grad():
        output_ids = generation_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    
    # Decode the output
    output_text = tokenizer.decode(output_ids[0].tolist())
    return output_text