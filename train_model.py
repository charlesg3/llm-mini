#!/usr/bin/env python3
"""
Training module for the GPT model.
This module provides functionality for training the GPT model on token data.
"""

import os
import torch
from model import GPT, get_gpt2_micro_config, get_gpt2_mini_config
from config import get_config
from token_dataset import TokenDataset
from training import Trainer, sample_from_model


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
    
    # Check for available GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 0:
        print(f"Found {num_gpus} GPUs available for training")
        # Adjust batch size based on number of GPUs if multi-GPU is enabled
        if args.multi_gpu and num_gpus > 1:
            # Set batch size to 4 per GPU to avoid OOM errors
            batch_size = 4 * num_gpus
            print(f"Using batch size of {batch_size} (4 per GPU across {num_gpus} GPUs)")
    else:
        print("No GPUs found, using CPU for training")
    
    # Create trainer with reduced batch size to avoid potential memory issues
    batch_size = min(batch_size, 4)  # Limit batch size to avoid OOM
    print(f"Using batch size of {batch_size} for training")
    
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        checkpoint_dir=args.output_dir,
        use_multi_gpu=args.multi_gpu,
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
    parser.add_argument("--multi-gpu", action="store_true", help="Use multiple GPUs for training if available")
    
    args = parser.parse_args()
    train_model(args)