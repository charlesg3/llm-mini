#!/usr/bin/env python3
"""
Fine-tuning module for the GPT model on assistant conversation data.
This module provides functionality for training and evaluating the GPT model on assistant conversations.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import GPT, get_gpt2_mini_config
from config import get_config
from training import Trainer, sample_from_model


class AssistantDataset(Dataset):
    """
    Dataset for training the GPT model on assistant conversations.
    """
    def __init__(self, json_file, tokenizer, block_size):
        """
        Initialize the dataset.
        
        Args:
            json_file: Path to the JSON file containing assistant conversations
            tokenizer: Tokenizer to use for encoding the text
            block_size: Size of the context window for the model
        """
        import json
        
        self.block_size = block_size
        self.tokenizer = tokenizer
        
        # Load conversations from JSON file
        print(f"Loading conversations from {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        # Process conversations into token sequences
        self.examples = []
        for conversation in tqdm(conversations, desc="Processing conversations"):
            # Format the conversation as a single text
            formatted_text = ""
            for message in conversation:
                # Each message is a dict with a single key (role) and value (text)
                role = list(message.keys())[0]
                text = message[role]
                formatted_text += f"{role}: {text}\n\n"
            
            # Tokenize the text
            tokens = self.tokenizer.encode(formatted_text)
            
            # Create examples of appropriate length
            for i in range(0, len(tokens) - block_size, block_size // 2):
                if i + block_size + 1 <= len(tokens):
                    self.examples.append(tokens[i:i + block_size + 1])
        
        print(f"Created {len(self.examples)} examples from {len(conversations)} conversations")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        chunk = self.examples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def train_assistant_model(args):
    """
    Train a model on the assistant conversation data.
    
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
        encoding_name = "cl100k_base"  # Default fallback
        print(f"Using default encoding: {encoding_name}")
    
    print(f"Using tokenizer encoding: {encoding_name}")
    tokenizer = tiktoken.get_encoding(encoding_name)
    
    # Create a model for assistant training
    config = get_gpt2_mini_config()
    config.vocab_size = tokenizer.n_vocab
    model = GPT(config)
    
    # Create dataset from assistant data
    assistant_data_file = args.input
    if not os.path.exists(assistant_data_file):
        print(f"Error: Assistant data file not found at {assistant_data_file}")
        print("Please run 'python main.py retrieve_assistant_data' first.")
        return False
    
    dataset = AssistantDataset(
        json_file=assistant_data_file,
        tokenizer=tokenizer,
        block_size=config.block_size
    )
    
    # Split into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        checkpoint_dir=args.output_dir,
        use_multi_gpu=args.multi_gpu if hasattr(args, 'multi_gpu') else False,
    )
    
    # Load checkpoint if specified
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            trainer.load_checkpoint(args.checkpoint)
        else:
            print(f"Warning: Checkpoint file {args.checkpoint} not found. Starting from scratch.")
    
    # Train for multiple epochs
    trainer.train(num_epochs=args.epochs, save_every=args.save_every, eval_every=args.eval_every)
    
    # Sample from the model
    if args.sample:
        sample_text = sample_from_model(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print("\nSample conversation:")
        print(sample_text)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune a GPT model on assistant conversation data")
    parser.add_argument("--input", type=str, default="data/assistant_data/assistant_data.json", help="Input assistant data file")
    parser.add_argument("--output-dir", type=str, default="checkpoints/assistant", help="Directory to save checkpoints")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--eval-every", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--sample", action="store_true", help="Sample from the model after training")
    parser.add_argument("--prompt", type=str, default="User: How do I implement a transformer model in PyTorch?\n\nAssistant:", help="Prompt for sampling")
    parser.add_argument("--max-tokens", type=int, default=200, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--multi-gpu", action="store_true", help="Use multiple GPUs for training if available")
    
    args = parser.parse_args()
    train_assistant_model(args)