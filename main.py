#!/usr/bin/env python3
"""
Main entry point for the LLM mini project.
This script provides a command-line interface for various data operations.
"""

import argparse
import sys
import time
from data_retrieval import download_fineweb_data, DEFAULT_NUM_SAMPLES
from retrieve_assistant_data import download_assistant_data
from tokenizer import tokenize
from config import get_config
from train_model import train_model
from fine_tune_assistant import train_assistant_model

def retrieve_data(args):
    """
    Handle the retrieve_data command by calling the download_fineweb_data function.
    
    Args:
        args: Command line arguments
    """
    start_time = time.time()
    print(f"Starting data retrieval at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    result = download_fineweb_data(
        num_samples=args.samples,
        output_file=args.output,
        cache_dir=args.cache_dir,
        force=args.force
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Data retrieval completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    return result

def retrieve_assistant_data_cmd(args):
    """
    Handle the retrieve_assistant_data command by calling the download_assistant_data function.
    
    Args:
        args: Command line arguments
    """
    start_time = time.time()
    print(f"Starting assistant data retrieval at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    result = download_assistant_data(
        output_dir=args.output_dir,
        force=args.force,
        lang=args.lang
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Assistant data retrieval completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    return result

def tokenize_data(args):
    """
    Handle the tokenize_data command by calling the tokenize function.
    
    Args:
        args: Command line arguments
    """
    start_time = time.time()
    print(f"Starting tokenization at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    token_counts, _ = tokenize(
        input_path=args.input,
        output_path=args.output,
        encoding_name=args.encoding
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tokenization completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    return len(token_counts) > 0

def train_cmd(args):
    """
    Handle the train command by calling the train_model function.
    
    Args:
        args: Command line arguments
    """
    start_time = time.time()
    print(f"Starting model training at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    result = train_model(args)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Model training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    return result

def finetune_cmd(args):
    """
    Handle the finetune command by calling the train_assistant_model function.
    
    Args:
        args: Command line arguments
    """
    start_time = time.time()
    print(f"Starting assistant model fine-tuning at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    result = train_assistant_model(args)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Assistant model fine-tuning completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    return result

def main():
    """
    Main entry point for the command-line interface.
    """
    # Get encoding from config without providing a default
    default_encoding = get_config('tokenizer/encoding')
    if default_encoding is None:
        print("Error: 'tokenizer/encoding' not found in configuration.")
        print("Please set this value in config/config.json under the 'tokenizer' section.")
        sys.exit(1)
    
    # Create the top-level parser
    parser = argparse.ArgumentParser(description="LLM Mini Project CLI")
    
    # Add description of available commands
    parser.description += "\n\nAvailable commands:\n" + \
                         "  retrieve_data            Download data from Hugging Face fineweb dataset\n" + \
                         "  retrieve_assistant_data  Download OpenAssistant Conversations Dataset\n" + \
                         "  tokenize_data            Tokenize text data using tiktoken\n" + \
                         "  train                    Train a GPT model on tokenized data\n" + \
                         "  finetune                 Fine-tune a GPT model on assistant conversations"
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute", metavar="command")
    
    # Create the parser for the "retrieve_data" command
    retrieve_parser = subparsers.add_parser("retrieve_data", 
                                           help="Download data from Hugging Face fineweb dataset",
                                           description="Download data from Hugging Face fineweb dataset.\n\n" +
                                                      "Optional arguments:\n" +
                                                      "  --force       Force overwrite of existing output file\n" +
                                                      f"  --samples     Number of samples to download (default: {DEFAULT_NUM_SAMPLES})\n" +
                                                      "  --output      Output file path (default: data/text.txt)\n" +
                                                      "  --cache-dir   Directory to cache parquet files (default: data/parquet_cache)")
    
    retrieve_parser.add_argument("--force", action="store_true", help="Force overwrite of existing output file")
    retrieve_parser.add_argument("--samples", type=int, default=DEFAULT_NUM_SAMPLES, help=f"Number of samples to download (default: {DEFAULT_NUM_SAMPLES})")
    retrieve_parser.add_argument("--output", type=str, default="data/text.txt", help="Output file path")
    retrieve_parser.add_argument("--cache-dir", type=str, default="data/parquet_cache", help="Directory to cache parquet files")
    
    # Create the parser for the "retrieve_assistant_data" command
    assistant_parser = subparsers.add_parser("retrieve_assistant_data",
                                           help="Download OpenAssistant Conversations Dataset",
                                           description="Download OpenAssistant Conversations Dataset.\n\n" +
                                                      "Optional arguments:\n" +
                                                      "  --force       Force overwrite of existing files\n" +
                                                      "  --output-dir  Directory to store the downloaded data (default: data/assistant_data)\n" +
                                                      "  --lang        Language code to filter for (default: 'en' for English)")
    
    assistant_parser.add_argument("--force", action="store_true", help="Force overwrite of existing files")
    assistant_parser.add_argument("--output-dir", type=str, default="data/assistant_data", help="Directory to store the downloaded data")
    assistant_parser.add_argument("--lang", type=str, default="en", help="Language code to filter for (default: 'en' for English)")
    
    # Create the parser for the "tokenize_data" command
    tokenize_parser = subparsers.add_parser("tokenize_data",
                                           help="Tokenize text data using tiktoken",
                                           description="Tokenize text data using tiktoken.\n\n" +
                                                      "Optional arguments:\n" +
                                                      "  --input       Input text file path (default: data/text.txt)\n" +
                                                      "  --output      Output tokens parquet file path (default: data/tokens.parquet)\n" +
                                                      f"  --encoding    Tiktoken encoding name (default: {default_encoding})")
    
    tokenize_parser.add_argument("--input", type=str, default="data/text.txt", help="Input text file path")
    tokenize_parser.add_argument("--output", type=str, default="data/tokens.parquet", help="Output tokens parquet file path")
    tokenize_parser.add_argument("--encoding", type=str, default=default_encoding, help=f"Tiktoken encoding name (default: {default_encoding})")
    
    # Create the parser for the "train" command
    train_parser = subparsers.add_parser("train",
                                        help="Train a GPT model on tokenized data",
                                        description="Train a GPT model on tokenized data.\n\n" +
                                                   "Optional arguments:\n" +
                                                   "  --input        Input tokens file (default: data/tokens.parquet)\n" +
                                                   "  --output-dir   Directory to save checkpoints (default: checkpoints/web)\n" +
                                                   "  --model-size   Model size (micro or mini, default: from config)\n" +
                                                   "  --batch-size   Batch size for training (default: from config)\n" +
                                                   "  --epochs       Number of epochs to train (default: from config)\n" +
                                                   "  --checkpoint   Path to checkpoint to resume from")
    
    train_parser.add_argument("--input", type=str, default="data/tokens.parquet", help="Input tokens file")
    train_parser.add_argument("--output-dir", type=str, default="checkpoints/web", help="Directory to save checkpoints")
    train_parser.add_argument("--model-size", type=str, choices=["micro", "mini"], help="Model size (default: from config)")
    train_parser.add_argument("--batch-size", type=int, help="Batch size (default: from config)")
    train_parser.add_argument("--epochs", type=int, help="Number of epochs (default: from config)")
    train_parser.add_argument("--save-every", type=int, help="Save checkpoint every N epochs (default: from config)")
    train_parser.add_argument("--eval-every", type=int, help="Evaluate every N epochs (default: from config)")
    train_parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    train_parser.add_argument("--sample", action="store_true", help="Sample from the model after training")
    train_parser.add_argument("--prompt", type=str, help="Prompt for sampling (default: from config)")
    train_parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate (default: from config)")
    train_parser.add_argument("--temperature", type=float, help="Sampling temperature (default: from config)")
    train_parser.add_argument("--multi-gpu", action="store_true", help="Use multiple GPUs for training if available")
    
    # Create the parser for the "finetune" command
    finetune_parser = subparsers.add_parser("finetune",
                                           help="Fine-tune a GPT model on assistant conversations",
                                           description="Fine-tune a GPT model on assistant conversations.\n\n" +
                                                      "Optional arguments:\n" +
                                                      "  --input        Input assistant data file (default: data/assistant_data/assistant_data.json)\n" +
                                                      "  --output-dir   Directory to save checkpoints (default: checkpoints/assistant)\n" +
                                                      "  --batch-size   Batch size for training (default: 4)\n" +
                                                      "  --epochs       Number of epochs to train (default: 5)\n" +
                                                      "  --checkpoint   Path to checkpoint to resume from")
    
    finetune_parser.add_argument("--input", type=str, default="data/assistant_data/assistant_data.json", help="Input assistant data file")
    finetune_parser.add_argument("--output-dir", type=str, default="checkpoints/assistant", help="Directory to save checkpoints")
    finetune_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    finetune_parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    finetune_parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs")
    finetune_parser.add_argument("--eval-every", type=int, default=1, help="Evaluate every N epochs")
    finetune_parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    finetune_parser.add_argument("--sample", action="store_true", help="Sample from the model after training")
    finetune_parser.add_argument("--prompt", type=str, default="User: How do I implement a transformer model in PyTorch?\n\nAssistant:", help="Prompt for sampling")
    finetune_parser.add_argument("--max-tokens", type=int, default=200, help="Maximum tokens to generate")
    finetune_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    finetune_parser.add_argument("--multi-gpu", action="store_true", help="Use multiple GPUs for training if available")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate command
    if args.command == "retrieve_data":
        result = retrieve_data(args)
        sys.exit(0 if result > 0 else 1)
    elif args.command == "retrieve_assistant_data":
        result = retrieve_assistant_data_cmd(args)
        sys.exit(0 if result else 1)
    elif args.command == "tokenize_data":
        result = tokenize_data(args)
        sys.exit(0 if result else 1)
    elif args.command == "train":
        result = train_cmd(args)
        sys.exit(0 if result else 1)
    elif args.command == "finetune":
        result = finetune_cmd(args)
        sys.exit(0 if result else 1)
    elif args.command is None:
        parser.print_help()
        sys.exit(1)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)

if __name__ == "__main__":
    main()