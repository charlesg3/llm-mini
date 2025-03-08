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
                         "                           Options: --force, --samples, --output, --cache-dir\n" + \
                         "  retrieve_assistant_data  Download OpenAssistant Conversations Dataset\n" + \
                         "                           Options: --force, --output-dir, --lang\n" + \
                         "  tokenize_data            Tokenize text data using tiktoken\n" + \
                         "                           Options: --input, --output, --encoding"
    
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
    elif args.command is None:
        parser.print_help()
        sys.exit(1)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)

if __name__ == "__main__":
    main()