#!/usr/bin/env python3
"""
Main entry point for the LLM mini project.
This script provides a command-line interface for various data operations.
"""

import argparse
import sys
from data_retrieval import download_fineweb_data

def retrieve_data(args):
    """
    Handle the retrieve_data command by calling the download_fineweb_data function.
    
    Args:
        args: Command line arguments
    """
    return download_fineweb_data(
        num_samples=args.samples,
        output_file=args.output,
        cache_dir=args.cache_dir,
        force=args.force
    )

def main():
    """
    Main entry point for the command-line interface.
    """
    # Create the top-level parser
    parser = argparse.ArgumentParser(description="LLM Mini Project CLI")
    
    # Add description of available commands
    parser.description += "\n\nAvailable commands:\n" + \
                         "  retrieve_data    Download data from Hugging Face fineweb dataset\n" + \
                         "                   Options: --force, --samples, --output, --cache-dir"
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute", metavar="command")
    
    # Create the parser for the "retrieve_data" command
    retrieve_parser = subparsers.add_parser("retrieve_data", 
                                           help="Download data from Hugging Face fineweb dataset",
                                           description="Download data from Hugging Face fineweb dataset.\n\n" +
                                                      "Optional arguments:\n" +
                                                      "  --force       Force overwrite of existing output file\n" +
                                                      "  --samples     Number of samples to download (default: 200)\n" +
                                                      "  --output      Output file path (default: data/text.txt)\n" +
                                                      "  --cache-dir   Directory to cache parquet files (default: data/parquet_cache)")
    
    retrieve_parser.add_argument("--force", action="store_true", help="Force overwrite of existing output file")
    retrieve_parser.add_argument("--samples", type=int, default=200, help="Number of samples to download")
    retrieve_parser.add_argument("--output", type=str, default="data/text.txt", help="Output file path")
    retrieve_parser.add_argument("--cache-dir", type=str, default="data/parquet_cache", help="Directory to cache parquet files")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate command
    if args.command == "retrieve_data":
        result = retrieve_data(args)
        sys.exit(0 if result > 0 else 1)
    elif args.command is None:
        parser.print_help()
        sys.exit(1)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)

if __name__ == "__main__":
    main()