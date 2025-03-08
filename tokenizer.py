#!/usr/bin/env python3
"""
Tokenizer module for the LLM mini project.
This script reads the training data, tokenizes it using tiktoken,
and stores the tokenized sequence in a parquet file.
"""

import os
import sys
from collections import Counter
import pandas as pd
import tiktoken

def tokenize(input_path="data/text.txt", output_path="data/tokens.parquet", encoding_name="cl100k_base"):
    """
    Read the data file, tokenize it using tiktoken, and store the tokenized sequence.
    
    Args:
        input_path (str): Path to the input data file
        output_path (str): Path to save the tokenized sequence
        encoding_name (str): Name of the tiktoken encoding to use
        
    Returns:
        tuple: (token_counts, encoding) - Counter of tokens and the tiktoken encoding
    """
    # Check if the file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Data file not found: {input_path}")
    
    # Read the file
    print(f"Reading data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Total characters: {len(text)}")
    
    # Get the tiktoken encoding
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception as e:
        print(f"Error loading tiktoken encoding '{encoding_name}': {e}")
        print("Available encodings:", tiktoken.list_encoding_names())
        raise
    
    # Tokenize the text
    print(f"Tokenizing text using {encoding_name} encoding...")
    tokens = encoding.encode(text)
    
    print(f"Total tokens: {len(tokens)}")
    
    # Count token frequencies
    token_counts = Counter(tokens)
    
    # Print the top 10 most frequent tokens
    print("\nTop 10 most frequent tokens:")
    print("-" * 60)
    print("| Token ID | Token Text                | Frequency |")
    print("-" * 60)
    
    for token_id, count in token_counts.most_common(10):
        # Get the token text and escape special characters
        token_bytes = encoding.decode_single_token_bytes(token_id)
        token_text = token_bytes.decode('utf-8', errors='replace')
        token_display = repr(token_text)[1:-1]  # Remove quotes from repr
        if len(token_display) > 20:
            token_display = token_display[:17] + "..."
        
        print(f"| {token_id:8d} | {token_display:24} | {count:9,d} |")
    
    print("-" * 60)
    print(f"Total unique tokens: {len(token_counts)}")
    
    # Create a DataFrame with the tokens
    tokens_df = pd.DataFrame({'token_id': tokens})
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save tokens to parquet file
    print(f"Saving {len(tokens)} tokens to {output_path}")
    tokens_df.to_parquet(output_path, index=False)
    
    # Save token frequencies to a separate file
    freq_path = os.path.join(os.path.dirname(output_path), 'token_frequencies.parquet')
    freq_df = pd.DataFrame({
        'token_id': list(token_counts.keys()),
        'frequency': list(token_counts.values())
    })
    print(f"Saving token frequencies ({len(token_counts)} entries) to {freq_path}")
    freq_df.to_parquet(freq_path, index=False)
    
    return token_counts, encoding

def main():
    """
    Main function to tokenize the data file.
    """
    try:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description="Tokenize text data for LLM training")
        parser.add_argument("--input", default="data/text.txt", help="Input text file path")
        parser.add_argument("--output", default="data/tokens.parquet", help="Output tokens parquet file path")
        parser.add_argument("--encoding", default="cl100k_base", help="Tiktoken encoding name")
        args = parser.parse_args()
        
        # Call the tokenize function
        tokenize(args.input, args.output, args.encoding)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()