#!/usr/bin/env python3
"""
Token dataset module for the GPT model.
This module provides a dataset class for loading and processing tokenized data.
"""

import torch
from torch.utils.data import Dataset
import pyarrow.parquet as pq


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