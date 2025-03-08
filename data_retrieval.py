#!/usr/bin/env python3
"""
Module for downloading webpages from the Hugging Face fineweb dataset.
This script downloads 200 webpages and stores them in data/text.txt.
It caches the parquet files in the data directory to avoid re-downloading.
"""

import os
import json
import requests
import tempfile
from tqdm import tqdm

# Global base URL for the fineweb dataset
BASE_URL = "https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/data/CC-MAIN-2013-20"

def download_parquet(parquet_file, cache_dir):
    """
    Download a parquet file if it doesn't exist in the cache directory.
    
    Args:
        parquet_file (str): Name of the parquet file to download
        cache_dir (str): Directory to cache downloaded parquet files
    
    Returns:
        str: Path to the cached parquet file, or None if download failed
    """
    # Path to cached parquet file
    cached_file_path = os.path.join(cache_dir, parquet_file)
    
    # Check if the file is already cached
    if os.path.exists(cached_file_path):
        print(f"Using cached {parquet_file}")
        return cached_file_path
    
    print(f"Downloading {parquet_file}...")
    url = f"{BASE_URL}/{parquet_file}"
    
    try:
        # Download the parquet file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        with open(cached_file_path, 'wb') as file:
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading") as progress_bar:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
        
        return cached_file_path
    
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {parquet_file}: {e}")
        return None

def download_fineweb_data(num_samples=200, output_file="data/text.txt", cache_dir="data/parquet_cache", force=False):
    """
    Download samples from the Hugging Face fineweb dataset.
    
    Args:
        num_samples (int): Number of webpages to download
        output_file (str): Path to the output file
        cache_dir (str): Directory to cache downloaded parquet files
        force (bool): If True, overwrite existing output file
    
    Returns:
        int: Number of successfully downloaded samples
    """
    # Check if output file already exists
    if os.path.exists(output_file) and not force:
        print(f"Output file {output_file} already exists. Use force=True to overwrite.")
        return 0
    # List of parquet files to try with the correct format "000_00000.parquet"
    parquet_files = [
        "000_00000.parquet", "000_00001.parquet", "000_00002.parquet", "000_00003.parquet", "000_00004.parquet",
        "000_00005.parquet", "000_00006.parquet", "000_00007.parquet", "000_00008.parquet", "000_00009.parquet",
        "000_00010.parquet", "000_00011.parquet", "000_00012.parquet", "000_00013.parquet", "000_00014.parquet",
        "000_00015.parquet", "000_00016.parquet", "000_00017.parquet", "000_00018.parquet", "000_00019.parquet"
    ]
    
    print(f"Downloading up to {num_samples} samples from the fineweb dataset...")
    
    try:
        # Make sure pandas and pyarrow are installed
        try:
            import pandas as pd
        except ImportError:
            print("Error: pandas and pyarrow are required. Install them with 'pip install pandas pyarrow'")
            return 0
        
        # Create output and cache directories if they don't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for parquet_file in parquet_files:
                if count >= num_samples:
                    break
                
                # Download or get cached parquet file
                cached_file_path = download_parquet(parquet_file, cache_dir)
                if not cached_file_path:
                    continue
                
                try:
                    # Read the parquet file
                    print(f"Processing {parquet_file}...")
                    df = pd.read_parquet(cached_file_path)
                    
                    # Get samples from this file
                    samples_needed = min(num_samples - count, len(df))
                    for i in tqdm(range(samples_needed), desc="Extracting webpages"):
                        if 'text' in df.columns:
                            text = df.iloc[i]['text']
                        elif 'content' in df.columns:
                            text = df.iloc[i]['content']
                        elif 'raw_content' in df.columns:
                            text = df.iloc[i]['raw_content']
                        else:
                            # Try to find a column that might contain text
                            text = None
                            for col in df.columns:
                                val = df.iloc[i][col]
                                if isinstance(val, str) and len(val) > 100:
                                    text = val
                                    break
                            
                            if text is None:
                                print(f"Warning: Could not find text content in row {i}")
                                print(f"Available columns: {df.columns.tolist()}")
                                continue
                        
                        f.write(f"{text}\n\n")
                        count += 1
                        
                        if count >= num_samples:
                            break
                
                except Exception as e:
                    print(f"Error processing {parquet_file}: {e}")
                    continue
        
        print(f"Successfully downloaded {count} webpages to {output_file}")
        return count
    
    except Exception as e:
        print(f"Error: {e}")
        return 0

if __name__ == "__main__":
    print("This module should be imported and used through main.py")
    print("Run 'python3 main.py retrieve_data --help' for usage information")
