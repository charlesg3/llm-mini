#!/usr/bin/env python3
"""
Configuration module for the LLM mini project.
This module provides functions to access configuration values from
either environment variables or the config/config.json file.
"""

import os
import json
from functools import lru_cache

@lru_cache(maxsize=32)
def _load_config_file():
    """
    Load configuration from config/config.json.
    Uses LRU cache to avoid reading the file multiple times.
    
    Returns:
        dict: Configuration dictionary
    """
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'config.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Warning: Could not load config file: {e}")
        return {}

def _get_from_dict(data_dict, key_path):
    """
    Get a value from a nested dictionary using a key path.
    
    Args:
        data_dict (dict): Dictionary to search in
        key_path (list): List of keys defining the path to the value
        
    Returns:
        The value at the specified path, or None if not found
    """
    current = data_dict
    for key in key_path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current

def get_config(key_path, default=None):
    """
    Get a configuration value from environment variables or config file.
    
    First checks if the key exists as an environment variable (with '/' replaced by '_' and uppercase).
    If not found, looks for the key in the config/config.json file.
    
    Args:
        key_path (str): Path to the configuration value, using '/' as separator
                        (e.g., "data_retrieval/base_url")
        default: Default value to return if the key is not found
        
    Returns:
        The configuration value, or the default if not found
    """
    # Check environment variables first (convert path to ENV_VAR format)
    env_key = key_path.replace('/', '_').upper()
    env_value = os.environ.get(env_key)
    if env_value is not None:
        return env_value
    
    # If not in environment, check config file
    key_parts = key_path.split('/')
    config = _load_config_file()
    value = _get_from_dict(config, key_parts)
    
    # Return the value or default
    return value if value is not None else default

if __name__ == "__main__":
    # Example usage
    print("Example usage of get_config:")
    print(f"data_retrieval/base_url: {get_config('data_retrieval/base_url')}")
    print(f"non_existent/key (with default): {get_config('non_existent/key', 'default_value')}")
