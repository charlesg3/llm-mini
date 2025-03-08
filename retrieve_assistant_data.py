#!/usr/bin/env python3
"""
Module for downloading the OpenAssistant Conversations Dataset (OASST1) from Hugging Face.
This script downloads the dataset, filters for English content only, and stores it in data/assistant_data.
"""

import os
import json
import requests
import tempfile
from tqdm import tqdm
from config import get_config
from collections import defaultdict

def download_assistant_data(output_dir="data/assistant_data", force=False, lang="en"):
    """
    Download the OpenAssistant Conversations Dataset (OASST1) from Hugging Face.
    
    Args:
        output_dir (str): Directory to store the downloaded data
        force (bool): If True, overwrite existing files
        lang (str): Language code to filter for (default: "en" for English)
    
    Returns:
        bool: True if download was successful, False otherwise
    """
    # Get the dataset URL from config
    dataset_url = get_config('assistant_data/url')
    if not dataset_url:
        raise ValueError("Dataset URL not found in configuration. Please set 'assistant_data/url' in config/config.json")
    
    print(f"Downloading OpenAssistant Conversations Dataset from {dataset_url}")
    print(f"Filtering for language: {lang}")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Use the Hugging Face datasets library to download the dataset
        try:
            from datasets import load_dataset
        except ImportError:
            print("Error: The 'datasets' library is required. Install it with 'pip install datasets'")
            return False
        
        # Check if files already exist
        output_file = os.path.join(output_dir, f"oasst1_{lang}.json")
        if os.path.exists(output_file) and not force:
            print(f"Output file {output_file} already exists. Use force=True to overwrite.")
            return False
        
        print("Loading dataset from Hugging Face...")
        dataset = load_dataset("OpenAssistant/oasst1")
        
        # Print dataset information
        print("\nDataset Information:")
        print(f"Number of splits: {len(dataset)}")
        for split_name, split_data in dataset.items():
            print(f"  - {split_name}: {len(split_data)} examples")
        
        # Process and save each split
        for split_name, split_data in dataset.items():
            split_output_file = os.path.join(output_dir, f"oasst1_{lang}_{split_name}.json")
            print(f"\nProcessing {split_name} split...")
            
            # Filter for the specified language
            print(f"Filtering for {lang} language...")
            filtered_data = []
            for i, example in enumerate(tqdm(split_data, desc=f"Filtering {split_name}")):
                if example.get('lang') == lang:
                    filtered_data.append(example)
            
            print(f"Found {len(filtered_data)} examples in {lang} (out of {len(split_data)} total)")
            
            # Save to JSON file
            with open(split_output_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, ensure_ascii=False, indent=2)
            
            print(f"Saved {len(filtered_data)} examples to {split_output_file}")
        
        # Create a simple text file with human/assistant pairs for training
        train_text_file = os.path.join(output_dir, f"oasst1_{lang}_train.txt")
        print(f"\nCreating training text file with {lang} human/assistant pairs...")
        
        # Get the training data
        train_data = dataset.get('train', None)
        if train_data is None:
            print("Warning: No training split found in the dataset")
            return True
        
        # Process conversations to extract human/assistant pairs
        with open(train_text_file, 'w', encoding='utf-8') as f:
            # Group by conversation ID
            conversations = {}
            for example in tqdm(train_data, desc="Grouping conversations"):
                # Skip non-English content
                if example.get('lang') != lang:
                    continue
                    
                conv_id = example.get('conversation_id', '')
                if conv_id not in conversations:
                    conversations[conv_id] = []
                conversations[conv_id].append(example)
            
            # Sort messages within each conversation
            for conv_id, messages in tqdm(conversations.items(), desc="Processing conversations"):
                # Sort by message_id (which indicates the order)
                messages.sort(key=lambda x: x.get('message_id', ''))
                
                # Write the conversation to the file
                f.write(f"# Conversation: {conv_id}\n")
                for msg in messages:
                    role = "Human: " if msg.get('role', '') == 'prompter' else "Assistant: "
                    text = msg.get('text', '').replace('\n', ' ').strip()
                    f.write(f"{role}{text}\n")
                f.write("\n\n")
        
        print(f"Created training text file with {len(conversations)} conversations at {train_text_file}")
        
        # Create the requested JSON format (list of lists of maps)
        # But this time properly link messages using parent_id and message_id
        assistant_data_file = os.path.join(output_dir, "assistant_data.json")
        print(f"\nCreating assistant data in the requested format at {assistant_data_file}...")
        
        # Combine train and validation data for processing
        all_data = []
        for split_name, split_data in dataset.items():
            for example in split_data:
                if example.get('lang') == lang:
                    all_data.append(example)
        
        print(f"Processing {len(all_data)} messages to build conversation threads...")
        
        # Create a dictionary to store messages by their message_id
        messages_by_id = {}
        # Create a dictionary to store child messages by their parent_id
        children_by_parent = defaultdict(list)
        # Keep track of root messages (those without a parent)
        root_messages = []
        
        # First pass: organize messages by ID and parent-child relationships
        for msg in tqdm(all_data, desc="Organizing messages"):
            message_id = msg.get('message_id')
            parent_id = msg.get('parent_id')
            
            # Store message by its ID
            messages_by_id[message_id] = msg
            
            # If it has a parent, add it to the children list of that parent
            if parent_id:
                children_by_parent[parent_id].append(message_id)
            else:
                # This is a root message (start of a conversation)
                root_messages.append(message_id)
        
        print(f"Found {len(root_messages)} conversation threads")
        
        # Function to recursively build a conversation thread
        def build_conversation_thread(message_id, thread=None):
            if thread is None:
                thread = []
            
            # Get the message
            msg = messages_by_id.get(message_id)
            if not msg:
                return thread
            
            # Add this message to the thread
            role = "User" if msg.get('role', '') == 'prompter' else "Assistant"
            text = msg.get('text', '').strip()
            thread.append({role: text})
            
            # Process children (should be just one in a linear conversation)
            for child_id in children_by_parent.get(message_id, []):
                build_conversation_thread(child_id, thread)
            
            return thread
        
        # Build all conversation threads
        formatted_conversations = []
        for root_id in tqdm(root_messages, desc="Building conversation threads"):
            conversation = build_conversation_thread(root_id)
            if len(conversation) > 1:  # Only include conversations with at least 2 messages
                formatted_conversations.append(conversation)
        
        print(f"Created {len(formatted_conversations)} valid conversation threads")
        
        # Save the formatted conversations to JSON
        with open(assistant_data_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_conversations, f, ensure_ascii=False, indent=2)
        
        print(f"Created assistant data file with {len(formatted_conversations)} conversations at {assistant_data_file}")
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download OpenAssistant Conversations Dataset")
    parser.add_argument("--output-dir", type=str, default="data/assistant_data", 
                        help="Directory to store the downloaded data")
    parser.add_argument("--force", action="store_true", 
                        help="Force overwrite of existing files")
    parser.add_argument("--lang", type=str, default="en",
                        help="Language code to filter for (default: 'en' for English)")
    
    args = parser.parse_args()
    
    success = download_assistant_data(args.output_dir, args.force, args.lang)
    if success:
        print("Download completed successfully!")
    else:
        print("Download failed or was skipped.")