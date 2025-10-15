#!/usr/bin/env python3
"""
Debug script to check logprobs structure
"""

import sys
import os
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

# Set up cartridges environment variables BEFORE importing
REPO_ROOT = Path(__file__).parent
CARTRIDGES_DIR = REPO_ROOT / "third_party" / "cartridges"

# Cartridges requires these environment variables
os.environ["CARTRIDGES_DIR"] = str(CARTRIDGES_DIR)
if "CARTRIDGES_OUTPUT_DIR" not in os.environ:
    os.environ["CARTRIDGES_OUTPUT_DIR"] = str(REPO_ROOT / "outputs")

# Add project paths
sys.path.insert(0, str(CARTRIDGES_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.memory.distillation.distill_into_cartridge import create_conversation_from_row


def debug_logprobs():
    """Debug the logprobs structure"""
    print("Debugging logprobs structure...")
    
    # Load dataset and tokenizer
    ds = load_dataset("stalaei/distillation-dataset-test", split="train")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    
    # Get first sample
    row = ds[0]
    
    # Test different min_prob_mass values
    for min_prob_mass in [0.5, 0.8, 0.9, 0.95, 0.99]:
        print(f"\n=== Testing min_prob_mass = {min_prob_mass} ===")
        conv = create_conversation_from_row(row, tokenizer, min_prob_mass=min_prob_mass)
        
        assistant_messages_with_logprobs = 0
        for msg in conv.messages:
            if msg.role == "assistant" and msg.top_logprobs:
                assistant_messages_with_logprobs += 1
                print(f"  Assistant message has logprobs: {msg.top_logprobs.token_id.shape}")
        
        print(f"  Assistant messages with logprobs: {assistant_messages_with_logprobs}")
    
    print(f"Conversation has {len(conv.messages)} messages")
    
    for i, msg in enumerate(conv.messages):
        print(f"\nMessage {i}: role={msg.role}")
        print(f"  Content length: {len(msg.content)}")
        print(f"  Token IDs: {len(msg.token_ids) if msg.token_ids else 0}")
        
        if msg.top_logprobs:
            print(f"  Top logprobs type: {type(msg.top_logprobs)}")
            print(f"  Top logprobs attributes: {dir(msg.top_logprobs)}")
            print(f"  token_idx shape: {msg.top_logprobs.token_idx.shape}")
            print(f"  token_id shape: {msg.top_logprobs.token_id.shape}")
            print(f"  logprobs shape: {msg.top_logprobs.logprobs.shape}")
            print(f"  shape: {msg.top_logprobs.shape}")
        else:
            print(f"  Top logprobs: None")


if __name__ == "__main__":
    debug_logprobs()
