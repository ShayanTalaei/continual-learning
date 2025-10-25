#!/usr/bin/env python3
"""
Quick test script for dataset conversion - minimal validation.
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
    # Default to ./outputs if not set
    os.environ["CARTRIDGES_OUTPUT_DIR"] = str(REPO_ROOT / "outputs")

# Add project paths
sys.path.insert(0, str(CARTRIDGES_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.memory.distillation.convert_to_cartridges import convert_row_to_conversation


def quick_test():
    """Quick test of the conversion process."""
    print("Quick Dataset Conversion Test")
    print("-" * 40)
    
    # Load one sample
    ds = load_dataset("stalaei/distillation-dataset-test", split="train")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    
    print(f"Dataset loaded: {len(ds)} samples")
    
    # Test first sample
    row = ds[0]
    print(f"Sample keys: {list(row.keys())}")
    
    if "messages" in row:
        print(f"Messages: {len(row['messages'])}")
        if row["messages"]:
            print(f"First message keys: {list(row['messages'][0].keys())}")
    
    # Try conversion
    try:
        conv = convert_row_to_conversation(row, tokenizer, min_prob_mass=0.99)
        print("✓ Conversion successful!")
        print(f"  Messages: {len(conv.messages)}")
        print(f"  System prompt: {len(conv.system_prompt)} chars")
        print(f"  Type: {conv.type}")
        
        # Check first message
        if conv.messages:
            msg = conv.messages[0]
            print(f"  First message: role={msg.role}, tokens={len(msg.token_ids) if msg.token_ids else 0}")
            print(f"  Top logprobs: {'present' if msg.top_logprobs else 'None'}")
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    quick_test()
