from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, NoReturn, Optional, cast
import argparse
import statistics
import random
import itertools
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from datasets import Dataset

def generate_mappings(num_mappings) -> dict:
    """
    Generate a dictionary mapping num_mappings random keys to num_mappings random values.
    Keys and values are single English words. No duplicate keys or values.
    """
    import nltk
    from nltk.corpus import words
    
    # Download words corpus if not already available
    try:
        word_list = words.words()
    except LookupError:
        nltk.download('words')
        word_list = words.words()
    
    # Filter to single words (no spaces, hyphens, etc.) and reasonable length
    single_words = [w.lower() for w in word_list if w.isalpha() and 3 <= len(w) <= 12]
    
    # Sample unique keys and values
    sampled_words = random.sample(single_words, num_mappings * 2)
    keys = sampled_words[:num_mappings]
    values = sampled_words[num_mappings:]
    
    # Create the mapping
    return dict(zip(keys, values))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic cities and push train/val splits to Hugging Face")
    parser.add_argument("--num-mappings", type=int, default=5000, help="Number of cities to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Tokenizer model id")
    parser.add_argument("--repo-id", type=str, default="Bradley/easy_mqar", help="Hugging Face repo id to push to")
    args = parser.parse_args()

    rng_seed = args.seed
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    mappings = generate_mappings(args.num_mappings)

    recs = []
    for idx, (key, value) in enumerate(mappings.items()):
        recs.append({
            "id": idx,
            "key": "Now, please respond with the value for the key within \\boxed{{}}: " + key,
            "value": value,
        })

    ds_train = Dataset.from_list(recs)
    ds_val = Dataset.from_list(recs)

    # Push both splits to the Hugging Face Hub
    print(f"Pushing dataset to {args.repo_id} (splits: train, val)...")
    ds_train.push_to_hub(args.repo_id, split="train")
    ds_val.push_to_hub(args.repo_id, split="val")
    print("Push complete.")
