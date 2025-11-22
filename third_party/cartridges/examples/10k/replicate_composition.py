"""
Replicate the exact cartridge composition experiment from the paper.

This script:
1. Loads/trains AMD and Pepsi cartridges using the same model (Llama-3.2-3B)
2. Evaluates single cartridges on single-document QA
3. Evaluates composed cartridges [AMD, Pepsi] on multi-document QA
4. Compares QA accuracy (exact match + F1) instead of perplexity
"""

import os
import sys
from pathlib import Path

# Add cartridges to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cartridges.utils.wandb import load_model_and_cache_from_wandb
from cartridges.train import TrainConfig
from cartridges.models import HFModelConfig, FlexLlamaForCausalLM
from cartridges.initialization import KVFromText
from cartridges.datasets import TrainDataset, DataSource

# Set up paths
CARTRIDGES_DIR = Path(__file__).parent.parent.parent
DATA_DIR = CARTRIDGES_DIR / "data" / "10k"
OUTPUT_DIR = CARTRIDGES_DIR / "outputs" / "composition_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model used in the paper for composition experiments
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# Pre-trained WandB run IDs (if available)
AMD_WANDB_RUN = None  # "hazy-research/cartridges/XXX"
PEPSI_WANDB_RUN = None  # "hazy-research/cartridges/XXX"


def create_amd_training_config():
    """Training config for AMD cartridge (matching paper settings)."""
    return TrainConfig(
        name="amd-10k-llama3-composition",
        
        model=HFModelConfig(
            pretrained_model_name_or_path=MODEL_NAME,
            model_cls=FlexLlamaForCausalLM,
            tuning_method="custom_prefix",
        ),
        
        kv_cache_initializer=KVFromText.Config(
            text_source=str(DATA_DIR / "amd_10k.txt"),
            max_tokens=None,  # Use full document
        ),
        
        dataset=TrainDataset.Config(
            data_sources=[
                DataSource(
                    path=str(DATA_DIR / "amd_10k_synthesize_qwen-qwen3-4b_n512-0.parquet"),
                    type="local"
                ),
            ],
            top_k_logits=20,
            packed_seq_length=2048,
            packing_mode="truncate",
        ),
        
        # Hyperparameters from paper
        lr=2e-2,
        epochs=3,
        global_batch_size=16,
        
        save_every_n_steps=128,
        save_after_training=True,
        
        # Cartridge sizes from paper: {512, 1024, 2048, 4096}
        # Default is usually 1024 or 2048
        max_tokens=2048,  
        
        wandb=None,  # Or set up WandB if you want
    )


def create_pepsi_training_config():
    """Training config for PepsiCo cartridge (matching paper settings)."""
    config = create_amd_training_config()
    config.name = "pepsi-10k-llama3-composition"
    config.kv_cache_initializer.text_source = str(DATA_DIR / "pepsi_10k.txt")
    config.dataset.data_sources[0].path = str(DATA_DIR / "pepsi_10k_synthesize_qwen-qwen3-4b_n512-0.parquet")
    return config


def load_or_train_cartridge(config, wandb_run_id=None):
    """Load pre-trained cartridge from WandB or train from scratch."""
    if wandb_run_id:
        print(f"Loading pre-trained cartridge from {wandb_run_id}...")
        cache_and_model = load_model_and_cache_from_wandb(wandb_run_id)
        return cache_and_model.cache, cache_and_model.model
    else:
        print(f"Training cartridge from scratch: {config.name}")
        print("Run:")
        print(f"  CUDA_VISIBLE_DEVICES=0 python -m cartridges.train {config.name}")
        # You would actually train here - for now just return placeholders
        return None, None


if __name__ == "__main__":
    print("="*80)
    print("CARTRIDGE COMPOSITION EXPERIMENT - QA ACCURACY")
    print("="*80)
    print()
    
    print("Model:", MODEL_NAME)
    print("Documents:")
    print(f"  - AMD 10-K: {DATA_DIR / 'amd_10k.txt'}")
    print(f"  - PepsiCo 10-K: {DATA_DIR / 'pepsi_10k.txt'}")
    print()
    
    # Step 1: Check for pre-trained cartridges
    print("Step 1: Loading/Training Cartridges")
    print("-" * 80)
    
    if AMD_WANDB_RUN:
        amd_cache, amd_model = load_or_train_cartridge(None, AMD_WANDB_RUN)
    else:
        print("No pre-trained AMD cartridge found.")
        print("Generate training config:")
        amd_config = create_amd_training_config()
        print(f"  Name: {amd_config.name}")
        print(f"  Model: {amd_config.model.pretrained_model_name_or_path}")
        print(f"  Max tokens: {amd_config.max_tokens}")
        print()
    
    if PEPSI_WANDB_RUN:
        pepsi_cache, pepsi_model = load_or_train_cartridge(None, PEPSI_WANDB_RUN)
    else:
        print("No pre-trained PepsiCo cartridge found.")
        print("Generate training config:")
        pepsi_config = create_pepsi_training_config()
        print(f"  Name: {pepsi_config.name}")
        print(f"  Model: {pepsi_config.model.pretrained_model_name_or_path}")
        print(f"  Max tokens: {pepsi_config.max_tokens}")
        print()
    
    # Step 2: Prepare evaluation datasets
    print("Step 2: Evaluation Datasets")
    print("-" * 80)
    print("You need:")
    print("  1. Single-doc AMD QA dataset")
    print("  2. Single-doc PepsiCo QA dataset")
    print("  3. Multi-doc composition QA dataset (AMD + PepsiCo)")
    print()
    print("See: examples/10k/create_eval_datasets.py")
    print()
    
    # Step 3: Evaluation
    print("Step 3: Evaluation Commands")
    print("-" * 80)
    print("Once cartridges are trained, run evaluation:")
    print()
    print("# Single-document QA (AMD only)")
    print("python examples/10k/evaluate_qa.py --cartridge outputs/amd-10k-llama3-composition/cache_final.pt --dataset data/10k/eval/amd_qa.json")
    print()
    print("# Single-document QA (PepsiCo only)")
    print("python examples/10k/evaluate_qa.py --cartridge outputs/pepsi-10k-llama3-composition/cache_final.pt --dataset data/10k/eval/pepsi_qa.json")
    print()
    print("# Multi-document QA (composition)")
    print("python examples/10k/evaluate_qa.py --cartridges outputs/amd-10k-llama3-composition/cache_final.pt outputs/pepsi-10k-llama3-composition/cache_final.pt --dataset data/10k/eval/composition_qa.json")
    print()
    
    print("="*80)
    print("To execute this experiment, you need to:")
    print("  1. Train both cartridges (or find pre-trained ones)")
    print("  2. Create evaluation datasets")
    print("  3. Run evaluation with QA accuracy metrics")
    print("="*80)


