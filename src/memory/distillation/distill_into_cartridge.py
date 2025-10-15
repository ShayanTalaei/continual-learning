#!/usr/bin/env python3
"""
Train a cartridge from a distillation dataset using knowledge distillation.

This script:
1. Loads a dataset from HuggingFace (in intermediate or cartridges format)
2. Converts to cartridges format if needed
3. Trains a KV cache cartridge using knowledge distillation
4. Uploads the trained cartridge to HuggingFace

Usage:
    python -m src.memory.distillation.distill_into_cartridge --config configs/distillation/cartridge_train.yaml
"""

import sys
import argparse
import yaml
from pathlib import Path
from typing import Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
import tempfile
import os

# Set up cartridges environment variables BEFORE importing
REPO_ROOT = Path(__file__).parent.parent.parent.parent
CARTRIDGES_DIR = REPO_ROOT / "third_party" / "cartridges"

# Cartridges requires these environment variables
os.environ["CARTRIDGES_DIR"] = str(CARTRIDGES_DIR)
if "CARTRIDGES_OUTPUT_DIR" not in os.environ:
    # Default to ./outputs if not set
    os.environ["CARTRIDGES_OUTPUT_DIR"] = str(REPO_ROOT / "outputs")

# Add third_party/cartridges to path
sys.path.insert(0, str(CARTRIDGES_DIR))

from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import HfApi, create_repo

# Import cartridges components
from cartridges.train import TrainConfig, train
from cartridges.datasets import TrainDataset, DataSource
from cartridges.models.config import HFModelConfig
from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
from cartridges.cache import KVCacheFactory
from cartridges.initialization.text import KVFromText
from cartridges.initialization.random import KVFromRandomVectors
from cartridges.structs import Conversation, write_conversations
from cartridges.utils.wandb import WandBConfig

# Import our converter
from src.memory.distillation.convert_to_cartridges import convert_row_to_conversation


# ============================================================================
# Configuration Schemas
# ============================================================================

class InputDatasetConfig(BaseModel):
    """Configuration for input dataset source."""
    repo_id: str = Field(..., description="HuggingFace dataset repo ID")
    split: str = Field(default="train", description="Dataset split to use")
    is_converted: bool = Field(
        default=False, 
        description="If True, dataset is already in cartridges format. If False, will convert from intermediate format."
    )


class OutputConfig(BaseModel):
    """Configuration for output artifacts."""
    local_dir: str = Field(..., description="Local directory to save training outputs")
    hf_repo_id: Optional[str] = Field(None, description="HuggingFace model repo ID for uploading cartridge")
    hf_private: bool = Field(default=True, description="Whether to create a private HF repo")
    upload_to_hf: bool = Field(default=True, description="Whether to upload to HuggingFace after training")


class KVCacheInitConfig(BaseModel):
    """Configuration for KV cache initialization."""
    method: Literal["random", "text"] = Field(default="random", description="Initialization method")
    num_tokens: int = Field(default=128, description="Number of tokens in the cartridge")
    num_frozen_tokens: int = Field(default=1, description="Number of tokens to freeze (prevents forgetting)")
    
    # For text initialization
    init_text: Optional[str] = Field(None, description="Text to initialize from (for method='text')")
    init_text_file: Optional[str] = Field(None, description="File containing init text (overrides init_text)")


class TrainingConfig(BaseModel):
    """Configuration for training hyperparameters."""
    epochs: int = Field(default=5, description="Number of training epochs")
    global_batch_size: int = Field(default=8, description="Total batch size across all devices")
    lr: float = Field(default=1e-4, description="Learning rate")
    weight_decay: float = Field(default=0.0, description="Weight decay")
    optimizer: Literal["adam"] = Field(default="adam", description="Optimizer type")
    
    # Checkpointing
    save_every_n_steps: Optional[int] = Field(default=100, description="Save checkpoint every N steps")
    save_after_training: bool = Field(default=True, description="Save final checkpoint after training")
    keep_last_n_saved: int = Field(default=3, description="Number of checkpoints to keep")
    
    # Device
    device: str = Field(default="cuda", description="Device to train on")
    seed: int = Field(default=42, description="Random seed")
    
    # Distributed training
    distributed_backend: Literal["nccl", "gloo"] = Field(
        default="nccl", 
        description="Distributed backend: 'nccl' for GPU (faster), 'gloo' for CPU/fallback"
    )


class DatasetConfig(BaseModel):
    """Configuration for dataset processing."""
    packing_mode: Literal["pad", "truncate"] = Field(default="pad", description="Sequence packing mode")
    packed_seq_length: int = Field(default=2048, description="Maximum sequence length")
    targets: Literal["logits", "tokens"] = Field(default="logits", description="Training target type")
    top_k_logits: int = Field(default=20, description="Number of top-k logits to keep")
    min_prob_mass: float = Field(default=0.99, description="Minimum probability mass for logprobs conversion")


class WandBConfigWrapper(BaseModel):
    """Configuration for Weights & Biases logging."""
    enabled: bool = Field(default=False, description="Enable WandB logging")
    project: Optional[str] = Field(None, description="WandB project name")
    entity: Optional[str] = Field(None, description="WandB entity")
    name: Optional[str] = Field(None, description="WandB run name")


class DistillationConfig(BaseModel):
    """Main configuration for cartridge distillation training."""
    
    # Input/Output
    input_dataset: InputDatasetConfig
    output: OutputConfig
    
    # Model
    model_name: str = Field(..., description="Model name (e.g., meta-llama/Llama-3.1-8B-Instruct)")
    
    # KV Cache
    kv_cache: KVCacheInitConfig
    
    # Training
    training: TrainingConfig
    
    # Dataset
    dataset: DatasetConfig
    
    # WandB (optional)
    wandb: WandBConfigWrapper = Field(default_factory=lambda: WandBConfigWrapper(
        enabled=False,
        project=None,
        entity=None,
        name=None
    ))


# ============================================================================
# Helper Functions
# ============================================================================

def create_conversation_from_row(row: Dict[str, Any], tokenizer: AutoTokenizer, min_prob_mass: float = 0.99) -> Conversation:
    """Create a Conversation object from a dataset row.
    
    Args:
        row: Dataset row dictionary
        tokenizer: Tokenizer for processing logprobs
        min_prob_mass: Probability mass threshold for flattening logprobs
        
    Returns:
        Conversation object
    """
    from src.memory.distillation.convert_to_cartridges import convert_openai_logprobs_to_top_logprobs
    
    messages = []
    for message in row["messages"]:
        content = message["content"]
        role = message["role"]
        token_ids = message.get("token_ids")
        
        # Process logprobs for assistant messages
        top_logprobs = None
        if role == "assistant" and message.get("logprobs"):
            top_logprobs_obj = convert_openai_logprobs_to_top_logprobs(message["logprobs"], tokenizer)
            if top_logprobs_obj:
                top_logprobs = top_logprobs_obj.flatten(threshold=min_prob_mass)
                if top_logprobs and len(top_logprobs.token_id) == 0:
                    print(f"[Distill] Warning: Empty logprobs after flattening with threshold {min_prob_mass}")
                    top_logprobs = None
        
        messages.append(Conversation.Message(
            content=content,
            role=role,
            token_ids=token_ids,
            top_logprobs=top_logprobs
        ))
    
    return Conversation(
        messages=messages,
        system_prompt=row["system_prompt"],
        metadata=row["metadata"],
        type=row["type"],
    )

def load_and_convert_dataset(
    config: DistillationConfig,
    tokenizer: AutoTokenizer,
    temp_dir: Path,
) -> str:
    """Load dataset from HF and convert to cartridges format if needed.
    
    Args:
        config: Distillation configuration
        tokenizer: Tokenizer for the model
        temp_dir: Temporary directory for intermediate files
        
    Returns:
        Path to the cartridges-format dataset (parquet file)
    """
    print(f"[Distill] Loading dataset from HuggingFace: {config.input_dataset.repo_id}")
    
    # Load from HuggingFace
    ds = load_dataset(config.input_dataset.repo_id, split=config.input_dataset.split)
    print(f"[Distill] Loaded {len(ds)} samples")
    
    if config.input_dataset.is_converted:
        # Already in cartridges format, save to temp location
        print("[Distill] Dataset is already in cartridges format")
        output_path = temp_dir / "dataset.parquet"
        
        # Convert HF dataset to Conversation objects
        conversations = []
        for i in range(len(ds)):
            row = ds[i]
            conv = create_conversation_from_row(row, tokenizer, config.dataset.min_prob_mass)
            conversations.append(conv)
        
        # Check if we have any logprobs
        total_logprobs = sum(1 for conv in conversations for msg in conv.messages if msg.top_logprobs)
        print(f"[Distill] Total messages with logprobs: {total_logprobs}")
        
        if total_logprobs == 0:
            print("[Distill] Warning: No logprobs found! This may cause training issues.")
        
        write_conversations(conversations, str(output_path))
        print(f"[Distill] Converted {len(conversations)} conversations to: {output_path}")
        return str(output_path)
    
    else:
        # Need to convert from intermediate format
        print("[Distill] Converting from intermediate format to cartridges format...")
        conversations = []
        
        for i in range(len(ds)):
            row = ds[i]
            try:    
                conv = convert_row_to_conversation(row, tokenizer, config.dataset.min_prob_mass)
                conversations.append(conv)
            except Exception as e:
                print(f"[Distill] Warning: Failed to convert sample {i}: {e}")
                continue
            
            if (i + 1) % 20 == 0:
                print(f"[Distill] Converted {i + 1}/{len(ds)} samples")
        
        # Save to temp parquet file
        output_path = temp_dir / "dataset.parquet"
        write_conversations(conversations, str(output_path))
        print(f"[Distill] Converted {len(conversations)} conversations to: {output_path}")
        
        return str(output_path)


def create_kv_cache_factory(config: KVCacheInitConfig, temp_dir: Path) -> KVCacheFactory.Config:
    """Create KV cache factory configuration.
    
    Args:
        config: KV cache initialization config
        temp_dir: Temporary directory for storing cache state
        
    Returns:
        KVCacheFactory.Config for initializing the cache
    """
    if config.method == "random":
        print(f"[Distill] Using random initialization for {config.num_tokens} tokens")
        return KVFromRandomVectors.Config(
            max_tokens=config.num_tokens,
            num_frozen_tokens=config.num_frozen_tokens,
        )
    
    elif config.method == "text":
        # Get initialization text
        if config.init_text_file:
            print(f"[Distill] Loading init text from: {config.init_text_file}")
            with open(config.init_text_file, "r") as f:
                init_text = f.read()
        elif config.init_text:
            print(f"[Distill] Using provided init text (length: {len(config.init_text)})")
            init_text = config.init_text
        else:
            raise ValueError("For method='text', must provide either init_text or init_text_file")
        
        # Write init text to a temporary file
        init_text_file = temp_dir / "init_text.txt"
        with open(init_text_file, "w") as f:
            f.write(init_text)
        
        return KVFromText.Config(
            max_tokens=config.num_tokens,
            text_source=str(init_text_file),
            num_frozen_tokens=config.num_frozen_tokens,
        )
    
    else:
        raise ValueError(f"Unknown KV cache initialization method: {config.method}")


def upload_cartridge_to_hf(
    cartridge_path: str,
    repo_id: str,
    private: bool = True,
    model_name: Optional[str] = None,
    config: Optional[DistillationConfig] = None,
) -> str:
    """Upload trained cartridge to HuggingFace Hub.
    
    Args:
        cartridge_path: Path to the .pt cartridge file
        repo_id: HuggingFace model repo ID
        private: Whether to create a private repo
        model_name: Model name for the README
        config: Training config for the README
        
    Returns:
        URL of the uploaded model
    """
    print(f"[Distill] Uploading cartridge to HuggingFace: {repo_id}")
    
    # Get HF token
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="model", private=private, exist_ok=True, token=token)
        print(f"[Distill] Created/verified repo: {repo_id}")
    except Exception as e:
        print(f"[Distill] Warning: Could not create repo: {e}")
    
    # Upload the cartridge file
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=cartridge_path,
        path_in_repo="cartridge.pt",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload trained cartridge from memory distillation"
    )
    print(f"[Distill] Uploaded cartridge.pt to {repo_id}")
    
    # Create README with usage instructions
    readme_content = f"""---
tags:
- cartridge
- memory-distillation
- kv-cache
model_name: {model_name or 'unknown'}
---

# Memory Distillation Cartridge

This cartridge was trained using knowledge distillation from a teacher model's memory.

## Usage

```python
from cartridges.cache import TrainableCache

# Load the cartridge
cache = TrainableCache.from_pretrained(
    "hf://{repo_id}/cartridge.pt",
    device="cuda"
)

# Use in your KVMemoryAgent
from src.memory.kv_cache import KVCacheMemory, KVCacheMemoryConfig, KVArtifact

config = KVCacheMemoryConfig(
    artifact=KVArtifact(
        id="{repo_id}",
        source="huggingface",
    ),
    tokenizer_name="{model_name}",
    model_name="{model_name}",
)

memory = KVCacheMemory(config)
```

## Training Details

- **Model**: {model_name}
- **Num Tokens**: {config.kv_cache.num_tokens if config else 'N/A'}
- **Frozen Tokens**: {config.kv_cache.num_frozen_tokens if config else 'N/A'}
- **Training Epochs**: {config.training.epochs if config else 'N/A'}
- **Learning Rate**: {config.training.lr if config else 'N/A'}
- **Dataset**: {config.input_dataset.repo_id if config else 'N/A'}

## Training Configuration

```yaml
{yaml.dump(config.model_dump() if config else {}, default_flow_style=False)}
```
"""
    
    # Upload README
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add README with usage instructions"
    )
    print(f"[Distill] Uploaded README.md")
    
    url = f"https://huggingface.co/{repo_id}"
    print(f"[Distill] ✓ Cartridge uploaded successfully: {url}")
    return url


# ============================================================================
# Main Training Function
# ============================================================================

def run_distillation(config: DistillationConfig):
    """Run the full cartridge distillation pipeline.
    
    Args:
        config: Distillation configuration
    """
    print("=" * 80)
    print("CARTRIDGE DISTILLATION TRAINING")
    print("=" * 80)
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.input_dataset.repo_id}")
    print(f"Output: {config.output.local_dir}")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(config.output.local_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config for reproducibility
    config_path = output_dir / "distillation_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False)
    print(f"[Distill] Config saved to: {config_path}")
    
    # Load tokenizer
    print(f"[Distill] Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Load and convert dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        dataset_path = load_and_convert_dataset(config, tokenizer, temp_path)
        
        # Create KV cache factory config
        kv_cache_factory_config = create_kv_cache_factory(config.kv_cache, temp_path)
        
        # Create cartridges TrainConfig
        print("[Distill] Setting up training configuration...")
        train_config = TrainConfig(
            name=f"distill_{Path(config.input_dataset.repo_id).name}",
            output_dir=str(output_dir),
            
            # Model
            model=HFModelConfig(
                pretrained_model_name_or_path=config.model_name,
                model_cls=FlexLlamaForCausalLM,  # Use custom model that supports TrainableCache
            ),
            
            # Dataset
            dataset=TrainDataset.Config(
                data_sources=[DataSource(path=dataset_path, type="local")],
                packing_mode=config.dataset.packing_mode,
                packed_seq_length=config.dataset.packed_seq_length,
                targets=config.dataset.targets,
                top_k_logits=config.dataset.top_k_logits,
            ),
            
            # Training
            global_batch_size=config.training.global_batch_size,
            epochs=config.training.epochs,
            device=config.training.device,
            distributed_backend=config.training.distributed_backend,
            optimizer=config.training.optimizer,
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
            
            # KV Cache
            kv_cache_initializer=kv_cache_factory_config,
            
            # Checkpointing
            save_every_n_steps=config.training.save_every_n_steps,
            save_after_training=config.training.save_after_training,
            keep_last_n_saved=config.training.keep_last_n_saved,
            save_to_wandb=False,  # We'll handle HF upload separately
            
            # WandB
            wandb=WandBConfig(
                project=config.wandb.project or "cartridges-distillation",
                entity=config.wandb.entity,
                name=config.wandb.name or f"distill_{Path(config.input_dataset.repo_id).name}",
            ) if config.wandb.enabled else None,
            
            # Misc
            seed=config.training.seed,
        )
        
        print("[Distill] Starting training...")
        print("=" * 80)
        
        # Run training
        train(train_config)
        
        print("=" * 80)
        print("[Distill] Training completed!")
        
    # Upload to HuggingFace if requested
    if config.output.upload_to_hf and config.output.hf_repo_id:
        # Find the final cartridge file
        cartridge_file = output_dir / "cache-final.pt"
        if not cartridge_file.exists():
            # Look for latest checkpoint
            checkpoints = sorted(output_dir.glob("cache-step*.pt"))
            if checkpoints:
                cartridge_file = checkpoints[-1]
            else:
                print("[Distill] Warning: No cartridge file found to upload")
                return
        
        print(f"[Distill] Found cartridge file: {cartridge_file}")
        
        upload_cartridge_to_hf(
            str(cartridge_file),
            config.output.hf_repo_id,
            private=config.output.hf_private,
            model_name=config.model_name,
            config=config,
        )
    
    print("\n" + "=" * 80)
    print("DISTILLATION COMPLETE!")
    print("=" * 80)
    print(f"Local output: {output_dir}")
    if config.output.upload_to_hf and config.output.hf_repo_id:
        print(f"HuggingFace: https://huggingface.co/{config.output.hf_repo_id}")
    print("=" * 80)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """CLI entry point for cartridge distillation."""
    parser = argparse.ArgumentParser(
        description="Train a cartridge from distillation dataset using knowledge distillation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python -m src.memory.distillation.distill_into_cartridge --config configs/distillation/cartridge_train.yaml
"""
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file with DistillationConfig parameters"
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    config = DistillationConfig(**config_dict)
    
    # Run distillation
    try:
        run_distillation(config)
    except Exception as e:
        print(f"\nERROR: Distillation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

