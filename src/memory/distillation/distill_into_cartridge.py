#!/usr/bin/env python3
"""
Train a cartridge from a distillation dataset using knowledge distillation.

This script:
1. Loads a dataset from HuggingFace (in intermediate or cartridges format)
2. Converts to cartridges format if needed
3. Trains a KV cache cartridge using knowledge distillation
4. Uploads the trained cartridge to HuggingFace

Usage:
    python -m src.memory.distillation.distill_into_cartridge
    python -m src.memory.distillation.distill_into_cartridge input_dataset.local_path=/path/to/dataset.jsonl
"""

# import torch
# torch._inductor.config.max_autotune_gemm_backends = ["ATEN", "TRITON", "CPP"]
# torch._inductor.config.max_autotune = True
# torch._inductor.config.epilogue_fusion = True

import sys
import pydra
import yaml
from pathlib import Path
from typing import Optional, Literal, Dict, Any
import tempfile
import os

# Set up cartridges environment variables BEFORE importing
REPO_ROOT = Path(__file__).parent.parent.parent.parent
CARTRIDGES_DIR = REPO_ROOT / "third_party" / "cartridges"
TOKASAURUS_DIR = REPO_ROOT / "third_party" / "tokasaurus"

# Cartridges requires these environment variables
os.environ["CARTRIDGES_DIR"] = str(CARTRIDGES_DIR)
if "CARTRIDGES_OUTPUT_DIR" not in os.environ:
    # Default to ./outputs if not set
    os.environ["CARTRIDGES_OUTPUT_DIR"] = str(REPO_ROOT / "outputs")

# Add third_party/cartridges to path
sys.path.insert(0, str(CARTRIDGES_DIR))
sys.path.insert(0, str(TOKASAURUS_DIR))

from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import HfApi, create_repo

# Import tokasaurus components
from tokasaurus.common_types import ServerConfig as TokaServerConfig

# Import cartridges components
from cartridges.train import TrainConfig, train, GenerationEvalConfig, LossEvalConfig
from cartridges.datasets import TrainDataset, ShayanTrainDataset, ShayanStreamingTrainDataset, DataSource
from cartridges.models.config import HFModelConfig
from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
from cartridges.cache import KVCacheFactory
from cartridges.initialization.text import KVFromText
from cartridges.initialization.random import KVFromRandomVectors
from cartridges.structs import Conversation, write_conversations
from cartridges.utils.wandb import WandBConfig
from cartridges.data.finer.evals import FinerGenerateDataset


# ============================================================================
# Pydra Configuration Classes
# ============================================================================

class InputDatasetConfig(pydra.Config):
    """Configuration for input dataset source."""
    def __init__(self):
        super().__init__()
        # Either use HuggingFace dataset or local pre-processed dataset
        self.repo_id = None  # HuggingFace dataset repo ID
        self.split = "train"  # Dataset split to use (for HF datasets)
        self.local_path = pydra.REQUIRED  # Path to pre-processed parquet file
        self.filter_incorrect = False  # Whether to filter incorrect answers
        self.ground_truth_target = False  # Whether to use the ground truth target for the output
    
    def finalize(self):
        if not self.repo_id and not self.local_path:
            raise ValueError("Must provide either repo_id or local_path")
        if self.repo_id and self.local_path:
            raise ValueError("Cannot provide both repo_id and local_path")


class OutputConfig(pydra.Config):
    """Configuration for output artifacts."""
    def __init__(self):
        super().__init__()
        self.local_dir = "/scratch/m000122/stalaei/continual-learning/cartridges"  # Local directory to save training outputs
        self.hf_repo_id = "stalaei/finer-cartridge-v1"  # HuggingFace model repo ID for uploading cartridge
        self.hf_private = True  # Whether to create a private HF repo
        self.upload_to_hf = True  # Whether to upload to HuggingFace after training


class KVCacheInitConfig(pydra.Config):
    """Configuration for KV cache initialization."""
    def __init__(self):
        super().__init__()
        self.method = "random"  # Initialization method: "random" or "text"
        self.num_tokens = pydra.REQUIRED  # Number of tokens in the cartridge
        self.num_frozen_tokens = 4  # Number of tokens to freeze (prevents forgetting)
        
        # For text initialization
        self.init_text = None  # Text to initialize from (for method='text')
        self.init_text_file = pydra.REQUIRED  # File containing init text (overrides init_text)


class TrainingConfig(pydra.Config):
    """Configuration for training hyperparameters."""
    def __init__(self):
        super().__init__()
        self.epochs = 100  # Number of training epochs
        self.global_batch_size = 8  # Total batch size across all devices
        self.lr = 5e-4  # Learning rate
        self.weight_decay = 0.0  # Weight decay
        self.optimizer = "adam"  # Optimizer type
        self.gradient_checkpointing = True  # Enable activation (gradient) checkpointing to reduce memory usage

        # Temperature
        self.train_temperature = 1.0  # Temperature for training
        self.val_temperature = 1.0  # Temperature for validation
        
        # Checkpointing
        self.save_every_n_steps = 100  # Save checkpoint every N steps
        self.save_after_training = True  # Save final checkpoint after training
        self.keep_last_n_saved = 3  # Number of checkpoints to keep
        
        # Device
        self.device = "cuda"  # Device to train on
        self.seed = 42  # Random seed
        
        # Distributed training
        self.distributed_backend = "nccl"  # Distributed backend: 'nccl' for GPU (faster), 'gloo' for CPU/fallback

        self.train_without_logits = False  # Whether to train without logits


class DatasetConfig(pydra.Config):
    """Configuration for dataset processing."""
    def __init__(self):
        super().__init__()
        self.packing_mode = "fixed_batch_size_then_pad"  # Sequence packing mode
        self.packed_seq_length = 32000  # Maximum sequence length
        self.targets = "logits"  # Training target type
        self.top_k_logits = 20  # Number of top-k logits to keep
        self.min_prob_mass = 0.8  # Minimum probability mass for logprobs conversion
        self.batch_size = 8  # Batch size

class WandBConfigWrapper(pydra.Config):
    """Configuration for Weights & Biases logging."""
    def __init__(self):
        super().__init__()
        self.enabled = True  # Enable WandB logging
        self.project = "cartridge-distillation"  # WandB project name
        self.entity = "stalaei-stanford-university"  # WandB entity
        self.name = "finer-cartridge-v1"  # WandB run name


class DistillationConfig(pydra.Config):
    """Main configuration for cartridge distillation training."""
    
    def __init__(self):
        super().__init__()
        # Input/Output
        self.input_dataset = InputDatasetConfig()
        self.val_dataset = InputDatasetConfig()
        self.val_dataset.local_path = "/mnt/data/shayan_memory/finer_val_data_gen_full_memory/dataset.jsonl"
        self.output = OutputConfig()
        
        # Model
        self.model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Model name
        
        # KV Cache
        self.kv_cache = KVCacheInitConfig()
        
        # Training
        self.training = TrainingConfig()
        
        # Dataset
        self.dataset = DatasetConfig()
        
        # WandB (optional)
        self.wandb = WandBConfigWrapper()

        # Evaluation
        self.do_loss_evals = True  # Whether to do loss evals
        self.do_train_gen_eval = False  # Whether to do generation evals
        self.do_val_gen_eval = True  # Whether to do generation evals
        self.num_train_generate_problems = 250
        self.train_gen_split = "train_ICL"
        self.val_gen_split = "val"
        self.generate_before_training = True
        self.generate_eval_every_n_steps = 50
        self.num_generate_problems = 1000
        self.generate_temperature = 0.0
        self.generate_batch_size = 32
        
        # Name
        self.run_name = None

        self.system_prompt_path = pydra.REQUIRED

        self.streaming_dataset = False
        self.generation_server_type = "hf"
        self.toka_server_config = None
        self.toka_kv_cache_num_tokens = 200_000

        self.dataloader_num_workers = 1
    
    def no_evals(self):
        self.do_loss_evals = False
        self.do_train_gen_eval = False
        self.do_val_gen_eval = False
    
    def train_gen_eval(self):
        self.do_train_gen_eval = True
        self.num_train_eval_problems = 250
        self.train_gen_split = "train_ICL"
    
    def init_from_text(self):
        self.kv_cache.method = "text"
        self.kv_cache.init_text_file = "/mnt/home/bradleyb/continual-learning/src/memory/distillation/kv_cache_init_texts/v1.txt"

    def finalize(self):
        if self.run_name is None:
            if self.input_dataset.local_path:
                self.run_name = f"distill_{Path(self.input_dataset.local_path).stem}"
            else:
                self.run_name = f"distill_{Path(self.input_dataset.repo_id or 'unknown').name}"
        
        self.run_dir = Path(self.output.local_dir) / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        if self.toka_server_config is not None:
            self.toka_server_overrides += [
                f"cartridge_dir={self.run_dir}",
            ]
            pydra.apply_overrides(self.toka_server_config, self.toka_server_overrides)
    
    def matx(self):
        self.output.local_dir = "/matx/u/bcabrown/shayan_memory/outputs"
        self.input_dataset.packed_seq_length = 8000  # Maximum sequence length
        self.val_dataset.packed_seq_length = 8000  # Maximum sequence length
        self.training.global_batch_size = 32  # Total batch size across all devices
        self.input_dataset.batch_size = 2
        self.val_dataset.batch_size = 2
        self.generate_batch_size = 2
        self.toka_kv_cache_num_tokens = 100_000
    
    def streaming_dataset(self):
        self.streaming_dataset = True
        self.dataloader_num_workers = 8

    def toka(self):
        self.generation_server_type = "toka"
        self.toka_server_config = TokaServerConfig()
        self.toka_server_overrides = [
            f"model={self.model_name}",
            f"tokenizer={self.model_name}",
            f"trust_remote_code=True",
            f"kv_cache_num_tokens={self.toka_kv_cache_num_tokens}",
            f"torch_compile=False",
        ]
        pydra.apply_overrides(self.toka_server_config, self.toka_server_overrides)
        self.generate_batch_size = 200


# ============================================================================
# Helper Functions
# ============================================================================


def get_dataset_path(dataset) -> str:
    """Get the path to the dataset for training.
    
    Args:
        config: Distillation configuration
        
    Returns:
        Path to the cartridges-format dataset (parquet file)
    """
    if dataset.local_path:
        # Use pre-processed local dataset
        dataset_path = Path(dataset.local_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Local dataset not found: {dataset_path}")
        print(f"[Distill] Using pre-processed dataset: {dataset_path}")
        return str(dataset_path)
    
    elif dataset.repo_id:
        # Load from HuggingFace (assume already in cartridges format)
        print(f"[Distill] Loading dataset from HuggingFace: {dataset.repo_id}")
        ds = load_dataset(dataset.repo_id, split=dataset.split)
        print(f"[Distill] Loaded {len(ds)} samples")
        
        # Save to temp location for training
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_path = temp_path / "dataset.parquet"
            
            # Convert HF dataset to Conversation objects
            conversations = []
            for i in range(len(ds)):
                row = ds[i]
                # Assume the HF dataset is already in cartridges format
                conv = Conversation(**row)
                conversations.append(conv)
            
            write_conversations(conversations, str(output_path))
            print(f"[Distill] Prepared {len(conversations)} conversations for training")
            return str(output_path)
    
    else:
        raise ValueError("Must provide either local_path or repo_id in input_dataset config")


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
{yaml.dump(config.to_dict() if config else {}, default_flow_style=False)}
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
        yaml.dump(config.to_dict(), f, default_flow_style=False)
    print(f"[Distill] Config saved to: {config_path}")
    
    # Load tokenizer
    print(f"[Distill] Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Get dataset path
    train_dataset_path = get_dataset_path(config.input_dataset)

    dataset_cls = ShayanStreamingTrainDataset if config.streaming_dataset else ShayanTrainDataset

    if config.do_loss_evals:
        val_dataset_path = get_dataset_path(config.val_dataset)
        loss_evals = [
            LossEvalConfig(
                dataset=dataset_cls.Config(
                    data_sources=[DataSource(path=val_dataset_path, type="local")],
                    packing_mode=config.dataset.packing_mode,
                    packed_seq_length=config.dataset.packed_seq_length,
                    targets=config.dataset.targets,
                    top_k_logits=config.dataset.top_k_logits,
                    batch_size=config.dataset.batch_size,
                    system_prompt_path=config.system_prompt_path,
                    train_without_logits=config.training.train_without_logits,
                ),
                name_for_wandb="finer_val_loss",
            )
        ]
    else:
        loss_evals = []

    generate_evals = []
    # TODO: generalize beyond finer
    if config.do_val_gen_eval:
        generate_evals.append(
            GenerationEvalConfig(
                dataset=FinerGenerateDataset.Config(
                    num_problems=config.num_generate_problems,
                    system_prompt_path=config.system_prompt_path,
                    dataset_split=config.val_gen_split,
                ),
                name_for_wandb="finer",
                generate_max_new_tokens=1024,
                num_samples=1,
                temperature=config.generate_temperature,
                batch_size=config.generate_batch_size,
            )
        )
    if config.do_train_gen_eval:
        generate_evals.append(
            GenerationEvalConfig(
                dataset=FinerGenerateDataset.Config(
                    num_problems=config.num_train_generate_problems,
                    system_prompt_path=config.system_prompt_path,
                    dataset_split=config.train_gen_split,
                ),
                name_for_wandb="finer_train",
                generate_max_new_tokens=1024,
                num_samples=1,
                temperature=config.generate_temperature,
                batch_size=config.generate_batch_size,
            )
        )

    # Create KV cache factory config
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        kv_cache_factory_config = create_kv_cache_factory(config.kv_cache, temp_path)
        
        # Create cartridges TrainConfig
        print("[Distill] Setting up training configuration...")
        # Generate run name from dataset source
        
        train_config = TrainConfig(
            name=config.run_name,
            output_dir=str(output_dir),
            run_dir=str(config.run_dir),
            
            train_temperature=config.training.train_temperature,
            val_temperature=config.training.val_temperature,
            
            # Model
            model=HFModelConfig(
                pretrained_model_name_or_path=config.model_name,
                model_cls=FlexLlamaForCausalLM,  # Use custom model that supports TrainableCache
            ),
            
            # Dataset
            dataset=dataset_cls.Config(
                data_sources=[DataSource(path=train_dataset_path, type="local")],
                packing_mode=config.dataset.packing_mode,
                packed_seq_length=config.dataset.packed_seq_length,
                targets=config.dataset.targets,
                top_k_logits=config.dataset.top_k_logits,
                batch_size=config.dataset.batch_size,
                filter_incorrect=config.input_dataset.filter_incorrect,
                ground_truth_target=config.input_dataset.ground_truth_target,
                system_prompt_path=config.system_prompt_path,
                train_without_logits=config.training.train_without_logits,
            ),

            # Loss evals
            loss_eval_every_n_steps=100,
            loss_evals=loss_evals,

            # Generate evals
            generation_server_type=config.generation_server_type,
            toka_server_config=config.toka_server_config,
            generate_before_training=config.generate_before_training,
            generate_eval_every_n_steps=config.generate_eval_every_n_steps,
            generate_evals=generate_evals,
            
            # Training
            global_batch_size=config.training.global_batch_size,
            epochs=config.training.epochs,
            device=config.training.device,
            distributed_backend=config.training.distributed_backend,
            optimizer=config.training.optimizer,
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
            gradient_checkpointing=config.training.gradient_checkpointing,
            
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
                name=config.wandb.name or run_name,
            ) if config.wandb.enabled else None,
            
            # Misc
            seed=config.training.seed,

            dataloader_num_workers=config.dataloader_num_workers,
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

@pydra.main(DistillationConfig)
def main(config: DistillationConfig):
    """CLI entry point for cartridge distillation."""
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

