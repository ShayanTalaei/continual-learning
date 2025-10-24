# Installing Cartridges Dependencies

This guide explains how to install the necessary dependencies for cartridge training into your existing `continual_learning` conda environment.

## Quick Installation

Activate your conda environment and install all dependencies:

```bash
# Activate your environment
conda activate continual_learning

# Install cartridges dependencies
pip install -r third_party/cartridges/requirements-distillation.txt
```

## Detailed Installation

If you prefer to install step by step:

```bash
conda activate continual_learning

# Core PyTorch and ML libraries
pip install torch  # Or use conda for CUDA-specific versions
pip install transformers>=4.49.0,<=4.55
pip install datasets
pip install numpy
pip install pandas
pip install pyarrow

# Cartridges-specific dependencies
pip install openai
pip install einops
pip install tqdm
pip install wandb
pip install pydrantic
pip install tiktoken
pip install peft
pip install evaluate
pip install markdown

# HuggingFace and utilities
pip install huggingface_hub
pip install pyyaml

# Optional but recommended
pip install matplotlib
pip install seaborn
```

## Verify Installation

Test that cartridges can be imported:

```bash
cd /sailhome/stalaei/code/continual-learning

python3 << 'EOF'
import sys
import os
from pathlib import Path

# Set environment variables
REPO_ROOT = Path.cwd()
os.environ["CARTRIDGES_DIR"] = str(REPO_ROOT / "third_party" / "cartridges")
os.environ["CARTRIDGES_OUTPUT_DIR"] = str(REPO_ROOT / "outputs")
sys.path.insert(0, str(REPO_ROOT / "third_party" / "cartridges"))

# Test imports
from cartridges.train import TrainConfig
from cartridges.cache import TrainableCache
from cartridges.structs import Conversation
print("✅ All cartridges imports successful!")
EOF
```

## GPU Support

If you need specific CUDA versions for PyTorch:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or use conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'XXX'`

Install the missing package:
```bash
pip install XXX
```

### Issue: Version conflicts

If you encounter version conflicts, try installing in this order:
1. PyTorch first (specific CUDA version if needed)
2. transformers with version constraint
3. Other dependencies

```bash
# Example
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.55
pip install -r third_party/cartridges/requirements-distillation.txt
```

### Issue: `pydrantic` not found

`pydrantic` might need to be installed from source:
```bash
pip install pydrantic
# or if that fails:
pip install git+https://github.com/pydrantic/pydrantic.git
```

## What Gets Installed

- **torch**: Deep learning framework
- **transformers**: Hugging Face transformers library for LLMs
- **datasets**: Hugging Face datasets library
- **pandas/pyarrow**: Data manipulation and parquet file handling
- **openai**: OpenAI API client (used for logprobs format compatibility)
- **einops**: Tensor operations
- **wandb**: Experiment tracking (optional, can be disabled)
- **peft**: Parameter-efficient fine-tuning
- **pydrantic**: Configuration management
- **huggingface_hub**: Upload/download models and datasets
- **pyyaml**: YAML configuration files

## Next Steps

After installation, you can:

1. **Test the distillation pipeline**:
   ```bash
   python -m src.memory.distillation.distill_into_cartridge \
     --config configs/distillation/cartridge_train_example.yaml
   ```

2. **Run multi-GPU training**:
   ```bash
   torchrun --nproc_per_node=4 \
     -m src.memory.distillation.distill_into_cartridge \
     --config configs/distillation/cartridge_train_example.yaml
   ```

See `src/memory/distillation/WORKFLOW.md` for complete usage instructions.


