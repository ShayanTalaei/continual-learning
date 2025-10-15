#!/bin/bash
# Install cartridges dependencies into continual_learning conda environment
#
# Usage:
#   chmod +x install_cartridges_deps.sh
#   ./install_cartridges_deps.sh

set -e  # Exit on error

echo "=========================================="
echo "Installing Cartridges Dependencies"
echo "=========================================="
echo ""

# Check if we're in a conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "❌ Error: No conda environment activated!"
    echo "Please run: conda activate continual_learning"
    exit 1
fi

echo "✓ Conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Check if requirements file exists
REQUIREMENTS_FILE="third_party/cartridges/requirements-distillation.txt"
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "❌ Error: Requirements file not found: $REQUIREMENTS_FILE"
    exit 1
fi

echo "Installing dependencies from: $REQUIREMENTS_FILE"
echo ""

# Install dependencies
pip install -r "$REQUIREMENTS_FILE"

echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="
echo ""

# Verify installation
python3 << 'EOF'
import sys
import os
from pathlib import Path

REPO_ROOT = Path.cwd()
os.environ["CARTRIDGES_DIR"] = str(REPO_ROOT / "third_party" / "cartridges")
os.environ["CARTRIDGES_OUTPUT_DIR"] = str(REPO_ROOT / "outputs")
sys.path.insert(0, str(REPO_ROOT / "third_party" / "cartridges"))

print("Testing imports...")

try:
    from cartridges.train import TrainConfig
    print("  ✓ cartridges.train")
except ImportError as e:
    print(f"  ❌ cartridges.train: {e}")
    sys.exit(1)

try:
    from cartridges.cache import TrainableCache
    print("  ✓ cartridges.cache")
except ImportError as e:
    print(f"  ❌ cartridges.cache: {e}")
    sys.exit(1)

try:
    from cartridges.structs import Conversation
    print("  ✓ cartridges.structs")
except ImportError as e:
    print(f"  ❌ cartridges.structs: {e}")
    sys.exit(1)

try:
    import torch
    print(f"  ✓ torch {torch.__version__}")
except ImportError as e:
    print(f"  ❌ torch: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"  ✓ transformers {transformers.__version__}")
except ImportError as e:
    print(f"  ❌ transformers: {e}")
    sys.exit(1)

try:
    import datasets
    print(f"  ✓ datasets")
except ImportError as e:
    print(f"  ❌ datasets: {e}")
    sys.exit(1)

try:
    import pandas
    print(f"  ✓ pandas")
except ImportError as e:
    print(f"  ❌ pandas: {e}")
    sys.exit(1)

try:
    import huggingface_hub
    print(f"  ✓ huggingface_hub")
except ImportError as e:
    print(f"  ❌ huggingface_hub: {e}")
    sys.exit(1)

print("")
print("✅ All imports successful!")
print("")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
EOF

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Test data generation:"
echo "     python -m src.memory.distillation.data_generation \\"
echo "       --config configs/distillation/example_gen.yaml"
echo ""
echo "  2. Train cartridge (single GPU):"
echo "     python -m src.memory.distillation.distill_into_cartridge \\"
echo "       --config configs/distillation/cartridge_train_example.yaml"
echo ""
echo "  3. Train cartridge (multi-GPU):"
echo "     torchrun --nproc_per_node=4 \\"
echo "       -m src.memory.distillation.distill_into_cartridge \\"
echo "       --config configs/distillation/cartridge_train_example.yaml"
echo ""
echo "See INSTALL_CARTRIDGES.md and src/memory/distillation/WORKFLOW.md for details."
echo ""


