# Training Cartridges on AMD and PepsiCo 10-K Documents

This directory contains all the necessary files to train cartridges on AMD and PepsiCo 10-K filings.

## Files

### Documents
- `data/10k/amd_10k.txt` - AMD 10-K filing (FY 2024, 408,465 chars)
- `data/10k/pepsi_10k.txt` - PepsiCo 10-K filing (FY 2024, 488,586 chars)

### Synthesis Configs
- `examples/10k/amd_synthesize.py` - Generate Q&A pairs from AMD 10-K
- `examples/10k/pepsi_synthesize.py` - Generate Q&A pairs from PepsiCo 10-K

### Training Configs
- `examples/10k/amd_train.py` - Train AMD cartridge
- `examples/10k/pepsi_train.py` - Train PepsiCo cartridge

## Pipeline

### Step 1: Start Tokasaurus Server (Required for Synthesis)

You need a running Tokasaurus server with Qwen3-4B to generate synthetic Q&A pairs:

```bash
cd /home/ubuntu/ding/continual-learning/third_party/tokasaurus
toka model=Qwen/Qwen3-4b kv_cache_num_tokens='(512 * 1024)' port=10210
```

### Step 2: Run Data Synthesis

Generate Q&A training pairs from the documents:

```bash
cd /home/ubuntu/ding/continual-learning/third_party/cartridges
export CARTRIDGES_DIR=$(pwd)
export TOKASAURUS_URL=http://localhost:10210

# Synthesize AMD data
python examples/10k/amd_synthesize.py

# Synthesize Pepsi data (after AMD completes)
python examples/10k/pepsi_synthesize.py
```

This will create:
- `data/10k/amd_10k_synthesize_qwen-qwen3-4b_n512-0.parquet`
- `data/10k/pepsi_10k_synthesize_qwen-qwen3-4b_n512-0.parquet`

### Step 3: Train Cartridges

Train the cartridges on separate GPUs:

```bash
cd /home/ubuntu/ding/continual-learning/third_party/cartridges
export CARTRIDGES_DIR=$(pwd)
export CARTRIDGES_OUTPUT_DIR=$(pwd)/outputs

# Train AMD cartridge on GPU 0
CUDA_VISIBLE_DEVICES=0 python examples/10k/amd_train.py

# Train Pepsi cartridge on GPU 1 (can run in parallel)
CUDA_VISIBLE_DEVICES=1 python examples/10k/pepsi_train.py
```

### Step 4: Test Cartridges

After training, cartridges will be saved in:
- `outputs/amd-10k-cartridge-<timestamp>/`
- `outputs/pepsi-10k-cartridge-<timestamp>/`

Test with Tokasaurus:

```python
import requests

# Query using AMD cartridge
response = requests.post(
    "http://localhost:10210/custom/cartridge/chat/completions",
    json={
        "model": "default",
        "messages": [{"role": "user", "content": "What was AMD's revenue in FY 2024?"}],
        "max_tokens": 200,
        "cartridges": [{"id": "amd_10k", "source": "local"}]
    }
)
print(response.json())
```

## Hardware Requirements

- **GPUs**: 4x NVIDIA GB200 (189GB VRAM each) available on slurm-compute-node-015
- **Memory**: 1.6 TiB RAM
- **Storage**: ~10GB for models, ~1GB for cartridges

## Training Time Estimates

- Synthesis: ~1-2 hours per document (512 samples)
- Training: ~30-60 minutes per cartridge (3 epochs)

## Notes

- Synthesis requires a running Tokasaurus/LLM server
- Training uses Qwen3-4B model (~8GB VRAM)
- Cartridges are saved in Tokasaurus-compatible format

