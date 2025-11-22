# Cartridge Composition Experiment - AMD & PepsiCo 10-K

Complete setup to replicate the cartridge composition experiment from the paper using FinanceBench data.

## Overview

This experiment trains two cartridges (AMD and PepsiCo) and evaluates:
1. **Single-document QA**: Each cartridge on its own document
2. **Composition QA**: Combined cartridges `[AMD, PepsiCo]` on multi-document questions
3. **Metrics**: QA accuracy (Exact Match + F1) instead of perplexity

## Prerequisites

- Conda environment: `composition`
- GPU: 4x NVIDIA GB200 (on `slurm-compute-node-015`)
- Python packages: `pydrantic`, `torch`, `transformers`, `PyPDF2`

## Step-by-Step Instructions

### Step 1: Setup FinanceBench Data

```bash
# SSH into compute node
ssh slurm-compute-node-015

# Activate conda environment
conda activate composition

# Navigate to cartridges directory
cd /home/ubuntu/ding/continual-learning/third_party/cartridges
export CARTRIDGES_DIR=$(pwd)

# Install dependencies if needed
pip install PyPDF2 pydrantic

# Download FinanceBench and extract AMD/PepsiCo 10-Ks
python examples/10k/setup_financebench.py
```

**This will:**
- Clone https://github.com/patronus-ai/financebench
- Extract AMD and PepsiCo 10-K PDFs from FinanceBench
- Save as: `data/10k/amd_10k_financebench.txt` and `data/10k/pepsi_10k_financebench.txt`
- Extract Q&A pairs for evaluation

### Step 2: Start Tokasaurus Server for Synthesis

```bash
# In a separate terminal/tmux session
cd /home/ubuntu/ding/continual-learning/third_party/tokasaurus
conda activate composition

# Start server (this will run in foreground)
toka model=Qwen/Qwen3-4b kv_cache_num_tokens='(512 * 1024)' port=10210
```

### Step 3: Generate Training Data (Synthesis)

```bash
# Back in the first terminal
cd /home/ubuntu/ding/continual-learning/third_party/cartridges
export CARTRIDGES_DIR=$(pwd)
export TOKASAURUS_URL=http://localhost:10210

# Synthesize AMD training data (~1-2 hours for 512 samples)
python examples/10k/amd_synthesize.py

# Synthesize PepsiCo training data (~1-2 hours)
python examples/10k/pepsi_synthesize.py
```

**Output:**
- `data/10k/amd_10k_synthesize_qwen-qwen3-4b_n512-0.parquet`
- `data/10k/pepsi_10k_synthesize_qwen-qwen3-4b_n512-0.parquet`

### Step 4: Stop Tokasaurus (Synthesis Done)

```bash
# In the Tokasaurus terminal, press Ctrl+C to stop
```

### Step 5: Train Cartridges

```bash
cd /home/ubuntu/ding/continual-learning/third_party/cartridges
export CARTRIDGES_DIR=$(pwd)
export CARTRIDGES_OUTPUT_DIR=$(pwd)/outputs

# Train AMD cartridge on GPU 0 (runs in background)
CUDA_VISIBLE_DEVICES=0 nohup python examples/10k/amd_train.py > /tmp/amd_train.log 2>&1 &

# Train PepsiCo cartridge on GPU 1 (runs in background)
CUDA_VISIBLE_DEVICES=1 nohup python examples/10k/pepsi_train.py > /tmp/pepsi_train.log 2>&1 &

# Monitor training
tail -f /tmp/amd_train.log
tail -f /tmp/pepsi_train.log

# Wait for both to complete (~30-60 minutes each)
wait
```

**Output:**
- `outputs/amd-10k-cartridge-<timestamp>/cache_final.pt`
- `outputs/pepsi-10k-cartridge-<timestamp>/cache_final.pt`

### Step 6: Evaluate QA Accuracy

```bash
# TODO: Create evaluation script
# This will evaluate:
# 1. AMD cartridge alone on AMD questions
# 2. PepsiCo cartridge alone on PepsiCo questions
# 3. Composed [AMD, PepsiCo] on multi-document questions

python examples/10k/evaluate_composition_qa.py \
    --amd-cartridge outputs/amd-10k-cartridge-<timestamp>/cache_final.pt \
    --pepsi-cartridge outputs/pepsi-10k-cartridge-<timestamp>/cache_final.pt \
    --eval-data data/10k/eval/
```

## File Structure

```
third_party/cartridges/
├── data/
│   ├── financebench/              # Cloned FinanceBench repo
│   └── 10k/
│       ├── amd_10k_financebench.txt              # AMD 10-K from FinanceBench
│       ├── pepsi_10k_financebench.txt            # PepsiCo 10-K from FinanceBench
│       ├── amd_10k_synthesize_*.parquet          # Synthesized training data
│       ├── pepsi_10k_synthesize_*.parquet        # Synthesized training data
│       └── eval/
│           ├── amd_qa_financebench.json          # AMD Q&A from FinanceBench
│           └── pepsi_qa_financebench.json        # PepsiCo Q&A from FinanceBench
├── examples/10k/
│   ├── setup_financebench.py      # Download & extract FinanceBench data
│   ├── amd_synthesize.py          # Generate AMD training data
│   ├── pepsi_synthesize.py        # Generate PepsiCo training data
│   ├── amd_train.py               # Train AMD cartridge
│   ├── pepsi_train.py             # Train PepsiCo cartridge
│   ├── evaluate_composition_qa.py # Evaluate QA accuracy (TODO)
│   └── COMPOSITION_EXPERIMENT.md  # This file
└── outputs/
    ├── amd-10k-cartridge-*/       # Trained AMD cartridge
    └── pepsi-10k-cartridge-*/     # Trained PepsiCo cartridge
```

## Key Configuration

- **Model**: Qwen/Qwen3-4b (for synthesis and training)
- **Cartridge size**: Default (typically 1024-2048 tokens)
- **Training**: 3 epochs, lr=2e-2, batch_size=16
- **Synthesis**: 512 Q&A pairs per document
- **Chunk size**: Fixed 8192 tokens per chunk (for self-study synthesis)

## Expected Results

The paper showed that composition of independently-trained cartridges:
- Works "off-the-shelf" without retraining
- Substantially outperforms single cartridges on multi-document questions
- Outperforms ICL which struggles with context length limits

## Troubleshooting

**"Module not found" errors:**
```bash
conda activate composition
pip install pydrantic PyPDF2 torch transformers datasets
```

**Tokasaurus connection errors:**
```bash
# Check if server is running
curl http://localhost:10210/health

# If not, restart server
toka model=Qwen/Qwen3-4b kv_cache_num_tokens='(512 * 1024)' port=10210
```

**GPU memory issues:**
```bash
# Check GPU usage
nvidia-smi

# Clear cache if needed
python -c "import torch; torch.cuda.empty_cache()"
```

## Next Steps

1. ✅ Setup FinanceBench data
2. ✅ Generate training data via synthesis
3. ✅ Train AMD and PepsiCo cartridges
4. ⏳ Create evaluation script for QA accuracy
5. ⏳ Run evaluation and compare results

## References

- Paper: "Cartridges: Trainable KV Caches for Long-Context LLM Inference"
- FinanceBench: https://github.com/patronus-ai/financebench
- Cartridges Repo: https://github.com/HazyResearch/cartridges

