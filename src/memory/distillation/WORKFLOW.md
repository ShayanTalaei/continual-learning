# Memory Distillation → Cartridge Training Workflow

This document describes the complete workflow for distilling agent memory into a trained cartridge.

## Overview

```
Agent Checkpoint → Data Generation → Conversion → Cartridge Training → HuggingFace → KVMemoryAgent
```

## Detailed Steps

### Step 1: Generate Distillation Dataset

Generate training data from a HistoryAgent checkpoint using the teacher model.

```bash
python -m src.memory.distillation.data_generation \
  --config configs/distillation/example_gen.yaml
```

**Config:** `configs/distillation/example_gen.yaml`
- `checkpoint_dir`: Path to HistoryAgent checkpoint with memory snapshots
- `top_logprobs`: Number of top-k logprobs to capture (e.g., 20)
- `num_threads`: Parallel processing threads for faster generation
- `hf_repo_id`: (Optional) Upload to HuggingFace

**Output:**
- JSONL dataset with messages, system_prompt, metadata, type
- OpenAI-style logprobs for knowledge distillation
- Can be uploaded directly to HuggingFace

**Example output:**
```json
{
  "messages": [
    {"role": "system", "content": "...", "token_ids": null, "logprobs": null},
    {"role": "user", "content": "...", "token_ids": null, "logprobs": null},
    {"role": "assistant", "content": "...", "token_ids": null, "logprobs": {...}}
  ],
  "system_prompt": "You are a helpful assistant...",
  "metadata": {...},
  "type": "memory_distillation"
}
```

### Step 2: Convert to Cartridges Format (if needed)

If you didn't upload to HF in Step 1, or need to convert locally:

```bash
python src/memory/distillation/convert_to_cartridges.py \
  --input outputs/distillation/data/example_run/dataset.jsonl \
  --output outputs/distillation/data/example_run/dataset.parquet \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --min-prob-mass 0.99
```

**This step:**
- Converts OpenAI logprobs → `FlatTopLogprobs` (sparse numpy arrays)
- Tokenizes all messages → adds token_ids
- Saves as parquet with proper structure

**Note:** If using `distill_into_cartridge.py` with `is_converted: false`, this happens automatically!

### Step 3: Train Cartridge with Knowledge Distillation

Train a compressed KV cache cartridge using the distillation dataset.

**Single GPU:**
```bash
python -m src.memory.distillation.distill_into_cartridge \
  --config configs/distillation/cartridge_train_example.yaml
```

**Multi-GPU (e.g., 4 GPUs):**
```bash
torchrun --nproc_per_node=4 \
  -m src.memory.distillation.distill_into_cartridge \
  --config configs/distillation/cartridge_train_example.yaml
```

**Multi-Node (e.g., 2 nodes, 4 GPUs each):**
```bash
# Node 0 (master):
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
  --master_addr=MASTER_NODE_IP --master_port=29500 \
  -m src.memory.distillation.distill_into_cartridge \
  --config configs/distillation/cartridge_train_example.yaml

# Node 1:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
  --master_addr=MASTER_NODE_IP --master_port=29500 \
  -m src.memory.distillation.distill_into_cartridge \
  --config configs/distillation/cartridge_train_example.yaml
```

**Config:** `configs/distillation/cartridge_train_example.yaml`
```yaml
input_dataset:
  repo_id: "stalaei/distillation-dataset-test"
  is_converted: false  # Auto-convert if needed

output:
  local_dir: "./outputs/cartridges/my-cartridge-v1"
  hf_repo_id: "stalaei/my-cartridge-v1"
  upload_to_hf: true

model_name: "meta-llama/Llama-3.1-8B-Instruct"

kv_cache:
  method: "random"
  num_tokens: 128
  num_frozen_tokens: 1

training:
  epochs: 5
  global_batch_size: 8
  lr: 1e-4
  top_k_logits: 20  # Match data generation!
```

**What happens:**
1. Loads dataset from HuggingFace
2. Converts to cartridges format if needed
3. Initializes KV cache (random or from text)
4. Trains using **knowledge distillation loss**:
   ```python
   loss = -teacher_probs * log(student_probs)
   ```
5. Saves checkpoints during training
6. Uploads final cartridge to HuggingFace

**Output:**
- `cache-final.pt` or `cache-step{N}.pt`: Trained cartridge
- `distillation_config.yaml`: Training configuration
- HuggingFace model repo with README

### Step 4: Use Cartridge in KVMemoryAgent

Now you can use the trained cartridge in your agent!

```python
from src.memory.kv_cache import KVCacheMemory, KVCacheMemoryConfig, KVArtifact
from src.agent.kv_memory_agent import KVMemoryAgent, KVMemoryAgentConfig

# Configure memory with your cartridge
memory_config = KVCacheMemoryConfig(
    artifact=KVArtifact(
        id="stalaei/my-cartridge-v1",
        source="huggingface",
        force_redownload=False,
    ),
    tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
)

# Create agent with KV memory
agent_config = KVMemoryAgentConfig(
    memory_config=memory_config,
    system_prompt="You are a helpful assistant. Use the attached cartridge as your long-term memory.",
    lm_config={"model": "meta-llama/Llama-3.1-8B-Instruct", ...},
)

agent = KVMemoryAgent(agent_config)
```

The cartridge will be automatically downloaded from HuggingFace and loaded into the agent's memory!

## Key Concepts

### Knowledge Distillation

The cartridges training uses the logprobs from your teacher model to train the student (cartridge):

- **Teacher**: Your large model that generated the responses (e.g., Llama-3.1-8B)
- **Student**: The cartridge (compressed KV cache)
- **Loss**: Cross-entropy between teacher and student distributions

This preserves not just the top-1 answer but the **uncertainty** and alternative choices!

### Cartridge Size vs Accuracy

- **More tokens** (e.g., 256): Better compression, slower inference
- **Fewer tokens** (e.g., 64): Less memory, faster inference
- **Frozen tokens**: Prevent catastrophic forgetting during training

### Auto-Conversion

Set `is_converted: false` in the training config to automatically convert intermediate datasets during training. No need to run `convert_to_cartridges.py` separately!

## Complete Example

**Single GPU:**
```bash
# 1. Generate dataset from checkpoint
python -m src.memory.distillation.data_generation \
  --config configs/distillation/example_gen.yaml

# 2. Train cartridge (auto-converts if needed)
python -m src.memory.distillation.distill_into_cartridge \
  --config configs/distillation/cartridge_train_example.yaml

# 3. Use in agent (see Python code above)
```

**Multi-GPU (4 GPUs):**
```bash
# 1. Generate dataset (with 20 threads for speed)
python -m src.memory.distillation.data_generation \
  --config configs/distillation/example_gen.yaml

# 2. Train cartridge on 4 GPUs
torchrun --nproc_per_node=4 \
  -m src.memory.distillation.distill_into_cartridge \
  --config configs/distillation/cartridge_train_example.yaml

# 3. Use in agent (see Python code above)
```

## File Reference

- `src/memory/distillation/data_generation.py` - Generate distillation dataset
- `src/memory/distillation/convert_to_cartridges.py` - Convert format (optional)
- `src/memory/distillation/distill_into_cartridge.py` - Train cartridge
- `configs/distillation/example_gen.yaml` - Data generation config
- `configs/distillation/cartridge_train_example.yaml` - Training config

## Multi-GPU Training

The training script **automatically detects** multi-GPU setups via the `LOCAL_RANK` environment variable set by `torchrun`. No config changes needed!

**What happens automatically:**
- ✅ Model wrapped in DistributedDataParallel (DDP)
- ✅ Data distributed across GPUs with DistributedSampler
- ✅ Gradients synchronized across GPUs
- ✅ Only rank 0 saves checkpoints and logs
- ✅ Effective batch size = `global_batch_size / num_gpus` per GPU

**Important:**
- Set `global_batch_size` to the **total** batch size across all GPUs
- Example: `global_batch_size: 32` on 4 GPUs = 8 samples per GPU
- Use `distributed_backend: "nccl"` for GPU training (default, fastest)
- Use `distributed_backend: "gloo"` only for CPU/debugging

## Tips

1. **Match top_logprobs**: Ensure `top_logprobs` in data generation matches `top_k_logits` in training
2. **Use multithreading**: Set `num_threads: 20` for faster data generation
3. **Experiment with size**: Try different `num_tokens` values (64, 128, 256)
4. **Monitor training**: Enable WandB to track loss and generation quality
5. **Save intermediate datasets**: Upload to HF after generation for reuse
6. **Multi-GPU training**: Use `torchrun` for automatic distributed training across GPUs
7. **Batch size tuning**: On multi-GPU, set `global_batch_size` high (e.g., 32, 64) for better GPU utilization

## Troubleshooting

**Issue**: `ValueError: CARTRIDGES_DIR is not set`
- **Solution**: The script automatically sets this. If you see this error, ensure you're running the script from the repo root or set manually:
  ```bash
  export CARTRIDGES_DIR=/path/to/continual-learning/third_party/cartridges
  export CARTRIDGES_OUTPUT_DIR=/path/to/outputs  # optional
  ```

**Issue**: Import errors for cartridges
- **Solution**: The script adds cartridges to sys.path at runtime, ignore linter warnings

**Issue**: Out of memory during training
- **Solution**: Reduce `global_batch_size` or `packed_seq_length`

**Issue**: Poor cartridge quality
- **Solution**: 
  - Increase `num_tokens` (more capacity)
  - Train longer (`epochs`)
  - Ensure `top_k_logits` matches data generation
  - Check if dataset has logprobs

**Issue**: Conversion fails
- **Solution**: Ensure dataset has `system_prompt` and `type` fields (use latest data_generation.py)

