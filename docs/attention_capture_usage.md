# Running Attention Capture with history_agent_cities.yaml

## Prerequisites

1. **Environment Setup**: Activate the conda environment and set required environment variables:
```bash
export CARTRIDGES_DIR=/u/stalaei/code/continual-learning/third_party/cartridges
export CARTRIDGES_OUTPUT_DIR=/tmp/cartridges_output  # or your preferred output directory
export PYTHONPATH=/u/stalaei/code/continual-learning/third_party/cartridges:$PYTHONPATH
```

2. **Memory Snapshot**: You need a memory snapshot file (JSONL format) from a previous training run. This contains the agent's memory state.

## Option 1: Using an Existing Memory Snapshot

If you have a memory snapshot from a previous run (typically saved in `outputs/.../memories/memory_{id}.jsonl`):

```bash
/projects/bfsg/stalaei/conda/envs/continual_learning/bin/python -m src.attention_capture.run_eval_cli \
    --config configs/attention_eval/history_agent_cities.yaml \
    --memory-snapshot /path/to/memory_700.jsonl \
    --output-dir outputs/attention_eval/history_agent_cities \
    --model-type llama \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --device cuda \
    --mode generate
```

## Option 2: Creating a Memory Snapshot First

If you don't have a memory snapshot yet, you need to run the training loop first:

```bash
# Step 1: Run the training loop to generate a memory snapshot
/projects/bfsg/stalaei/conda/envs/continual_learning/bin/python -m src.main \
    --config configs/attention_eval/history_agent_cities.yaml

# This will create a memory snapshot in outputs/attention_eval/history_agent_cities/memories/memory_{id}.jsonl

# Step 2: Run attention capture with the generated snapshot
/projects/bfsg/stalaei/conda/envs/continual_learning/bin/python -m src.attention_capture.run_eval_cli \
    --config configs/attention_eval/history_agent_cities.yaml \
    --memory-snapshot outputs/attention_eval/history_agent_cities/memories/memory_700.jsonl \
    --output-dir outputs/attention_eval/history_agent_cities \
    --model-type llama \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --device cuda \
    --mode generate
```

## Option 3: Testing with Empty Memory

For testing purposes, you can use an empty memory snapshot:

```bash
# Create an empty memory snapshot
echo '[]' > /tmp/empty_memory.jsonl

# Run with empty memory
/projects/bfsg/stalaei/conda/envs/continual_learning/bin/python -m src.attention_capture.run_eval_cli \
    --config configs/attention_eval/history_agent_cities.yaml \
    --memory-snapshot /tmp/empty_memory.jsonl \
    --output-dir /tmp/test_attention_capture/eval_test \
    --model-type llama \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --max-samples 5 \
    --device cuda \
    --mode generate
```

## Command-Line Options

### Required Arguments
- `--config`: Path to the YAML configuration file (e.g., `configs/attention_eval/history_agent_cities.yaml`)
- `--memory-snapshot`: Path to the memory snapshot JSONL file
- `--output-dir`: Directory where results will be written

### Optional Arguments
- `--model-type`: Model family (`llama` or `qwen`), default: `llama`
- `--model-name`: HuggingFace model identifier, default: `meta-llama/Llama-3.1-8B-Instruct`
- `--device`: Torch device (`cuda`, `cuda:0`, `cpu`), default: `cuda` if available
- `--dtype`: Computation dtype (`bfloat16`, `float16`, `float32`), default: `bfloat16`
- `--mode`: Forward pass mode (`train` or `generate`), default: `generate`
- `--max-samples`: Maximum number of validation samples to process (default: all)
- `--cartridge-path`: Optional path to a pre-trained cartridge cache (`.torch` file)
- `--save-qkv`: Save Q/K/V tensors to disk (increases storage requirements)
- `--query-span-tag`: Message tag for query tokens (default: `current_observation`)
- `--layer-idx`: Layer index for aggregation (-1 = last layer), default: `-1`

## Output Files

The command generates:
- `conversation_*.json`: Per-conversation attention summaries
- `metadata.json`: Run metadata and configuration
- `attention_heatmap.png`: Aggregated attention heatmap (if multiple conversations)
- `attention_matrix.pt`: Attention matrix tensor (if multiple conversations)

## Example: Full Command with All Options

```bash
export CARTRIDGES_DIR=/u/stalaei/code/continual-learning/third_party/cartridges
export CARTRIDGES_OUTPUT_DIR=/tmp/cartridges_output
export PYTHONPATH=/u/stalaei/code/continual-learning/third_party/cartridges:$PYTHONPATH

/projects/bfsg/stalaei/conda/envs/continual_learning/bin/python -m src.attention_capture.run_eval_cli \
    --config configs/attention_eval/history_agent_cities.yaml \
    --memory-snapshot outputs/attention_eval/history_agent_cities/memories/memory_700.jsonl \
    --output-dir outputs/attention_eval/history_agent_cities \
    --model-type llama \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --device cuda \
    --dtype bfloat16 \
    --mode generate \
    --max-samples 128 \
    --query-span-tag current_observation \
    --layer-idx -1
```

## Troubleshooting

1. **FileNotFoundError for memory snapshot**: Make sure the memory snapshot path is correct. You can create an empty one for testing: `echo '[]' > /tmp/empty_memory.jsonl`

2. **CUDA out of memory**: Reduce `--max-samples` or use a smaller model

3. **ModuleNotFoundError**: Make sure `CARTRIDGES_DIR` and `PYTHONPATH` are set correctly

4. **Hangs during forward pass**: This should be fixed now! If you still see hangs, check that `capture.record()` is called before `flex_attention_forward()` in the model code.

