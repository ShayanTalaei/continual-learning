# Memory Distillation Dataset Generation

This module generates training datasets for distilling a HistoryAgent's behavior into a smaller model using the [cartridges](../../third_party/cartridges) framework.

## Overview

The dataset generation process has two stages:

1. **Generation Stage** (`data_generation.py`): Creates an intermediate JSONL dataset with all raw information
2. **Conversion Stage** (`convert_to_cartridges.py`): Converts to cartridges-compatible parquet format

This two-stage design keeps our codebase clean by avoiding direct cartridges dependencies in the main generation code.

## Quick Start

### 1. Generate Intermediate Dataset

```bash
python -m src.memory.distillation.data_generation \
  --config configs/distillation/example_gen.yaml
```

This will:
- Load a HistoryAgent checkpoint with memory snapshots
- For each sample in the history, construct prompts using a memory formation strategy
- Call the LM to generate responses with logprobs
- Save raw data to JSONL format

### 2. Convert to Cartridges Format

```bash
python src/memory/distillation/convert_to_cartridges.py \
  --input /path/to/dataset.jsonl \
  --output /path/to/dataset.parquet \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --min-prob-mass 0.99
```

This will:
- Load the tokenizer for the specified model
- Tokenize all message content
- Convert OpenAI-style logprobs to cartridges TopLogprobs format
- Flatten logprobs to sparse representation (keeping top-k that sum to min-prob-mass)
- Write to parquet format compatible with cartridges training

## Configuration

### Data Generation Config (`configs/distillation/example_gen.yaml`)

Key parameters:

- **`checkpoint_dir`**: Path to HistoryAgent checkpoint containing `memory_*.jsonl` snapshots
- **`output_dir`**: Where to save the intermediate dataset
- **`output_format`**: `"jsonl"` (recommended) or `"parquet"`
- **`strategy`**: Memory formation strategy (`"exclude_current"`, `"full_memory"`, etc.)
- **`lm_model`**: Model name for generation (must match your Tokasaurus server)
- **`top_logprobs`**: Number of top logprobs to capture (e.g., 5, 10, 20)
- **`max_samples`**: Limit on samples to generate (null = all samples)

### Conversion Parameters

- **`--model`**: Model name for tokenizer (should match generation model)
- **`--min-prob-mass`**: Probability mass threshold for sparse logprobs (default: 0.99)
  - Higher = more logprobs kept per token (larger file size)
  - Lower = fewer logprobs kept (smaller file, potential information loss)

## Intermediate Format

The intermediate JSONL format looks like:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant...",
      "token_ids": null,
      "logprobs": null
    },
    {
      "role": "user",
      "content": "What is 2+2?",
      "token_ids": null,
      "logprobs": null
    },
    {
      "role": "assistant",
      "content": "4",
      "token_ids": null,
      "logprobs": {
        "content": [
          {
            "token": "4",
            "logprob": -0.1,
            "top_logprobs": [
              {"token": "4", "logprob": -0.1},
              {"token": "four", "logprob": -2.3},
              ...
            ]
          }
        ]
      }
    }
  ],
  "system_prompt": "You are a helpful assistant...",
  "metadata": {
    "sample_idx": 0,
    "metrics": {
      "duration": 1.2,
      "input_tokens": 150,
      "output_tokens": 20
    }
  },
  "type": "memory_distillation"
}
```

## Cartridges Format

The final cartridges parquet format contains `Conversation` objects with:

```python
Conversation(
    messages=[
        Message(
            role="system",
            content="...",
            token_ids=[128000, 882, ...],  # Tokenized content
            top_logprobs=None  # System message doesn't need logprobs
        ),
        Message(
            role="user",
            content="...",
            token_ids=[128000, 1234, ...],
            top_logprobs=None  # User message doesn't need logprobs
        ),
        Message(
            role="assistant",
            content="...",
            token_ids=[128000, 5678, ...],
            top_logprobs=FlatTopLogprobs(  # Sparse representation
                token_idx=np.array([0, 0, 1, 1, ...]),
                token_id=np.array([5678, 91, 9012, 42, ...]),
                logprobs=np.array([-0.1, -2.3, -0.05, -3.1, ...]),
                shape=(num_tokens, top_k)
            )
        )
    ],
    system_prompt="...",
    metadata={...},
    type="memory_distillation"
)
```

## Memory Formation Strategies

Strategies control what memory context is provided when generating the training data:

- **`exclude_current`**: Exclude the current sample from memory (forces model to generalize)
- **`full_memory`**: Use all available memory
- **`rolling_window`**: Use only the last K memory entries
- **`failure_focus`**: Prioritize failed samples in memory

Strategies are defined in `src/memory/distillation/strategies/`.

## Design Rationale

**Why two stages?**

1. **Clean separation of concerns**: Generation logic doesn't need cartridges imports
2. **Debugging**: Can inspect intermediate JSONL format easily
3. **Flexibility**: Can re-convert with different tokenizers/settings without regenerating
4. **Efficiency**: Generation is expensive (LM calls), conversion is cheap

**Why keep raw logprobs in intermediate format?**

The OpenAI-style logprobs contain all necessary information and are human-readable. The conversion to cartridges' sparse format is a lossy compression step that should be done last.

## Training with Cartridges

Once you have the parquet dataset, use it with cartridges training:

```python
from cartridges.train import TrainConfig
from cartridges.datasets import TrainDataset

config = TrainConfig(
    data_sources=["/path/to/dataset.parquet"],
    # ... other training config
)
config.run()
```

See [cartridges documentation](../../third_party/cartridges/docs/training.md) for details.

## Troubleshooting

**"No memory snapshots found"**
- Ensure checkpoint_dir points to a specific episode checkpoint (e.g., `ep_000200`)
- Check that `memory_*.jsonl` files exist in that directory

**"Failed to extract logprobs"**
- Ensure `top_logprobs` is set in the generation config
- Verify your Tokasaurus server supports logprobs

**"Token encoding mismatch"**
- Ensure the `--model` parameter in conversion matches the generation `lm_model`
- Some tokens may not encode cleanly; these are marked with token_id=-1

**Large file sizes**
- Reduce `top_logprobs` (e.g., from 20 to 5)
- Increase `min_prob_mass` in conversion (e.g., from 0.99 to 0.95)
- Use fewer samples with `max_samples`
