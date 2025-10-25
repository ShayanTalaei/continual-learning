---
noteId: "2246d0709dd011f0b67f67467df34666"
tags: []

---

## Quickstart

Get started with continual learning experiments in 3 steps.

### 1. Installation

```bash
# Clone repository
git clone https://github.com/your-org/continual-learning.git
cd continual-learning

# Install dependencies
pip install -r requirements.txt

# Set environment variables (for API keys)
export GOOGLE_API_KEY="your-gemini-key"  # For Gemini
export OPENAI_API_KEY="your-openai-key"  # For OpenAI
```

### 2. Run Your First Experiment

**Simple History Agent on synthetic data:**

```bash
python -m src.main --config configs/toy/history_list.yaml
```

This trains a `HistoryAgent` (remembers past experiences) on a small dataset.

**Results** (in `outputs/toy/{timestamp}/`):
- `metrics.json` - Final performance metrics
- `run.log` - Detailed execution log
- `scores/train/scores.jsonl` - Per-step scores
- `memories/memory_*.jsonl` - Final memory state
- `config.yaml` - Config snapshot

### 3. Try Different Configurations

**Memoryless baseline:**
```bash
python -m src.main --config configs/finer/memoryless.yaml
```

**ReflexionAgent (learns from mistakes):**
```bash
python -m src.main --config configs/finer/reflexion.yaml
```

**With validation:**
```bash
python -m src.main --config configs/omega/explorative/history_list.yaml
```

---

## Example Workflows

### Math Problem Solving

```yaml
# configs/my_math.yaml
runtime:
  max_envs_to_visit: 100
  scores_path: scores.jsonl

train_dataset:
  task_type: numeric
  hf_dataset: "keirp/aime_1983_2024"
  split: "train"
  max_samples: 100

agent:
  type: history_agent
  lm_config:
    model: "gemini-2.5-flash"
    temperature: 0.7
  memory_config:
    _type: history_list
    max_length: 100
  history_k: 10

output:
  results_dir: outputs/math_experiment
```

Run:
```bash
python -m src.main --config configs/my_math.yaml
```

### Local LLM with Tokasaurus

1. **Start Tokasaurus server:**
```bash
toka model=meta-llama/Llama-3.1-8B-Instruct port=8080
```

2. **Run experiment:**
```yaml
# configs/local_llm.yaml
agent:
  lm_config:
    model: "toka:meta-llama/Llama-3.1-8B-Instruct"
    base_url: "http://localhost:8080"
    protocol: "openai"
```

```bash
python -m src.main --config configs/local_llm.yaml
```

See [Tokasaurus Setup Guide](guides/tokasaurus-setup.md) for details.

### With Checkpointing

```yaml
runtime:
  max_envs_to_visit: 1000
  checkpoint_dir: checkpoints
  checkpoint_every_episodes: 100
  checkpoint_keep_last: 5
```

Resume from checkpoint:
```yaml
runtime:
  resume_from: "outputs/run1/20240115_103045/checkpoints/latest"
```

---

## Monitoring Progress

### Real-Time Scores

```bash
# Watch training progress
tail -f outputs/my_run/*/scores/train/scores.jsonl
```

### Compute Metrics

```python
import json

# Load scores
scores = []
with open("outputs/my_run/20240115_103045/scores/train/scores.jsonl") as f:
    for line in f:
        scores.append(json.loads(line)["score"])

# Compute statistics
print(f"Mean: {sum(scores) / len(scores):.3f}")
print(f"Total: {len(scores)} steps")
```

### View Memory

```bash
# See what agent remembered
cat outputs/my_run/20240115_103045/memories/memory_*.jsonl | head -20
```

### Inspect LLM Calls

```yaml
# Enable LLM logging
agent:
  lm_config:
    log_calls: true
```

```bash
# View prompts and responses
cat outputs/my_run/20240115_103045/llm_calls/train/actions/action_*.json
```

---

## Next Steps

- **[Configuration Reference](reference/config.md)** - All configuration options
- **[Agents](concepts/agents.md)** - Agent types and memory
- **[Environments](concepts/environments.md)** - Task types and datasets
- **[Checkpointing Guide](guides/checkpointing.md)** - Save and resume training
- **[Validation Guide](guides/validation.md)** - Monitor performance
- **[Multi-Step Environments](guides/multi-step-environments.md)** - Interactive tasks

---

## Troubleshooting

**"Module not found":**
```bash
# Install missing dependencies
pip install -r requirements.txt
```

**"API key not found":**
```bash
# Set environment variables
export GOOGLE_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

**"Out of memory":**
```yaml
# Reduce memory size
agent:
  memory_config:
    max_length: 50  # Smaller memory
```

**"Training too slow":**
```yaml
# Reduce dataset size
train_dataset:
  max_samples: 100
```

For more help, see the [full documentation](README.md).


