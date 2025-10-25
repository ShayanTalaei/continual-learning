## Validation Guide

This guide shows you how to set up and use validation to monitor agent performance during training.

### Quick Start

**Basic validation** (every 50 episodes):
```yaml
runtime:
  validation_freq: 50

validation_dataset:
  type: qa
  hf_dataset: "my/validation-set"
  split: "validation"
```

**Fast parallel validation** (100 workers):
```yaml
runtime:
  validation_freq: 50
  validation_num_workers: 100
  run_validation_at_start: true
```

---

### Why Use Validation

**Monitor learning progress:**
- Track performance on held-out data
- Detect overfitting early
- Measure generalization vs. memorization

**Checkpoint best models:**
- Combine with `checkpoint_strategy: top_k_val`
- Keep only high-performing checkpoints
- Easy model selection

**Debug training:**
- Baseline performance before training
- Compare training vs. validation curves
- Identify when to stop training

---

### Basic Validation Setup

**Minimal configuration:**
```yaml
runtime:
  max_envs_to_visit: 1000
  validation_freq: 100

train_dataset:
  hf_dataset: "my/dataset"
  split: "train"
  # ... other config

validation_dataset:
  hf_dataset: "my/dataset"
  split: "validation"
  # ... other config (usually same as train)
```

This runs validation every 100 training episodes.

**Validation runs:**
- Episode 100: validate on full validation set
- Episode 200: validate again
- Episode 300: validate again
- ...

---

### Validation Frequency

**Choose frequency based on:**
- **Training set size**: Smaller = more frequent
- **Episode duration**: Longer = less frequent
- **Compute budget**: More workers = can validate more often

**Examples:**
```yaml
# Fast training (< 1000 episodes)
runtime:
  validation_freq: 50

# Medium training (1000-5000 episodes)
runtime:
  validation_freq: 100

# Long training (> 5000 episodes)
  validation_freq: 250
```

**Disable validation:**
```yaml
runtime:
  validation_freq: null  # or omit entirely
```

---

### Parallel Validation

Validation runs in parallel using thread pools (agent clones share memory):

**Configure workers:**
```yaml
runtime:
  validation_num_workers: 100
```

**Worker count guidelines:**
- **Default**: Uses CPU count if not specified
- **Low parallelism**: 10-20 workers (conservative)
- **Medium parallelism**: 50-100 workers (recommended)
- **High parallelism**: 200+ workers (fast hardware)

**Memory sharing:**
- All validation workers share the same agent memory (read-only)
- No memory updates during validation
- Independent short-term state per worker
- Thread-safe implementation

**Performance:**
```
1 worker:    ~1000 episodes/hour
10 workers:  ~10k episodes/hour
100 workers: ~100k episodes/hour (limited by GIL/LLM API)
```

---

### Validation at Start

Run baseline validation before any training:

```yaml
runtime:
  run_validation_at_start: true
  validation_freq: 50
```

**Timeline:**
```
Start: Validation on val set (episode 0, baseline)
Episode 50: Validation
Episode 100: Validation
...
```

**Use when:**
- Documenting baseline performance
- Comparing different initialization strategies
- Debugging: verify validation works before training

---

### Validation Dataset Configuration

**Same dataset, different split:**
```yaml
train_dataset:
  hf_dataset: "cais/mmlu"
  hf_subset: "abstract_algebra"
  split: "train"
  max_samples: 1000

validation_dataset:
  hf_dataset: "cais/mmlu"
  hf_subset: "abstract_algebra"
  split: "validation"
  max_samples: 200  # Smaller for faster validation
```

**Different dataset:**
```yaml
train_dataset:
  hf_dataset: "org/train-data"
  split: "train"

validation_dataset:
  hf_dataset: "org/val-data"  # Different dataset
  split: "test"
```

**Local files:**
```yaml
train_dataset:
  dataset_path: "data/train.jsonl"

validation_dataset:
  dataset_path: "data/val.jsonl"
```

---

### Validation Outputs

**Score files** (when `scores_path` enabled):
```
results_dir/YYYYMMDD_HHMMSS/scores/val/
├── 0_seen_episodes_scores.jsonl   # Baseline (if run_validation_at_start)
├── 50_seen_episodes_scores.jsonl  # After 50 training episodes
├── 100_seen_episodes_scores.jsonl
└── ...
```

**Score format:**
```jsonl
{"timestamp": "...", "mode": "val", "episode_index": 1, "step_index": 1, "score": 1.0, ...}
{"timestamp": "...", "mode": "val", "episode_index": 2, "step_index": 1, "score": 0.0, ...}
...
```

**LLM call logs** (when `log_calls: true`):
```
results_dir/YYYYMMDD_HHMMSS/llm_calls/validation/
├── val_0/    # Baseline validation
│   └── actions/
├── val_50/   # After 50 episodes
│   └── actions/
└── val_100/  # After 100 episodes
    └── actions/
```

**Runtime logs:**
```
[INFO] Starting validation at train episode 50 on 200 examples
[INFO] Validation finished: mean_score_val=0.72 total=200
```

---

### Monitoring Validation

**View validation scores:**
```bash
# Latest validation
tail -20 results_dir/20240115_103045/scores/val/100_seen_episodes_scores.jsonl

# Count correct answers
grep '"score": 1.0' results_dir/20240115_103045/scores/val/100_seen_episodes_scores.jsonl | wc -l
```

**Compute mean validation score:**
```python
import json

scores = []
with open("scores/val/100_seen_episodes_scores.jsonl") as f:
    for line in f:
        scores.append(json.loads(line)["score"])

print(f"Mean: {sum(scores) / len(scores):.3f}")
```

**Track validation curve:**
```python
import glob
import json

val_files = sorted(glob.glob("scores/val/*_scores.jsonl"))
means = []

for path in val_files:
    scores = []
    with open(path) as f:
        for line in f:
            scores.append(json.loads(line)["score"])
    means.append(sum(scores) / len(scores))
    episodes = int(path.split("/")[-1].split("_")[0])
    print(f"Episodes {episodes}: {means[-1]:.3f}")
```

---

### Validation with Checkpointing

**Top-K validation checkpoints:**
```yaml
runtime:
  validation_freq: 50
  validation_num_workers: 100
  checkpoint_dir: checkpoints
  checkpoint_strategy: top_k_val
  checkpoint_keep_last: 5
```

**How it works:**
1. Train 50 episodes
2. Run validation → mean_score_val = 0.45
3. Save checkpoint with score
4. Train 50 more episodes
5. Run validation → mean_score_val = 0.52
6. Save checkpoint with score
7. Continue...
8. Keep only top-5 checkpoints by validation score

See [Checkpointing Guide](checkpointing.md) for details.

---

### Advanced Patterns

#### Validation-Only Run

Validate without training:

```yaml
runtime:
  max_envs_to_visit: 0  # No training
  run_validation_at_start: true
  validation_num_workers: 100

validation_dataset:
  # ... your validation data
```

#### Different Validation Cadence

```yaml
runtime:
  max_envs_to_visit: 5000
  validation_freq: 100  # Every 100 during 0-1000
  # Then manually adjust and resume:
  # validation_freq: 500  # Less frequent after 1000
```

#### Multiple Validation Sets

Run validation on multiple datasets by resuming:

```bash
# Validation on set A
python -m src.main --config config_val_a.yaml

# Validation on set B (resume from checkpoint)
python -m src.main --config config_val_b.yaml
```

---

### Troubleshooting

#### "Validation runs too slow"

**Solution 1: Increase workers**
```yaml
runtime:
  validation_num_workers: 200
```

**Solution 2: Reduce validation set**
```yaml
validation_dataset:
  max_samples: 500  # Smaller sample
```

**Solution 3: Less frequent validation**
```yaml
runtime:
  validation_freq: 200  # Validate less often
```

#### "Validation scores not saved"

**Cause:** `scores_path` not set

**Fix:**
```yaml
runtime:
  scores_path: scores.jsonl
```

#### "Memory updated during validation"

**Cause:** Should never happen (validation freezes memory)

**Debug:**
- Check agent `training` flag is False
- Verify `memory.eval()` called
- Report bug if issue persists

#### "Validation different from training"

**Expected:** Validation should be harder (generalization test)

**Unexpected differences:**
- Check dataset splits are actually different
- Verify instruction templates match
- Ensure evaluation modes consistent

---

### Best Practices

1. **Start with baseline:** Use `run_validation_at_start: true`

2. **Match train/val configs:** Keep evaluation settings identical
   ```yaml
   train_dataset:
     task_type: numeric
     eval_tolerance: 1e-6
   
   validation_dataset:
     task_type: numeric
     eval_tolerance: 1e-6  # Same as train
   ```

3. **Reasonable validation size:** 
   - Too small: noisy estimates
   - Too large: slow validation
   - Sweet spot: 200-1000 examples

4. **Parallelize aggressively:** Validation is embarrassingly parallel

5. **Monitor both curves:** Plot train + validation to detect overfitting

6. **Use for early stopping:** Stop when validation plateaus or degrades

7. **Document validation choices:** 
   - Why this validation set?
   - How representative is it?
   - Any known biases?

---

### Examples

**Example 1: Standard setup**
```yaml
runtime:
  max_envs_to_visit: 1000
  validation_freq: 50
  validation_num_workers: 100
  run_validation_at_start: true
  scores_path: scores.jsonl

train_dataset:
  hf_dataset: "cais/mmlu"
  split: "train"
  max_samples: 1000

validation_dataset:
  hf_dataset: "cais/mmlu"
  split: "validation"
  max_samples: 500
```

**Example 2: Fast validation for debugging**
```yaml
runtime:
  max_envs_to_visit: 100
  validation_freq: 10
  validation_num_workers: 50

validation_dataset:
  # ... config
  max_samples: 50  # Small for speed
```

**Example 3: Production training**
```yaml
runtime:
  max_envs_to_visit: 10000
  validation_freq: 100
  validation_num_workers: 200
  run_validation_at_start: true
  checkpoint_strategy: top_k_val
  checkpoint_keep_last: 10
```

---

### See Also

- [Runtime Concepts](../concepts/runtime.md) - Validation system details
- [Checkpointing Guide](checkpointing.md) - Combine with top-k checkpoints
- [Configuration Reference](../reference/config.md) - All validation options

