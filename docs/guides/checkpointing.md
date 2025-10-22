## Checkpointing Guide

This guide shows you how to use checkpointing to save and resume training runs.

### Quick Start

**Enable checkpointing** (save every 50 episodes, keep last 5):
```yaml
runtime:
  checkpoint_dir: checkpoints
  checkpoint_every_episodes: 50
  checkpoint_keep_last: 5
```

**Resume from checkpoint**:
```yaml
runtime:
  resume_from: "outputs/my_run/20240115_103045/checkpoints/latest"
```

That's it! Your training will now save checkpoints and can resume from them.

---

### When to Use Checkpointing

**Use checkpointing when:**
- Training runs longer than 1 hour (recovery from failures)
- Experimenting with hyperparameters (resume from best checkpoint)
- Limited compute time windows (pause and resume across sessions)
- Want to analyze intermediate states (inspect memory at different points)
- Running expensive experiments (don't lose progress)

**Skip checkpointing when:**
- Quick experiments (< 100 episodes)
- Prototyping (no need for persistence)
- Disk space is limited
- Debugging configuration (faster iteration without I/O overhead)

---

### Basic Checkpointing

**Minimal configuration:**
```yaml
runtime:
  max_envs_to_visit: 1000
  checkpoint_dir: checkpoints
  checkpoint_every_episodes: 100
```

This saves a checkpoint every 100 episodes. All checkpoints are kept (no automatic deletion).

**Result structure:**
```
outputs/my_run/20240115_103045/checkpoints/
├── ep_000100/
├── ep_000200/
├── ep_000300/
└── latest → ep_000300
```

---

### Last-N Strategy

Keep only the most recent N checkpoints (saves disk space):

```yaml
runtime:
  checkpoint_dir: checkpoints
  checkpoint_every_episodes: 50
  checkpoint_strategy: last_n
  checkpoint_keep_last: 5
```

**Timeline example:**
```
Episode 50:  ep_000050/ created
Episode 100: ep_000100/ created
Episode 150: ep_000150/ created
Episode 200: ep_000200/ created
Episode 250: ep_000250/ created (5 checkpoints now)
Episode 300: ep_000300/ created, ep_000050/ deleted (oldest removed)
Episode 350: ep_000350/ created, ep_000100/ deleted
```

**Use when:**
- Disk space is limited
- You only need recent checkpoints for recovery
- Training is long but stable (don't need old checkpoints)

---

### Top-K Validation Strategy

Keep the K best-performing checkpoints based on validation scores:

```yaml
runtime:
  validation_freq: 50
  checkpoint_dir: checkpoints
  checkpoint_strategy: top_k_val
  checkpoint_keep_last: 3
```

**Requirements:**
- Must enable validation (`validation_freq`)
- Validation must have a `validation_dataset`

**Timeline example:**
```
Episode 50  (val=0.45): ep_000050/ saved
Episode 100 (val=0.52): ep_000100/ saved
Episode 150 (val=0.48): ep_000150/ saved (3 checkpoints)
Episode 200 (val=0.55): ep_000200/ saved, ep_000050/ deleted (lowest score)
Episode 250 (val=0.51): ep_000250/ saved, ep_000150/ deleted (now 2nd lowest)
Final: ep_000100/, ep_000200/, ep_000250/ (top 3 by validation score)
```

**Use when:**
- Validation performance is critical
- Want to keep best-performing models only
- Doing hyperparameter search (keep best configs)

---

### Resume from Checkpoint

**Find checkpoint path:**
```bash
# Use latest
ls outputs/my_run/20240115_103045/checkpoints/latest

# Or specific episode
ls outputs/my_run/20240115_103045/checkpoints/ep_000500
```

**Resume configuration:**
```yaml
runtime:
  resume_from: "outputs/my_run/20240115_103045/checkpoints/ep_000500"
  max_envs_to_visit: 2000  # Continue to 2000 total episodes
```

**What happens:**
1. Checkpoint manifest loaded
2. Agent memory restored from snapshot
3. Episode counter set to 500
4. Episodes 1-500 skipped
5. Training resumes from episode 501
6. New checkpoints saved in new timestamped directory

**Important:** 
- Original checkpoints NOT modified
- New run gets new timestamp directory
- Progress continues from checkpoint's episode index

---

### Checkpoint on Start

Save initial state before any training:

```yaml
runtime:
  checkpoint_on_start: true
  checkpoint_dir: checkpoints
```

Creates `ep_000000/` with initial agent state.

**Use when:**
- Debugging checkpoint/resume logic
- Comparing initial vs. final states
- Documenting experiment starting conditions

---

### Inspecting Checkpoints

**Checkpoint contents:**
```
ep_000100/
├── runtime.json       # Episode count, step count, timestamp
├── agent.json         # Agent type, config
└── memory_100.jsonl   # Memory snapshot (one entry per line)
```

**View runtime manifest:**
```bash
cat checkpoints/ep_000100/runtime.json
```

Example output:
```json
{
  "timestamp": "2024-01-15T10:30:45.123456Z",
  "episode_index": 100,
  "train_steps_total": 234,
  "agent_type": "HistoryAgent",
  "memory_snapshot_path": "memory_100.jsonl",
  "val_score": 0.52
}
```

**View memory snapshot:**
```bash
head -5 checkpoints/ep_000100/memory_100.jsonl
```

Example output:
```jsonl
{"type": "Observation", "content": "Solve: 2+2=?"}
{"type": "Action", "content": "4"}
{"type": "Feedback", "content": {"score": 1.0, "message": "Correct!"}}
{"type": "Reflection", "content": "Simple arithmetic works well..."}
...
```

---

### Advanced Patterns

#### Multiple Checkpoints for Comparison

```yaml
runtime:
  checkpoint_dir: checkpoints
  checkpoint_every_episodes: 10  # Frequent checkpoints
  checkpoint_keep_last: 20       # Keep many for analysis
```

Useful for analyzing learning curves and memory evolution.

#### Checkpoint + Validation Baseline

```yaml
runtime:
  checkpoint_dir: checkpoints
  checkpoint_on_start: true       # Save initial state
  run_validation_at_start: true   # Baseline validation
  validation_freq: 50
  checkpoint_every_episodes: 50
```

Documents both initial and intermediate performance.

#### Resume with Different Config

You can resume a checkpoint but change some settings:

```yaml
runtime:
  resume_from: "outputs/run1/20240115_103045/checkpoints/ep_000500"
  max_envs_to_visit: 2000
  validation_freq: 25  # More frequent validation
```

Memory and progress restored, but runtime settings can differ.

#### Multi-Stage Training

```bash
# Stage 1: Train to 500 episodes
python -m src.main --config stage1.yaml

# Stage 2: Resume with different hyperparameters
# Edit config: lower learning rate, change prompts, etc.
python -m src.main --config stage2.yaml
```

---

### Troubleshooting

#### "Checkpoint not found"

**Check path:**
```bash
ls outputs/my_run/20240115_103045/checkpoints/
```

**Verify latest pointer:**
```bash
cat outputs/my_run/20240115_103045/checkpoints/latest.json
```

#### "Training restarts from episode 0"

**Cause:** `start_episode_index` not set correctly

**Fix:** Verify checkpoint manifest has correct `episode_index`

```bash
cat checkpoints/ep_000500/runtime.json | grep episode_index
```

#### "Memory not restored"

**Cause:** Memory snapshot file missing or incompatible

**Check snapshot exists:**
```bash
ls checkpoints/ep_000500/memory_*.jsonl
```

**Verify memory type matches:**
```bash
cat checkpoints/ep_000500/agent.json | grep memory_type
```

#### "Old checkpoints not deleted"

**Cause:** `checkpoint_keep_last` not set or strategy mismatch

**Fix:**
```yaml
runtime:
  checkpoint_strategy: last_n  # Ensure strategy is set
  checkpoint_keep_last: 5      # Must be > 0
```

#### "Validation scores not tracked"

**Cause:** Top-k strategy requires validation

**Fix:**
```yaml
runtime:
  validation_freq: 50          # Enable validation
  checkpoint_strategy: top_k_val
```

---

### Best Practices

1. **Start with last_n:** Simple and effective for most use cases

2. **Set reasonable frequency:** 
   - Fast experiments: every 100 episodes
   - Long runs: every 50 episodes
   - Very long runs: every 20-30 episodes

3. **Keep enough checkpoints:** 
   - Minimum: 3 (recent recovery)
   - Recommended: 5-10 (rollback options)
   - Analysis: 20+ (learning curve studies)

4. **Use top_k_val for hyperparameter search:** Automatically keeps best configs

5. **Monitor disk space:**
   ```bash
   du -sh outputs/my_run/*/checkpoints/
   ```

6. **Document resume decisions:** Note why resuming and what changed

7. **Test resume early:** Verify checkpoint/resume works before long runs

8. **Backup important checkpoints:** Copy to permanent storage before cleanup

---

### Examples

**Example 1: Long training with recovery**
```yaml
runtime:
  max_envs_to_visit: 10000
  checkpoint_dir: checkpoints
  checkpoint_every_episodes: 100
  checkpoint_keep_last: 10
```

**Example 2: Hyperparameter search**
```yaml
runtime:
  max_envs_to_visit: 1000
  validation_freq: 50
  checkpoint_dir: checkpoints
  checkpoint_strategy: top_k_val
  checkpoint_keep_last: 5
```

**Example 3: Resume and extend**
```yaml
runtime:
  resume_from: "outputs/run1/20240115_103045/checkpoints/ep_001000"
  max_envs_to_visit: 5000  # Extend to 5000 total
  checkpoint_every_episodes: 100
```

---

### See Also

- [Checkpointing Concepts](../concepts/checkpointing.md) - Detailed technical documentation
- [Configuration Reference](../reference/config.md) - All checkpoint options
- [Validation Guide](validation.md) - Validation for top-k strategy

