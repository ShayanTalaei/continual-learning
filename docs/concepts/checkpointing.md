## Checkpointing

Checkpointing enables saving and resuming agent training state, including memory and progress counters. This is essential for long-running experiments, recovery from failures, and systematic hyperparameter exploration.

### Core Concepts

**Checkpoint**: A snapshot of agent state at a specific training episode, including:
- Runtime manifest (episode index, step counts, timestamps)
- Agent configuration
- Memory state (e.g., HistoryList contents)
- Optional validation scores (for top-k strategies)

**CheckpointManager**: Orchestrates checkpoint saving, pruning, and strategy enforcement
- Delegates to agent's `save_checkpoint()` / `load_checkpoint()` methods
- Manages checkpoint retention policies
- Tracks validation scores for top-k pruning
- Maintains `latest` pointer for easy resume

### Checkpoint Strategies

#### 1. Last-N Strategy (`"last_n"`)
**Use case**: Keep recent checkpoints for resume capability, discard old ones to save space

**Behavior**:
- Save every `checkpoint_every_episodes` episodes
- Keep only the last `checkpoint_keep_last` checkpoints
- Delete older checkpoints automatically
- Independent of validation scores

**Configuration**:
```yaml
runtime:
  checkpoint_dir: checkpoints
  checkpoint_every_episodes: 50
  checkpoint_strategy: last_n
  checkpoint_keep_last: 5
```

**Example timeline**:
```
Episode 50  → checkpoint saved → ep_000050/
Episode 100 → checkpoint saved → ep_000100/
Episode 150 → checkpoint saved → ep_000150/
Episode 200 → checkpoint saved → ep_000200/
Episode 250 → checkpoint saved → ep_000250/
Episode 300 → checkpoint saved → ep_000300/ (ep_000050/ deleted)
Episode 350 → checkpoint saved → ep_000350/ (ep_000100/ deleted)
```

#### 2. Top-K Validation Strategy (`"top_k_val"`)
**Use case**: Keep best-performing checkpoints based on validation scores

**Behavior**:
- Save on every validation run (triggered by `validation_freq`)
- Associate validation score with each checkpoint
- Keep only the top-k checkpoints by validation score
- Break ties by episode index (prefer later episodes)
- Delete checkpoints with lower validation scores

**Configuration**:
```yaml
runtime:
  validation_freq: 50
  checkpoint_dir: checkpoints
  checkpoint_strategy: top_k_val
  checkpoint_keep_last: 3
```

**Example timeline**:
```
Episode 50  (val_score=0.45) → ep_000050/ saved
Episode 100 (val_score=0.52) → ep_000100/ saved
Episode 150 (val_score=0.48) → ep_000150/ saved
Episode 200 (val_score=0.55) → ep_000200/ saved (ep_000050/ deleted, lowest score)
Episode 250 (val_score=0.53) → ep_000250/ saved (ep_000150/ deleted)
Final: ep_000100/, ep_000200/, ep_000250/ (top 3 by score)
```

### Checkpoint Directory Structure

```
checkpoints/
├── ep_000050/
│   ├── runtime.json          # Runtime state manifest
│   ├── agent.json            # Agent configuration snapshot
│   └── memory_50.jsonl       # Memory module snapshot
├── ep_000100/
│   ├── runtime.json
│   ├── agent.json
│   └── memory_100.jsonl
├── latest → ep_000100        # Symlink to most recent checkpoint
└── latest.json               # JSON pointer (fallback for non-symlink systems)
```

**runtime.json**:
```json
{
  "timestamp": "2024-01-15T10:30:45.123456Z",
  "episode_index": 100,
  "train_steps_total": 1234,
  "agent_type": "HistoryAgent",
  "memory_snapshot_path": "memory_100.jsonl",
  "val_score": 0.52
}
```

**agent.json**:
```json
{
  "agent_type": "HistoryAgent",
  "agent_config": {
    "type": "history_agent",
    "lm_config": {...},
    "memory_config": {...},
    "history_k": 10,
    "system_prompt": "..."
  },
  "memory_type": "HistoryList",
  "memory_snapshot_path": "memory_100.jsonl"
}
```

### Resume from Checkpoint

**Basic resume**:
```yaml
runtime:
  resume_from: "outputs/my_run/20240115_103045/checkpoints/ep_000100"
  # or resume_from: "outputs/my_run/20240115_103045/checkpoints/latest"
```

**Resume behavior**:
1. Load checkpoint manifest from `resume_from` path
2. Resolve `latest` symlink/pointer if specified
3. Restore agent via `agent.load_checkpoint()`
   - Memory restored from snapshot
   - Configuration preserved
4. Set `start_episode_index` from manifest
5. Skip already-processed episodes in training loop
6. Continue training from next episode
7. Save new checkpoints in current run's directory

**Important notes**:
- Resumed run gets a new timestamp directory
- Old checkpoints are not modified
- New checkpoints saved alongside new logs/scores
- Progress counters (`num_seen_episodes`) restored from manifest

### Agent Checkpoint API

Agents implement default checkpoint behavior via `Agent` base class:

**Save checkpoint**:
```python
def save_checkpoint(self, checkpoint_dir: str, snapshot_id: int) -> dict:
    """Save agent state to checkpoint directory.
    
    Returns:
        manifest: Dict with agent metadata
    """
    # Default implementation:
    # 1. Create checkpoint directory
    # 2. Save agent config to agent.json
    # 3. Delegate to memory.save_snapshot() if memory exists
    # 4. Write agent manifest with snapshot path
    # 5. Return manifest
```

**Load checkpoint**:
```python
def load_checkpoint(self, checkpoint_dir: str) -> None:
    """Restore agent state from checkpoint directory.
    
    Default implementation:
    # 1. Read agent.json manifest
    # 2. Find memory snapshot file
    # 3. Call memory.load_snapshot() if available
    # 4. Restore memory instance
    """
```

**Custom checkpoint data**:
Subclasses can override to save additional state:
```python
def save_checkpoint(self, checkpoint_dir: str, snapshot_id: int) -> dict:
    manifest = super().save_checkpoint(checkpoint_dir, snapshot_id)
    # Save custom state
    manifest["custom_field"] = self._custom_data
    return manifest
```

### Memory Snapshots

Memory modules implement snapshot API for persistence:

**HistoryList snapshot**:
```python
def save_snapshot(self, directory: str, snapshot_id: int) -> str:
    """Save memory to JSONL file.
    
    Returns:
        path: Absolute path to saved file
    """
    # Format: memory_{snapshot_id}.jsonl
    # One JSON object per line (one entry per line)

def load_snapshot(cls, path: str) -> HistoryList:
    """Class method to load memory from file.
    
    Returns:
        New HistoryList instance with restored state
    """
```

**Snapshot format** (JSONL):
```jsonl
{"type": "Observation", "content": "Solve this problem..."}
{"type": "Action", "content": "The answer is 42"}
{"type": "Feedback", "content": {"score": 1.0, "message": "Correct!"}}
{"type": "Reflection", "content": "I should always check my work..."}
```

### Configuration Reference

**Minimal checkpointing** (every 100 episodes, keep last 3):
```yaml
runtime:
  checkpoint_dir: checkpoints
  checkpoint_every_episodes: 100
  checkpoint_keep_last: 3
```

**Top-k validation checkpoints** (keep best 5):
```yaml
runtime:
  validation_freq: 50
  checkpoint_dir: checkpoints
  checkpoint_strategy: top_k_val
  checkpoint_keep_last: 5
```

**Checkpoint on start** (useful for debugging):
```yaml
runtime:
  checkpoint_dir: checkpoints
  checkpoint_on_start: true
  checkpoint_every_episodes: 50
```

**Resume from checkpoint**:
```yaml
runtime:
  resume_from: "/path/to/checkpoints/ep_000500"
  # Training continues from episode 501
  # New checkpoints saved in current run directory
```

### Best Practices

1. **Strategy selection**:
   - Use `last_n` for reliable recent state (default for most cases)
   - Use `top_k_val` when validation performance is critical
   - Combine with frequent validation for top-k effectiveness

2. **Checkpoint frequency**:
   - More frequent = better resume granularity, more disk space
   - Less frequent = less overhead, coarser resume points
   - Typical: every 50-100 episodes for medium runs

3. **Retention count**:
   - Balance disk space vs. rollback options
   - `keep_last: 5` provides ~5 rollback points
   - For top-k, higher count preserves more diversity

4. **Resume workflow**:
   - Always check `latest` checkpoint first
   - Verify manifest contents before resume
   - Consider starting a new run for major config changes

5. **Disk space**:
   - Monitor checkpoint directory size
   - Memory snapshots grow with `max_length`
   - Use `checkpoint_keep_last` to cap storage

### Troubleshooting

**Checkpoint not found**:
- Verify `checkpoint_dir` path is correct
- Check `latest.json` pointer if symlink doesn't exist
- Ensure checkpoint was actually saved (check logs)

**Resume skips wrong episodes**:
- Verify `start_episode_index` matches checkpoint `episode_index`
- Check that dataset hasn't changed between runs
- Ensure `max_envs_to_visit` not limiting iteration

**Memory not restored**:
- Verify memory snapshot file exists in checkpoint directory
- Check memory type compatibility (HistoryList → HistoryList)
- Ensure `load_snapshot()` class method exists

**Validation scores not tracked**:
- Verify `checkpoint_strategy: top_k_val` is set
- Ensure `validation_freq` is configured
- Check that validation is actually running (not skipped)

### Implementation Notes

**Thread safety**: Checkpointing happens on main thread after episode completion (no concurrency issues)

**Atomicity**: Manifests written atomically via temp file + rename pattern

**Portability**: `latest.json` provides fallback when symlinks unsupported (Windows, some network filesystems)

**Backward compatibility**: Old checkpoints remain loadable after code updates (best-effort basis)

