---
noteId: "3c3db2009dd011f0b67f67467df34666"
tags: []

---

## Runtime

The `RunTime` class orchestrates the training loop, validation, scoring, and progress tracking for continual learning experiments.

### Core Responsibilities

1. **Episode execution**: Runs environments sequentially through agent using Gym-like `reset()`/`step()` interface
2. **Progress tracking**: Displays progress bars via tqdm, tracks episode/step counters
3. **Logging**: Writes structured logs to file with configurable verbosity
4. **Metrics aggregation**: Computes mean scores and per-episode statistics
5. **Score persistence**: Streams scores to JSONL files for real-time monitoring
6. **Validation**: Runs parallel validation episodes at configurable intervals
7. **Checkpointing**: Saves agent/memory state periodically (via CheckpointManager)

### Episode Loop

For each environment in the training dataset:
1. `env.reset()` → initial observation
2. `agent.reset()` → clear episode-specific state
3. While not done:
   - `agent.act(obs)` → action
   - `env.step(action)` → next_obs, feedback, done, info
   - `agent.observe(next_obs, feedback, done)` → update memory (if training)
   - Record step result and score
4. `agent.end_episode()` → optional cleanup (e.g., reflection)

**Step limits**: `max_steps_per_episode` caps multi-step environments (safety for unbounded tasks)

### Validation System

**Purpose**: Evaluate agent performance on held-out data without updating memory

**Key features:**
- **Parallel execution**: Validates using thread pool with configurable workers
- **Memory isolation**: Agent clones share memory (read-only during validation)
- **Training mode freezing**: Agent in eval mode, memory updates disabled
- **Structured logging**: Validation logs organized by episode count (e.g., `val_0`, `val_10`, `val_20`)

**Configuration:**
- `validation_freq`: Run validation every N training episodes (e.g., 10)
- `validation_num_workers`: Parallel validation workers (e.g., 100 for fast eval)
- `run_validation_at_start`: Run validation before any training (baseline)
- `validation_dataset`: Separate dataset config for validation data

**Implementation notes:**
- Validation results buffered and flushed in deterministic order (episode, step)
- Agent cloning via `clone_for_episode(training=False, share_memory=True)`
- Mean validation score tracked for checkpoint strategies

### Score Tracking

**Streaming scores** (`scores_path`): Real-time JSONL files for monitoring
- **Train**: `scores/train/scores.jsonl` - one line per training step
- **Validation**: `scores/val/{N}_seen_episodes_scores.jsonl` - per validation run

**Score record fields** (when `verbose_score_logging: true`):
```python
{
  "timestamp": "2024-01-15T10:30:45.123456Z",
  "mode": "train" | "val",
  "episode_index": 42,
  "step_index": 1,
  "score": 0.0-1.0,
  "episode_cum_score": 2.5,
  "env_id": "task_42",
  "env_type": "qa",
  "observation": "...",  # optional
  "action": "...",       # optional
  "feedback": {...},     # optional
  "info": {...},         # optional
  "lm_model": "gpt-4o",
  "agent_type": "HistoryAgent",
  "step_start": "...",
  "step_end": "...",
  "duration_ms": 1234.5
}
```

**Minimal mode** (`verbose_score_logging: false`): Only core fields (timestamp, mode, episode/step indices, scores, env identifiers)

### Configuration

**Core options:**
- `max_envs_to_visit`: Limit training episodes (for quick runs/debugging)
- `max_steps_per_episode`: Cap steps per episode (safety for multi-step tasks)
- `verbose`: Enable progress bars and console logging

**Score tracking:**
- `scores_path`: Base path for score files (resolved under results_dir)
- `verbose_score_logging`: Include full observation/action/feedback in scores (default: true)

**Validation:**
- `validation_freq`: Validate every N episodes (omit to disable)
- `validation_num_workers`: Parallel worker count (default: uses `num_parallel_episodes`)
- `run_validation_at_start`: Validate before training (default: false)

**Checkpointing** (see [Checkpointing](checkpointing.md)):
- `checkpoint_dir`: Directory for checkpoints
- `checkpoint_every_episodes`: Save frequency
- `checkpoint_strategy`: `"last_n"` or `"top_k_val"`
- `checkpoint_keep_last`: Number of checkpoints to retain
- `checkpoint_on_start`: Save checkpoint before training
- `resume_from`: Resume from checkpoint path
- `start_episode_index`: Internal resume counter

### Output Structure

```
results_dir/YYYYMMDD_HHMMSS/
├── run.log                    # Structured runtime logs
├── config.yaml                # Effective configuration snapshot
├── metrics.json               # Final aggregated metrics
├── memories/                  # Memory snapshots (if configured)
│   └── memory_{episode}.jsonl
├── llm_calls/                 # LLM call logs (if log_calls: true)
│   ├── train/
│   │   ├── actions/           # Action generation calls
│   │   └── reflections/       # Reflection calls (ReflexionAgent)
│   └── validation/
│       └── val_{N}/
│           └── actions/
├── scores/
│   ├── train/
│   │   └── scores.jsonl
│   └── val/
│       ├── 0_seen_episodes_scores.jsonl
│       ├── 10_seen_episodes_scores.jsonl
│       └── ...
└── checkpoints/               # Agent state snapshots (if configured)
    ├── ep_000010/
    │   ├── runtime.json
    │   ├── agent.json
    │   └── memory_10.jsonl
    ├── ep_000020/
    └── latest → ep_000020     # Symlink to most recent
```

### Example Configuration

```yaml
runtime:
  max_envs_to_visit: 1000         # Train on 1000 episodes
  max_steps_per_episode: 50       # Cap at 50 steps per episode
  verbose: true
  scores_path: scores.jsonl        # Enable score tracking
  verbose_score_logging: true      # Full score details
  
  # Validation
  validation_freq: 50              # Validate every 50 episodes
  validation_num_workers: 100      # 100 parallel workers
  run_validation_at_start: true    # Baseline validation
  
  # Checkpointing
  checkpoint_dir: checkpoints
  checkpoint_every_episodes: 50
  checkpoint_strategy: last_n
  checkpoint_keep_last: 5
  checkpoint_on_start: false

output:
  results_dir: outputs/my_experiment
  log_level: INFO
```

### Metrics Returned

The `run()` method returns:
```python
{
  "mean_score": 0.75,              # Average score across all steps
  "train_steps": 1234,             # Total training steps executed
  "train_episodes": 1000,          # Total training episodes completed
  "episodes": [                    # Per-episode step details
    [{"obs": "...", "action": "...", "feedback": {...}, "done": true}, ...],
    ...
  ]
}
```


