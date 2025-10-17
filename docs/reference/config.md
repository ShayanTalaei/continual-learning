## Configuration Reference

Complete reference for YAML configuration files used to define continual learning experiments.

### Top-Level Structure

```yaml
runtime:           # RunTimeConfig - orchestration and execution
train_dataset:     # EnvDatasetConfig - training data
validation_dataset # EnvDatasetConfig - validation data (optional)
agent:             # AgentConfig - agent type and configuration
output:            # OutputConfig - logging and output directories
```

---

## Runtime Configuration

Controls experiment execution, validation, checkpointing, and scoring.

```yaml
runtime:
  # Episode limits
  max_envs_to_visit: int | null      # Limit training episodes (null = all)
  max_steps_per_episode: int | null  # Cap steps per episode
  
  # Logging
  verbose: bool = true                # Progress bars and console logs
  
  # Score tracking
  scores_path: str | null             # Enable score streaming (e.g., "scores.jsonl")
  verbose_score_logging: bool = true  # Include full obs/action/feedback in scores
  
  # Validation
  validation_freq: int | null         # Validate every N episodes (null = disabled)
  validation_num_workers: int | null  # Parallel validation workers
  run_validation_at_start: bool = false  # Run validation before training
  
  # Checkpointing
  checkpoint_dir: str | null          # Checkpoint directory
  checkpoint_every_episodes: int | null  # Save frequency
  checkpoint_strategy: str = "last_n"  # "last_n" or "top_k_val"
  checkpoint_keep_last: int = 0       # Number to retain (0 = keep all)
  checkpoint_on_start: bool = false   # Save initial checkpoint
  
  # Resume
  resume_from: str | null             # Path to checkpoint directory
  start_episode_index: int = 0        # Internal: starting episode (for resume)
```

**Notes**:
- `scores_path` resolved relative to timestamped results directory
- `checkpoint_dir` resolved relative to results directory if not absolute
- Validation workers default to CPU count if not specified
- `checkpoint_strategy: "top_k_val"` requires validation enabled

**Examples**:
```yaml
# Minimal training
runtime:
  max_envs_to_visit: 100
  verbose: true

# Full featured
runtime:
  max_envs_to_visit: 1000
  max_steps_per_episode: 50
  verbose: true
  scores_path: scores.jsonl
  validation_freq: 50
  validation_num_workers: 100
  run_validation_at_start: true
  checkpoint_dir: checkpoints
  checkpoint_every_episodes: 50
  checkpoint_strategy: last_n
  checkpoint_keep_last: 5

# Resume from checkpoint
runtime:
  resume_from: "outputs/run1/20240115_103045/checkpoints/ep_000500"
  max_envs_to_visit: 2000  # Continue training
```

---

## Dataset Configuration

Defines data sources and environment construction.

### Common Fields (All Dataset Types)

```yaml
type: str                   # Dataset type: "qa", "finer", "omega_math", "alfworld"
verbose: bool = true        # Console logging
```

### QA Dataset (General Purpose)

```yaml
train_dataset:
  type: qa  # Optional, inferred from other fields
  
  # Data source (choose one)
  dataset_path: str | null         # Local JSONL file
  hf_dataset: str | null           # HuggingFace dataset name
  hf_subset: str | null            # HuggingFace subset/config
  split: str = "train"             # Dataset split
  
  # Field mapping
  input_field: str = "question"    # Question field name
  target_field: str = "answer"     # Answer field name
  choices_field: str | null        # MCQ choices field
  id_field: str | null             # ID field
  meta_fields: list[str] = []      # Additional metadata fields
  
  # Environment routing
  task_type: str = "exact"         # "exact", "numeric", "mcq", "custom"
  env_class: str | null            # Force specific env class
  
  # Instruction formatting
  instruction_template: str | null # Format string with {question}
  
  # Sampling
  max_samples: int | null          # Limit dataset size
  shuffle: bool = false            # Shuffle before sampling
  seed: int = 42                   # Random seed for shuffle
```

**Task types**:
- `"exact"` → `QAEnv` (exact string match)
- `"numeric"` → `MathQAEnv` (numeric comparison with tolerance)
- `"mcq"` → `MCQEnv` (multiple choice)
- `"custom"` → requires `env_class`

**Examples**:
```yaml
# AIME math problems
train_dataset:
  task_type: numeric
  hf_dataset: "keirp/aime_1983_2024"
  split: "train"
  input_field: "problem"
  target_field: "answer"
  instruction_template: "Solve this math problem. Put your final answer in \\boxed{{}}.\n\nProblem: {question}"

# MMLU multiple choice
train_dataset:
  task_type: mcq
  hf_dataset: "cais/mmlu"
  hf_subset: "abstract_algebra"
  split: "test"
  input_field: "question"
  target_field: "answer"
  choices_field: "choices"

# Local JSONL
train_dataset:
  dataset_path: "data/my_tasks.jsonl"
  input_field: "question"
  target_field: "answer"
  task_type: exact
```

### Finer Dataset (Financial Entity Recognition)

```yaml
train_dataset:
  type: finer
  
  # Data source
  dataset_path: str | null         # Local JSONL
  hf_dataset: str | null           # HuggingFace dataset
  hf_subset: str | null            # Subset
  split: str = "test"
  
  # Field mapping
  input_field: str = "context"
  target_field: str = "target"
  instruction_template: str | null
  
  # Options
  prepend_options: bool = false    # Prepend US GAAP tag options
  
  # Sampling
  max_samples: int | null
  shuffle: bool = false
  seed: int = 42
```

### OMEGA Math Dataset (Advanced Math)

```yaml
train_dataset:
  type: omega_math
  
  # Data source
  hf_dataset: str                  # Required
  hf_subset: str | null
  split: str = "test"
  
  # Field mapping
  input_field: str = "input"
  target_field: str = "target"
  instruction_template: str | null
  
  # Evaluation
  eval_mode: str = "auto"          # "auto", "numeric_tol", "tuple_tol", "set_tol", "matrix_tol", "expr_equiv"
  eval_tolerance: float = 1e-6
  expect_boxed: bool = false
  
  # Output format
  output_type: str = "text"        # "text" or "json" (json expects {final_answer, rationale})
  
  # Feedback
  feedback_type: str = "final_answer"  # "final_answer", "ground_truth_solution", "llm_feedback_from_ground_truth_solution"
  feedback_lm_config: LMConfig | null  # LM for generating feedback
  enable_llm_feedback: bool = true
  
  # Sampling
  max_samples: int | null
  shuffle: bool = false
  seed: int = 42
  
  # Subset filtering
  num_subsets: int | null          # Limit number of task subsets
  samples_per_subset: int | null   # Limit samples per subset
```

**Evaluation modes**:
- `"auto"`: Infer from target (matrix → tuple → set → numeric → exact)
- `"numeric_tol"`: Numeric with tolerance
- `"tuple_tol"`: Ordered tuples with tolerance
- `"set_tol"`: Unordered sets with tolerance
- `"matrix_tol"`: 2D matrices with tolerance
- `"expr_equiv"`: Symbolic expression equivalence (SymPy)

### ALFWorld Dataset (Embodied Tasks)

```yaml
train_dataset:
  type: alfworld
  
  # ALFWorld-specific
  base_config_path: str            # Path to ALFWorld config YAML
  split: str = "eval_out_of_distribution"
  num_episodes: int = 1            # Number of episodes to load
  prompts_path: str | null         # Path to prompts JSON (few-shot examples)
  
  verbose: bool = true
```

---

## Agent Configuration

Defines agent type, language model, memory, and behavior.

### Common Fields (All Agents)

```yaml
agent:
  type: str                       # "history_agent", "memoryless_agent", "reflexion_agent"
  verbose: bool = true            # Console logging
  
  # Language model
  lm_config:
    model: str                    # Model identifier (determines client)
    temperature: float = 0.2
    max_output_tokens: int = 2048
    log_calls: bool = false
    max_retries: int = 5
    starting_delay: float = 1.0
    backoff_factor: float = 2.0
    max_delay: float = 10.0
    # Provider-specific fields (see LM Config section)
  
  # System prompt
  system_prompt: str | null       # Override default, supports {file:path}
```

### HistoryAgent

```yaml
agent:
  type: history_agent
  
  lm_config: { ... }
  
  memory_config:
    _type: history_list
    max_length: int = 100         # Max entries in history
  
  history_k: int | null           # Show last k entries (null = all)
  system_prompt: str | null
  verbose: bool = true
```

### MemorylessAgent

```yaml
agent:
  type: memoryless_agent
  
  lm_config: { ... }
  system_prompt: str | null
  verbose: bool = true
```

### ReflexionAgent

```yaml
agent:
  type: reflexion_agent
  
  lm_config: { ... }
  
  memory_config:
    _type: history_list
    max_length: int = 200
  
  history_k: int | null
  
  # Reflection settings
  enable_reflection: bool = true
  reflect_on_failure_only: bool = false
  failure_threshold: float = 1.0
  
  # Prompts
  agent_system_prompt: str | null           # Main task prompt
  reflection_system_prompt: str | null      # Override reflection prompt
  reflection_few_shot_examples: str | null  # Path or inline examples
  
  verbose: bool = true
```

---

## Language Model Configuration

### Gemini

```yaml
lm_config:
  model: "gemini-2.5-flash"       # or "gemini-1.5-pro", etc.
  temperature: 0.7
  max_output_tokens: 2048
  log_calls: true
  thinking_budget: int | null     # Optional: thinking tokens
  max_retries: 5
```

**Authentication**: Requires `GOOGLE_API_KEY` environment variable

### OpenAI

```yaml
lm_config:
  model: "gpt-4o"                 # or "gpt-4o-mini", "o1-preview"
  temperature: 0.7
  max_output_tokens: 2048
  log_calls: true
  max_retries: 5
```

**Authentication**: Requires `OPENAI_API_KEY` environment variable

### Tokasaurus (Local)

```yaml
lm_config:
  model: "toka:meta-llama/Llama-3.1-8B-Instruct"  # Prefix with "toka:"
  temperature: 0.7
  max_output_tokens: 2048
  log_calls: true
  
  # Tokasaurus-specific
  base_url: "http://localhost:8080"
  protocol: "openai"              # or "toka"
  stop_sequences: ["FEEDBACK", "OBSERVATION"]
  timeout_s: 900.0
  enable_health_check: false
  
  max_retries: 5
  starting_delay: 1.0
  backoff_factor: 2.0
  max_delay: 10.0
```

**Server setup**: See [Tokasaurus Setup Guide](../guides/tokasaurus-setup.md)

---

### vLLM

```yaml
lm_config:
  model: "vllm:Qwen/Qwen3-1.7B"   # Prefix with "vllm:"
  temperature: 0.0
  max_output_tokens: 1024
  log_calls: true

  # Mode: local (in-process) vs server (OpenAI-compatible)
  use_server: false                  # false=in-process; true=HTTP server
  base_url: "http://localhost:8000" # required when use_server: true
  protocol: "openai"
  api_key: null                      # optional if server enforces auth
  timeout_s: 900.0

  # Local vLLM settings (ignored in server mode)
  tensor_parallel_size: 1
  max_model_len: 2048                # keep modest for better concurrency/VRAM
  dtype: "float16"                  # or "bfloat16" if supported
  gpu_memory_utilization: 0.5
  trust_remote_code: true            # needed by some models
  stop_sequences:
    - "FEEDBACK"
    - "OBSERVATION"
  cache_dir: "/path/to/hf-cache"   # optional HF cache root
```

Notes:
- Local mode (use_server: false): runs the engine in-process; no base_url needed. Concurrency is limited by GPU KV cache; prefer batching multiple prompts per call for throughput.
- Server mode (use_server: true): start the vLLM server separately, e.g.: `vllm serve Qwen/Qwen3-1.7B --dtype float16 --max-model-len 2048 --host 0.0.0.0 --port 8000`. The client sends OpenAI-style chat requests; the server safely queues and continuous-batches concurrent requests.

---

## Output Configuration

```yaml
output:
  results_dir: str | null         # Base results directory
  log_level: str = "INFO"         # "DEBUG", "INFO", "WARN", "ERROR"
```

**Notes**:
- Actual directory: `{results_dir}/{YYYYMMDD_HHMMSS}/`
- Can use `LOGS_DIR` environment variable as base
- Config snapshot saved as `config.yaml` in results directory

---

## Complete Example

```yaml
runtime:
  max_envs_to_visit: 1000
  max_steps_per_episode: 50
  verbose: true
  scores_path: scores.jsonl
  verbose_score_logging: true
  validation_freq: 50
  validation_num_workers: 100
  run_validation_at_start: true
  checkpoint_dir: checkpoints
  checkpoint_every_episodes: 50
  checkpoint_strategy: last_n
  checkpoint_keep_last: 5
  checkpoint_on_start: false

train_dataset:
  type: omega_math
  hf_dataset: "allenai/omega-explorative"
  split: "train"
  input_field: "input"
  target_field: "target"
  eval_mode: "auto"
  output_type: "json"
  feedback_type: "llm_feedback_from_ground_truth_solution"
  feedback_lm_config:
    model: "gpt-4o-mini"
    temperature: 0.7
    max_output_tokens: 512
  max_samples: 1000
  num_subsets: 4
  samples_per_subset: 50

validation_dataset:
  type: omega_math
  hf_dataset: "allenai/omega-explorative"
  split: "validation"
  input_field: "input"
  target_field: "target"
  eval_mode: "auto"
  output_type: "json"
  max_samples: 200

agent:
  type: reflexion_agent
  verbose: true
  lm_config:
    model: "toka:meta-llama/Llama-3.1-8B-Instruct"
    temperature: 0.7
    max_output_tokens: 2048
    log_calls: true
    base_url: "http://localhost:8080"
    protocol: "openai"
    stop_sequences: ["FEEDBACK"]
    max_retries: 5
  memory_config:
    _type: history_list
    max_length: 200
  history_k: 50
  enable_reflection: true
  reflect_on_failure_only: false
  agent_system_prompt: "{file:prompts/system.txt}"
  reflection_few_shot_examples: "{file:prompts/reflection_examples.txt}"

output:
  results_dir: outputs/omega_reflexion
  log_level: INFO
```

---

## Path Resolution

- **Relative paths**: Resolved relative to YAML file location (for `dataset_path`)
- **Results directory**: Appended with timestamp `YYYYMMDD_HHMMSS`
- **scores_path**: Resolved under timestamped results directory
- **checkpoint_dir**: Resolved under timestamped results directory
- **File injection** (`{file:path}`): Resolved relative to current working directory
- **Environment variable**: `LOGS_DIR` prepended to `results_dir` if set

---

## Output Directory Structure

```
{results_dir}/{YYYYMMDD_HHMMSS}/
├── config.yaml               # Config snapshot
├── run.log                   # Runtime logs
├── metrics.json              # Final metrics
├── scores/
│   ├── train/
│   │   └── scores.jsonl
│   └── val/
│       ├── 0_seen_episodes_scores.jsonl
│       └── 50_seen_episodes_scores.jsonl
├── llm_calls/                # If log_calls: true
│   ├── train/
│   │   ├── actions/
│   │   └── reflections/
│   └── validation/
│       └── val_0/
├── memories/                 # Final memory snapshot
│   └── memory_1000.jsonl
└── checkpoints/              # If checkpointing enabled
    ├── ep_000050/
    ├── ep_000100/
    └── latest → ep_000100
```


