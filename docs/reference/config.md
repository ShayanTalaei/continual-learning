## Run configuration

Top-level keys:
- `runtime`: RunTimeConfig
- `train_dataset`: QAEnvDatasetConfig (training dataset)
- `validation_dataset`: QAEnvDatasetConfig (optional validation dataset)
- `agent`: includes `type`, `lm_config`, `memory_config`, `history_k?`, `system_prompt?`, `verbose?`
- `output`: results_dir?, log_level?

Runtime keys (subset):
- `max_envs_to_visit?`, `max_steps_per_episode?`, `verbose?`
- `scores_path?`: if set, write streaming scores.jsonl and scores.json snapshot
- `validation_freq?`: run validation every N training episodes
- `validation_num_workers?`: number of parallel workers for validation
- `run_validation_at_start?`: run validation before training starts

LM config keys (subset):
- `model`, `temperature?`, `max_output_tokens?`, `log_calls?`
- `max_retries?`, `starting_delay?`, `backoff_factor?`, `max_delay?`

Example:
```yaml
runtime:
  max_envs_to_visit: 10
  max_steps_per_episode: 1
  verbose: true
  scores_path: scores.jsonl
  validation_freq: 5
  validation_num_workers: 4
  run_validation_at_start: false

train_dataset:
  dataset_path: data/synthetic/encryption/train_100.jsonl
  question_field: question
  answer_field: answer
  instruction_template: "Question: {question}\nAnswer concisely."
  task_type: exact
  verbose: true

validation_dataset:
  dataset_path: data/synthetic/encryption/train_100.jsonl
  question_field: question
  answer_field: answer
  instruction_template: "Question: {question}\nAnswer concisely."
  task_type: exact
  verbose: true

agent:
  type: history_agent
  lm_config:
    model: gemini-2.5-flash
    temperature: 0.0
    max_output_tokens: 1024
    log_calls: true
    max_retries: 5
    starting_delay: 1.0
    backoff_factor: 2.0
    max_delay: 10.0
  memory_config:
    _type: history_list
    max_length: 200
  history_k: 10
  system_prompt: "You are a helpful assistant."
  verbose: true

output:
  results_dir: outputs/run1
  log_level: INFO
```

Notes:
- At runtime, the actual results directory is suffixed with a timestamp `YYYYMMDD_HHMMSS`, and a snapshot of the effective config is saved as `config.yaml` inside it.
- If `scores_path` is relative, it is resolved under the timestamped results directory.
- Memory snapshots are automatically saved as `memory_{episode_count}.jsonl` in the results directory.
- LLM calls are logged to `llm_calls/` subdirectory when `log_calls: true`.
- Validation scores are organized under `scores/validation/` with per-episode files.


