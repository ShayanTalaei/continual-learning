## Run configuration

Top-level keys:
- `runtime`: RunTimeConfig
- `dataset`: QAEnvDatasetConfig
- `agent`: includes `type`, `lm_config`, `memory_config`, `history_k?`, `system_prompt?`, `verbose?`
- `output`: results_dir?, save_memory_path?, log_level?

Example:
```yaml
runtime:
  max_envs_to_visit: 10
  max_steps_per_episode: 1
  verbose: true

dataset:
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
  memory_config:
    _type: history_list
    max_length: 200
  history_k: 10
  system_prompt: "You are a helpful assistant."
  verbose: true

output:
  results_dir: outputs/run1
  save_memory_path: outputs/run1/memory.jsonl
  log_level: INFO
```


