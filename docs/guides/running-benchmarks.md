## Running Benchmarks

- Choose `task_type`: exact → QAEnv, numeric → MathQAEnv, mcq → MCQEnv.
- Optionally set `env_class` to force a specific env.
- Load from HF datasets (hf_name/config/split) or local `dataset_path` (jsonl or load_from_disk).
- Provide field mappings: `question_field`, `answer_field`, `choices_field?`, `id_field?`.


