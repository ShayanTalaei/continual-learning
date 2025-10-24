---
noteId: "2ab34f609dd311f0b67f67467df34666"
tags: []

---

## Continual Learning

A lightweight framework for studying continual learning: an agent interacts with environments, receives feedback, and updates its memory at run time to improve over time. The project separates design-time choices (memory mechanisms, update rules, LM selection) from run-time adaptation, enabling head-to-head comparisons and easy extensibility.

Key idea at step t: memory is updated after each observation–action–feedback tuple to guide the next step.

### Highlights
- Task-agnostic agents with pluggable memory and language model
- Bounded HistoryList memory with atomic entries (Observation, Action, Feedback)
- Gym-style runtime loop with tqdm progress and structured logging (file + optional console)
- Flexible dataset adapter (HuggingFace datasets or local JSONL)
- Benchmark-ready environment subclasses (QA, numeric/boxed, MCQ)
- YAML-driven runs via a single entrypoint

### Quickstart
```bash
python -m src.main --config configs/toy_inmemory.yaml
```

Outputs (by default):
- Metrics: `outputs/.../metrics.json`
- Memory: `outputs/.../memory.jsonl`
- Logs (file): `outputs/.../run.log`

### Repository Structure

- [docs/](docs/)
  - [README.md](docs/README.md) — documentation index
  - [quickstart.md](docs/quickstart.md)
  - concepts/
    - [runtime.md](docs/concepts/runtime.md)
    - [agents.md](docs/concepts/agents.md)
    - [memory.md](docs/concepts/memory.md)
    - [environments.md](docs/concepts/environments.md)
    - [language_model.md](docs/concepts/language_model.md)
    - [logging.md](docs/concepts/logging.md)
  - reference/
    - [config.md](docs/reference/config.md)
  - guides/
    - [running-benchmarks.md](docs/guides/running-benchmarks.md)
    - [synthetic-data.md](docs/guides/synthetic-data.md)
  - cookbooks/
    - [encryption-history-list.md](docs/cookbooks/encryption-history-list.md)
  - design/
    - [roadmap.md](docs/design/roadmap.md)

- [configs/](configs/)
  - [toy_inmemory.yaml](configs/toy_inmemory.yaml)
  - [encryption_history_list.yaml](configs/encryption_history_list.yaml)

- [src/](src/)
  - [main.py](src/main.py) — CLI entrypoint for YAML-driven runs
  - [run_time.py](src/run_time.py) — runtime loop (reset/step, logging, tqdm)
  - [run_config.py](src/run_config.py) — run-level config schema
  - agent/
    - [agent.py](src/agent/agent.py) — base agent interface
    - [memory_agent.py](src/agent/memory_agent.py) — task-agnostic agent using memory
    - [history_agent.py](src/agent/history_agent.py) — HistoryList-specialized agent
    - [registry.py](src/agent/registry.py) — agent type registry
  - data/
    - [env.py](src/data/env.py) — Environment + EnvDataset bases
    - [qa_dataset.py](src/data/qa_dataset.py) — QA dataset adapter (HF or JSONL)
    - envs/
      - [qa_env.py](src/data/envs/qa_env.py)
      - [math_qa_env.py](src/data/envs/math_qa_env.py)
      - [mcq_env.py](src/data/envs/mcq_env.py)
    - synthetic_task_generators/
      - [encryption.py](src/data/synthetic_task_generators/encryption.py)
  - lm/
    - [language_model.py](src/lm/language_model.py)
    - [gemini_client.py](src/lm/gemini_client.py)
    - [lm_factory.py](src/lm/lm_factory.py)
  - memory/
    - [memory_module.py](src/memory/memory_module.py)
    - [history_list.py](src/memory/history_list.py)
    - [memory_factory.py](src/memory/memory_factory.py)
  - utils/
    - [logger.py](src/utils/logger.py) — file logger + optional per-component console

### How it fits together
1) Dataset yields environments (QA/numeric/MCQ) via `QAEnvDataset`.
2) `RunTime` orchestrates: `obs → agent.act → env.step → agent.observe` until `done`.
3) The agent builds prompts using recent memory entries and an optional system prompt.
4) Logging writes to a run-scoped file; console verbosity can be toggled per component.

### Learn more
- Start with [docs/quickstart.md](docs/quickstart.md)
- Explore concepts in [docs/concepts/](docs/concepts/)
- See the config schema in [docs/reference/config.md](docs/reference/config.md)
- Try the cookbook in [docs/cookbooks/encryption-history-list.md](docs/cookbooks/encryption-history-list.md)


