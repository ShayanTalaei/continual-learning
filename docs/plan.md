---
noteId: "de64d0209d8711f0b67f67467df34666"
tags: []

---

## Continual Learning Plan (Trimmed)

### Goals
- **Generic** abstractions for `Environment`, `Agent`, `Memory`, and `LanguageModel`.
- **Minimal** runnable QA demo using a single-turn QA environment.
- **Modular** so we can later plug in multi-step envs, other memories, and other LMs.

### Implemented (see structured docs)
- Quickstart: `docs/quickstart.md`
- Concepts: `docs/concepts/` (runtime, agents, memory, environments, language_model, logging)
- Reference: `docs/reference/config.md`
- Guides: `docs/guides/` (running-benchmarks, synthetic-data)
- Cookbooks: `docs/cookbooks/` (encryption-history-list)

### High-level notes
- Agents are task-agnostic; `HistoryAgent` specializes prompt/update for `HistoryList`.
- Datasets route to env subclasses via `task_type`/`env_class`; `.jsonl` or HF datasets supported.
- Logging writes to a run-scoped file; console verbosity per component.

### Recently Implemented ✅
- **Memory snapshots**: `save_snapshot()` and `load_snapshot()` APIs for persistent memory state.
- **Validation system**: Parallel validation with configurable frequency and worker count.
- **Agent modes**: Training/evaluation modes with context managers and memory freezing.
- **Agent cloning**: `clone_for_episode()` for parallel execution with isolated state.
- **LLM call logging**: Structured JSON logging with context-aware organization.
- **Thread-safe logging**: Generic JSON logging utilities with per-file locks.
- **Config organization**: Moved configs to organized folders by dataset type.
- **Score organization**: Separate training/validation score files with episode-based naming.

### Remaining TODOs
- Implement `ReflexionAgent` and episode-level reflection.
- Multi-trial runtime orchestration (`num_trials`, `early_stop_on_success`, `carry_memory_across_trials`).
- Per-task metrics reporting (e.g., MCQ confusion stats).
- Seed management in `main` (set random/np/torch if present).
- Optional JSON formatter and rotating logs in logger utilities.

### Memory Modules ✅

#### Base
- `MemoryModule`
  - Purpose: Abstract storage for agent memory.
  - Fields:
    - `config: MemoryModuleConfig` — type and parameters for the concrete memory.
  - Functions:
    - `update(entry: Entry) -> None`
      - Purpose: Insert or merge an entry into memory.
    - `recall() -> Any`
      - Purpose: Return memory contents or a view suitable for prompting.

- `MemoryModuleConfig`
  - Fields:
    - `_type: str` — memory type discriminator, e.g., `"history_list"`.

#### History List
- `HistoryList(MemoryModule)`
  - Purpose: Append-only bounded list of typed entries.
  - Fields:
    - `history_list: list[Entry]` — stored entries.
    - `config: HistoryListConfig` — includes max length.
  - Functions:
    - `update(entry: Entry) -> None`
      - Purpose: Append and truncate to `max_length` if necessary.
    - `recall() -> list[Entry]`
      - Purpose: Return the list of entries for prompt construction.

- `HistoryListConfig(MemoryModuleConfig)`
  - Fields:
    - `_type: Literal["history_list"] = "history_list"` — discriminator.
    - `max_length: int = 100` — storage cap.

#### Entries (typed objects)
- `Entry` (discriminated union)
  - Variants:
    - `ExperienceEntry`
      - Fields:
        - `_type: Literal["experience"]`
        - `obs: str` — observation at step.
        - `action: str` — agent output.
        - `feedback: dict` — environment evaluation at step.
    - `ReflectionEntry`
      - Fields:
        - `_type: Literal["reflection"]`
        - `content: str` — distilled advice/rules from prior experience.

#### Factory
- `build_memory(memory_config: MemoryModuleConfig) -> MemoryModule`
  - Purpose: Instantiate concrete memory module based on `_type`.
  - Behavior: `_type == "history_list"` → `HistoryList(memory_config)`; else raise.

---

### Agents

#### Base
- `Agent`
  - Purpose: Minimal agent protocol.
  - Fields:
    - `config: AgentConfig` — agent options.
  - Functions:
    - `act(obs: str) -> str`
      - Purpose: Produce the next action given current observation.
    - `observe(obs: str | None, feedback: dict, done: bool) -> None`
      - Purpose: Receive transition aftermath, update internal state/memory.
    - `reset() -> None`
      - Purpose: Reset internal state between episodes.
    - `end_episode() -> None`
      - Purpose: Optional end-of-episode hook for summary/cleanup.
    - `train() / eval()`
      - Purpose: Toggle the agent's training mode (affects whether long-term memory is updated).
    - `eval_mode()` (context manager)
      - Purpose: Temporarily set eval mode within a `with` block; restores previous mode on exit.
    - `clone_for_episode(training: bool, share_memory: bool = True) -> Agent`
      - Purpose: Create a per-episode clone with independent short-term state (e.g., `_trajectory`), optional shared long-term memory, and explicit training flag.

- `AgentConfig`
  - Purpose: Base class for concrete agent configs.
  - Common fields (shared across agents):
    - `lm_config: LMConfig` — language model configuration.
    - `system_prompt: Optional[str]` — override default system instruction.
    - `verbose: bool = True` — console logging verbosity.

#### MemoryAgent (abstract base, task-agnostic)
- `MemoryAgent[C <: AgentConfig](Agent[C])`
  - Purpose: Abstract orchestrator for agents that use a `MemoryModule` and an `LM`.
  - Fields:
    - `memory: MemoryModule` — pluggable memory backend (any type via factory).
    - `lm: LanguageModel` — pluggable LM client.
    - `config: MemoryAgentConfig` — includes LM and memory configs.
    - `_last_action: str | None` — cache last action for `observe`.
    - `_trajectory: list[Any]` — per-episode buffer for reflections.
  - Functions:
    - `act(obs: str) -> str`
      - Purpose: Format prompt using recent memory entries and return LM output as action.
      - Steps: recall memory → format N latest entries → build `(system_prompt, user_prompt)` → `lm.call(...)`.
    - `observe(obs: str | None, feedback: dict, done: bool) -> None`
      - Purpose: Delegate event creation to hooks and update memory.
      - Always appends to `_trajectory` for episode-level reflection.
      - Updates to long-term memory are performed only when `self.training` is True.
    - `end_episode() -> None`
      - Purpose: Default no-op; subclass may implement reflection here.
  - Training/Eval semantics:
    - In eval mode: long-term memory (e.g., `HistoryList`) is frozen (no `memory.update(...)`).
    - Short-term state (e.g., `_trajectory`, `_last_action`) can still be written per-episode.
    - Provide a zero-side-effect `eval_act(obs: str) -> str` in concrete subclasses for efficiency; default fallback can use `with agent.eval_mode(): act(obs)`.
  - Hooks (must be implemented by concrete agents):
    - `build_system_prompt() -> str`
    - `build_user_prompt(obs: str, history: list[Any], k: int | None) -> str`
    - `create_observation_event(obs: str) -> Any`
    - `create_action_event(action: str) -> Any`
    - `create_feedback_event(feedback: dict) -> Any`
  - Logging convention (recommended):
    - The base logs Observation/Action/Feedback through the hook-created events.

#### HistoryAgent (concrete over HistoryList)
- `HistoryAgent(MemoryAgent[HistoryAgentConfig])`
  - Purpose: Concrete agent operating over `HistoryList` memory.
  - Fields:
    - `memory_config: HistoryListConfig`, `history_k: int`, `system_prompt: Optional[str]` in its config.
  - Behavior:
    - Prompt: formats last `k` entries as `"<TYPE>: <content>"`, followed by `Q: <obs>`.
    - Update: writes atomic entries via hooks: Observation, Action, Feedback.
    - `end_episode()`: no-op for now; extension point for Reflexion.

#### MemorylessAgent (no persistent memory)
- `MemorylessAgent(Agent[AgentConfig])`
  - Purpose: Agent without persistent memory across steps/episodes.
  - Fields:
    - Inherits common fields from `AgentConfig`: `lm_config`, `system_prompt?`, `verbose?`.
  - Behavior:
    - Act: builds prompt from only the current observation and calls the LM.
    - Observe: maintains an ephemeral `_trajectory` per step and clears it each step (no writes to a memory backend).
    - End of episode: no-op.

#### ReflexionAgent
- `ReflexionAgent(MemoryAgent)`
  - Purpose: Adds reflection using the LM to write `ReflectionEntry`s.
  - Fields:
    - `config: ReflexionAgentConfig` — adds reflection options.
  - Functions:
    - `update_memory(obs, action, feedback, done) -> None`
      - Purpose: Optionally perform per-step micro-reflection (if enabled), then call base to append experience.
    - `end_episode() -> None`
      - Purpose: Synthesize episode reflections from `_trajectory` using LM and append a `ReflectionEntry`.
    - `build_reflection_prompt(trajectory: list[ExperienceEntry]) -> tuple[str, str]`
      - Purpose: Produce prompts that elicit concise “dos/don’ts” based on the episode.

- `ReflexionAgentConfig(MemoryAgentConfig)`
  - Fields:
    - `reflection_config: ReflectionConfig` — mode and prompting for reflections.

- `ReflectionConfig`
  - Fields:
    - `enabled: bool = True` — master toggle.
    - `mode: Literal["episode_end", "per_step", "both"] = "episode_end"` — trigger mode.
    - `system_prompt: Optional[str]` — override default system instruction.
    - `max_tokens: Optional[int]` — budget for reflections.

---

### Language Model ✅

- `LanguageModel`
  - Purpose: Abstracts LM provider.
  - Fields:
    - `config: LMConfig` — runtime parameters and model choice.
  - Functions:
    - `call(system_prompt: str, user_prompt: str) -> str`
      - Purpose: Synchronous call returning model text output.
    - `enable_call_logging(calls_dir: str) -> None`
      - Purpose: Enable call logging to `calls_dir` (under timestamped results dir) when `LMConfig.log_calls` is true.

- `LMConfig`
  - Fields:
    - `model: str` — provider/model identifier (e.g., `"gemini-1.5-pro"`).
    - `temperature: float = 0.2`
    - `max_output_tokens: int = 2048`
    - `log_calls: bool = false` — when true, input and output of LM calls are saved as JSON files under `llm_calls/` in the run's results directory.

- `GeminiConfig(LMConfig)`
  - Fields:
    - `thinking_budget: Optional[int]` — enables Gemini thinking if supported.

- `get_lm_client(lm_config: LMConfig) -> LanguageModel`
  - Purpose: Return a concrete client (e.g., `GeminiClient`) based on `lm_config.model`.
  - Runtime wiring: `main` will call `agent.lm.enable_call_logging(<results_dir>/llm_calls)` when `lm_config.log_calls` is true.

---

### Runtime Logic and Scoring ✅

- `RunTime`
  - Purpose: Orchestrate running a dataset of environments through an agent.
  - Fields:
    - `config: RunTimeConfig`
    - `env_dataset: EnvDataset`
    - `agent: Agent`
  - Functions:
    - `run() -> dict`
      - Purpose: Iterate over environments, run episodes, aggregate metrics.
      - Returns: summary metrics (e.g., `mean_score`) and per-env logs.
    - `run_episode(environment: Environment) -> list[StepResult]`
      - Purpose: Execute one full episode using the Gym-style loop; returns recorded steps.

- `RunTimeConfig`
  - Fields:
    - `max_envs_to_visit: Optional[int]` — subset for quick runs.
    - `max_steps_per_episode: Optional[int]` — safety cap for multi-turn envs.
    - `scores_path: Optional[str]` — when set, write streaming `scores.jsonl` and snapshot `scores.json`.
    - `runtime_type: Literal["sequential", "parallel"] = "sequential"` — parallel only valid for memoryless agent.
    - `num_parallel_episodes: Optional[int]` — cap parallelism when `runtime_type="parallel"`.
    - `validation_freq: Optional[int]` — run validation every N training episodes (omit to disable).
    - `validation_num_workers: Optional[int]` — parallelism for validation (defaults to `num_parallel_episodes` if unset).

- `StepResult`
  - Fields:
    - `obs: str | None` — observation after the step.
    - `action: str` — agent action for this step.
    - `feedback: dict` — evaluation payload from the env.
    - `done: bool` — terminal flag.

### Future Extensions
- Retrieval DB memory, dynamic cheatsheet memory, cartridge (prefix-tuning) memory.
- Multi-step environments; richer feedback (partial credit, rationales).
- Learnable update functions and summarization.

### Assumptions Kept Minimal
- Single-turn QA per env for phase 1.
- Exact string match for correctness.
- Plain-text prompts (no JSON tool calls).



---

## Next Stage Implementation Goals

### 1) Reflexion Multi-Trials (Runtime-Orchestrated)
- Goals:
  - Support multiple trials per environment to allow between-trial reflection and improvement.
  - Keep orchestration in runtime; agents remain focused on per-episode logic.
- Additions to `RunTimeConfig`:
  - `num_trials: Optional[int] = 1` — number of times to re-run each environment.
  - `early_stop_on_success: bool = False` — advance to next env when success observed.
  - `carry_memory_across_trials: bool = True` — whether to reset agent memory between trials.
- Runtime flow per env:
  - for `t in range(num_trials)`:
    - if `t > 0` and not `carry_memory_across_trials`: reset agent memory.
    - run episode loop
    - `agent.end_episode()` to trigger reflections
    - if `early_stop_on_success` and last feedback indicates success: break
- Optional: `Agent` memory snapshot API for persistence across sessions (save/load).

### 2) Hybrid Memory and Dynamic Cheatsheet Agent
- Memory modules:
  - `CheatsheetMemory`: stores a single evolving cheatsheet string; optional versioning.
  - `ExperienceDBMemory`: structured store of `(obs, action, feedback, metadata)` with similarity search.
  - `HybridMemory`: composes sub-memories and exposes convenience methods:
    - `get_cheatsheet() -> str`
    - `set_cheatsheet(text: str) -> None`
    - `append_experience(entry: ExperienceEntry) -> None`
    - `retrieve_examples(query: Any, k: int) -> list[ExperienceEntry]`
- Retrieval service (optional alternative):
  - `RetrievalService` with `build_index`, `query_top_k` to avoid coupling retrieval to memory backend.
- Agent: `DynamicCheatsheetAgent(MemoryAgent)`
  - Act flow:
    - Pull cheatsheet from memory.
    - Optionally retrieve top-k exemplars from DB memory or service.
    - Build prompt with cheatsheet + exemplars + recent experiences.
    - Generate answer (optionally multi-round and code-exec; see below).
    - Curate/update cheatsheet via a second LM call; write `CheatsheetEntry` (or `ReflectionEntry`).
    - Return final answer.
  - Observe/update:
    - Append `ExperienceEntry`; optionally summarize/compact cheatsheet on `end_episode()`.
- Optional code execution plugin:
  - `CodeExecutionService.run(code, timeout) -> str`, gated by config flags.
  - Agent detects execution flag in LM output and replaces it with execution result before finalizing.
- Config additions:
  - `DynamicCheatsheetAgentConfig(MemoryAgentConfig)`:
    - `use_retrieval: bool`, `retrieve_top_k: int`
    - `use_code_execution: bool`, `code_timeout_sec: int`
    - `generator_template: Optional[str]`, `cheatsheet_template: Optional[str]`

These additions keep runtime control simple and make memory and agent behavior modular enough to express both Reflexion and Dynamic Cheatsheet patterns.

---

## Benchmark Integration Plan (Task-Agnostic Agents, Task-Specific Envs) ✅

### Goals
- Support real benchmarks (e.g., AIME, GPQA, MMLU-Pro) using flexible dataset loading.
- Keep agents task-agnostic; encapsulate evaluation and normalization in env subclasses.
- Provide a uniform feedback schema and simple runtime aggregation.

### Dataset Ingestion ✅
- Primary loader: HuggingFace `datasets`.
  - Accept `dataset_path` (for `load_from_disk`) or `hf_name/config/split` (for `load_dataset`).
- Extend `QAEnvDatasetConfig` with:
  - `task_type: Literal["exact", "numeric", "mcq", "custom"]`
  - `split: Optional[str]`
  - `input_field: str`, `target_field: str`
  - `choices_field: Optional[str]`, `id_field: Optional[str]`, `meta_fields: list[str] = []`
  - `env_class: Optional[str]` to force a specific env subclass
  - `prompt_template: Optional[str]` (optional hint only)
  - `instruction_template: Optional[str]` (format string with `{question}`) passed to `QAEnv`

### Environment Hierarchy ✅
- `QAEnv` (base, single-turn)
  - Hooks: `evaluate`, optional `normalize_input/normalize_target/normalize_action`.
- `MathQAEnv(QAEnv)`
  - Numeric/boxed answer normalization, tolerance handling, whitespace/LaTeX stripping.
- `MCQEnv(QAEnv)`
  - Multiple-choice support with `choices`; case-insensitive letter/text match; normalization helpers.
- Future: `MultiTurnQAEnv` for multi-hop tasks (out of current scope).

### Routing and Construction ✅
- In `QAEnvDataset.load_dataset`:
  - Load rows via `datasets`.
  - If `env_class` is set: instantiate that env explicitly.
  - Else select by `task_type`:
    - `exact` → `QAEnv`
    - `numeric` → `MathQAEnv`
    - `mcq` → `MCQEnv`
    - `custom` → require `env_class` or raise
  - Build `metadata` per-row including id, choices (if any), and extra fields from `meta_fields`.

### Prompting and Agents
- Agents remain task-agnostic; they may accept `system_prompt` overrides.
- History agents format memory generically: `<TYPE>: <content>` lines and `Q: <obs>`.
- Task-specific prompt nuances should be encoded either in `system_prompt` or in an env-specific subclass if strictly necessary.

### Feedback Schema and Aggregation ✅
- Uniform feedback across envs: `{score: number, target: str, message: str, extra?: dict}`.
  - Binary evaluations: `score` ∈ {1, 0}.
  - Non-binary evaluations: `score` normalized to [0, 1] when possible.
- Backward compatibility: if `correct`/`is_correct` present, map to `score` with a deprecation log.
- Runtime aggregates `mean_score` over steps (and provides per-episode totals/means).

### Online Score Monitoring Outputs ✅
- `scores.jsonl`: one line per step containing `{ episode_index, step_index, score, episode_cum_score, timestamp }`.
- `scores.json`: snapshot with per-episode totals/means and overall aggregates.

### Mapping to Benchmarks
- AIME: `task_type="numeric"` → `MathQAEnv` (normalization/tolerance).
- GPQA/MMLU-Pro: `task_type="mcq"` → `MCQEnv` (choices + letter/text matching).
- Others can plug in via `custom` + `env_class`.

### Step-by-Step Integration Sequence
1) ✅ Extend `QAEnvDatasetConfig` with fields above and support both `load_from_disk` and `load_dataset`.
2) ✅ Implement `MathQAEnv` and `MCQEnv` (inherit `QAEnv`, override `evaluate`, add normalizers as needed).
3) ✅ Implement routing in `QAEnvDataset.load_dataset` using `task_type`/`env_class`.
4) Add minimal configuration examples in docs for AIME, GPQA, MMLU-Pro.
5) Smoke-test each mode on a tiny subset; verify accuracy aggregation and memory logging.

---

## Unified Runs via YAML Configs ✅

### Goals
- Define full runs declaratively in YAML under `configs/`.
- Single main entrypoint that loads a YAML, builds components, runs, and saves results.

### Config Structure (YAML) ✅
- Top-level keys:
  - `runtime`: fields of `RunTimeConfig`
  - `train_dataset`: fields of `QAEnvDatasetConfig`
  - `validation_dataset`: fields of `QAEnvDatasetConfig`
  - `lm`: fields of `LMConfig`/`GeminiConfig`
  - `memory`: fields of `MemoryModuleConfig` (e.g., `{ _type: "history_list", max_length: 200 }`)
  - `agent`: `{ type: "history_agent", ... }` plus its config fields
  - `output`: `{ results_dir, save_memory_path, log_level }`
  - `seed`: optional int

### Schema Mapping ✅
- `RunConfig`
  - `runtime: RunTimeConfig`
  - `train_dataset: QAEnvDatasetConfig`
  - `validation_dataset: QAEnvDatasetConfig`
  - `lm: LMConfig|GeminiConfig`
  - `memory: MemoryModuleConfig`
  - `agent: { type: str, ... }` (fields merged for that agent)
  - `output: OutputConfig`
  - `seed: Optional[int]`
- `OutputConfig`: `{ results_dir?: str, save_memory_path?: str, log_level?: "DEBUG"|"INFO"|"WARN"|"ERROR" }`
  - Behavior: At runtime the actual results directory is suffixed with a timestamp `YYYYMMDD_HHMMSS` and the effective config is copied as `config.yaml` inside it.

### Entrypoint Flow ✅
1) Load YAML file; resolve relative paths to YAML dir.
2) Validate to `RunConfig`.
3) Initialize logging and output dirs; set seed (if provided).
4) Build: `lm`, `memory`, `agent` (via registry), `dataset`, `runtime`.
5) Run `runtime.run()`; write `metrics.json` and optional memory dump.

### Agent Registry ✅
- Map `agent.type` → `(ConfigClass, AgentClass)`
  - `history_agent` → `(HistoryAgentConfig, HistoryAgent)`
  - `memory_agent` → `(MemoryAgentConfig, MemoryAgent)`
  - `memoryless_agent` → `(MemorylessAgentConfig, MemorylessAgent)`
  - future: `reflexion_agent` → `ReflexionAgent`

### Examples ✅
- `configs/aime_numeric.yaml`, `configs/gpqa_mcq.yaml`, `configs/toy_inmemory.yaml`.

---

## Remaining TODOs
- Add example configs for AIME/GPQA/MMLU-Pro.
- Implement `ReflexionAgent` and episode-level reflection.
- Multi-trial runtime orchestration (`num_trials`, `early_stop_on_success`, `carry_memory_across_trials`).
- Per-task metrics reporting (e.g., MCQ confusion stats).
- Seed management in `main` (set random/np/torch if present).
- Optional JSON formatter and rotating logs in logger utilities.

---

## Logging Strategy ✅

### Goals
- Single run-scoped log file (plus stdout) capturing all components.
- Simple dependency injection: create the logger in `main`, pass to root objects.
- Extensible formatting/handlers later without touching business logic.

### Design
- `utils/logger.py` provides:
  - `setup_logger(name: str, level: str, log_path: str, extra: dict | None = None) -> logging.Logger`
  - `child(logger: logging.Logger, name: str, extra: dict | None = None) -> logging.Logger`
- Root classes accept an optional logger and store it:
  - `RunTime.__init__(..., logger: Optional[Logger] = None)`
  - `Agent.__init__(..., logger: Optional[Logger] = None)`
  - `EnvDataset.__init__(..., logger: Optional[Logger] = None)`
- In `main`:
  - Create `root_logger = setup_logger("run", level, results_dir/run.log)`
  - Pass `child(root_logger, "runtime")`, `child(root_logger, "agent")`, `child(root_logger, "dataset")` to components.
- Components use `self.logger` for all logging; do not reconfigure handlers.

### Extensibility
- Swap formatter to JSON later.
- Add more handlers (rotation/cloud) without changing components.
- Use adapters or child loggers to add context (env_id, step) where needed.
