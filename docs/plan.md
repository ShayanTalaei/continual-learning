---
noteId: "de64d0209d8711f0b67f67467df34666"
tags: []

---

## Minimal Plan: QA Task with History-List Memory Agent

### Goals
- **Generic** abstractions for `Environment`, `Agent`, `Memory`, and `LanguageModel`.
- **Minimal** runnable QA demo using a single-turn QA environment.
- **Modular** so we can later plug in multi-step envs, other memories, and other LMs.

### Core Interfaces (aligned with Gym-style patterns) ✅
- `Environment.reset() -> obs` initializes the environment and returns the first observation.
- `Environment.step(action) -> (obs, feedback, done, info)` advances the environment. `feedback` is a dict (task-specific), `done` indicates episode termination.
- `EnvDataset`: loads a list of `Environment` instances.
- `Agent.act(obs: str) -> str` returns the next action.
- `Agent.observe(obs: str, feedback: dict, done: bool) -> None` receives the transition aftermath (for memory updates, learning signals).
- `Agent.reset() -> None` resets internal state between episodes.
- `MemoryModule`: `update(entry) -> None`, `recall() -> Any`.
- `LanguageModel`: `call(system_prompt: str, user_prompt: str) -> str`.
  
Task-specific evaluation lives in the environment implementation itself (e.g., `MathQAEnv.step` can compute correctness by parsing boxed answers). If desired, expose a helper `Environment.evaluate(action) -> dict` used inside `step`.
  
`StepResult` (runtime record): `{obs: str, action: str, feedback: dict, done: bool}`.

### Concrete Implementations for Phase 1
- `HistoryList` memory: append-only list of entries with optional max length; returns the list for recall.
- `QAEnv` and `QAEnvDataset`: single-turn QA pairs `(question, answer)`; `step` returns ground-truth to compute feedback.
- `MemoryAgent`:
  - Builds prompts from `(recall, current obs)`.
  - Calls LM to get an answer.
  - On feedback, adds a structured entry `(obs, action, feedback)` to history.
  - Provides optional `ReflectionPolicy` hook(s) to run at episode end (and optionally per step) to synthesize guidance and write `_type="reflection"` entries.
- `LanguageModel` implementations:
  - `GeminiClient` (already scaffolded). We will rely on Gemini directly for this project and skip mocks for now.
  
Task-specific environment subclassing for evaluation (example):
- `MathQAEnv(QAEnv)`: overrides `evaluate(action)` to extract boxed answers or parse numeric forms, then compares against canonical target. Keeps `QAEnvDataset` simple while isolating evaluation logic in the env.

### Run Loop ✅
`RunTime.run()`:
1. Iterate over environments from `EnvDataset` (optionally limit via config).
2. For each environment (episode loop):
   - `obs = environment.reset()`; `agent.reset()`
   - `done = False`
   - while not `done`:
     - `action = agent.act(obs)`
     - `obs, feedback, done, info = environment.step(action)`  (env computes task-specific feedback)
     - `agent.observe(obs, feedback, done)`  (may trigger per-step reflection if enabled)
     - record `StepResult`
   - After loop ends, call `agent.end_episode()` (triggers episode-level reflection if configured).
   - Aggregate per-env metrics from recorded `StepResult`s.

### Prompt Format (minimal, plain text) ✅
- System: "You are a helpful QA assistant. Use prior history when useful."
- User: a compact block:
  - Optional history lines (last `k`, e.g., 10): `Q: ... A: ... F: ...`
  - Current: `Q: <obs>`

### Data Flow ✅
1. `EnvDataset` yields `QAEnv(question, answer, metadata)`.
2. `RunTime` asks `Agent` to act using `LanguageModel` with memory-augmented prompt.
3. `RunTime` computes feedback and sends to `Agent`.
4. `Agent` updates `HistoryList` with an `Entry` capturing `(obs, action, feedback)`.

### Minimal Types ✅
- `Entry` with fields: `_type` (e.g., "experience"), `content` (stringified record). Keep simple for now.
- `HistoryListConfig` with `max_length: int = 100`.
- `MemoryAgentConfig`:
  - `memory_config: HistoryListConfig`
  - `lm_config: LMConfig`
  - `history_k: int = 10` (how many to include in prompt)
- `Feedback` dict shape: `{correct: bool, target: str, message: str}` (extensible by envs)
- `StepResult` as described above

### Agent Hierarchy and Memory Updates ✅
- Base `Agent`: `act`, `observe`, `reset` as defined above.
- `MemoryAgent(Agent)`: centralizes memory handling and exposes a single extension point:
  - `update_memory(obs, action, feedback, done) -> None` (called by `observe`).
  - Default behavior (no-reflection): append `ExperienceEntry` to memory.
- `ReflexionAgent(MemoryAgent)`: overrides `update_memory` (and optionally `end_episode`) to run reflection using the LM and append `ReflectionEntry` in addition to experiences.
  - Minimal policy: reflections at episode end summarizing what worked/failed.
  - Optional extension: per-step micro-reflections if desired.

### Config Hierarchy (Pydantic) ✅
- `RunTimeConfig`:
  - `max_envs_to_visit: Optional[int]`
  - `max_steps_per_episode: Optional[int]` (safety cap)
- `EnvDatasetConfig` (base):
  - `dataset_path: Optional[str]` (enables `QAEnvDataset` to load via `datasets`)
- `QAEnvDatasetConfig(EnvDatasetConfig)`:
  - `question_field: str`, `answer_field: str`
- `AgentConfig` (base) — used generically by `Agent[C]`.
- `HistoryAgentConfig(AgentConfig)`:
  - `lm_config: LMConfig`
  - `memory_config: HistoryListConfig`
  - `history_k: int = 10`
- `ReflexionAgentConfig(HistoryAgentConfig)`:
  - `reflection_config: ReflectionConfig`
- `LMConfig` → `GeminiConfig` (adds `thinking_budget`, etc.).
- `MemoryModuleConfig` → `HistoryListConfig`.
- `ReflectionConfig`:
  - `enabled: bool = False`
  - `mode: Literal["episode_end", "per_step", "both"] = "episode_end"`
  - `max_tokens: Optional[int]` (if needed)
  - `system_prompt: Optional[str]` (override default reflexion prompt)

### Memory Entry Modularization ✅
- Use a single generic entry object to keep the memory flexible:
  - `Entry`: `{ _type: str, content: str }`
    - Atomic logging pattern (recommended):
      - `_type="Observation"`, `content=<obs>`
      - `_type="Action"`, `content=<action>`
      - `_type="Feedback"`, `content=<feedback-json-or-text>`
    - Other examples:
      - `_type="Reflection"`, `content="short advice text"`
      - `_type="Summary"`, `_type="Retrieval"`, etc.
- The `HistoryList` stores `List[Entry]` and truncates to `max_length`.

### Milestones (implements in order)
1) Interfaces and Configs ✅
   - Update `Environment` to Gym-style: `reset() -> obs`, `step(action) -> (obs, feedback, done, info)`.
   - Update `Agent` to: `act(obs) -> action`, `observe(obs, feedback, done)`, `reset()`, and add `end_episode()`.
   - Define config classes:
     - `RunTimeConfig`: `max_envs_to_visit`, `max_steps_per_episode`.
     - `EnvDatasetConfig` and `QAEnvDatasetConfig`.
     - `LMConfig`/`GeminiConfig`.
     - `MemoryModuleConfig`/`HistoryListConfig`.
     - `AgentConfig` → `MemoryAgentConfig` → `ReflexionAgentConfig`.

2) Memory: HistoryList ✅
   - Ensure `HistoryList` stores a discriminated union `Entry` type: `ExperienceEntry | ReflectionEntry`.
   - Expose `update(entry)` and `recall()`; enforce `max_length` truncation.
   - Fix `memory_factory` to select by `_type` or `type` consistently with `HistoryListConfig`.

3) QA Environment and Dataset ✅
   - `QAEnv(question, answer, metadata)` implementing `reset` and single-step `step`:
     - On `reset`, return the question; internal flag `done=False`.
     - On `step(action)`, compute feedback: exact match by default; return `(obs=None, feedback, done=True, info={"target": answer})`.
   - `QAEnvDataset` loads an in-memory list of QA pairs for the demo.
   - Optional subclass `MathQAEnv(QAEnv)` overriding evaluation (e.g., boxed answer extraction) later.

4) Language Model ✅
   - Implement `LanguageModel` base signature; rely on `GeminiClient` for calls.
   - Ensure `lm_factory.get_lm_client` returns `GeminiClient` given `GeminiConfig`.

5) Agents ✅
   - `MemoryAgent`:
     - Holds `memory`, `lm`, `history_k`.
     - `act(obs)`: build prompt from last `k` entries + current obs, call LM, return action.
     - `observe(obs, feedback, done)`: call `update_memory(obs, action, feedback, done)`.
     - `update_memory(...)` default: append `ExperienceEntry`.
     - `end_episode()`: default no-op.
   - `ReflexionAgent(MemoryAgent)`:
     - Override `update_memory` (optionally) if per-step reflections are needed.
     - Override `end_episode()` to synthesize reflections from the trajectory using the LM and append a `ReflectionEntry`.

6) RunTime Loop ✅
   - `run()` iterates envs; for each:
     - `obs = env.reset()`; `agent.reset()`
     - loop steps with safety cap from `RunTimeConfig`:
       - `action = agent.act(obs)`
       - `obs, feedback, done, info = env.step(action)`
       - `agent.observe(obs, feedback, done)`
       - record `StepResult`
       - if `done`: break
     - `agent.end_episode()`
     - accumulate metrics (e.g., accuracy) from feedbacks.

7) Demo and Instructions
   - Add a minimal demo in docs (or `examples/`) showing how to:
     - Build `QAEnvDataset` from a few QA pairs.
     - Instantiate `ReflexionAgent` or `MemoryAgent` with `HistoryList` and `GeminiClient`.
     - Run `RunTime` and print accuracy and saved memory entries.

---

## Implementation Spec (Bottom-Up)

### Environments ✅

#### Base Interfaces ✅
- `Environment`
  - Purpose: Standard Gym-style text environment protocol for single-/multi-turn tasks.
  - Functions:
    - `reset() -> str`
      - Purpose: Initialize episode state and return the first observation string.
    - `step(action: str) -> tuple[str | None, dict, bool, dict]`
      - Purpose: Advance environment with the agent action.
      - Returns: `(next_obs, feedback, done, info)`
        - `next_obs: Optional[str]` — the next observation, or `None` if terminal.
        - `feedback: dict` — task-specific evaluation, minimally `{correct: bool, target: str, message: str}`.
        - `done: bool` — whether the episode has terminated.
        - `info: dict` — auxiliary details (e.g., normalized answers, metadata ids).
    - `evaluate(action: str) -> dict` (optional helper)
      - Purpose: Centralize evaluation logic for reusability; `step` can call this internally.

- `EnvDataset`
  - Purpose: Provide a collection of `Environment` instances for the runtime.
  - Fields:
    - `config: EnvDatasetConfig` — dataset configuration.
    - `dataset: list[Environment]` — list of ready-to-run envs.
  - Functions:
    - `load_dataset() -> list[Environment]`
      - Purpose: Build the list of environments from source (in-memory or files).
    - `get_dataset() -> list[Environment]`
      - Purpose: Accessor for the constructed environments.

#### QA Specializations
- `QAEnv(Environment)`
  - Fields:
    - `question: str` — the prompt presented to the agent.
    - `answer: str` — ground-truth answer string.
    - `metadata: dict` — dataset/source meta.
    - `_done: bool` — internal terminal flag (single-turn).
  - Functions:
    - `reset() -> str`
      - Purpose: Set `_done=False` and return `question`.
    - `step(action: str) -> (None, dict, True, dict)`
      - Purpose: Single-step episode; compute feedback via exact string match.
      - Feedback: `{correct: action == answer, target: answer, message: "exact-match"}`.
      - Info: `{}` or `{"question_id": ..., "source": ...}` from `metadata`.
    - `evaluate(action: str) -> dict`
      - Purpose: Isolated evaluation used by `step`; enables easy overrides.

- `QAEnvDataset(EnvDataset)`
  - Fields:
    - `items: list[dict]` — in-memory rows with `question`, `answer`, `metadata`.
  - Functions:
    - `load_dataset() -> list[Environment]`
      - Purpose: Wrap each row as a `QAEnv` instance and return the list.

- Optional later: `MathQAEnv(QAEnv)`
  - Purpose: Override `evaluate` to extract boxed or normalized numeric answers before comparison.

---

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

- `AgentConfig`
  - Purpose: Base class for concrete agent configs.

#### MemoryAgent (generic base, task-agnostic)
- `MemoryAgent[C <: AgentConfig](Agent[C])`
  - Purpose: Task-agnostic orchestrator for agents that use a `MemoryModule` and an `LM`.
  - Fields:
    - `memory: MemoryModule` — pluggable memory backend (any type via factory).
    - `lm: LanguageModel` — pluggable LM client.
    - `config: MemoryAgentConfig` — includes LM and memory configs.
    - `_last_action: str | None` — cache last action for `observe`.
    - `_trajectory: list[ExperienceEntry]` — per-episode buffer for reflections.
  - Functions:
    - `act(obs: str) -> str`
      - Purpose: Format prompt using recent memory entries and return LM output as action.
      - Steps: recall memory → format N latest entries → build `(system_prompt, user_prompt)` → `lm.call(...)`.
    - `observe(obs: str | None, feedback: dict, done: bool) -> None`
      - Purpose: Create `ExperienceEntry` from `(obs_prev, _last_action, feedback)` and call `update_memory`.
      - Also appends to `_trajectory` for episode-level reflection.
    - `update_memory(obs: str | None, action: str, feedback: dict, done: bool) -> None`
      - Purpose: Extension point; default appends `ExperienceEntry` to memory.
    - `end_episode() -> None`
      - Purpose: Default no-op; subclass may implement reflection here.
  - Hooks (to be implemented/overridden by concrete agents):
    - `build_system_prompt() -> str`
    - `build_user_prompt(obs: str, history: list[Entry], k: int) -> str`
    - `update_memory_with_entry(entry: Entry) -> None`
    - `end_episode() -> None`
  - Logging convention (recommended, still task-agnostic):
    - The base `observe` can log atomic entries for Observation, Action, Feedback; concrete agents can extend/override.

#### HistoryAgent (specialized, task-agnostic formatting)
- `HistoryAgent(MemoryAgent[HistoryAgentConfig])`
  - Purpose: Concrete agent operating over `HistoryList` memory.
  - Fields:
    - `memory_config: HistoryListConfig`, `history_k: int`, `system_prompt: Optional[str]` in its config.
  - Behavior:
    - Prompt: formats last `k` entries as `"<TYPE>: <content>"`, followed by a generic line for the next observation (e.g., `Observation: <obs>`). Avoids QA-specific labels.
    - Update: on `observe`, writes atomic entries: Observation, Action, Feedback.
    - `end_episode()`: no-op for now; extension point for Reflexion.

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

- `LMConfig`
  - Fields:
    - `model: str` — provider/model identifier (e.g., `"gemini-1.5-pro"`).
    - `temperature: float = 0.2`
    - `max_output_tokens: int = 2048`

- `GeminiConfig(LMConfig)`
  - Fields:
    - `thinking_budget: Optional[int]` — enables Gemini thinking if supported.

- `get_lm_client(lm_config: LMConfig) -> LanguageModel`
  - Purpose: Return a concrete client (e.g., `GeminiClient`) based on `lm_config.model`.

---

### Runtime Logic ✅

- `RunTime`
  - Purpose: Orchestrate running a dataset of environments through an agent.
  - Fields:
    - `config: RunTimeConfig`
    - `env_dataset: EnvDataset`
    - `agent: Agent`
  - Functions:
    - `run() -> dict`
      - Purpose: Iterate over environments, run episodes, aggregate metrics.
      - Returns: summary metrics (e.g., overall accuracy) and per-env logs.
    - `run_episode(environment: Environment) -> list[StepResult]`
      - Purpose: Execute one full episode using the Gym-style loop; returns recorded steps.

- `RunTimeConfig`
  - Fields:
    - `max_envs_to_visit: Optional[int]` — subset for quick runs.
    - `max_steps_per_episode: Optional[int]` — safety cap for multi-turn envs.

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
- History agents format memory generically: `<TYPE>: <content>` lines and `Observation: <obs>`.
- Task-specific prompt nuances should be encoded either in `system_prompt` or in an env-specific subclass if strictly necessary.

### Feedback Schema and Aggregation ✅
- Uniform feedback across envs: `{correct: bool, target: str, message: str, extra?: dict}`.
- Runtime aggregates overall accuracy via `feedback.correct`.
- Per-task metrics can be added later by reading `feedback.extra`.

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
  - `dataset`: fields of `QAEnvDatasetConfig`
  - `lm`: fields of `LMConfig`/`GeminiConfig`
  - `memory`: fields of `MemoryModuleConfig` (e.g., `{ _type: "history_list", max_length: 200 }`)
  - `agent`: `{ type: "history_agent", ... }` plus its config fields
  - `output`: `{ results_dir, save_memory_path, log_level }`
  - `seed`: optional int

### Schema Mapping ✅
- `RunConfig`
  - `runtime: RunTimeConfig`
  - `dataset: QAEnvDatasetConfig`
  - `lm: LMConfig|GeminiConfig`
  - `memory: MemoryModuleConfig`
  - `agent: { type: str, ... }` (fields merged for that agent)
  - `output: OutputConfig`
  - `seed: Optional[int]`
- `OutputConfig`: `{ results_dir?: str, save_memory_path?: str, log_level?: "DEBUG"|"INFO"|"WARN"|"ERROR" }`

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
