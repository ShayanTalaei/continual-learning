## Agents

### Base Agent Types

- **`Agent`**: Base class with core interface:
  - `act(obs)` → action
  - `observe(obs, feedback, done)` → updates internal state
  - `reset()` → reset between episodes
  - `end_episode()` → optional end-of-episode hook
  - `train()` / `eval()` → mode switching
  - `eval_mode()` → context manager for temporary eval mode
  - `clone_for_episode(training, share_memory)` → parallel execution support

- **`MemoryAgent`**: Abstract task-agnostic base for memory-backed agents. Orchestrates LM calls and memory usage via hooks:
  - `build_system_prompt()` → constructs system prompt
  - `build_user_prompt(obs, history, k)` → formats user prompt with memory context
  - `create_observation_event(obs)` → creates observation entry for memory
  - `create_action_event(action)` → creates action entry for memory
  - `create_feedback_event(feedback)` → creates feedback entry for memory
  
  Concrete agents implement these hooks to customize prompts and memory entries.

### Concrete Agent Implementations

- **`MemorylessAgent`**: No persistent memory across steps/episodes. Prompts only with current observation. Ideal for:
  - Baseline comparisons
  - Parallel validation (no memory coordination needed)
  - Tasks where history is not beneficial

- **`HistoryAgent`**: Specializes `MemoryAgent` for `HistoryList` memory:
  - Formats prompt lines as `<TYPE>: <content>`
  - Writes atomic Observation/Action/Feedback entries
  - Configurable history window via `history_k` (shows last k entries)
  - Simple append-only memory strategy

- **`ReflexionAgent`**: Self-reflective agent inspired by "Reflexion: Language Agents with Verbal Reinforcement Learning" (Shinn et al., 2023):
  - **Key feature**: Generates verbal reflections after episodes using LM to analyze trajectory
  - **Memory integration**: Stores reflections in same memory as experiences
  - **Prompt structure**: Reflections prominently featured before recent history
  - **Configurable triggers**: Reflect on all episodes or failures only
  - **Training-mode gated**: Reflections only generated during training
  
  **Configuration options:**
  - `enable_reflection`: Master toggle (default: true)
  - `reflect_on_failure_only`: Only reflect when score < threshold (default: false)
  - `failure_threshold`: Score threshold for failure detection (default: 1.0)
  - `reflection_system_prompt`: Override default reflection prompt
  - `reflection_few_shot_examples`: Optional examples for reflection generation
  
  **Reflection prompt**: Task-agnostic, analyzes observations/actions/feedback to extract lessons.

### Agent Modes

- **Training mode**: Memory updates enabled, agent learns from experience
- **Evaluation mode**: Memory updates disabled (frozen), agent performs inference only
- **Context managers**: Use `with agent.eval_mode():` to temporarily switch to eval mode; restores previous mode on exit

**Implementation notes:**
- Short-term state (e.g., `_trajectory`, `_last_action`) can still be written in eval mode
- Long-term memory (e.g., `HistoryList`) is frozen in eval mode
- Mode is propagated to memory module via `memory.train()` / `memory.eval()`

### Agent Cloning

- **Purpose**: Create isolated agent instances for parallel episode execution
- **API**: `clone_for_episode(training: bool, share_memory: bool = True)`
- **Behavior**:
  - Shares LM client (saves resources)
  - Optionally shares memory module (safe for eval when training=False)
  - Independent short-term state (trajectory, last action)
  - Explicit training flag controls memory updates

**Use cases:**
- Parallel validation with shared memory
- Independent episode runs with isolated memory

### Configuration

**Common fields** (inherited from `AgentConfig`):
- `lm_config`: Language model configuration
- `system_prompt`: Override default system instruction
  - Supports file injection: `{file:path/to/prompt.txt}`
- `verbose`: Console logging verbosity (default: true)

**Agent-specific configs:**
- `HistoryAgentConfig`: adds `memory_config`, `history_k`
- `ReflexionAgentConfig`: adds reflection options (see above)
- `MemorylessAgentConfig`: minimal, just common fields

**Examples:**
```yaml
# HistoryAgent
agent:
  type: history_agent
  lm_config:
    model: gemini-2.5-flash
    temperature: 0.7
  memory_config:
    _type: history_list
    max_length: 200
  history_k: 10
  system_prompt: "You are a helpful assistant."

# ReflexionAgent
agent:
  type: reflexion_agent
  lm_config:
    model: gpt-4o-mini
    temperature: 0.2
  memory_config:
    _type: history_list
    max_length: 200
  history_k: 50
  enable_reflection: true
  reflect_on_failure_only: false
  reflection_few_shot_examples: "${file:prompts/reflection_examples.txt}"
```


