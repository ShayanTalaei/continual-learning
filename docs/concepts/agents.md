## Agents

- `Agent`: base with `act`, `observe`, `reset`, `end_episode`, `train()`, `eval()`, `eval_mode()` context manager, and `clone_for_episode()`.
- `MemoryAgent`: abstract, task-agnostic; orchestrates LM calls and memory usage via hooks:
  - `build_system_prompt()`
  - `build_user_prompt(obs, history, k)`
  - `create_observation_event(obs)` / `create_action_event(action)` / `create_feedback_event(feedback)`
  Concrete agents implement these to shape prompts and memory entries.
- `MemorylessAgent`: no persistent memory; prompts only with the current observation; supports parallel episode execution during validation.
- `HistoryAgent`: specializes to `HistoryList`; formats prompt lines as `<TYPE>: <content>` and writes atomic Observation/Action/Feedback entries.

### Agent Modes
- **Training mode**: Memory updates are enabled, agent learns from experience.
- **Evaluation mode**: Memory updates are disabled, agent performs inference only.
- **Context managers**: Use `agent.eval_mode()` to temporarily switch to evaluation mode.

### Agent Cloning
- `clone_for_episode(training, share_memory)`: Creates isolated agent instances for parallel execution.
- Clones share the LM client and optionally share memory modules.
- Each clone maintains independent short-term state (trajectory, last action).

### Configuration
- `system_prompt` can be overridden in config; console verbosity via `agent.verbose`.
- Common fields (`lm_config`, `system_prompt`, `verbose`) are inherited from base `AgentConfig`.


