## Agents

- `Agent`: base with `act`, `observe`, `reset`, `end_episode`.
- `MemoryAgent`: abstract, task-agnostic; orchestrates LM calls and memory usage via hooks:
  - `build_system_prompt()`
  - `build_user_prompt(obs, history, k)`
  - `create_observation_event(obs)` / `create_action_event(action)` / `create_feedback_event(feedback)`
  Concrete agents implement these to shape prompts and memory entries.
- `HistoryAgent`: specializes to `HistoryList`; formats prompt lines as `<TYPE>: <content>` and writes atomic Observation/Action/Feedback entries.
- `system_prompt` can be overridden in config; console verbosity via `agent.verbose`.


