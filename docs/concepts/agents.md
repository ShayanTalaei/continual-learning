## Agents

- `Agent`: base with `act`, `observe`, `reset`, `end_episode`.
- `MemoryAgent`: task-agnostic; builds prompts from memory; logs Observation/Action/Feedback entries.
- `HistoryAgent`: specializes to HistoryList; generic prompt lines `<TYPE>: <content>`.
- `system_prompt` can be overridden in config; console verbosity via `agent.verbose`.


