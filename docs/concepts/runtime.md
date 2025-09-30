---
noteId: "3c3db2009dd011f0b67f67467df34666"
tags: []

---

## Runtime

- Orchestrates env episodes using Gym-like `reset/step`.
- Progress bar via tqdm; logs to file; console verbosity via config.
- Returns metrics and per-step logs.

Key config: `max_envs_to_visit`, `max_steps_per_episode`, `verbose`.


