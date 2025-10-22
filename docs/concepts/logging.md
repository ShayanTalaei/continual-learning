---
noteId: "604c31d09dd011f0b67f67467df34666"
tags: []

---

## Logging

- File logs always enabled via `setup_logger("run", level, results_dir/run.log)`.
- Console logs added per component with `add_console` when `*.verbose` is true.
- Use `child(root, "runtime"|"agent"|"dataset")`; components do not reconfigure handlers.


