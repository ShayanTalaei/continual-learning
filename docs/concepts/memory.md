---
noteId: "4bd824209dd011f0b67f67467df34666"
tags: []

---

## Memory

- `MemoryModule` interface; `HistoryList` implementation with max_length.
- `Entry { type: str, content: str }` with atomic Observation/Action/Feedback entries.
- Factory: `build_memory` via `_type` discriminator.


