---
noteId: "2246d0709dd011f0b67f67467df34666"
tags: []

---

## Quickstart

1) Generate (optional):
```bash
python -m src.data.synthetic_task_generators.encryption --num_samples 100 --string_length 8 --save_path data/synthetic/encryption/train_100.jsonl
```

2) Run a config:
```bash
python -m src.main --config configs/toy_inmemory.yaml
```

3) Results:
- Metrics: `outputs/toy/metrics.json`
- Memory: `outputs/toy/memory.jsonl`
- Logs: `outputs/toy/run.log`


