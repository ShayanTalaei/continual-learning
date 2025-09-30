---
noteId: "6d8868a09dd011f0b67f67467df34666"
tags: []

---

## Synthetic Data

- Generate encryption dataset:
```bash
python -m src.data.synthetic_task_generators.encryption --num_samples 1000 --string_length 12 --save_path data/synthetic/encryption/train.jsonl
```
- Load via config using `dataset_path: data/synthetic/encryption/train.jsonl`.


