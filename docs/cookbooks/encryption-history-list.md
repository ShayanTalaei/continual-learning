---
noteId: "71cbdeb09dd011f0b67f67467df34666"
tags: []

---

## Encryption + HistoryList (Cookbook)

1) Generate:
```bash
python -m src.data.synthetic_task_generators.encryption --num_samples 100 --string_length 8 --save_path data/synthetic/encryption/train_100.jsonl
```

2) Configure `configs/encryption_history_list.yaml` with `dataset.dataset_path` pointing to the jsonl.

3) Run:
```bash
python -m src.main --config configs/encryption_history_list.yaml
```

4) Inspect outputs in `outputs/synthetic/encryption/`.


