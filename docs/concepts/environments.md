---
noteId: "54f55cd09dd011f0b67f67467df34666"
tags: []

---

## Environments

- `QAEnv`: base single-turn; `instruction_template` formats question.
- `MathQAEnv`: numeric/boxed normalization with tolerance.
- `MCQEnv`: letter/text matching; choices in metadata.
- `QAEnvDataset`: loads from HF datasets or .jsonl; routes via `task_type`/`env_class`.


