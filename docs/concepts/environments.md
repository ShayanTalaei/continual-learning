---
noteId: "54f55cd09dd011f0b67f67467df34666"
tags: []

---

## Environments

Environments encapsulate tasks that agents interact with. All environments follow a Gym-like interface with `reset()` and `step(action)` methods.

### Base Environment Classes

- **`Environment`**: Abstract base with standard Gym API
  - `reset() -> str`: Initialize episode, return initial observation
  - `step(action) -> (obs, feedback, done, info)`: Execute action, return results
  - Fields: `env_id`, `env_type`, `metadata`

- **`EnvDataset`**: Container for loading and organizing multiple environment instances
  - `load_dataset() -> List[Environment]`: Load environments from various sources
  - `get_dataset() -> List[Environment]`: Access loaded environments
  - Supports HuggingFace datasets, local JSONL, and custom loaders

### Single-Turn QA Environments

These environments present a question, receive one action (the answer), and terminate.

- **`QAEnv`**: Base single-turn QA environment
  - Formats question using `instruction_template` (e.g., `"Question: {question}\nAnswer:"`)
  - Evaluates via exact string match (normalized)
  - Feedback schema: `{score: float, target: str, message: str}`
  - Used for general QA tasks

- **`MathQAEnv(QAEnv)`**: Mathematical QA with specialized normalization
  - Numeric answer extraction and comparison with tolerance
  - Boxed answer extraction: `\\boxed{answer}`
  - LaTeX marker stripping
  - Whitespace normalization
  - Tolerance-based comparison for numerical answers
  - Used for AIME, math competition problems

- **`MCQEnv(QAEnv)`**: Multiple-choice question environment
  - Stores choices in metadata
  - Case-insensitive letter matching (A, B, C, D)
  - Text matching against choice content
  - Normalization helpers for both formats
  - Used for MMLU, GPQA benchmarks

- **`FinerEnv(QAEnv)`**: Financial entity recognition (US GAAP tag prediction)
  - Exact string match on normalized GAAP tags
  - Supports boxed answer extraction
  - Whitespace-normalized comparison
  - Used for FinLoRA Finer dataset

- **`OmegaMathEnv(QAEnv)`**: Advanced math environment with multiple evaluation modes
  - **Evaluation modes**: `auto`, `numeric_tol`, `normalized_exact`, `tuple_tol`, `set_tol`, `matrix_tol`, `expr_equiv`
  - **Auto mode**: Infers type from target (matrix → tuple → set → numeric → expression → exact)
  - **Structured answer support**: Tuples, sets, matrices with tolerance-based comparison
  - **Symbolic equivalence**: Uses SymPy for algebraic expression comparison
  - **Optional features**:
    - JSON output schema with rationale + final answer
    - Ground truth solution feedback
    - LLM-generated tutoring feedback
    - Response schema for structured outputs
  - Used for OMEGA Math benchmark variants

### Multi-Step Environments

These environments support multiple turns of interaction before task completion.

- **`ALFWorldEnv`**: Embodied household task environment
  - **Based on**: ALFWorld interactive text-based simulation
  - **Actions**: Text commands (`"go to X"`, `"take Y from Z"`, `"use A B"`)
  - **Special action**: `"think: ..."` for deliberation without environment change
  - **Few-shot prompting**: Injects 2 task-specific examples at episode start
  - **Feedback**: Success/failure with observation text
  - **Max steps**: Configurable cap per episode (default: 50)
  - **Prompt templates**: Task-type-specific examples (put, clean, heat, cool, examine, puttwo)
  - Used for embodied reasoning and planning tasks

### Dataset Loading and Routing

**`QAEnvDataset`**: Flexible dataset loader with automatic environment routing
- **Sources**:
  - HuggingFace datasets: `hf_dataset`, `hf_subset`, `split`
  - Local JSONL: `dataset_path`
  - Disk-saved datasets: `load_from_disk`
- **Field mapping**: `input_field`, `target_field`, `choices_field`, `id_field`, `meta_fields`
- **Routing strategies**:
  1. `task_type`: `exact` → QAEnv, `numeric` → MathQAEnv, `mcq` → MCQEnv
  2. `env_class`: Explicitly specify environment class
  3. `type` field: Routes to registered environment types

**Specialized dataset loaders:**
- `FinerEnvDataset`: FinLoRA Finer data with optional GAAP tag prepending
- `OmegaMathEnvDataset`: OMEGA Math with subset filtering, structured feedback
- `ALFWorldEnvDataset`: ALFWorld episodes with prompt injection

### Configuration Examples

**QA Environment (AIME-style):**
```yaml
train_dataset:
  task_type: numeric
  hf_dataset: "keirp/aime_1983_2024"
  split: "train"
  input_field: "problem"
  target_field: "answer"
  instruction_template: "Solve this math problem. Put your final answer in \\boxed{}.\n\nProblem: {question}"
```

**MCQ Environment (MMLU):**
```yaml
train_dataset:
  task_type: mcq
  hf_dataset: "cais/mmlu"
  hf_subset: "abstract_algebra"
  split: "test"
  input_field: "question"
  target_field: "answer"
  choices_field: "choices"
```

**ALFWorld Multi-Step:**
```yaml
train_dataset:
  type: alfworld
  base_config_path: "third_party/reflexion/alfworld_runs/base_config.yaml"
  split: "eval_out_of_distribution"
  num_episodes: 10
  prompts_path: "src/data/prompts/alfworld/alfworld_3prompts.json"
```

**OMEGA Math with LLM Feedback:**
```yaml
train_dataset:
  type: omega_math
  hf_dataset: "allenai/omega-explorative"
  split: "test"
  eval_mode: "auto"
  output_type: "json"  # Agent outputs {final_answer: str, rationale: str}
  feedback_type: "llm_feedback_from_ground_truth_solution"
  feedback_lm_config:
    model: "gpt-4o-mini"
    temperature: 0.7
```

### Feedback Schema

All environments return feedback as a dictionary with standardized keys:
- `score`: Numeric score (0.0-1.0 for binary correct/incorrect, can be partial credit)
- `target`: Ground truth answer
- `message`: Human-readable feedback message
- `extra` (optional): Additional metadata (error types, parse details, etc.)

**Backward compatibility**: Environments accept legacy `correct`/`is_correct` keys but log deprecation warnings.


