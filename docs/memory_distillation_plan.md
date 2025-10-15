### Memory Distillation to KV Prefix (Cartridge): Architecture and Implementation Plan

This document specifies a KV-only memory distillation system for continual-learning agents. We generate teacher data (text + top-k logits) from historical trajectories, train a compact KV prefix (“cartridge”) offline, and serve it at inference through Tokasaurus cartridge endpoints. Integration is modular and agent-friendly.

References:
- Tokasaurus Cartridges (serving API, sources: local/HF/W&B): https://github.com/ScalingIntelligence/tokasaurus/tree/geoff/cartridges?tab=readme-ov-file#cartridges
- Cartridges (training code, datasets, Flex models): https://github.com/HazyResearch/cartridges

---

## High-Level Overview

1) Data generation (teacher): Build a dataset from `HistoryList` memory using configurable strategies. For each sample, reconstruct the HistoryAgent-style prompt and call a text LM (Tokasaurus chat) to collect assistant completions and optional top-k token logits.

2) Training (student): Use the Cartridges training loop to optimize a `TrainableCache` (prefix KV) on the generated dataset while freezing the base model. This happens entirely in PyTorch (no Tokasaurus involved) and supports AMP/DDP.

3) Serving and evaluation (student): Package the trained KV as a cartridge artifact (local/HF/W&B). `KVMemoryAgent` uses `TokasaurusClient` to call cartridge-enabled endpoints for inference, paying near-zero token cost for the long memory.

Design goals:
- KV-only inference (no textual fallback)
- Clean separation of concerns: agents for inference, Cartridges for training, Tokasaurus for serving
- Strategy-driven data generation to control leakage and emphasis
- Minimal invasive changes to existing agent abstractions

---

## Architecture

Core components we add:
- `src/memory/distillation/` (data-gen pipeline)
  - Strategies to form sample-specific memory views from a `HistoryList`
  - Replay dataset to rebuild agent prompts without executing real envs
  - Orchestrator to call Tokasaurus for teacher outputs and save a Cartridges-compatible dataset
- `KVCacheMemory` module for cartridge descriptors (id + source)
- `KVMemoryAgent` that uses cartridge memory at inference
- `TokasaurusClient` extensions for cartridge endpoints and top-k logprobs capture
- Thin adapter that maps our train config to Cartridges’ `TrainConfig` and invokes training in-process

---

## Data Generation Pipeline (Teacher)

Location: `src/memory/distillation/`

- History loader
  - Discover latest `memory_*.jsonl` under a checkpoint dir
  - Load via existing `HistoryList.load_snapshot()` and parse `Entry(type, content)`

- Strategies (`src/memory/distillation/strategies/`)
  - `MemoryFormationStrategy` interface:
    - `iter_samples(full_history) -> Iterable[SampleSpec]`
    - `build_memory_for_sample(full_history, spec) -> list[Entry]`
  - Implementations:
    - `FullMemoryStrategy`: use entire history
    - `ExcludeCurrentStrategy`: exclude current (obs, action, feedback)
    - `RollingWindowStrategy`: last K entries (type-filtering optional)
    - `FailureFocusStrategy`: emphasize windows around negative feedback
  - Composable and configurable via Pydantic configs

- Prompt builder
  - Reproduce `HistoryAgent` prompt shape: system instructions + “previous experiences” + current obs
  - Output OpenAI-style `messages` for Tokasaurus chat endpoints

- Tokasaurus client call (teacher output)
  - Use vanilla chat endpoint (no cartridges) to avoid leakage from student memory
  - Add `top_logprobs` support to receive per-token distributions from server
  - Collect: assistant text, `token_ids`, `topk_token_ids`, `topk_logprobs`, `topk_token_idxs`, and usage metadata

- Dataset writer
  - Save each sample as a Conversation-like row:
    - `messages`: system/user/assistant
    - Assistant can include tokenization/logprob traces if requested
    - `metadata`: strategy, indices, memory sizes, etc.
  - Write parquet or jsonl, then optionally push to HF Hub (dataset)

---

## Training the Cartridge (Student)

Approach: use Cartridges training loop, invoked as a library. We retain all perf-critical details (FlexAttention, packing, AMP/DDP) and minimize maintenance risk.

- Adapter
  - `DistillTrainConfig` (our Pydantic) → Cartridges `TrainConfig`
  - Data source points to generated parquet/jsonl (`DataSource(type="local", path=...)`)
  - `kv_cache_initializer`: `KVFromRandomVectors`, `KVFromText` (optional), or `KVFromPretrained`
  - Loss target can consume our top-k logits for distillation (logit-based objective already supported by Cartridges datasets)

- Execution
  - Call `cartridges.train.train(config)` in-process
  - Save `cache-step*.pt` and `cache_last.pt` (trainable KV)

- Registration (to serve with Tokasaurus)
  - `source="local"`: materialize under server `./cartridges/<cartridge_id>/`
  - `source="huggingface"`: upload artifact to HF repo
  - `source="wandb"`: point to W&B artifact id/run
  - Record `{ id, source, force_redownload? }` as the cartridge descriptor

---

## Serving and Agent Integration (Student)

- `KVCacheMemory`
  - Config: `artifact: { id: str, source: Literal["wandb","huggingface","local"], force_redownload?: bool }`
  - `recall()` returns `{ "type": "kv", "cartridges": [artifact] }`

- `KVMemoryAgent`
  - Builds minimal prompts (short system/user) and attaches cartridge via the LM client
  - No writes into KV memory (cartridge is static between distills)

- `TokasaurusClient` extension
  - If `cartridges` param present → POST to `/v1/cartridge/chat/completions`
  - Payload includes OpenAI-style `messages` and a `cartridges` list per docs
  - Also supports `top_logprobs` for analysis/diagnostics when needed

---

## Configuration Surface (New/Updated)

- Data generation (`src/memory/distillation/config.py`)
  - `checkpoint_dir: str`
  - `strategy: { name, window_k?, exclude_current?, failure_focus? }`
  - `lm_config` (Tokasaurus chat, temperature, max_tokens)
  - `top_logprobs: int | None` (collect logits)
  - `system_prompt: str | None` (override)
  - `output_path: str`, `hf_repo_id: str | None`

- Training adapter
  - Model id, initializer type/args, batch sizes, LR, epochs, eval cadences
  - Dataset path from data-gen output

- Agent/runtime
  - `agent.type: kv_memory_agent | hybrid_kv_history_agent`
  - `kv_memory.memory_config.artifact: { id, source, force_redownload? }`
  - Distillation schedule (when to re-run data-gen + training and swap cartridge)

---

## Step-by-Step Implementation Plan

Phase 0 — Scaffolding and client
1. Add `src/memory/distillation/` with submodules: `strategies/`, `history_loader.py`, `prompt_builder.py`, `replay_env.py`, `data_generation.py`, `exporters/hf_uploader.py`, `config.py`.
2. Extend `src/lm/tokasaurus_client.py`:
   - `call(system, user, cartridges: list[dict] | None = None, top_logprobs: int | None = None)`
   - `call_with_trace(...) -> (text, trace)` that returns token/logprob details

Phase 1 — Data generation
3. Implement strategies and prompt builder mirroring `HistoryAgent` formatting.
4. Implement replay dataset to iterate `SampleSpec` and build `messages`.
5. Implement data-gen orchestrator: loop samples → call Tokasaurus → save rows (messages + optional logits) → optional HF upload.

Phase 2 — Cartridge training integration
6. Implement a thin adapter to construct Cartridges `TrainConfig` from our `DistillTrainConfig` and call `cartridges.train.train`.
7. Add a small helper to register the resulting cache as a cartridge (local/HF/W&B) and produce the `{ id, source }` descriptor.

Phase 3 — Agent integration and evaluation
8. Implement `KVCacheMemory` and wire it in `memory_factory`.
9. Implement `KVMemoryAgent` and (optionally) `HybridKVHistoryAgent`.
10. Add examples/configs to run eval episodes with cartridges via Tokasaurus.

Phase 4 — Quality and automation
11. Add distillation scheduler (episode-count/size/plateau triggers) and a `tools/distill_memory.py` to run the cycle end-to-end.
12. Benchmarks: compare KV inference vs long-context prompts on held-out tasks.

---

## Actionable Roadmap (Detailed)

1) Extend Tokasaurus client (serving + teacher logits)
   - Files: `src/lm/tokasaurus_client.py`, `src/lm/lm_factory.py`
   - Add methods:
     - `call(system, user, cartridges: list[dict] | None = None, top_logprobs: int | None = None) -> str`
     - `call_with_trace(system, user, cartridges: list[dict] | None = None, top_logprobs: int | None = None) -> tuple[str, dict]`
   - Behavior:
     - If `cartridges` set → POST `/v1/cartridge/chat/completions` with `messages` and `cartridges` (sources: local/HF/W&B)
     - Else → POST `/v1/chat/completions`
     - If `top_logprobs` set → request server-side logprobs and return token/logprob traces (flattened to arrays matching Cartridges expectations)
   - Validation: small smoke tests against a local Tokasaurus instance and the cartridges endpoints per docs

2) Create `KVMemoryAgent` (KV-only inference)
   - Files: `src/agent/kv_memory_agent.py`, `src/memory/kv_cache.py`, `src/memory/memory_factory.py`, `src/agent/registry.py`
   - Add `KVCacheMemoryConfig` with `artifact: { id, source, force_redownload? }`
   - `KVCacheMemory.recall()` returns `{ "type": "kv", "cartridges": [artifact] }`
   - `KVMemoryAgent`:
     - `build_system_prompt()`: minimal task instruction
     - `build_user_prompt()`: no long history; only current observation
     - `act()`: passes `cartridges` from memory recall to `TokasaurusClient`
   - Validation: run inference with a known cartridge (can use a tiny/random cartridge for a dry run)

3) Implement data generation for distillation
   - Files: `src/memory/distillation/` (new folder)
     - `config.py`: Pydantic configs for strategies and run
     - `history_loader.py`: checkpoint discovery and `HistoryList` loading
     - `strategies/`: `base.py`, `full_memory.py`, `exclude_current.py`, `rolling_window.py`, `failure_focus.py`
     - `prompt_builder.py`: build `messages` matching `HistoryAgent` format
     - `replay_env.py`: iterate samples and build prompts without real env
     - `data_generation.py`: orchestrator to call Tokasaurus and write parquet/jsonl
     - `exporters/hf_uploader.py`: optional HF upload
   - Output schema per row:
     - `messages`: [{role:"system",content}, {role:"user",content}, {role:"assistant",content, token_ids?, topk_token_ids?, topk_logprobs?, topk_token_idxs?}]
     - `metadata`: { strategy, window_k, exclude_current, sample_idx, memory_len, ... }
   - Validation: generate a small dataset from a known history checkpoint

4) Hook training via Cartridges
   - Files: `tools/distill_memory.py` (runner), `src/memory/distillation/train_adapter.py`
   - `train_adapter.py` maps our config → Cartridges `TrainConfig` and calls `cartridges.train.train`
   - Provide example configs (model id, batch sizes, lr, epochs)
   - Save `cache_last.pt` and register the cartridge:
     - local: place under server `./cartridges/<id>/`
     - HF: push to repo
     - W&B: record artifact
   - Validation: run a short training job and verify `cache_last.pt`

5) Evaluate with `KVMemoryAgent`
   - Update an experiment config to use `KVMemoryAgent` + `KVCacheMemory` with the new artifact
   - Run eval episodes; compare with HistoryAgent long-context baseline on held-out tasks
   - Metrics: accuracy/score, tokens used, latency

6) Operationalization and polish
   - Distillation schedule triggers (episode count, token budget, plateau)
   - `tools/register_cartridge.py` helper to standardize artifact registration
   - Docs and examples: minimal end-to-end tutorial (generate → train → serve → eval)


---

## Design Choices and Tradeoffs

- Keep training in the Cartridges stack (recommended):
  - Pros: mature FlexAttention integration, stable performance, less maintenance; produces artifacts Tokasaurus can serve directly.
  - Cons: less granular control over bespoke losses/schedules; bridging configs.
- Reimplement trainer in our tree (defer): high effort and risk (KV packing, masks, AMP/DDP, model wrappers). Consider only if we need deeply customized online/continual updates.

---

## Risks and Mitigations

- KV inference support: ensure Tokasaurus server is on a cartridges-capable branch and configured properly (see README).
- Data leakage: use `ExcludeCurrentStrategy` to avoid memorizing the exact step used for supervision.
- Privacy: add optional redaction filters over memory entries during export.
- Drift: schedule periodic re-distillation; support multiple cartridges by domain.

---

## Minimal Interfaces (Sketches)

Strategy and sample specs:
```python
class SampleSpec(BaseModel):
    observation: str
    target_action: str | None
    feedback: dict | None
    memory_view: dict
    meta: dict

class MemoryFormationStrategy:
    def iter_samples(self, full_history: list[Entry]) -> Iterable[SampleSpec]: ...
    def build_memory_for_sample(self, full_history: list[Entry], spec: SampleSpec) -> list[Entry]: ...
```

Tokasaurus client additions:
```python
def call(self, system_prompt: str, user_prompt: str,
         cartridges: list[dict] | None = None,
         top_logprobs: int | None = None) -> str: ...

def call_with_trace(self, system_prompt: str, user_prompt: str,
                    cartridges: list[dict] | None = None,
                    top_logprobs: int | None = None) -> tuple[str, dict]: ...
```

KV memory recall:
```python
{"type": "kv", "cartridges": [{"id": "...", "source": "local|huggingface|wandb"}]}
```

---

## Acceptance Criteria

- Dataset generation writes Cartridges-compatible rows with optional top-k logprobs.
- Cartridges training runs in-process and produces `cache_last.pt`.
- Tokasaurus serves the trained cartridge; `KVMemoryAgent` achieves at least parity with long-context prompting at lower token costs.
- Distillation cycles can be scheduled and swapped without agent code changes.

### Memory Distillation to KV Prefix: Design and Roadmap

This document proposes a clean, modular design to distill the `HistoryList` memory into a compact KV-prefix for efficient inference, leveraging the `third_party/cartridges` project for cache (prefix) training and generation. It also describes integration into our `continual-learning` agents, data pipelines, and configs.

---

## Motivation

- **Problem**: `HistoryList` grows without bound; prompts become long, expensive, and less effective.
- **Goal**: Periodically distill long histories into a fixed-size KV-prefix that the model can attend to at near-zero token cost.
- **Approach**: Train a learnable KV cache (prefix) using our interaction traces as supervision (self-study/teacher-forcing style), then use that cache during inference. Continue appending new experiences to `HistoryList`, and re-distill when needed.

---

## Key References in Repos

- Continual-learning memory and agents
  - `src/memory/memory_module.py` (base): snapshot/eval/training semantics
  - `src/memory/history_list.py` (simple append-only memory)
  - `src/agent/memory_agent.py` (LM + Memory orchestration)
  - `src/agent/history_agent.py` (prompting over `HistoryList`)
- Cartridges cache and training
  - `third_party/cartridges/cartridges/cache.py` (TrainableCache, KVCacheFactory)
  - `third_party/cartridges/cartridges/train.py` (training loop; `CacheAndModel` wrapper)
  - `third_party/cartridges/cartridges/datasets.py` (conversation-to-tokens, batching)
  - `third_party/cartridges/cartridges/initialization/{text,random,pretrained}.py` (cache init)
  - `third_party/cartridges/cartridges/generation.py` (flex_generate with cache)

---

## High-Level Architecture

- **Two memory types**:
  - `HistoryList` (existing): append experiences and feedback; canonical log of agent learning.
  - `KVCacheMemory` (new): holds the learned non-text KV-prefix, used at inference time.

- **Three agents**:
  - `HistoryAgent` (existing): baseline, prompts with recent `HistoryList` entries only.
  - `KVMemoryAgent` (new): prompts using KV-prefix as primary memory; minimal or no `HistoryList` in prompt.
  - `HybridKVHistoryAgent` (new): uses KV-prefix for inference and still writes new experiences to `HistoryList`; periodically triggers distillation.

- **Offline trainer** (new): builds training examples from `HistoryList` traces and runs Cartridges training to produce a cache checkpoint. Plugs back the resulting cache into `KVCacheMemory`.

---

## Data Generation (Self-Study over History)

We adapt Cartridges’ self-study idea to our traces:

- Source: Episodes recorded by `HistoryAgent` (`Entry(type, content)` for Observation, Action, Feedback).
- Convert to Cartridges `Conversation` items where the goal is to predict high-quality actions given prior experiences.
  - Minimal path: Teacher-forcing on the best-known actions/answers (from env feedback).
  - Optional: Construct multi-turn contexts with several prior experience snippets, then the current “question”, target answer is the correct or improved action.
  - No need for top-logprobs; Cartridges supports retokenizing targets.

Proposed converter API:

- `history_export.conversations_from_history(history: list[Entry], policy: ExportPolicy) -> list[Conversation]`
  - Policies: windowed (last K per example), curriculum (earliest→latest), failure-focused (examples with negative feedback emphasized), mixed.
  - Optionally add a system message summarizing task rules we already surfaced in prompts.

Dataset assembly for Cartridges:

- Write conversations to `.jsonl` or `.pkl` that `cartridges/datasets.py` can read via `DataSource`.
- Provide a small generation-eval split from recent held-out episodes for on-the-fly validation with `evaluate_generations`.

---

## KV Prefix Training (Cartridges)

- Initialization options:
  - Text-derived prefix via `KVFromText` using a distilled textual summary of the `HistoryList` (e.g., reflections or a rulesheet)
  - Random vectors via `KVFromRandomVectors` for ablation
  - Warm-start from a previous cache via `KVFromPretrained`

- Training loop: use `cartridges/train.py` with `CacheAndModel` when `tuning_method == "custom_prefix"`.
  - Freeze model weights; optimize only `TrainableCache` params
  - Loss over targets (logits or tokens), packed sequences for efficient flex attention
  - Periodic eval and generation for sanity checks

- Outputs:
  - Cache checkpoints (`cache-step*.pt`) and `cache_last.pt`
  - Metadata with tokenizer/model identifiers

---

## Inference Integration (KV-only)

We only support non-text KV prefixes for inference:

- KV-cache inference (local or server-backed):
  - Model must support `past_key_values` with a `TrainableCache`-compatible ABI (Cartridges Flex models)
  - Two integration paths:
    1. Local client path: load HF model in-process and call with `past_key_values=cache` (aligns with `cartridges/generation.flex_generate`)
    2. Server path via Tokasaurus Cartridges API: use cartridge-enabled chat/completions endpoints with a `cartridges` field. See Tokasaurus cartridges docs for details ([Tokasaurus cartridges README](https://github.com/ScalingIntelligence/tokasaurus/tree/geoff/cartridges?tab=readme-ov-file#cartridges)).

---

## New Components and Interfaces

### Memory: KVCacheMemory

- `KVCacheMemoryConfig(MemoryModuleConfig)`
  - `_type = "kv_cache"`
  - `artifact: { id: str, source: Literal["wandb","huggingface","local"], force_redownload?: bool }`
  - For `source == "local"`, we maintain the absolute path to a directory the Tokasaurus server will mount or watch (see Serving section below)
  - `tokenizer_name: str` and `model_name: str` (for validation only)
  - `max_length: int | None` (optional: bounded runtime appends if we decide to allow KV growth)

- `KVCacheMemory(MemoryModule)`
  - `recall()` returns a structured handle:
    - `{ "type": "kv", "cartridges": [ { id, source, force_redownload? } ] }`
  - Snapshots: store a small manifest with tokenizer/model ids and the cartridge descriptor. For `local` source, also persist/copy the cache files under the agreed server directory convention.

### Agents

- `KVMemoryAgent(MemoryAgent)`
  - `build_system_prompt()` includes minimal instructions; user prompt omits long history
  - At `act()`, calls LM client with KV handle (KV-only), passing a `cartridges` list
  - `observe()` remains identical to base; does not write to KV memory (KV is static between distillations)

- `HybridKVHistoryAgent(MemoryAgent)`
  - Uses KV for inference while continuing to append to `HistoryList`
  - Periodically triggers distillation when thresholds fire
    - token budget exceeded; N episodes; elapsed time; or score plateau
  - Trigger spawns an offline distillation job; when complete, swaps `KVCacheMemory.cache_path` to the new checkpoint

### Distillation Orchestrator

- `tools/distill_memory.py` (runner script)
  - Export history → build Cartridges dataset
  - Prepare `TrainConfig` (model id, batch sizes, cache initializer)
  - Launch training (single-GPU first)
  - On success: register the trained cache as a Cartridge artifact:
    - Option A (local): materialize under the Tokasaurus server `./cartridges/<cartridge_id>/` with expected file layout; set memory artifact `{ id: <cartridge_id>, source: "local" }`
    - Option B (HuggingFace): upload to HF repo and set `{ id: <hf_repo_id>, source: "huggingface" }`
    - Option C (W&B): rely on W&B artifact id and set `{ id: <wandb_run_or_artifact>, source: "wandb" }`
  - Update `KVCacheMemory` to reference the new artifact descriptor

---

## Prompting Details

- History-based prompts (for dataset construction):
  - Each example includes a compact set of prior experiences formatted similarly to `HistoryAgent`’s prompt, but trimmed and standardized
  - System prompt template can match our current one to reduce domain shift
  - Weight failure cases more (more learning signal)

- Inference prompts:
  - KV mode: very short system prompt with role and safety; context is entirely in the KV

---

## Training Objective and Curriculum

- Primary: Next-action prediction under the distilled context (teacher-forced actions considered “gold”): minimizes cross-entropy of assistant tokens
- Optional augmentations:
  - Negative sampling: pair incorrect action with feedback; target is corrected action
  - Rule extraction pre-stage: generate rules/constraints from history; use `KVFromText` to seed cache with the textual rules, then fine-tune
  - Mixed window sizes, emphasis on hard examples (low-score/long-tail states)

---

## Integration with LM Clients

- Clients:
  - `GeminiClient`: not compatible with KV; do not use for KV memory inference.
  - `TokasaurusClient`: add cartridge-aware calls using cartridge endpoints (no server changes needed):
    - Keep existing `openai` path for vanilla chat completions at `/v1/chat/completions`.
    - Add cartridge chat completions at `/v1/cartridge/chat/completions` when a `cartridges` param is provided.
    - New signature:
      - `call(system_prompt: str, user_prompt: str, cartridges: list[dict] | None = None) -> str`
      - If `cartridges` is not None, payload example:
        ```json
        {
          "model": "default",
          "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
          ],
          "max_tokens": 512,
          "temperature": 0.0,
          "cartridges": [
            {"id": "my_local_id", "source": "local", "force_redownload": false}
          ]
        }
        ```
    - Sources supported: `"wandb"`, `"huggingface"`, `"local"` (see docs: [Tokasaurus cartridges README](https://github.com/ScalingIntelligence/tokasaurus/tree/geoff/cartridges?tab=readme-ov-file#cartridges))
  - `LocalFlexHFClient` (new): full KV support using Cartridges Flex models in-process.
    - Loads HF model; accepts a local `cache_path` for `TrainableCache.from_pretrained`.
    - API: `call(system: str, user: str, cache_path: str | None = None) -> str`

- `get_lm_client()`:
  - Route `model` values like `"flex:meta-llama/Llama-3.2-3B-Instruct"` to `LocalFlexHFClient`.
  - For Tokasaurus with cartridges, keep `model: "toka:<model_id>"` and pass `cartridges` when available.

---

## Config and Orchestration

- New YAML knobs:
  - Agent selection: `agent.type: hybrid_kv_history_agent | kv_memory_agent`
  - KV memory config with `artifact: { id, source, force_redownload? }`
  - Distillation schedule:
    - `distill.when: { max_history_tokens: 20000, every_episodes: 500, min_improvement: 0.01 }`
    - `distill.model: meta-llama/Llama-3.2-3B-Instruct`
    - `distill.cache_init: { type: text|random|pretrained, args: ... }`  # init source only; not used at inference
    - `distill.output_dir`

- Runtime behavior (Hybrid agent):
  - Online: act/observe like `HistoryAgent`, but act uses KV prefix only via `cartridges`
  - Background: when trigger, run `tools/distill_memory.py` (or a modal job) with a snapshot of history
  - Post-train: update `KVCacheMemory.artifact` to the new cartridge id/source

---

## Risks, Constraints, and Mitigations

- **KV inference availability**: Provider chat APIs don’t accept `past_key_values`; use local Flex client or a Tokasaurus server with KV support.
- **GPU and latency**: Cache training requires a GPU; keep models small for first runs; use short packed lengths.
- **Drift**: If task distribution shifts, cache can degrade; schedule periodic refresh and/or maintain multiple domain-specific caches.
- **Safety**: Ensure the distilled prefix doesn’t memorize sensitive data from history; include a redaction step in exporter.

---

## Implementation Roadmap (Phased)

Phase 0 — Export + Offline Train (KV-only)
- Build `history_export` to produce Cartridges `Conversation`s from `HistoryList` dumps
- Run `cartridges/train.py` (tuning method: custom_prefix) → produce `cache_last.pt`
- Add `KVCacheMemory` that returns `{ type: "kv", cache_path }` from `recall()`

Phase 1 — Hybrid Agent + Scheduling
- Implement `HybridKVHistoryAgent` that uses `KVCacheMemory` at inference and appends to `HistoryList`
- Add distillation triggers and runner script; sync outputs to a durable path and update `KVCacheMemory`

Phase 2 — KV Inference Clients
- Implement `LocalFlexHFClient` that loads the HF model and applies `TrainableCache`
- Extend `TokasaurusClient` to use cartridge-enabled endpoints and pass `cartridges` (local/HF/W&B)
- Add configs and small benchmarks to validate speed/quality vs long-context baselines

Phase 3 — Quality Improvements
- Curriculum, failure-focused sampling, ablations on init, cache size sweeps, multi-domain caches
- Optional: integrate slice dashboards from `third_party/cartridges/viz`

---

## Minimal Interfaces and File Touches (Proposed)

- Memory
  - `src/memory/kv_cache.py`: `KVCacheMemory`, `KVCacheMemoryConfig`
  - `src/memory/memory_factory.py`: wire `_type == "kv_cache"`

- Agents
  - `src/agent/kv_memory_agent.py`: `KVMemoryAgent`
  - `src/agent/hybrid_kv_history_agent.py`: `HybridKVHistoryAgent`
  - `src/agent/registry.py`: register new agents

- LM
  - `src/lm/flex_hf_client.py`: local HF client with KV support
  - `src/lm/lm_factory.py`: route `model` like `"flex:<hf_id>"`

- Distillation
  - `tools/distill_memory.py`: end-to-end export + train + register
  - `src/data/export/history_export.py`: history→conversations
  - `configs/distill/*.yaml`: training recipes
  - Optionally, `tools/register_cartridge.py`: helper to push to HF/W&B or place under server `./cartridges/<id>/`

No breaking changes to existing `HistoryAgent` flows; the KV path is opt-in.

---

## Acceptance Criteria and Checks

- KV inference:
  - Measurable speedups vs long-context prompts with equal or better quality on held-out tasks
  - Cache swapping at runtime works without crashes; memory snapshots persist and reload

- Reproducibility:
  - Distillation runner saves datasets, configs, and cache checkpoints under a run directory
  - Agents save/load memory snapshots consistently (`save_snapshot`/`load_snapshot`)

---

## Open Questions

- How aggressively should we prune/summarize `HistoryList` once KV is strong? Keep a rolling buffer for new learning vs preserving full logs?
- Should we learn multiple specialized caches (by task/domain) and route at inference?
- How to extend Tokasaurus (or deploy vLLM/sglang) to accept and reuse a precomputed cache across turns for low-latency serving?

---

## Appendix: Code Touch Points (References)

Existing code worth consulting during implementation:

```1:44:src/agent/history_agent.py
class HistoryAgent(MemoryAgent):
    def build_user_prompt(self, obs: str, history: List[Any], k: int | None) -> str:
        lines: List[str] = []
        lines.append("Here is a list of your previous experiences:")
        recent: List[Entry] = history[-k:] if k is not None else history
        # ... build prompt with history entries ...
```

```1:58:third_party/cartridges/cartridges/cache.py
class TrainableCache(nn.Module):
    def update(self, new_keys, new_values, new_seq_ids, layer_idx, skip_append=False):
        # ... appends and returns packed K/V including trainable and frozen tokens ...
```

```120:179:third_party/cartridges/cartridges/train.py
class CacheAndModel(nn.Module):
    def forward(self, input_ids, seq_ids, position_ids):
        out = self.model(
            input_ids=input_ids,
            seq_ids=seq_ids,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=self.cache,
        )
        return out
```


