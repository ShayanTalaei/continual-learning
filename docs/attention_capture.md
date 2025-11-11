<!-- Detailed documentation for the attention capture pipeline -->

# Attention Capture Pipeline

This document explains the additions made to support extracting attention statistics from cartridges models. It walks through the control flow, data dependencies, and provides a step-by-step recipe for running the new tooling.

## Overview

The pipeline adds three major components:

1. **Cache enhancements** (`third_party/cartridges/cartridges/cache.py`)  
   Track per-layer sequence ids for the KV cache and optionally expose them when attention computations request diagnostic data.

2. **Attention capture utility** (`third_party/cartridges/cartridges/models/attention_capture.py`)  
   A light-weight recorder object that stores queries, keys, values, and context metadata on demand.

3. **Forward-pass integration** (`third_party/cartridges/cartridges/models/**/modeling_*.py`)  
   Hooks wired into flex attention so that, when a capture object is provided, each decoder layer reports what the model attended to.

4. **CLI modules** (`src.attention_capture.check_cli`, `src.attention_capture.run_eval_cli`)  
   Python entrypoints (invoked via `python -m ...`) that drive single-conversation inspection as well as validation replays with checkpointed agents, persisting summaries, tensors, and plots.

5. **Reusable helpers** (`src/attention_capture/`)  
   Shared tokenisation, metrics, plotting, and prompt-reconstruction utilities used by both CLIs and available to notebooks.

The figure below summarises the data flow.

```
Conversation(s) ──▶ Tokeniser ──▶ Packed tensors (input_ids, seq_ids, position_ids)
        │                                     │
        │                         (Optionally)│
        │                                     ▼
        │                      TrainableCache initialised
        │                                     │
        ▼                                     ▼
  AttentionCapture ──▶ Flex*Model.forward ──▶ Attention layers ──▶ flex_attention_forward
         ▲                                        │                        │
         │                                        └───── captures q/k/v, seq metadata
         │
         └───── src.attention_capture.check_cli summarises and saves outputs
```

---

## Module-by-module details

### 1. Trainable Cache updates (`cartridges/cache.py`)

- **State tracking**  
  Each layer now records the sequence ids associated with appended tokens (`self._layer_seq_ids`). This mirrors the packed KV tensors so we can align attention matrices with the original tokens.

- **Extended API**  
  `TrainableCache.update(..., return_seq_ids=True)` returns a third tensor containing the concatenated sequence ids in the same order as the returned keys/values. The flag defaults to `False`, preserving the existing interface until diagnostics are requested.

- **Cache clearing**  
  `clear()` now resets the per-layer sequence ids alongside the tensors, ensuring repeated calls start from a clean state.

### 2. Attention capture utility (`cartridges/models/attention_capture.py`)

- `AttentionRecord` dataclass stores:
  - layer index and mode (train/generate)
  - optional Q/K/V tensors (squeezed batch dimension, moved to CPU for convenience)
  - query/kv sequence ids and cache size
  - scaling factor, GQA flag, and mask diagnostics

- `AttentionCapture` exposes a single method `record(...)`. During a forward pass, attention modules pass tensors into this recorder. The recorder stores references (detached) so downstream analysis can operate without autograd interference.

- Configuration knobs:
  - `store_qkv`: disable to reduce memory usage if only metadata is required.
  - `move_to_cpu`: keeps tensors off GPU once captured.

### 3. Model integration (`cartridges/models/attention.py`, `modeling_llama.py`, `modeling_qwen3.py`)

1. **Flex attention hook**  
   `flex_attention_forward` now accepts extra parameters (`attention_capture`, `layer_idx`, `seq_ids`, `kv_seq_ids`, `cache_len`). After computing the attention output, it conditionally calls `attention_capture.record(...)`.

2. **Batch dataclasses**  
   `LlamaBatch` and `Qwen3Batch` gained an optional `attention_capture` field, allowing the capture object to flow through the decoder stack.

3. **Attention modules** (`LlamaAttention`, `Qwen3Attention`)
   - When a cache is present and capture is requested, they call `TrainableCache.update(..., return_seq_ids=True)` to obtain aligned key/value tensors and the kv sequence ids.
   - Without capture, existing behaviour is untouched.
   - The modules pass all relevant metadata into `flex_attention_forward`.

4. **Model forward** (`FlexLlamaModel`, `FlexQwen3Model` and their Causal LM wrappers)
   - Each forward method accepts `attention_capture` and plumbs it into the batch.
   - The high-level CausalLM wrappers expose the same parameter, enabling external callers (e.g. the CLI) to activate capture.

### 4. Analysis scripts

Both CLIs share the refactored helpers under `src/attention_capture/` but target distinct workflows:

1. **`src.attention_capture.check_cli`**  
   - Accepts ad-hoc conversations (JSONL, text, literal prompt).  
   - Captures per-layer tensor data and cartridge vs normal attention mass.  
   - Saves JSON summaries and, when requested, `.pt` blobs containing weights/QKV.

2. **`src.attention_capture.run_eval_cli`**  
   - Loads a `history_agent` configuration, restores memory from a snapshot, and rebuilds validation prompts without contacting the serving LM.  
   - Executes the cartridges model with attention capture enabled across the selected validation subset.  
   - Aggregates attention by tagged spans (system prompt, individual memories, current observation, cartridge tokens) and renders matplotlib heatmaps plus machine-readable matrices for downstream analysis.

---

## Step-by-step usage guide

1. **Install dependencies**  
   Ensure the cartridges third-party package and its requirements are installed in your environment (follow the repo’s `INSTALL_CARTRIDGES.md` if needed).

2. **Prepare inputs**  
   - For realistic runs: export conversations in JSONL where each entry contains a `messages` array (see `cartridges.structs.Conversation` for reference).  
   - For quick tests: supply a plain text prompt via `--prompt "My question"`.

3. **(Optional) Obtain cartridge weights**  
   If you want attention statistics with a specific cartridge, point `--cartridge-path` to the `kv_cache.torch` artifact.

4. **Run the script**  
   ```bash
   python -m src.attention_capture.check_cli \
       --model-type llama \
       --model-name meta-llama/Llama-3.2-3B-Instruct \
       --cartridge-path /path/to/kv_cache.torch \
       --input-jsonl /path/to/convos.jsonl \
       --output-dir outputs/attn_stats \
       --save-qkv \
       --save-attention-weights \
       --mode train
   ```
   Adjust flags to suit your workflow (e.g. `--mode generate`, `--device cpu`, `--summary-only`).

5. **Inspect outputs**  
   - `outputs/attn_stats/conversation_0000.json` summarises cartridge vs normal attention mass per layer.  
   - `outputs/attn_stats/conversation_0000.pt` contains PyTorch tensors (if enabled) for deeper or visual analysis.

6. **Replay validation prompts with a saved history agent**  
   ```bash
   python -m src.attention_capture.run_eval_cli \
       --config configs/attention_eval/history_agent_cities.yaml \
       --memory-snapshot /projects/bfsg/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/checkpoints/ep_000700/memory_700.jsonl \
       --output-dir outputs/attn_eval/l8b_history_agent \
       --model-type llama \
       --model-name meta-llama/Llama-3.1-8B-Instruct \
       --layer-idx -1 \
       --query-span-tag current_observation \
       --max-samples 40
   ```
   This command replays the validation split defined in the config, captures attention tensors for each sample, and emits both per-conversation artefacts and an aggregate heatmap showing how strongly the latest question attends to the system prompt, every stored memory, and any cartridge slots.

7. **Smoke-test with ad-hoc prompts**  
   A tiny JSONL is bundled at `configs/attention_eval/sample_prompts.jsonl` for quick local checks:
   ```bash
   python -m src.attention_capture.check_cli \
       --model-type llama \
       --model-name meta-llama/Llama-3.1-8B-Instruct \
       --input-jsonl configs/attention_eval/sample_prompts.jsonl \
       --output-dir outputs/attn_stats/sample_prompts \
       --save-attention-weights
   ```
   Replace this file with your own conversations when running real analyses—it simply showcases the expected JSONL shape.

---

## Extending the pipeline

- **Alternative aggregations**  
  Extend `src/attention_capture/pipeline.py` to compute custom statistics (e.g. per-token entropy, top-k cartridge tokens).

- **Batch processing**  
  Currently each conversation is processed independently for clarity. You can vectorise the script by packing multiple conversations into a single forward pass, reusing the same capture flow.

- **Visualization**  
  Reuse `src/attention_capture/plotting.py` utilities or load the emitted heatmap matrices for bespoke dashboards.

- **Additional models**  
  To support other cartridge-aware architectures, expose the capture flag in their forward paths and register them in `MODEL_LOADERS`.

- **Attention-focused plots**  
  The aggregated JSON (`layer_xx_<span>_attention.json`) and PNG heatmaps generated by `src.attention_capture.run_eval_cli` track how query tokens distribute mass across each tagged span, enabling quick comparison across validation examples.

---

## Troubleshooting

- **Missing tensors in capture records**  
  Ensure the script runs with `AttentionCapture(store_qkv=True)` (default). If running custom code, pass the capture object into `Flex*Model.forward(..., attention_capture=capture)`.

- **Flex attention hangs while capturing**  
  The recorder now operates entirely outside the compiled flex-attention kernels, so no special flag is required. If you modify the call sites, ensure Q/K/V tensors are detached before invoking `AttentionCapture.record_from_eager(...)`.

- **(Update)** We experimented with a full eager-attention fallback to avoid the tracing hang. While it unblocked the recorder, long packed sequences quickly exhausted GPU memory (the dense score tensor scales with `heads × seq_len × kv_len`, which can exceed hundreds of GiB). The current implementation keeps flex attention for efficiency, records tensors outside the compiled graph, and reconstructs aggregates in streamed chunks to avoid OOMs.

- **Shape mismatches while reconstructing attention weights**  
  Use the returned `kv_seq_ids` to verify alignment—if the capture flow bypassed the cache (e.g. no cartridge), heads and sequence lengths should match the query tensors directly.

- **GPU memory pressure**  
  Aggregation now operates in streamed chunks, so only per-tag summaries are stored. Full attention weight tensors are no longer persisted; keep `--save-attention-weights` unset to avoid unnecessary work.

---

With these components, you can generate reproducible attention statistics for both cartridge and non-cartridge tokens, enabling further analysis or visualisation pipelines.

