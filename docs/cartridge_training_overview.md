# Cartridge Training Overview

This document summarizes the goal of the cartridge distillation pipeline in `src/memory/distillation/distill_into_cartridge.py` and the main architectural knobs exposed by the Cartridges library (see `third_party/cartridges`).

## Goal
- Distill a teacher model’s behavior into a compact, trainable KV cache “cartridge”.
- Initialize cartridge keys/values (K/Vs), optionally add learned positional biases, and train with distillation data.
- Optionally upload the resulting cartridge to Hugging Face for reuse.

## Key Components & Knobs

### Cache Initialization
- **Random vectors** (`KVFromRandomVectors`): uniform random K/V tensors; set `max_tokens`, `num_frozen_tokens`.
- **Text-derived** (`KVFromText` / `KVFromRandomText`): tokenize a text prompt through the base model to produce K/Vs; set `text_source`, `system_prompt_template`, `max_tokens`, `cartridge_start_position`.
- **Pretrained** (`KVFromPretrained`): load a previously saved cartridge checkpoint (e.g., from W&B).

### Parametrization of Trainable Tokens
Let the initialized cartridge for a given layer be `K0, V0` with shape `(1, n_heads, T, d)`, where `T = num_frozen_tokens + num_trainable_tokens`. The first `num_frozen_tokens` are fixed; the rest are trainable. Choose via `parametrization_type`:

- **offset** (`OffsetParametrization`):  
  Split `K0 = [K_frozen, K_ref]`, `V0 = [V_frozen, V_ref]`. Learn offsets `ΔK, ΔV` (initialized at 0):
  ```
  K_train = K_ref + ΔK
  V_train = V_ref + ΔV
  ```
  Frozen K/V stay unchanged and are prepended during attention.

- **mlp_residual** (`ResidualMLPParametrization`):  
  Apply a small MLP to the reference part and add a residual:
  ```
  K_train = MLP_K(K_ref) + K_ref
  V_train = MLP_V(V_ref) + V_ref
  ```
  Knobs:
  - `hidden_multiplier`: sets hidden width = `head_dim * hidden_multiplier`.
  - `activation`: `relu` or `gelu`.
  - `share_across_layers`: share MLP weights across layers vs per-layer MLPs.
  - `zero_init_last_layer`: zero the last linear layer to start exactly at the reference (stability/preservation of init).
- **mlp_residual_gated** (`GatedResidualMLPParametrization`):  
  Adds learnable gates α on top of the residual MLP to keep the cartridge “off” by default:
  ```
  K_train = K_ref + α_K * MLP_K(K_ref)
  V_train = V_ref + α_V * MLP_V(V_ref)
  ```
  Gates are learnable scalars with configurable granularity:
  - `gate_granularity`: `global`, `per_layer`, or `per_head` (default `per_head`).
  - `gate_init`: default `0.0` (suppresses cartridge changes at init); set tiny (>0) for a small starting effect.
  All MLP knobs from `mlp_residual` still apply (hidden multiplier, activation, sharing, zero-init last layer).

### Frozen vs Trainable Tokens
- `num_frozen_tokens` keeps the first tokens fixed to prevent forgetting; remaining tokens are trainable.
- Cartridge tokens get sequence id `-1` so all queries can attend to them.

### Positional Embeddings on Cartridge K/Vs
- Optional learned per-layer bias added to cartridge K/Vs (broadcast over tokens):
  ```
  K_pos = K + Pk[layer]   ,   V_pos = V + Pv[layer]
  ```
- Controlled by `positional_embeddings`:
  - `enabled`
  - `init_mode` (`zero` or `random`) and `random_std`
  - Parameters shaped `(1, n_heads, 1, head_dim)` per layer.

### Attention Combination Modes (Llama backend)
When `use_unrotated_queries_for_cartridges` is enabled, attention is split into two masked calls:

1) Build masks:
   - `block_mask`: excludes cartridge keys (normal context).
   - `cartridge_mask`: only cartridge keys (seq id = -1).

2) Use rotated queries for normal attention and unrotated for cartridge attention:
   ```
   out_norm, lse_norm  = Attn(query_rot,   K, V, block_mask)
   out_cart, lse_cart  = Attn(query_unrot, K, V, cartridge_mask)
   ```

3) Combine per `cartridge_attention_mode`:
   - `global_softmax` (default): approximate one softmax over both groups via log-sum-exp:
     ```
     logZ   = logaddexp(lse_norm, lse_cart)
     w_norm = exp(lse_norm - logZ)
     w_cart = exp(lse_cart - logZ)
     attn   = w_norm * out_norm + w_cart * out_cart
     ```
     Pros: smooth, closer to a single softmax; avoids recomputing over concatenated keys.  
     Cons: approximation assumes separable logits across the two groups.
   - `separate_sum`: two independent softmaxes, summed:
     ```
     attn = out_norm + out_cart
     ```
     Pros: decouples cartridge/context; can keep strong cartridge influence.  
     Cons: not jointly normalized—can over-weight cartridges if both are confident.
   - Optional gate α (config: `cartridge_attention_gate_*`):  
     ```
     attn = (1 - α) * out_norm + α * out_cart              # separate_sum
     attn = (1 - α) * out_norm + α * (w_norm*out_norm + w_cart*out_cart)  # global_softmax
     ```
     Granularity: `global`, `per_layer`, `per_head` (default `per_head`).  
     Gate init: default `0.0` (context-only at start); small positive values give gentle cartridge influence.

If `use_unrotated_queries_for_cartridges` is off, a single attention call is used with the standard block mask (cartridge tokens are simply part of K/V with seq id = -1).

### Distillation Script Knobs (`distill_into_cartridge.py`)
- **Model**: `model_name`, `non_cartridge_start_position_id_offset`, `use_unrotated_queries_for_cartridges`, `cartridge_attention_mode`, `cartridge_attention_gate_enabled`, `cartridge_attention_gate_granularity`, `cartridge_attention_gate_init`.
- **KV cache**: `method` (`random`/`text`), `num_tokens`, `num_frozen_tokens`, `cartridge_start_position`, parametrization fields (including `parametrization_type` `mlp_residual_gated`, gate granularity/init), positional embedding fields.
- **Dataset**: packing mode/length, targets (`logits`), top-k logits, batch size, optional streaming.
- **Training**: epochs, global batch size, lr, weight decay, optimizer, gradient checkpointing, device/backend, save cadence.
- **Eval**: loss evals, generation evals, eval type (`finer` / `synth_cities`), grouped eval configs, temperatures, batch sizes.
- **Logging/Outputs**: WandB toggles, output dir, HF upload (`hf_repo_id`, `hf_private`).

## Typical Recipes
- **Fast experiment**: random init, `offset`, small `num_tokens`, `num_frozen_tokens=1`, positional embeddings off, `global_softmax`, gates disabled.
- **Higher capacity**: text init with meaningful prompt, `mlp_residual` (wider multiplier), positional embeddings enabled, larger `num_tokens`.
- **Safety against forgetting / minimal interference**: `mlp_residual_gated` with `gate_init=0`, `per_head` gates, `zero_init_last_layer`, `global_softmax` + attention gate; increase `num_frozen_tokens`.

## Validation / Safety Checks
- Defaults preserve old behavior: gates disabled → matches prior configs.
- When gates are enabled, start with `gate_init=0` and verify loss/eval parity to baseline (cartridge off).
- Track gate magnitudes and MLP output norms (`logging_metrics`) to ensure cartridges only activate when trained.
- Run paired evals with and without cartridges to catch ICL regressions early.
- Tokasaurus eval support: unrotated queries + `global_softmax`/`separate_sum` are supported; gating is supported after adding `cartridge_attention_gate_*` in tokasaurus configs and attention path (per this change).

