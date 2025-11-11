# Attention Capture Fix: Implementation Summary

## Problem Statement

The attention capture pipeline was experiencing two critical issues:

1. **Hangs during forward pass**: The `capture.record_from_eager()` calls after `flex_attention_forward()` were causing the model to hang when compiled with `torch.compile()`. Even with `@torch._dynamo.disable()`, CPU operations inside compiled graph contexts can cause deadlocks.

2. **OOM (Out of Memory) errors**: Attempting to materialize full attention matrices for long sequences would exhaust GPU memory, as the dense score tensor scales with `heads × seq_len × kv_len`.

## Solution Strategy

The fix involves three key changes:

1. **Store Q/K/V tensors before compiled operations**: Move tensor storage to happen immediately after Q/K/V computation, before calling `flex_attention_forward()`. This ensures storage happens outside any compiled graph.

2. **Synchronous CPU movement**: Move tensors to CPU synchronously during storage to avoid graph compilation issues.

3. **Chunked attention computation**: Compute attention masses in chunks after the forward pass completes, aggregating results online to avoid materializing full attention matrices.

## Implementation Status

### ✅ Completed Changes

#### 1. Updated `AttentionCapture.record()` method
**File:** `third_party/cartridges/cartridges/models/attention_capture.py`

- Changed CPU movement to synchronous (`non_blocking=False`) to avoid graph compilation issues
- Simplified `record_from_eager()` method (kept for backward compatibility, marked as deprecated)
- The `record()` method now immediately moves tensors to CPU synchronously when `move_to_cpu=True`

#### 2. Updated `LlamaAttention.forward()`
**File:** `third_party/cartridges/cartridges/models/llama/modeling_llama.py`

- **Moved** `capture.record()` call to **before** `flex_attention_forward()` (immediately after Q/K/V computation)
- **Removed** `capture.record_from_eager()` call after `flex_attention_forward()`
- **Removed** unused parameters from `flex_attention_forward()` call (`attention_capture`, `layer_idx`, `seq_ids`, `kv_seq_ids`, `cache_len`)
- **Cleaned up** local `flex_attention_forward()` function to remove unused attention capture parameters

#### 3. Updated `Qwen3Attention.forward()`
**File:** `third_party/cartridges/cartridges/models/qwen/modeling_qwen3.py`

- **Added** `capture.record()` call before `flex_attention_forward()` (same pattern as Llama)
- **Removed** unused parameters from `flex_attention_forward()` call

#### 4. Cleaned up `flex_attention_forward()` signatures
**Files:**
- `third_party/cartridges/cartridges/models/attention.py` (shared function)
- `third_party/cartridges/cartridges/models/llama/modeling_llama.py` (local function)

- **Removed** unused parameters: `attention_capture`, `layer_idx`, `seq_ids`, `kv_seq_ids`, `cache_len`
- **Removed** unused `AttentionCapture` import from `attention.py`

#### 5. Verified chunked computation
**File:** `src/attention_capture/pipeline.py`

- No changes needed - `_compute_attention_masses()` already implements chunked computation correctly
- Processes query tokens in chunks (default `chunk_size=128`)
- Aggregates attention masses online to avoid materializing full attention matrices

## Key Implementation Details

### Storage Timing
- Q/K/V tensors are now stored **immediately after computation**, before any compiled operations
- This ensures storage happens outside the compiled `flex_attention` graph

### CPU Movement
- Tensors are moved to CPU **synchronously** during storage (`non_blocking=False`)
- This prevents graph compilation from trying to trace CPU operations

### Computation Timing
- Attention mass computation happens **after** the model forward completes
- Computation is done in `_compute_attention_masses()` which processes chunks and aggregates online

### Chunking Strategy
- Query tokens are processed in chunks (configurable `chunk_size`, default 128)
- For each chunk:
  - Compute attention scores: `Q_chunk @ K^T * scaling`
  - Apply causal and sequence masks
  - Compute softmax to get attention weights
  - Aggregate cartridge vs normal attention masses
- Results are accumulated across chunks to produce final statistics

## Next Steps: Testing

### Prerequisites
1. Access to a GPU node (flex_attention requires CUDA)
2. Conda environment `continual_learning` activated
3. Model weights available (e.g., `meta-llama/Llama-3.1-8B-Instruct`)

### Test Cases

#### Test 1: Basic Functionality (No Hangs)
**Command:**
```bash
python -m src.attention_capture.check_cli \
    --model-type llama \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --prompt "What is the capital of France?" \
    --output-dir /tmp/test_attention_capture/basic_test \
    --mode generate \
    --summary-only \
    --device cuda
```

**Expected:**
- ✅ No hangs during forward pass
- ✅ Output JSON file created with attention statistics
- ✅ Completion in reasonable time (< 5 minutes for single prompt)

#### Test 2: With Cartridge Cache
**Command:**
```bash
python -m src.attention_capture.check_cli \
    --model-type llama \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --cartridge-path /path/to/kv_cache.torch \
    --prompt "What is the capital of France?" \
    --output-dir /tmp/test_attention_capture/cartridge_test \
    --mode generate \
    --summary-only \
    --device cuda
```

**Expected:**
- ✅ No hangs
- ✅ Cartridge attention mass > 0 in output
- ✅ Correct handling of cartridge vs normal tokens

#### Test 3: Long Sequence (Memory Test)
**Command:**
```bash
python -m src.attention_capture.check_cli \
    --model-type llama \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --input-jsonl /path/to/long_conversation.jsonl \
    --output-dir /tmp/test_attention_capture/long_seq_test \
    --mode generate \
    --summary-only \
    --device cuda
```

**Expected:**
- ✅ No OOM errors even with long sequences
- ✅ Chunked computation prevents memory issues
- ✅ Attention statistics computed correctly

#### Test 4: Validation Replay (Full Pipeline)
**Command:**
```bash
python -m src.attention_capture.run_eval_cli \
    --config configs/attention_eval/history_agent_cities.yaml \
    --memory-snapshot /path/to/memory_700.jsonl \
    --output-dir /tmp/test_attention_capture/eval_test \
    --model-type llama \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --max-samples 5 \
    --mode generate \
    --device cuda
```

**Expected:**
- ✅ No hangs during processing
- ✅ Multiple conversations processed successfully
- ✅ Aggregated attention heatmaps generated
- ✅ Per-conversation JSON files created

#### Test 5: Train Mode
**Command:**
```bash
python -m src.attention_capture.check_cli \
    --model-type llama \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --prompt "What is the capital of France?" \
    --output-dir /tmp/test_attention_capture/train_test \
    --mode train \
    --summary-only \
    --device cuda
```

**Expected:**
- ✅ No hangs in train mode
- ✅ Attention statistics computed correctly

### Verification Checklist

For each test, verify:

- [ ] **No hangs**: Process completes without hanging indefinitely
- [ ] **No OOM**: Memory usage stays reasonable (monitor with `nvidia-smi`)
- [ ] **Output files created**: JSON summary files are generated
- [ ] **Correct statistics**: Attention masses are reasonable (0-1 range, sum to ~1)
- [ ] **Cartridge detection**: When cache is used, cartridge attention mass is detected
- [ ] **Chunked computation**: For long sequences, verify memory doesn't spike

### Debugging Tips

If hangs occur:
1. Check if tensors are being stored before `flex_attention_forward()` (add print statements)
2. Verify `move_to_cpu=True` and `non_blocking=False` in `AttentionCapture`
3. Check if model is actually using compiled flex_attention (should see compilation messages)

If OOM occurs:
1. Reduce `chunk_size` in `_compute_attention_masses()` (default 128)
2. Check if full attention weights are being saved (should be disabled)
3. Verify chunked computation is actually running (add logging)

### Performance Benchmarks

After successful testing, document:
- Time per conversation (should be similar to before, maybe slightly faster without hangs)
- Memory usage for different sequence lengths
- Maximum sequence length that can be processed without OOM

## Files Modified

1. `third_party/cartridges/cartridges/models/attention_capture.py`
2. `third_party/cartridges/cartridges/models/llama/modeling_llama.py`
3. `third_party/cartridges/cartridges/models/qwen/modeling_qwen3.py`
4. `third_party/cartridges/cartridges/models/attention.py`

## Files Verified (No Changes Needed)

1. `src/attention_capture/pipeline.py` - Chunked computation already implemented correctly

## Backward Compatibility

- `record_from_eager()` method is kept for backward compatibility but marked as deprecated
- All existing API calls should continue to work
- The main change is internal: storage happens earlier in the forward pass

## Summary

The implementation moves tensor storage to happen **before** compiled operations, ensuring no interference with `torch.compile()`. The chunked attention computation was already correctly implemented and continues to work as before. The key insight is that **any CPU operations or state mutations must happen outside the compiled graph** to avoid hangs.

