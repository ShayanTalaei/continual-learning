# Modularize Attention Computation and Add Individual Pair Extraction

## Overview

Refactor `_compute_attention_masses` to separate calculation from summarization, and add extraction of individual attention pairs for flexible visualization.

## Size Analysis

For each conversation:

- **Pairs**: N generated tokens × M KV positions (cartridge + prompt)
- **Per pair**: ~175 bytes (4 ints + 2 strings + 1 float)
- **Example**: 100 targets × 1000 sources = 100k pairs × 175 bytes ≈ **17.5 MB per layer**
- **32 layers**: ~560 MB per conversation
- **Recommendation**: Save pairs per layer in separate files, or add compression

## Implementation Plan

### 1. Create `_compute_attention_weights()` function

**File**: `src/attention_capture/pipeline.py`

Extract the core attention computation logic:

- Input: `AttentionRecord`, masks, spans, etc.
- Output: Dictionary with:
  - `weights`: `torch.Tensor` shape `(seq_len, kv_len)` - attention weights averaged across heads
  - `target_positions`: `torch.Tensor` - positions of target tokens in original sequence
  - `source_positions`: `torch.Tensor` - positions of source tokens in KV cache
  - `source_tags`: `List[Optional[str]]` - tag for each source position (from message_spans)
  - `metadata`: Dict with layer_idx, cache_len, etc.

**Key changes**:

- Compute attention scores and weights in chunks
- Average across heads: `weights.mean(dim=0)` to get `(seq_len, kv_len)`
- Map KV positions to source tags using message_spans
- Return raw weights instead of aggregated masses

### 2. Create `_summarize_attention_weights()` function

**File**: `src/attention_capture/pipeline.py`

Extract summarization logic:

- Input: weights dict from `_compute_attention_weights()`, span_masks, etc.
- Output: `LayerAttentionSummary` and tensor payload (same as current)

**Key changes**:

- Take pre-computed weights instead of computing them
- Compute cartridge masses, tag masses, etc. from the weights
- Keep existing aggregation logic (per-head, per-token, means)

### 3. Create `_extract_attention_pairs()` function

**File**: `src/attention_capture/pipeline.py`

Extract individual attention pairs:

- Input: 
  - weights dict from `_compute_attention_weights()`
  - `tokenizer` for decoding
  - `input_ids` for target token IDs
  - `kv_input_ids` or mapping for source token IDs
- Output: `List[Dict]` where each dict has:
  ```python
  {
      "target_token_id": int,
      "target_token_text": str,
      "source_token_id": int,
      "source_token_text": str,
      "source_tag": Optional[str],
      "weight": float,
      "layer_idx": int,
      "generation_step": Optional[int],  # if applicable
  }
  ```

**Key implementation**:

- Iterate over target positions (seq_len)
- For each target, iterate over source positions (kv_len)
- Extract weight, decode tokens, get source tag
- Handle chunking by processing each chunk and concatenating results
- Filter out zero-weight pairs if needed (optional optimization)

### 4. Refactor `_compute_attention_masses()`

**File**: `src/attention_capture/pipeline.py`

Update to use new functions:

```python
def _compute_attention_masses(...):
    weights_dict = _compute_attention_weights(...)
    summary, tensors = _summarize_attention_weights(weights_dict, ...)
    return summary, tensors
```

### 5. Update `capture_attention_for_messages()`

**File**: `src/attention_capture/pipeline.py`

Add individual pair extraction:

- After computing attention for each record, also extract pairs
- Collect pairs per layer
- Store in result or save directly

**Key changes**:

- Call `_extract_attention_pairs()` for each record
- Need to pass tokenizer and input_ids
- Track generation step for each record
- Collect pairs: `Dict[int, List[Dict]]` keyed by layer_idx

### 6. Update `write_conversation_outputs()`

**File**: `src/attention_capture/pipeline.py`

Add saving of attention pairs:

- New parameter: `save_attention_pairs: bool = False`
- Save pairs to: `conversation_{idx:04d}_attention_pairs.json`
- Structure: `{"layer_0": [pairs...], "layer_31": [pairs...], ...}`
- Include metadata: conversation_index, num_pairs_per_layer, size_estimate

### 7. Update function signatures and calls

**Files**: `src/attention_capture/pipeline.py`, `src/attention_capture/run_eval_cli.py`

- Add `save_attention_pairs` parameter to `capture_attention_for_messages()`
- Pass through from CLI if needed
- Update `ConversationAttentionResult` to optionally include pairs (or save directly)

## File Structure Changes

**New files**: None

**Modified files**:

- `src/attention_capture/pipeline.py`: Refactor functions, add pair extraction
- `src/attention_capture/types.py`: Optionally add field for pairs (or save directly)

**Output files** (per conversation):

- `conversation_{idx:04d}.json` (existing)
- `conversation_{idx:04d}.pt` (existing)
- `conversation_{idx:04d}_attention_pairs.json` (new) - per-layer pairs

## Testing Considerations

- Verify pairs match aggregated masses (sum of pair weights ≈ tag masses)
- Check memory usage with large conversations
- Ensure chunking works correctly (pairs concatenate properly)
- Verify token decoding matches original tokens
- Test with/without cartridges, with/without tags

## Notes

- Keep backward compatibility: existing code should work unchanged
- Pairs are optional (controlled by flag)
- Consider adding filtering/thresholding later if size becomes issue
- May want to add compression for large files

## Implementation Checklist

- [ ] Create `_compute_attention_weights()` function that returns raw attention weights and metadata, averaging across heads
- [ ] Create `_summarize_attention_weights()` function that takes weights and creates `LayerAttentionSummary`
- [ ] Create `_extract_attention_pairs()` function that extracts individual target x source pairs with decoded tokens and tags
- [ ] Refactor `_compute_attention_masses()` to use the new separated functions
- [ ] Update `capture_attention_for_messages()` to extract and collect attention pairs per layer
- [ ] Update `write_conversation_outputs()` to save attention pairs to JSON file

## Key Implementation Details

### Attention Weight Computation

The `_compute_attention_weights()` function should:
1. Process Q/K/V tensors from `AttentionRecord`
2. Compute attention scores: `scores = Q @ K^T * scaling`
3. Apply masks (causal, sequence, prompt, cartridge)
4. Compute softmax: `weights = softmax(scores)`
5. Average across heads: `weights.mean(dim=0)` → `(seq_len, kv_len)`
6. Map KV positions to source tags using `message_spans` and `cartridge_len`
7. Return dictionary with weights and metadata

### Source Token ID Mapping

Challenge: Need to map KV cache positions back to original token IDs.

Options:
1. Store `kv_input_ids` in the cache (if available)
2. Reconstruct from `message_spans` and original `input_ids`
3. Store mapping during prompt processing

For cartridges (seq_id == -1), source_token_id might be None or a special value.

### Pair Extraction

The `_extract_attention_pairs()` function should:
1. Iterate over each target position (0 to seq_len-1)
2. For each target, iterate over each source position (0 to kv_len-1)
3. Extract weight from `weights[target_idx, source_idx]`
4. Decode target token: `tokenizer.decode([target_token_id])`
5. Decode source token: `tokenizer.decode([source_token_id])` (if available)
6. Get source tag from pre-computed mapping
7. Create dict and append to list
8. Handle chunking by processing chunks separately and concatenating

### Memory Considerations

- For large conversations, consider:
  - Streaming pairs to disk instead of keeping all in memory
  - Filtering zero-weight pairs
  - Processing one layer at a time
  - Using generators/yield for pair extraction

