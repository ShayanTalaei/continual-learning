from __future__ import annotations

import json
import math
from bisect import bisect_right
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast, TypedDict

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding

from cartridges.cache import TrainableCache, AttnConfig
from cartridges.models.attention_capture import AttentionCapture, AttentionRecord

from .types import (
    ChatMessage,
    TokenSpan,
    TokenizedConversation,
    LayerAttentionSummary,
    ConversationAttentionSummary,
    ConversationAttentionResult,
)

from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM

# LLAMA_CARTRIDGE_TEMPLATE matches the template used in tokasaurus and cartridges codebases
LLAMA_CARTRIDGE_TEMPLATE = """\
{%- for message in messages %}
    {%- if  (message.role == 'assistant') %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}{% generation %}{{- message['content'] | trim + '<|eot_id|>' }}{% endgeneration %}

    {%- else %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
        
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"""


class StepTimer:
    def __init__(self) -> None:
        self._timings: Dict[str, float] = {}
        self._indent: int = 0

    @contextmanager
    def track(self, step: str):
        indent = "  " * self._indent
        print(f"{indent}[StepTimer] start {step}")
        self._indent += 1
        start = perf_counter()
        try:
            yield
        finally:
            duration = perf_counter() - start
            self._timings[step] = self._timings.get(step, 0.0) + duration
            self._indent = max(0, self._indent - 1)
            indent = "  " * self._indent
            print(f"{indent}[StepTimer] end {step}: {duration:.3f}s")

    def as_dict(self) -> Dict[str, float]:
        return dict(self._timings)

def load_model(
    model_name: str,
    device: Union[str, torch.device],
):
    resolved_device = torch.device(device)
    model = FlexLlamaForCausalLM.from_pretrained(model_name)
    model = model.to(resolved_device)  # type: ignore[call-arg]
    model.eval()
    return model


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name)


def load_cartridges_from_ids(
    cartridge_ids: List[str],
    cartridge_dir: str,
    device: Union[str, torch.device],
) -> List[Tuple[TrainableCache, str]]:
    """Load cartridges from local paths based on cartridge IDs.
    
    Args:
        cartridge_ids: List of cartridge ID strings
        cartridge_dir: Base directory where cartridge files are stored
        device: Device to load cartridges onto
        
    Returns:
        List of (cache, cartridge_id) tuples
    """
    resolved_device = torch.device(device)
    cartridges = []
    base_path = Path(cartridge_dir)
    
    for cartridge_id in cartridge_ids:
        # Cartridges are stored at: {cartridge_dir}/{cartridge_id}/cartridge.pt
        cartridge_path = base_path / cartridge_id / "cartridge.pt"
        
        if not cartridge_path.exists():
            raise FileNotFoundError(
                f"Cartridge file not found for ID '{cartridge_id}'. "
                f"Expected path: {cartridge_path}"
            )
        
        cache = TrainableCache.from_pretrained(
            str(cartridge_path),
            device=resolved_device.type if resolved_device.type != "cuda" else None,
        )
        cache = cache.to(device=resolved_device)
        cache.eval()
        cartridges.append((cache, cartridge_id))
    
    return cartridges




def tokenize_messages(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[ChatMessage],
    device: Union[str, torch.device],
) -> TokenizedConversation:
    timer = StepTimer()
    resolved_device = torch.device(device)
    with timer.track("build_message_dicts"):
        message_dicts = [m.as_dict() for m in messages]
    
    # Determine if we should add generation prompt (like tokasaurus does)
    # Add generation prompt if the last message is from the user
    ends_with_user = len(messages) > 0 and messages[-1].role == "user"
    add_generation_prompt = ends_with_user
    
    prompt = tokenizer.apply_chat_template(
        message_dicts,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        chat_template=LLAMA_CARTRIDGE_TEMPLATE,
    )
    
    # Tokenize the rendered prompt ONCE
    with timer.track("tokenize_prompt"):
        encoding = cast(
            BatchEncoding,
            tokenizer(
                cast(str, prompt),
                add_special_tokens=False,
                return_tensors="pt",
            ),
        )
        tokenized = cast(torch.Tensor, encoding["input_ids"])
        if tokenized.dim() > 1:
            tokenized = tokenized.squeeze(0)
        full_token_ids = tokenized.tolist()
    
    # OPTIMIZED: Find message boundaries using special tokens instead of tokenizing each prefix
    with timer.track("compute_message_boundaries"):
        # Get special token IDs by encoding the tokens
        # Use encode with add_special_tokens=False to get just the token ID
        start_header_encoded = tokenizer.encode("<|start_header_id|>", add_special_tokens=False)
        end_header_encoded = tokenizer.encode("<|end_header_id|>", add_special_tokens=False)
        eot_encoded = tokenizer.encode("<|eot_id|>", add_special_tokens=False)
        
        # Extract the token ID (should be a single token)
        start_header_id = start_header_encoded[0] if start_header_encoded else None
        end_header_id = end_header_encoded[0] if end_header_encoded else None
        eot_id = eot_encoded[0] if eot_encoded else None
        
        if start_header_id is None or end_header_id is None or eot_id is None:
            raise ValueError("Could not find special tokens in tokenizer vocabulary")
        
        # Find message boundaries by searching for special tokens
        prefix_token_lengths: List[int] = []
        current_pos = 0
        message_idx = 0
        
        while current_pos < len(full_token_ids) and message_idx < len(messages):
            # Find next <|start_header_id|>
            try:
                start_idx = full_token_ids.index(start_header_id, current_pos)
            except ValueError:
                # No more messages found
                break
            
            # Find the matching <|end_header_id|> after the start
            try:
                end_idx = full_token_ids.index(end_header_id, start_idx + 1)
            except ValueError:
                # Malformed template, break
                break
            
            # Find the <|eot_id|> that ends this message
            try:
                eot_idx = full_token_ids.index(eot_id, end_idx + 1)
            except ValueError:
                # If no eot_id found, this might be the last message or generation prompt
                # Use the end of the sequence
                if message_idx == len(messages) - 1:
                    prefix_token_lengths.append(len(full_token_ids))
                break
            
            # The message ends at eot_id (inclusive, so +1)
            prefix_token_lengths.append(eot_idx + 1)
            current_pos = eot_idx + 1
            message_idx += 1
        
        # If we didn't find all messages, pad with the total length
        while len(prefix_token_lengths) < len(messages):
            prefix_token_lengths.append(len(full_token_ids))

    with timer.track("prepare_token_metadata"):
        input_ids = tokenized.to(resolved_device)
        total_length = int(input_ids.shape[0])
        seq_ids = torch.zeros(total_length, dtype=torch.long, device=resolved_device)
        position_ids = torch.arange(total_length, dtype=torch.long, device=resolved_device)

        spans: List[TokenSpan] = []
        prev = 0
        for idx, prefix_len in enumerate(prefix_token_lengths):
            if idx < len(messages):
                span = TokenSpan(
                    start=prev,
                    end=prefix_len,
                    message_index=idx,
                    tag=messages[idx].tag,
                    role=messages[idx].role,
                )
                spans.append(span)
                prev = prefix_len

        if add_generation_prompt and prev < total_length:
            spans.append(
                TokenSpan(
                    start=prev,
                    end=total_length,
                    message_index=None,
                    tag="generation_prompt",
                    role="assistant",
                )
            )

    return TokenizedConversation(
        input_ids=input_ids,
        seq_ids=seq_ids,
        position_ids=position_ids,
        message_spans=spans,
        total_length=total_length,
        timings=timer.as_dict(),
    )


class AttentionWeightsDict(TypedDict):
    weights: torch.Tensor
    target_positions: torch.Tensor
    source_positions: torch.Tensor
    source_tags: List[Optional[str]]
    metadata: Dict[str, Union[int, str]]


def _compute_attention_weights(
    record: AttentionRecord,
    chunk_size: int = 128,
    message_spans: Optional[Sequence[TokenSpan]] = None,
    cartridge_lengths: Optional[List[int]] = None,
    cartridge_ids: Optional[List[str]] = None,
) -> AttentionWeightsDict:
    """
    Compute attention weights averaged across heads.
    
    Returns a dictionary with:
    - weights: torch.Tensor shape (seq_len, kv_len) - attention weights averaged across heads
    - target_positions: torch.Tensor - positions of target tokens in original sequence
    - source_positions: torch.Tensor - positions of source tokens in KV cache
    - source_tags: List[Optional[str]] - tag for each source position
    - metadata: Dict with layer_idx, cache_len, etc.
    """
    if record.query is None or record.key is None or record.seq_ids is None:
        raise RuntimeError(
            "Attention capture record is missing tensors; ensure AttentionCapture.store_qkv is True."
        )

    query = record.query.to(dtype=torch.float32)
    key = record.key.to(dtype=torch.float32)
    seq_ids = record.seq_ids.to(dtype=torch.long)
    kv_seq_ids = (
        record.kv_seq_ids.to(dtype=torch.long) if record.kv_seq_ids is not None else seq_ids.clone()
    )

    if query.dim() == 4 and query.shape[0] == 1:
        query = query.squeeze(0)
    if key.dim() == 4 and key.shape[0] == 1:
        key = key.squeeze(0)
    if query.dim() != 3 or key.dim() != 3:
        raise RuntimeError("Captured tensors must have shape [num_heads, seq_len, head_dim].")

    num_heads, seq_len, head_dim = query.shape
    key_heads, kv_len, _ = key.shape

    if key_heads != num_heads:
        if num_heads % key_heads != 0:
            raise RuntimeError(
                f"Unable to align heads: query_heads={num_heads}, key_heads={key_heads}"
            )
        repeat_factor = num_heads // key_heads
        key = key.repeat_interleave(repeat_factor, dim=0)

    scaling = record.scaling
    if scaling is None:
        head_dim_value = record.head_dim or head_dim
        scaling = 1.0 / math.sqrt(head_dim_value)

    if chunk_size <= 0 or chunk_size >= seq_len:
        chunk_size = seq_len

    device = query.device
    kv_seq_ids_device = kv_seq_ids.to(device)
    seq_ids_device = seq_ids.to(device)
    kv_positions = torch.arange(kv_len, dtype=torch.long, device=device)
    
    # Compute total cartridge length
    total_cartridge_len = sum(cartridge_lengths) if cartridge_lengths else 0
    
    # Map KV positions to source tags uniformly
    source_tags: List[Optional[str]] = [None] * kv_len
    
    # Tag cartridge positions by their IDs
    if cartridge_lengths and cartridge_ids:
        cartridge_start = 0
        for cartridge_len, cartridge_id in zip(cartridge_lengths, cartridge_ids):
            cartridge_end = cartridge_start + cartridge_len
            for kv_pos in range(cartridge_start, min(cartridge_end, kv_len)):
                if kv_seq_ids_device[kv_pos].item() == -1:
                    source_tags[kv_pos] = f"cartridge_{cartridge_id}"
            cartridge_start = cartridge_end
    
    # Tag prompt message positions
    if message_spans is not None:
        for span in message_spans:
            if span.tag is None:
                continue
            span_kv_start = total_cartridge_len + span.start
            span_kv_end = total_cartridge_len + span.end
            for kv_pos in range(span_kv_start, min(span_kv_end, kv_len)):
                source_tags[kv_pos] = span.tag

    key_t = key.transpose(-1, -2)
    
    # Accumulate weights across chunks
    all_weights = []
    target_positions_list = []

    for q_start in range(0, seq_len, chunk_size):
        q_end = min(seq_len, q_start + chunk_size)
        q_chunk = query[:, q_start:q_end, :]
        seq_ids_chunk = seq_ids_device[q_start:q_end]
        q_positions_chunk = torch.arange(q_start, q_end, dtype=torch.long, device=device)

        same_sequence = kv_seq_ids_device.unsqueeze(0).eq(seq_ids_chunk.unsqueeze(1))
        causal_mask = kv_positions.unsqueeze(0) <= (q_positions_chunk + record.cache_len).unsqueeze(1)
        
        # Uniform mask: cartridges (seq_id=-1) are always allowed, prompt tokens follow causal + same_sequence
        allowed = kv_seq_ids_device.unsqueeze(0).eq(-1) | (same_sequence & causal_mask)

        scores = torch.matmul(q_chunk, key_t) * scaling
        scores = scores.masked_fill(~allowed.unsqueeze(0), -1e9)
        weights = torch.softmax(scores, dim=-1)  # (num_heads, chunk_seq_len, kv_len)
        
        # Average across heads: (chunk_seq_len, kv_len)
        weights_avg = weights.mean(dim=0)
        all_weights.append(weights_avg)
        target_positions_list.append(q_positions_chunk)

    # Concatenate weights from all chunks: (seq_len, kv_len)
    weights_concatenated = torch.cat(all_weights, dim=0)
    target_positions = torch.cat(target_positions_list, dim=0)

    return {
        "weights": weights_concatenated,
        "target_positions": target_positions,
        "source_positions": kv_positions,
        "source_tags": source_tags,
        "metadata": {
            "layer_idx": record.layer_idx,
            "mode": record.mode,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "kv_len": kv_len,
            "cache_len": record.cache_len,
        },
    }


def _summarize_attention_weights(
    weights_dict: AttentionWeightsDict,
    span_masks: Optional[Dict[str, torch.Tensor]] = None,
) -> LayerAttentionSummary:
    """
    Summarize attention weights into LayerAttentionSummary.
    
    Takes pre-computed weights and computes tag-averaged statistics.
    """
    weights = weights_dict["weights"]  # (seq_len, kv_len)
    source_tags = weights_dict["source_tags"]
    metadata = weights_dict["metadata"]
    
    device = weights.device
    seq_len = weights.shape[0]
    kv_len = weights.shape[1]
    num_heads = cast(int, metadata["num_heads"])
    
    # Build span masks from source_tags if not provided
    if span_masks is None:
        span_masks = {}
        if source_tags:
            # Group by tag
            tag_to_positions: Dict[str, List[int]] = {}
            for kv_pos, tag in enumerate(source_tags):
                if tag is not None:
                    if tag not in tag_to_positions:
                        tag_to_positions[tag] = []
                    tag_to_positions[tag].append(kv_pos)
            
            # Create masks for all tags uniformly
            for tag, positions in tag_to_positions.items():
                mask = torch.zeros(kv_len, dtype=torch.bool, device=device)
                for pos in positions:
                    mask[pos] = True
                span_masks[tag] = mask
    
    # Compute per-tag masses
    tag_mass_per_token: Dict[str, torch.Tensor] = {}
    total_tag_mass: Dict[str, torch.Tensor] = {}
    
    for tag, span_mask in span_masks.items():
        if tag is not None:
            tag_weights = weights * span_mask.to(dtype=weights.dtype).unsqueeze(0)
            tag_mass_per_token[tag] = tag_weights.sum(dim=1)  # (seq_len,)
            total_tag_mass[tag] = tag_mass_per_token[tag].sum()
    
    # Compute means (average across all decoded tokens)
    tag_mass_means: Dict[str, float] = {
        tag: (total_tag_mass[tag] / seq_len).item()
        for tag in span_masks.keys()
    }
    
    # Convert per-token masses to lists
    tag_mass_per_token_dict: Dict[str, List[float]] = {
        tag: tag_mass_per_token[tag].cpu().to(dtype=torch.float32).tolist()
        for tag in span_masks.keys()
    }

    summary = LayerAttentionSummary(
        layer_idx=cast(int, metadata["layer_idx"]),
        mode=cast(str, metadata["mode"]),
        num_heads=num_heads,
        seq_len=seq_len,
        cache_len=cast(int, metadata["cache_len"]),
        tag_mass_means=tag_mass_means,
        tag_mass_per_token=tag_mass_per_token_dict,
    )
    
    return summary


def _compute_attention_masses(
    record: AttentionRecord,
    chunk_size: int = 128,
    message_spans: Optional[Sequence[TokenSpan]] = None,
    cartridge_lengths: Optional[List[int]] = None,
    cartridge_ids: Optional[List[str]] = None,
) -> LayerAttentionSummary:
    """
    Compute attention masses using the refactored functions.
    """
    weights_dict = _compute_attention_weights(
        record=record,
        chunk_size=chunk_size,
        message_spans=message_spans,
        cartridge_lengths=cartridge_lengths,
        cartridge_ids=cartridge_ids,
    )
    
    # Summarize weights (span masks are built from source_tags in _summarize_attention_weights)
    summary = _summarize_attention_weights(
        weights_dict=weights_dict,
        span_masks=None,
    )
    
    return summary


def _prepare_cache_and_cartridges(
    cartridges: Optional[List[Tuple[TrainableCache, str]]],
    model,
    device: Union[str, torch.device],
) -> Tuple[TrainableCache, List[int], List[str]]:
    """Prepare cache and extract cartridge information.
    
    Args:
        cartridges: Optional list of (cache, cartridge_id) tuples
        model: Model to get config from
        device: Device to place cache on
        
    Returns:
        Tuple of (local_cache, cartridge_lengths, cartridge_ids)
    """
    resolved_device = torch.device(device)
    local_cache = None
    cartridge_lengths: List[int] = []
    cartridge_ids: List[str] = []
    
    if cartridges:
        # For now, use the first cartridge. In the future, we'll need to combine them properly
        local_cache, cartridge_id = cartridges[0]
        cartridge_lengths = [local_cache.num_cartridge_tokens()]
        cartridge_ids = [cartridge_id]
    
    if local_cache is None:
        # Initialize an empty cache for generation
        local_cache = TrainableCache(
            config=AttnConfig(
                n_layers=model.config.num_hidden_layers,
                n_heads=model.config.num_key_value_heads,
                head_dim=(
                    model.config.head_dim
                    if hasattr(model.config, "head_dim")
                    else model.config.hidden_size // model.config.num_attention_heads
                ),
            ),
        )
        local_cache = local_cache.to(device=resolved_device)
    
    if local_cache is not None:
        local_cache.clear()
    
    return local_cache, cartridge_lengths, cartridge_ids


def _process_prompt_tokens(
    model,
    tokenized: TokenizedConversation,
    cache: TrainableCache,
) -> None:
    """Process all prompt tokens through the model without attention capture.
    
    Args:
        model: The model to run forward pass on
        tokenized: Tokenized conversation with input_ids, seq_ids, position_ids
        cache: KV cache to populate
    """
    with torch.no_grad():
        model(
            input_ids=tokenized.input_ids,
            seq_ids=tokenized.seq_ids,
            position_ids=tokenized.position_ids,
            past_key_values=cache,
            use_cache=True,
            mode="generate",
            attention_capture=None,  # Don't capture attention for prompt processing
        )


def _generate_tokens_with_attention(
    model,
    tokenizer: PreTrainedTokenizerBase,
    tokenized: TokenizedConversation,
    cache: TrainableCache,
    stop_token_ids: List[int],
    max_tokens: Optional[int],
    layer_idxs: List[int],
) -> Tuple[List[int], List[AttentionRecord]]:
    """Generate tokens autoregressively while capturing attention.
    
    Args:
        model: The model to generate with
        tokenizer: Tokenizer for decoding tokens
        tokenized: Tokenized conversation
        cache: KV cache (already populated with prompt)
        stop_token_ids: List of token IDs that signal stop
        max_tokens: Maximum number of tokens to generate
        layer_idxs: List of layer indices to capture attention for
        
    Returns:
        Tuple of (generated_tokens, attention_records)
    """
    import os
    from time import perf_counter
    
    _PROFILE_ATTENTION = os.environ.get("PROFILE_ATTENTION", "0") == "1"
    
    resolved_device = model.device
    generated_tokens: List[int] = []
    all_attention_records: List[AttentionRecord] = []
    
    step = 0
    last_token_id = tokenized.input_ids[-1].item() if len(tokenized.input_ids) > 0 else None
    if last_token_id is None:
        raise RuntimeError("No tokens in prompt to generate from")
    
    current_position = tokenized.total_length - 1
    initial_cache_len = cache.num_tokens() if cache is not None else 0
    
    if _PROFILE_ATTENTION:
        print(f"[PROFILE] Starting decoding (initial_cache_len={initial_cache_len})")
    
    while True:
        # Check stopping conditions
        if max_tokens is not None and step >= max_tokens:
            break
        
        if _PROFILE_ATTENTION:
            t_step_start = perf_counter()
            current_cache_len = cache.num_tokens() if cache is not None else 0
            print(f"[PROFILE] === Decoding step {step + 1} (cache_len={current_cache_len}) ===")
        
        # Generate next token with attention capture
        capture_step = AttentionCapture(store_qkv=True, layers_to_capture=layer_idxs)
        with torch.no_grad():
            outputs = model(
                input_ids=torch.tensor([last_token_id], device=resolved_device, dtype=torch.long),
                seq_ids=torch.tensor([0], device=resolved_device, dtype=torch.long),
                position_ids=torch.tensor([current_position], device=resolved_device, dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
                mode="generate",
                attention_capture=capture_step,
            )
            
            # Get next token from logits
            logits = outputs.logits[0, -1, :]  # (vocab_size,)
            next_token = logits.argmax().item()
            
            # Store attention records from this step
            for record in capture_step.records:
                all_attention_records.append(record)
            generated_tokens.append(next_token)
            
            if _PROFILE_ATTENTION:
                t_step_end = perf_counter()
                print(f"[PROFILE] Decoding step {step + 1} total: {t_step_end - t_step_start:.4f}s")
            
            # Debug output: decode and print token information
            decoded_token = tokenizer.decode([next_token], skip_special_tokens=False)
            token_number = step + 1
            print(f"[Token {token_number}] Decoded: {repr(decoded_token)}")
            
            # Check if we should stop
            if next_token in stop_token_ids:
                break
            
            # Update for next iteration
            last_token_id = next_token
            current_position += 1
            step += 1
    
    if _PROFILE_ATTENTION:
        final_cache_len = cache.num_tokens() if cache is not None else 0
        print(f"[PROFILE] Finished decoding {step} tokens (final_cache_len={final_cache_len})")
    
    return generated_tokens, all_attention_records


def _aggregate_attention_by_layer(
    attention_records: List[AttentionRecord],
    tokenized: TokenizedConversation,
    cartridge_lengths: List[int],
    cartridge_ids: List[str],
    chunk_size: int = 128,
) -> List[LayerAttentionSummary]:
    """Aggregate attention records by layer and compute summaries.
    
    Args:
        attention_records: List of attention records from generation
        tokenized: Tokenized conversation with message spans
        cartridge_lengths: List of cartridge lengths
        cartridge_ids: List of cartridge IDs
        chunk_size: Chunk size for attention computation
        
    Returns:
        List of LayerAttentionSummary, one per layer
    """
    # Group records by layer
    records_by_layer: Dict[int, List[AttentionRecord]] = {}
    for record in attention_records:
        if record.layer_idx not in records_by_layer:
            records_by_layer[record.layer_idx] = []
        records_by_layer[record.layer_idx].append(record)
    
    layer_summaries: List[LayerAttentionSummary] = []
    
    for layer_idx, layer_records in records_by_layer.items():
        # Compute attention masses for each record, then average
        summaries = []
        
        for record in layer_records:
            summary = _compute_attention_masses(
                record,
                chunk_size=chunk_size,
                message_spans=tokenized.message_spans,
                cartridge_lengths=cartridge_lengths,
                cartridge_ids=cartridge_ids,
            )
            summaries.append(summary)
        
        # Average across all generated tokens
        if summaries:
            avg_summary = _average_layer_summaries(summaries)
            layer_summaries.append(avg_summary)
    
    return layer_summaries


def _average_layer_summaries(summaries: List[LayerAttentionSummary]) -> LayerAttentionSummary:
    """Combine multiple layer summaries into one.
    
    Each summary represents attention from one generated token.
    - tag_mass_means: averaged across all tokens
    - tag_mass_per_token: concatenated to preserve per-token values
    """
    if not summaries:
        raise ValueError("Cannot combine empty list of summaries")
    
    if len(summaries) == 1:
        return summaries[0]
    
    # All summaries should be for the same layer
    layer_idx = summaries[0].layer_idx
    mode = summaries[0].mode
    num_heads = summaries[0].num_heads
    cache_len = summaries[0].cache_len
    
    # Total sequence length is the sum of all individual seq_lens
    total_seq_len = sum(s.seq_len for s in summaries)
    
    # Collect all unique tags across all summaries
    all_tags = set()
    for s in summaries:
        all_tags.update(s.tag_mass_means.keys())
    
    # Combine per-tag data
    tag_mass_means: Dict[str, float] = {}
    tag_mass_per_token: Dict[str, List[float]] = {}
    
    for tag in all_tags:
        # Average tag mass means across all generated tokens
        tag_values = [s.tag_mass_means.get(tag, 0.0) for s in summaries]
        tag_mass_means[tag] = sum(tag_values) / len(summaries)
        
        # Concatenate per-token lists to preserve individual token attention values
        concatenated = []
        for s in summaries:
            concatenated.extend(s.tag_mass_per_token[tag])
    
        tag_mass_per_token[tag] = concatenated
    
    return LayerAttentionSummary(
        layer_idx=layer_idx,
        mode=mode,
        num_heads=num_heads,
        seq_len=total_seq_len,
        cache_len=cache_len,
        tag_mass_means=tag_mass_means,
        tag_mass_per_token=tag_mass_per_token,
    )


def capture_attention_for_messages(
    model,
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[ChatMessage],
    conversation_index: int,
    cartridges: Optional[List[Tuple[TrainableCache, str]]] = None,
    chunk_size: int = 128,
    max_tokens: Optional[int] = None,
    layer_idxs: List[int] = [],
) -> ConversationAttentionResult:
    """Capture attention weights for a conversation.
    
    Args:
        model: The model to use for generation
        tokenizer: Tokenizer for tokenization and decoding
        messages: Sequence of chat messages
        conversation_index: Index of this conversation
        cartridges: Optional list of (cache, cartridge_id) tuples
        chunk_size: Chunk size for attention computation
        max_tokens: Maximum number of tokens to generate
        layer_idxs: List of layer indices to capture attention for
        
    Returns:
        ConversationAttentionResult with attention summaries and metadata
    """
    timer = StepTimer()
    resolved_device = model.device
    
    # Determine EOS token ID
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else []
    if not isinstance(eos_token_id, list):
        eos_token_id = [eos_token_id]
    stop_token_ids = [int(tid) for tid in eos_token_id]
    
    with timer.track("tokenize_messages"):
        tokenized = tokenize_messages(
            tokenizer=tokenizer,
            messages=messages,
            device=resolved_device,
        )
    
    with timer.track("prepare_cache"):
        local_cache, cartridge_lengths, cartridge_ids = _prepare_cache_and_cartridges(
            cartridges=cartridges,
            model=model,
            device=resolved_device,
        )
    
    with timer.track("process_prompt"):
        _process_prompt_tokens(
            model=model,
            tokenized=tokenized,
            cache=local_cache,
        )
    
    with timer.track("generate_tokens"):
        generated_tokens, attention_records = _generate_tokens_with_attention(
            model=model,
            tokenizer=tokenizer,
            tokenized=tokenized,
            cache=local_cache,
            stop_token_ids=stop_token_ids,
            max_tokens=max_tokens,
            layer_idxs=layer_idxs,
        )
    
    if local_cache is not None:
        with timer.track("cache.clear_after_forward"):
            local_cache.clear()
    
    with timer.track("aggregate_attention"):
        layer_summaries = _aggregate_attention_by_layer(
            attention_records=attention_records,
            tokenized=tokenized,
            cartridge_lengths=cartridge_lengths,
            cartridge_ids=cartridge_ids,
            chunk_size=chunk_size,
        )
    
    # Build result
    conversation_summary = ConversationAttentionSummary(
        conversation_index=conversation_index,
        messages=[msg.as_dict() for msg in messages],
        layers=layer_summaries,
    )
    
    # Store generated tokens in tensors for later retrieval
    result = ConversationAttentionResult(
        summary=conversation_summary,
        tensors={"generated_tokens": {"tokens": torch.tensor(generated_tokens, device=resolved_device)}},  # Store generated tokens
        token_metadata=tokenized,
        timings=timer.as_dict(),
    )
    
    return result


def save_attention_results(
    result: ConversationAttentionResult,
    output_path: Path,
    tokenizer: PreTrainedTokenizerBase,
    model_name: Optional[str] = None,
) -> None:
    """Save attention results to JSON file.
    
    Args:
        result: ConversationAttentionResult to save
        output_path: Path to save JSON file to
        tokenizer: Tokenizer for decoding generated tokens
        model_name: Optional model name to include in metadata
    """
    # Extract generated tokens from tensors
    generated_token_ids = []
    decoded_tokens = []
    
    if "generated_tokens" in result.tensors and "tokens" in result.tensors["generated_tokens"]:
        generated_token_ids = result.tensors["generated_tokens"]["tokens"].cpu().tolist()
        decoded_tokens = [
            tokenizer.decode([token_id], skip_special_tokens=False)
            for token_id in generated_token_ids
        ]
    
    # Serialize message spans
    message_spans = [
        {
            "start": span.start,
            "end": span.end,
            "tag": span.tag,
            "role": span.role,
            "message_index": span.message_index,
        }
        for span in result.token_metadata.message_spans
    ]
    
    # Serialize layer summaries
    layers = []
    for layer_summary in result.summary.layers:
        layers.append({
            "layer_idx": layer_summary.layer_idx,
            "num_heads": layer_summary.num_heads,
            "seq_len": layer_summary.seq_len,
            "cache_len": layer_summary.cache_len,
            "tag_mass_means": layer_summary.tag_mass_means,
            "tag_mass_per_token": layer_summary.tag_mass_per_token,
        })
    
    # Extract cartridge IDs from tags (look for cartridge_* tags)
    cartridge_ids = []
    if result.summary.layers:
        for tag in result.summary.layers[0].tag_mass_means.keys():
            if tag.startswith("cartridge_"):
                cartridge_id = tag[len("cartridge_"):]
                if cartridge_id not in cartridge_ids:
                    cartridge_ids.append(cartridge_id)
    
    # Build output dictionary
    output_dict = {
        "conversation_index": result.summary.conversation_index,
        "metadata": {
            "num_layers": len(result.summary.layers),
            "num_generated_tokens": len(generated_token_ids),
            "cartridge_ids": cartridge_ids,
            "model_name": model_name,
        },
        "generated_tokens": {
            "token_ids": generated_token_ids,
            "decoded": decoded_tokens,
        },
        "message_spans": message_spans,
        "layers": layers,
        "timings": result.timings,
    }
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_dict, f, indent=2)

