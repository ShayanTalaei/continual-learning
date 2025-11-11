from __future__ import annotations

import json
import math
from bisect import bisect_right
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding

from cartridges.cache import TrainableCache
from cartridges.models.attention_capture import AttentionCapture, AttentionRecord

from .types import (
    ChatMessage,
    TokenSpan,
    TokenizedConversation,
    LayerAttentionSummary,
    ConversationAttentionSummary,
    ConversationAttentionResult,
)

MODEL_LOADERS: Dict[str, str] = {
    "llama": "cartridges.models.llama.modeling_llama.FlexLlamaForCausalLM",
    "qwen": "cartridges.models.qwen.modeling_qwen3.FlexQwen3ForCausalLM",
}


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


def import_model_class(dotted_path: str):
    module_name, class_name = dotted_path.rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def _resolve_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if isinstance(dtype, str):
        key = dtype.lower()
        if key in dtype_map:
            return dtype_map[key]
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_model(
    model_type: str,
    model_name: str,
    device: Union[str, torch.device],
    dtype: Union[str, torch.dtype] = torch.bfloat16,
):
    cls_path = MODEL_LOADERS[model_type]
    model_cls = import_model_class(cls_path)
    resolved_device = torch.device(device)
    resolved_dtype = _resolve_dtype(dtype)
    model = model_cls.from_pretrained(model_name)
    model = model.to(device=resolved_device, dtype=resolved_dtype)
    model.eval()
    return model


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name)


def load_cache(
    cache_path: str,
    device: Union[str, torch.device],
    dtype: Union[str, torch.dtype],
) -> TrainableCache:
    resolved_device = torch.device(device)
    resolved_dtype = _resolve_dtype(dtype)
    cache = TrainableCache.from_pretrained(
        cache_path,
        device=resolved_device.type if resolved_device.type != "cuda" else None,
    )
    cache = cache.to(device=resolved_device, dtype=resolved_dtype)
    cache.eval()
    return cache


def _prefix_char_lengths(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[ChatMessage],
) -> List[int]:
    message_dicts = [m.as_dict() for m in messages]
    lengths: List[int] = []
    for idx in range(len(message_dicts)):
        prefix = tokenizer.apply_chat_template(
            message_dicts[: idx + 1],
            tokenize=False,
            add_generation_prompt=False,
        )
        lengths.append(len(prefix))
    return lengths


def tokenize_messages(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[ChatMessage],
    device: Union[str, torch.device],
    add_generation_prompt: bool,
) -> TokenizedConversation:
    timer = StepTimer()
    resolved_device = torch.device(device)
    with timer.track("build_message_dicts"):
        message_dicts = [m.as_dict() for m in messages]
    prefix_token_lengths: List[int] = []
    tokenizer_is_fast = getattr(tokenizer, "is_fast", False)
    tokenized: torch.Tensor

    conversation_text: Optional[str] = None
    if tokenizer_is_fast:
        with timer.track("fast.compute_prefix_char_lengths"):
            prefix_char_lengths = _prefix_char_lengths(tokenizer, messages)
        with timer.track("fast.render_without_prompt"):
            rendered = tokenizer.apply_chat_template(
                message_dicts,
                tokenize=False,
                add_generation_prompt=False,
            )
        if not isinstance(rendered, str):
            tokenizer_is_fast = False
            prefix_token_lengths = []
        else:
            conversation_text = rendered
            with timer.track("fast.tokenize_without_prompt"):
                encoding = cast(
                    BatchEncoding,
                    tokenizer(
                        conversation_text,
                        add_special_tokens=False,
                        return_offsets_mapping=True,
                    ),
                )
            offset_mapping = encoding.get("offset_mapping")
            if offset_mapping is None:
                tokenizer_is_fast = False
                prefix_token_lengths = []
            else:
                with timer.track("fast.process_offsets"):
                    raw_offsets = offset_mapping
                    if raw_offsets and isinstance(raw_offsets[0], list):
                        raw_offsets = raw_offsets[0]
                    normalized_offsets: List[Tuple[int, int]] = []
                    for start, end in raw_offsets:
                        normalized_offsets.append((int(start), int(end)))

                    cumulative_char_ends: List[int] = []
                    running_end = 0
                    for start, end in normalized_offsets:
                        if start < 0 or end < 0:
                            cumulative_char_ends.append(running_end)
                            continue
                        running_end = max(running_end, end)
                        cumulative_char_ends.append(running_end)

                    prefix_token_lengths = [
                        bisect_right(cumulative_char_ends, char_len)
                        for char_len in prefix_char_lengths
                    ]

                if add_generation_prompt:
                    with timer.track("fast.render_with_prompt"):
                        rendered_with_prompt = tokenizer.apply_chat_template(
                            message_dicts,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    if not isinstance(rendered_with_prompt, str):
                        tokenizer_is_fast = False
                        prefix_token_lengths = []
                    else:
                        full_text = rendered_with_prompt
                        with timer.track("fast.tokenize_with_prompt"):
                            full_encoding = cast(
                                BatchEncoding,
                                tokenizer(
                                    full_text,
                                    add_special_tokens=False,
                                    return_tensors="pt",
                                ),
                            )
                        tokenized = cast(torch.Tensor, full_encoding["input_ids"])
                else:
                    if conversation_text is None:
                        tokenizer_is_fast = False
                        prefix_token_lengths = []
                    else:
                        with timer.track("fast.tokenize_without_prompt_tensor"):
                            tensor_encoding = cast(
                                BatchEncoding,
                                tokenizer(
                                    conversation_text,
                                    add_special_tokens=False,
                                    return_tensors="pt",
                                ),
                            )
                        tokenized = cast(torch.Tensor, tensor_encoding["input_ids"])

    if not tokenizer_is_fast:
        with timer.track("slow.compute_prefix_token_lengths"):
            for idx in range(len(message_dicts)):
                prefix_tokens = tokenizer.apply_chat_template(
                    message_dicts[: idx + 1],
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors=None,
                )
                prefix_token_lengths.append(len(prefix_tokens))
        with timer.track("slow.tokenize_full"):
            tokenized = cast(
                torch.Tensor,
                tokenizer.apply_chat_template(
                    message_dicts,
                    tokenize=True,
                    add_generation_prompt=add_generation_prompt,
                    return_tensors="pt",
                ),
            )

    with timer.track("prepare_token_metadata"):
        if tokenized.dim() > 1:
            tokenized = tokenized.squeeze(0)
        input_ids = tokenized.to(resolved_device)
        total_length = int(input_ids.shape[0])
        seq_ids = torch.zeros(total_length, dtype=torch.long, device=resolved_device)
        position_ids = torch.arange(total_length, dtype=torch.long, device=resolved_device)

        spans: List[TokenSpan] = []
        prev = 0
        for idx, prefix_len in enumerate(prefix_token_lengths):
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


def _compute_attention_masses(
    record: AttentionRecord,
    save_attention_weights: bool,
    chunk_size: int = 128,
) -> Tuple[LayerAttentionSummary, Dict[str, torch.Tensor]]:
    if record.query is None or record.key is None or record.seq_ids is None:
        raise RuntimeError(
            "Attention capture record is missing tensors; ensure AttentionCapture.store_qkv is True."
        )
    if save_attention_weights:
        raise NotImplementedError("Saving full attention weights is not supported in chunked capture mode.")

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
    cartridge_mask = kv_seq_ids_device.eq(-1)

    cartridge_mass_per_head_acc = torch.zeros(num_heads, dtype=torch.float64, device=device)
    cartridge_mass_per_token_acc = torch.zeros(seq_len, dtype=torch.float64, device=device)
    total_cartridge_mass = torch.zeros(1, dtype=torch.float64, device=device)
    total_normal_mass = torch.zeros(1, dtype=torch.float64, device=device)

    key_t = key.transpose(-1, -2)

    for q_start in range(0, seq_len, chunk_size):
        q_end = min(seq_len, q_start + chunk_size)
        q_chunk = query[:, q_start:q_end, :]
        seq_ids_chunk = seq_ids_device[q_start:q_end]
        q_positions_chunk = torch.arange(q_start, q_end, dtype=torch.long, device=device)

        same_sequence = kv_seq_ids_device.unsqueeze(0).eq(seq_ids_chunk.unsqueeze(1))
        causal_mask = kv_positions.unsqueeze(0) <= (q_positions_chunk + record.cache_len).unsqueeze(1)
        allowed = cartridge_mask.unsqueeze(0) | (same_sequence & causal_mask)

        scores = torch.matmul(q_chunk, key_t) * scaling
        scores = scores.masked_fill(~allowed.unsqueeze(0), -1e9)
        weights = torch.softmax(scores, dim=-1)

        cartridge_weights = weights * cartridge_mask.to(dtype=weights.dtype).unsqueeze(0).unsqueeze(0)
        cartridge_mass_chunk = cartridge_weights.sum(dim=-1)
        normal_mass_chunk = weights.sum(dim=-1) - cartridge_mass_chunk

        cartridge_mass_per_head_acc += cartridge_mass_chunk.sum(dim=1).to(dtype=torch.float64)
        cartridge_mass_per_token_acc[q_start:q_end] += cartridge_mass_chunk.sum(dim=0).to(dtype=torch.float64)
        total_cartridge_mass += cartridge_mass_chunk.sum().to(dtype=torch.float64)
        total_normal_mass += normal_mass_chunk.sum().to(dtype=torch.float64)

    cartridge_mass_mean = (total_cartridge_mass / (num_heads * seq_len)).item()
    normal_mass_mean = (total_normal_mass / (num_heads * seq_len)).item()
    cartridge_mass_per_head = (cartridge_mass_per_head_acc / seq_len).cpu().to(dtype=torch.float32)
    cartridge_mass_per_token = (cartridge_mass_per_token_acc / num_heads).cpu().to(dtype=torch.float32)

    summary = LayerAttentionSummary(
        layer_idx=record.layer_idx,
        mode=record.mode,
        num_heads=num_heads,
        seq_len=seq_len,
        cache_len=record.cache_len,
        cartridge_mass_mean=cartridge_mass_mean,
        normal_mass_mean=normal_mass_mean,
        cartridge_mass_per_head=cartridge_mass_per_head.tolist(),
        cartridge_mass_per_token=cartridge_mass_per_token.tolist(),
    )

    tensors_to_save: Dict[str, torch.Tensor] = {
        "cartridge_mass_per_head": cartridge_mass_per_head,
        "cartridge_mass_per_token": cartridge_mass_per_token,
        "kv_seq_ids": kv_seq_ids.cpu(),
        "seq_ids": seq_ids.cpu(),
    }
    if record.query is not None:
        tensors_to_save["query"] = record.query.cpu()
    if record.key is not None:
        tensors_to_save["key"] = record.key.cpu()
    if record.value is not None:
        tensors_to_save["value"] = record.value.cpu()

    return summary, tensors_to_save


def capture_attention_for_messages(
    model,
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[ChatMessage],
    conversation_index: int,
    device: Union[str, torch.device],
    mode: str,
    cache: Optional[TrainableCache] = None,
    add_generation_prompt: bool = True,
    save_qkv: bool = False,
    save_attention_weights: bool = False,
    chunk_size: int = 128,
) -> ConversationAttentionResult:
    timer = StepTimer()
    resolved_device = torch.device(device)
    with timer.track("tokenize_messages"):
        tokenized = tokenize_messages(
            tokenizer=tokenizer,
            messages=messages,
            device=resolved_device,
            add_generation_prompt=add_generation_prompt,
        )

    capture = AttentionCapture(store_qkv=True)
    local_cache = cache
    if local_cache is not None:
        with timer.track("cache.clear_before_forward"):
            local_cache.clear()

    with timer.track("model_forward"):
        with torch.no_grad():
            model(
                input_ids=tokenized.input_ids,
                seq_ids=tokenized.seq_ids,
                position_ids=tokenized.position_ids,
                past_key_values=local_cache,
                use_cache=local_cache is not None,
                mode=mode,
                attention_capture=capture,
            )

    if local_cache is not None:
        with timer.track("cache.clear_after_forward"):
            local_cache.clear()

    with timer.track("aggregate_attention"):
        layer_summaries: List[LayerAttentionSummary] = []
        tensors: Dict[str, Dict[str, torch.Tensor]] = {}
        for record in capture.records:
            summary, tensor_payload = _compute_attention_masses(
                record,
                save_attention_weights=save_attention_weights,
                chunk_size=chunk_size,
            )
            layer_summaries.append(summary)
            if save_qkv or save_attention_weights:
                tensors[str(summary.layer_idx)] = tensor_payload

    with timer.track("prepare_summary"):
        message_payload = [
            {
                "role": msg.role,
                "content": msg.content,
                "tag": msg.tag,
                "metadata": msg.metadata,
            }
            for msg in messages
        ]
        summary = ConversationAttentionSummary(
            conversation_index=conversation_index,
            messages=message_payload,
            layers=layer_summaries,
        )

    timings = timer.as_dict()
    if tokenized.timings:
        timings.update({f"tokenize.{k}": v for k, v in tokenized.timings.items()})

    return ConversationAttentionResult(
        summary=summary,
        tensors=tensors,
        token_metadata=tokenized,
        timings=timings,
    )


def _summary_to_dict(summary: ConversationAttentionSummary) -> Dict[str, object]:
    return {
        "conversation_index": summary.conversation_index,
        "num_layers": summary.num_layers,
        "messages": summary.messages,
        "layers": [
            {
                "layer_idx": layer.layer_idx,
                "mode": layer.mode,
                "num_heads": layer.num_heads,
                "seq_len": layer.seq_len,
                "cache_len": layer.cache_len,
                "cartridge_mass_mean": layer.cartridge_mass_mean,
                "normal_mass_mean": layer.normal_mass_mean,
                "cartridge_mass_per_head": layer.cartridge_mass_per_head,
                "cartridge_mass_per_token": layer.cartridge_mass_per_token,
            }
            for layer in summary.layers
        ],
    }


def _token_metadata_to_dict(metadata: TokenizedConversation) -> Dict[str, object]:
    return {
        "total_length": metadata.total_length,
        "timings": metadata.timings,
        "message_spans": [
            {
                "start": span.start,
                "end": span.end,
                "length": span.length,
                "message_index": span.message_index,
                "tag": span.tag,
                "role": span.role,
            }
            for span in metadata.message_spans
        ],
    }


def write_conversation_outputs(
    output_dir: Path,
    result: ConversationAttentionResult,
    save_qkv: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    idx = result.summary.conversation_index
    summary_path = output_dir / f"conversation_{idx:04d}.json"
    payload = _summary_to_dict(result.summary)
    payload["token_metadata"] = _token_metadata_to_dict(result.token_metadata)
    if result.timings:
        payload["timings"] = result.timings
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if save_qkv and result.tensors:
        tensor_path = output_dir / f"conversation_{idx:04d}.pt"
        torch.save(result.tensors, tensor_path)


