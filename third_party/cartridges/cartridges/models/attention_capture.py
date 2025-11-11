from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal

import torch


@dataclass
class AttentionRecord:
    layer_idx: int
    mode: Literal["train", "generate"]
    query: Optional[torch.Tensor]
    key: Optional[torch.Tensor]
    value: Optional[torch.Tensor]
    seq_ids: Optional[torch.Tensor]
    kv_seq_ids: Optional[torch.Tensor]
    cache_len: int
    scaling: Optional[float]
    enable_gqa: bool
    has_block_mask: bool
    num_query_heads: int
    num_key_heads: int
    head_dim: Optional[int]


class AttentionCapture:
    """Utility for recording attention inputs during a forward pass."""

    def __init__(
        self,
        store_qkv: bool = True,
        move_to_cpu: bool = True,
    ) -> None:
        self.store_qkv = store_qkv
        self.move_to_cpu = move_to_cpu
        self._records: List[AttentionRecord] = []

    @property
    def records(self) -> List[AttentionRecord]:
        return self._records

    def _process_tensor(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        processed = tensor.detach()
        if processed.dim() > 0 and processed.shape[0] == 1:
            processed = processed.squeeze(0)
        if self.move_to_cpu:
            # Move to CPU synchronously to avoid graph compilation issues
            processed = processed.to("cpu", non_blocking=False)
        return processed

    def record(
        self,
        *,
        layer_idx: int,
        mode: Literal["train", "generate"],
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        seq_ids: Optional[torch.Tensor],
        kv_seq_ids: Optional[torch.Tensor],
        cache_len: int,
        scaling: Optional[float],
        enable_gqa: bool,
        has_block_mask: bool,
    ) -> None:
        stored_query: Optional[torch.Tensor]
        stored_key: Optional[torch.Tensor]
        stored_value: Optional[torch.Tensor]
        if self.store_qkv:
            stored_query = self._process_tensor(query)
            stored_key = self._process_tensor(key)
            stored_value = self._process_tensor(value)
        else:
            stored_query = None
            stored_key = None
            stored_value = None

        record = AttentionRecord(
            layer_idx=layer_idx,
            mode=mode,
            query=stored_query,
            key=stored_key,
            value=stored_value,
            seq_ids=self._process_tensor(seq_ids),
            kv_seq_ids=self._process_tensor(kv_seq_ids),
            cache_len=cache_len,
            scaling=float(scaling) if scaling is not None else None,
            enable_gqa=enable_gqa,
            has_block_mask=has_block_mask,
            num_query_heads=query.shape[1],
            num_key_heads=key.shape[1],
            head_dim=query.shape[-1] if query.dim() > 0 else None,
        )
        self._records.append(record)


__all__ = ["AttentionCapture", "AttentionRecord"]

