from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch


@dataclass
class ChatMessage:
    """Chat message with optional semantic tag to track origin (e.g. memory_0)."""

    role: str
    content: str
    tag: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class TokenSpan:
    """Token span corresponding to a single message or template fragment."""

    start: int
    end: int
    message_index: Optional[int] = None
    tag: Optional[str] = None
    role: Optional[str] = None

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)


@dataclass
class TokenizedConversation:
    """Tokenized representation of a chat conversation with span metadata."""

    input_ids: torch.Tensor
    seq_ids: torch.Tensor
    position_ids: torch.Tensor
    message_spans: Sequence[TokenSpan]
    total_length: int
    timings: Dict[str, float] = field(default_factory=dict)


@dataclass
class LayerAttentionSummary:
    """Aggregated attention statistics for a single decoder layer."""

    layer_idx: int
    mode: str
    num_heads: int
    seq_len: int
    cache_len: int
    cartridge_mass_mean: float
    normal_mass_mean: float
    cartridge_mass_per_head: List[float]
    cartridge_mass_per_token: List[float]


@dataclass
class ConversationAttentionSummary:
    """High-level summary for one conversation forward pass."""

    conversation_index: int
    messages: Sequence[Dict[str, Any]]
    layers: Sequence[LayerAttentionSummary]

    @property
    def num_layers(self) -> int:
        return len(self.layers)


@dataclass
class ConversationAttentionResult:
    """Attention capture outputs for a conversation."""

    summary: ConversationAttentionSummary
    tensors: Dict[str, Dict[str, torch.Tensor]]
    token_metadata: TokenizedConversation
    timings: Dict[str, float] = field(default_factory=dict)

