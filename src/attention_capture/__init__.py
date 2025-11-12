"""Reusable utilities for capturing and analysing attention distributions."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap_cartridges() -> None:
    try:
        import cartridges  # type: ignore # noqa: F401
        return
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[2]
        cartridges_dir = repo_root / "third_party" / "cartridges"
        tokasaurus_dir = repo_root / "third_party" / "tokasaurus"

        for path in (cartridges_dir, tokasaurus_dir):
            if path.exists():
                str_path = str(path)
                if str_path not in sys.path:
                    sys.path.insert(0, str_path)

        os.environ.setdefault("CARTRIDGES_DIR", str(cartridges_dir))
        os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", str(repo_root / "outputs"))

        import cartridges  # type: ignore # noqa: F401


_bootstrap_cartridges()

from .types import (
    ChatMessage,
    TokenSpan,
    TokenizedConversation,
    LayerAttentionSummary,
    ConversationAttentionSummary,
    ConversationAttentionResult,
)
from .pipeline import (
    load_model,
    load_tokenizer,
    tokenize_messages,
    capture_attention_for_messages,
)

__all__ = [
    "ChatMessage",
    "TokenSpan",
    "TokenizedConversation",
    "LayerAttentionSummary",
    "ConversationAttentionSummary",
    "ConversationAttentionResult",
    "load_model",
    "load_tokenizer",
    "tokenize_messages",
    "capture_attention_for_messages",
]