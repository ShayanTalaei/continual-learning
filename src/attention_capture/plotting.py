from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

from .types import ConversationAttentionResult, TokenSpan


def aggregate_attention_by_span(
    result: ConversationAttentionResult,
    layer_idx: int,
    query_span_tag: str = "current_observation",
) -> Dict[str, float]:
    tensors = result.tensors.get(str(layer_idx))
    if tensors is None or "weights" not in tensors:
        raise ValueError(
            f"Attention weights not available for layer {layer_idx}. Set --save-attention-weights when capturing."
        )
    weights = tensors["weights"].float()  # (heads, query_len, key_len)
    spans: Sequence[TokenSpan] = result.token_metadata.message_spans

    query_span = next((span for span in spans if span.tag == query_span_tag), None)
    if query_span is None or query_span.length == 0:
        raise ValueError(f"Query span with tag '{query_span_tag}' not found or empty.")

    query_slice = slice(query_span.start, query_span.end)
    weights_query = weights[:, query_slice, :]  # (heads, query_len, key_len)

    span_masses: Dict[str, float] = {}
    for span in spans:
        tag = span.tag or f"span_{span.start}_{span.end}"
        if span.length == 0:
            span_masses[tag] = 0.0
            continue
        key_slice = slice(span.start, span.end)
        sub_weights = weights_query[:, :, key_slice]
        # Sum over key tokens, mean over heads and query tokens
        mass = sub_weights.sum(dim=-1).mean().item()
        span_masses[tag] = mass
    return span_masses


def build_attention_matrix(
    results: Sequence[ConversationAttentionResult],
    layer_idx: int,
    query_span_tag: str = "current_observation",
) -> Tuple[List[str], List[str], np.ndarray]:
    # Determine column order from first result, then extend with unseen tags.
    column_tags: List[str] = []
    for result in results:
        for span in result.token_metadata.message_spans:
            tag = span.tag or f"span_{span.start}_{span.end}"
            if tag not in column_tags:
                column_tags.append(tag)

    matrix = np.full((len(results), len(column_tags)), np.nan, dtype=np.float32)
    row_labels: List[str] = []

    for row_idx, result in enumerate(results):
        row_labels.append(f"conv_{result.summary.conversation_index:04d}")
        masses = aggregate_attention_by_span(
            result,
            layer_idx=layer_idx,
            query_span_tag=query_span_tag,
        )
        for col_idx, tag in enumerate(column_tags):
            if tag in masses:
                matrix[row_idx, col_idx] = masses[tag]

    return row_labels, column_tags, matrix


def plot_attention_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    output_path: Union[str, Path],
    title: str = "",
    cmap: str = "magma",
) -> None:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.ma.masked_invalid(matrix)
    vmax = np.nanmax(matrix)
    if not np.isfinite(vmax):
        vmax = 1.0

    fig_width = max(8.0, 0.6 * len(column_labels))
    fig_height = max(4.0, 0.4 * len(row_labels))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=vmax)

    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(column_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Context spans")
    ax.set_ylabel("Validation examples")
    if title:
        ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Attention mass (avg over heads/query tokens)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_attention_matrix(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    output_path: Union[str, Path],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rows": list(row_labels),
        "columns": list(column_labels),
        "matrix": matrix.tolist(),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

