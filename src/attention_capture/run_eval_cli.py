from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from tqdm import tqdm

from .agent_runner import (
    HistoryPromptBuilder,
    build_validation_dataset,
    iter_validation_prompts,
    load_history_memory,
    load_run_configuration,
    prepare_conversation_messages,
    resolve_system_prompt,
)
from .pipeline import (
    StepTimer,
    capture_attention_for_messages,
    load_cache,
    load_model,
    load_tokenizer,
    write_conversation_outputs,
)
from .plotting import (
    aggregate_attention_by_span,
    build_attention_matrix,
    plot_attention_heatmap,
    save_attention_matrix,
)
from .types import (
    ConversationAttentionResult,
    ConversationAttentionSummary,
    LayerAttentionSummary,
    TokenSpan,
    TokenizedConversation,
)
from src.agent.history_agent import HistoryAgentConfig


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run attention evaluation over validation prompts using a checkpointed history agent.",
    )
    parser.add_argument("--config", required=True, help="Path to run configuration YAML.")
    parser.add_argument(
        "--memory-snapshot",
        required=True,
        help="Path to the HistoryList snapshot to seed the agent memory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write summaries, tensors, and plots.",
    )
    parser.add_argument(
        "--model-type",
        choices=["llama", "qwen"],
        default="llama",
        help="Model family identifier.",
    )
    parser.add_argument(
        "--model-name",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF model name used for attention capture.",
    )
    parser.add_argument(
        "--cartridge-path",
        default=None,
        help="Optional TrainableCache path (kv_cache.torch) to preload.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string (e.g. cuda, cuda:1, cpu).",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Computation dtype for model + cache.",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "generate"],
        default="generate",
        help="Forward pass mode for the cartridges model.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of validation prompts to process.",
    )
    parser.add_argument(
        "--save-qkv",
        action="store_true",
        help="Persist captured Q/K/V tensors.",
    )
    parser.add_argument(
        "--save-attention-weights",
        action="store_true",
        help="Deprecated: full attention weight tensors are not saved in chunked capture mode.",
    )
    parser.add_argument(
        "--query-span-tag",
        default="current_observation",
        help="Message tag treated as the querying tokens when aggregating attention.",
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=-1,
        help="Decoder layer index for aggregation/plotting (-1 selects the last layer).",
    )
    parser.add_argument(
        "--add-generation-prompt",
        dest="add_generation_prompt",
        action="store_true",
        help="Append the generation prompt when tokenising.",
    )
    parser.add_argument(
        "--no-generation-prompt",
        dest="add_generation_prompt",
        action="store_false",
        help="Skip adding generation prompt tokens.",
    )
    parser.set_defaults(add_generation_prompt=True)
    return parser.parse_args(argv)


def _select_layer_index(results: List[ConversationAttentionResult], layer_idx: int) -> int:
    if not results:
        raise ValueError("No attention results available.")
    available = results[0].summary.num_layers
    resolved = layer_idx + available if layer_idx < 0 else layer_idx
    if resolved < 0 or resolved >= available:
        raise ValueError(f"Layer index {layer_idx} out of range (0..{available - 1}).")
    return resolved


def _extract_result_label(summary: ConversationAttentionSummary) -> str:
    for message in reversed(summary.messages):
        metadata = message.get("metadata") if isinstance(message, dict) else None
        if isinstance(metadata, dict):
            env_id = metadata.get("env_id")
            if env_id:
                return str(env_id)
    return ""


def _load_existing_results(
    output_dir: Path,
) -> Tuple[List[ConversationAttentionResult], List[str], Set[int]]:
    results: List[ConversationAttentionResult] = []
    labels: List[str] = []
    processed: Set[int] = set()

    for json_path in sorted(output_dir.glob("conversation_*.json")):
        stem_parts = json_path.stem.split("_")
        if len(stem_parts) < 2:
            continue
        try:
            idx = int(stem_parts[1])
        except ValueError:
            continue

        payload = json.loads(json_path.read_text(encoding="utf-8"))

        layers_payload = payload.get("layers", [])
        layers = [
            LayerAttentionSummary(
                layer_idx=int(layer["layer_idx"]),
                mode=str(layer.get("mode", "")),
                num_heads=int(layer.get("num_heads", 0)),
                seq_len=int(layer.get("seq_len", 0)),
                cache_len=int(layer.get("cache_len", 0)),
                cartridge_mass_mean=float(layer.get("cartridge_mass_mean", 0.0)),
                normal_mass_mean=float(layer.get("normal_mass_mean", 0.0)),
                cartridge_mass_per_head=list(layer.get("cartridge_mass_per_head", [])),
                cartridge_mass_per_token=list(layer.get("cartridge_mass_per_token", [])),
            )
            for layer in layers_payload
        ]

        summary = ConversationAttentionSummary(
            conversation_index=int(payload["conversation_index"]),
            messages=payload.get("messages", []),
            layers=layers,
        )

        token_metadata_payload = payload.get("token_metadata", {}) or {}
        total_length = int(token_metadata_payload.get("total_length", 0))
        span_payloads = token_metadata_payload.get("message_spans", []) or []
        spans = [
            TokenSpan(
                start=int(span.get("start", 0)),
                end=int(span.get("end", 0)),
                message_index=span.get("message_index"),
                tag=span.get("tag"),
                role=span.get("role"),
            )
            for span in span_payloads
        ]

        zero_tensor = torch.zeros(total_length, dtype=torch.long)
        token_metadata = TokenizedConversation(
            input_ids=zero_tensor.clone(),
            seq_ids=zero_tensor.clone(),
            position_ids=torch.arange(total_length, dtype=torch.long),
            message_spans=spans,
            total_length=total_length,
            timings=token_metadata_payload.get("timings", {}),
        )

        tensor_path = json_path.with_suffix(".pt")
        tensors: Dict[str, Dict[str, torch.Tensor]] = {}
        if tensor_path.exists():
            tensors = torch.load(tensor_path, map_location="cpu")

        result = ConversationAttentionResult(
            summary=summary,
            tensors=tensors,
            token_metadata=token_metadata,
            timings=payload.get("timings", {}),
        )

        results.append(result)
        labels.append(_extract_result_label(summary))
        processed.add(idx)

    return results, labels, processed


def main(argv: List[str] | None = None) -> None:
    timer = StepTimer()

    with timer.track("parse_args"):
        args = parse_args(argv)

    with timer.track("load_run_configuration"):
        run_conf, raw_conf = load_run_configuration(args.config)

    with timer.track("prepare_agent"):
        agent_conf = HistoryAgentConfig(**run_conf.agent)
        system_prompt = resolve_system_prompt(agent_conf.system_prompt or "")
        history_memory = load_history_memory(agent_conf.memory_config, args.memory_snapshot)
        builder = HistoryPromptBuilder(agent_conf)

    with timer.track("build_validation_dataset"):
        dataset = build_validation_dataset(run_conf.validation_dataset)
    if dataset is None:
        raise ValueError("Run configuration does not define a validation dataset.")

    dataset_items = dataset.get_dataset()
    dataset_size = len(dataset_items)
    max_samples = args.max_samples if args.max_samples is not None else dataset_size
    sample_limit = min(dataset_size, max_samples)

    with timer.track("load_model_and_tokenizer"):
        device = torch.device(args.device)
        model = load_model(args.model_type, args.model_name, device=device, dtype=args.dtype)
        tokenizer = load_tokenizer(args.model_name)

    cache = None
    if args.cartridge_path:
        with timer.track("load_cache"):
            cache = load_cache(args.cartridge_path, device=device, dtype=args.dtype)

    with timer.track("prepare_output_dir"):
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    with timer.track("write_run_metadata"):
        metadata = {
            "config_path": str(Path(args.config).resolve()),
            "memory_snapshot": str(Path(args.memory_snapshot).resolve()),
            "model_type": args.model_type,
            "model_name": args.model_name,
            "cartridge_path": args.cartridge_path,
            "dtype": args.dtype,
            "device": str(device),
            "mode": args.mode,
            "add_generation_prompt": args.add_generation_prompt,
            "max_samples": args.max_samples,
            "save_qkv": args.save_qkv,
            "save_attention_weights": args.save_attention_weights,
            "query_span_tag": args.query_span_tag,
            "layer_idx": args.layer_idx,
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    with timer.track("recall_history_entries"):
        history_entries = list(history_memory.recall())

    with timer.track("load_existing_results"):
        existing_results, existing_labels, processed_indices = _load_existing_results(output_dir)

    processed_indices = {idx for idx in processed_indices if idx < sample_limit}
    existing_pairs = [
        (result, label)
        for result, label in zip(existing_results, existing_labels)
        if result.summary.conversation_index < sample_limit
    ]
    existing_pairs.sort(key=lambda pair: pair[0].summary.conversation_index)

    results: List[ConversationAttentionResult] = [pair[0] for pair in existing_pairs]
    convo_labels: List[str] = [pair[1] for pair in existing_pairs]

    save_qkv = args.save_qkv
    save_attention_weights = args.save_attention_weights

    with timer.track("process_validation_prompts"):
        progress = tqdm(
            total=sample_limit,
            initial=min(len(processed_indices), sample_limit),
            desc="conversations",
            unit="conv",
        )
        for idx, (env, obs) in enumerate(iter_validation_prompts(dataset)):
            if idx >= sample_limit:
                break
            if idx in processed_indices:
                continue

            print(f"[run_eval] Processing conversation {idx} env={env.env_id}")

            conversation_step = f"conversation_{idx}"
            with timer.track(f"{conversation_step}.build_history_messages"):
                history_messages = builder.build_history_messages(obs, history_entries)
                if history_messages:
                    history_messages[-1].metadata.update(
                        {
                            "env_id": env.env_id,
                            "env_type": env.env_type,
                            "question": obs,
                        }
                    )
                conversation = prepare_conversation_messages(system_prompt, history_messages)

            with timer.track(f"{conversation_step}.capture_attention"):
                result = capture_attention_for_messages(
                    model=model,
                    tokenizer=tokenizer,
                    messages=conversation,
                    conversation_index=idx,
                    device=device,
                    mode=args.mode,
                    cache=cache,
                    add_generation_prompt=args.add_generation_prompt,
                    save_qkv=save_qkv,
                    save_attention_weights=save_attention_weights,
                )

            with timer.track(f"{conversation_step}.write_outputs"):
                write_conversation_outputs(
                    output_dir,
                    result,
                    save_qkv=save_qkv or save_attention_weights,
                )

            results.append(result)
            convo_labels.append(_extract_result_label(result.summary) or str(env.env_id))
            processed_indices.add(idx)
            progress.update(1)

        progress.close()

    if not results:
        print("No validation samples processed; exiting.")
        return

    combined = list(zip(results, convo_labels))
    combined.sort(key=lambda pair: pair[0].summary.conversation_index)
    combined = [pair for pair in combined if pair[0].summary.num_layers > 0]
    if not combined:
        raise ValueError("No captured layer data available for aggregation.")

    results = [pair[0] for pair in combined]
    convo_labels = [pair[1] for pair in combined]

    resolved_layer = _select_layer_index(results, args.layer_idx)
    layer_tag = f"layer_{resolved_layer:02d}"

    with timer.track("aggregate_results"):
        aggregated_path = output_dir / f"{layer_tag}_{args.query_span_tag}_attention.json"
        aggregated_payload: List[Dict[str, object]] = []
        for result, label in zip(results, convo_labels):
            masses = aggregate_attention_by_span(
                result,
                layer_idx=resolved_layer,
                query_span_tag=args.query_span_tag,
            )
            aggregated_payload.append(
                {
                    "conversation_index": result.summary.conversation_index,
                    "label": label,
                    "attention_mass": masses,
                }
            )
        aggregated_path.write_text(json.dumps(aggregated_payload, indent=2), encoding="utf-8")

        row_labels, column_labels, matrix = build_attention_matrix(
            results,
            layer_idx=resolved_layer,
            query_span_tag=args.query_span_tag,
        )
        row_labels = [
            label if label else default
            for label, default in zip(convo_labels, row_labels)
        ]
        heatmap_path = output_dir / f"{layer_tag}_{args.query_span_tag}_attention.png"
        plot_attention_heatmap(
            matrix,
            row_labels=row_labels,
            column_labels=column_labels,
            output_path=heatmap_path,
            title=f"Layer {resolved_layer} attention mass (query: {args.query_span_tag})",
        )

        matrix_json_path = output_dir / f"{layer_tag}_{args.query_span_tag}_attention_matrix.json"
        save_attention_matrix(matrix, row_labels, column_labels, matrix_json_path)

    with timer.track("write_timing_summary"):
        timing_path = output_dir / "timings.json"
        timing_path.write_text(json.dumps(timer.as_dict(), indent=2), encoding="utf-8")

    print(f"Wrote attention evaluation artefacts to {output_dir}")


if __name__ == "__main__":
    main()

