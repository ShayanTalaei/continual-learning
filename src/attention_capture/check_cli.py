"""Command-line entrypoint for inspecting cartridge attention mass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from .pipeline import (
    MODEL_LOADERS,
    capture_attention_for_messages,
    load_cache,
    load_model,
    load_tokenizer,
    write_conversation_outputs,
)
from .types import ChatMessage


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect attention mass over cartridge tokens.")
    parser.add_argument(
        "--model-type",
        choices=sorted(MODEL_LOADERS.keys()),
        default="llama",
        help="Model family to load.",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Pretrained model identifier (HF hub path or local checkpoint).",
    )
    parser.add_argument(
        "--cartridge-path",
        type=str,
        default=None,
        help="Optional path to a serialized TrainableCache (kv_cache.torch).",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-jsonl",
        type=str,
        help="Path to a JSONL file. Each line should be a conversation with a `messages` field or a raw list.",
    )
    input_group.add_argument(
        "--input-text",
        type=str,
        help="Path to a plain-text file interpreted as a single user message.",
    )
    input_group.add_argument(
        "--prompt",
        type=str,
        help="Literal prompt string for a single user turn.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory where statistics and optional tensors will be stored.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use (e.g. cuda, cuda:1, cpu).",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Computation dtype for the model (affects weights and cache).",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "generate"],
        default="train",
        help="Forward pass mode to execute within the model.",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help="Optional limit on number of conversations to process.",
    )
    parser.add_argument(
        "--save-qkv",
        action="store_true",
        help="Persist captured Q/K/V tensors to disk alongside summaries.",
    )
    parser.add_argument(
        "--save-attention-weights",
        action="store_true",
        help="Persist full attention weight tensors (heads x tokens x context).",
    )
    parser.add_argument(
        "--add-generation-prompt",
        action="store_true",
        help="If set, append the generation prompt when applying the tokenizer chat template.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only emit JSON summaries (override of --save-qkv / --save-attention-weights).",
    )
    return parser.parse_args(argv)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_conversations(args: argparse.Namespace) -> Iterable[List[Dict[str, str]]]:
    if args.input_jsonl:
        with Path(args.input_jsonl).open("r", encoding="utf-8") as handle:
            for line_idx, line in enumerate(handle):
                if args.max_conversations is not None and line_idx >= args.max_conversations:
                    break
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, list):
                    yield payload
                elif isinstance(payload, dict):
                    messages = payload.get("messages")
                    if not isinstance(messages, list):
                        raise ValueError(f"Expected `messages` list in JSONL object at line {line_idx + 1}")
                    yield messages
                else:
                    raise ValueError(f"Unsupported JSONL entry type: {type(payload)}")
    elif args.input_text:
        text = Path(args.input_text).read_text(encoding="utf-8").strip()
        yield [{"role": "user", "content": text}]
    else:
        yield [{"role": "user", "content": args.prompt}]


def to_chat_messages(messages: List[Dict[str, str]]) -> List[ChatMessage]:
    chat_messages: List[ChatMessage] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role is None or content is None:
            raise ValueError(f"Each message must include 'role' and 'content': {msg}")
        chat_messages.append(ChatMessage(role=role, content=content, tag=msg.get("tag")))
    return chat_messages


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    device = torch.device(args.device)

    model = load_model(args.model_type, args.model_name, device=device, dtype=args.dtype)
    tokenizer = load_tokenizer(args.model_name)

    cache = None
    if args.cartridge_path:
        cache = load_cache(args.cartridge_path, device=device, dtype=args.dtype)

    metadata = {
        "model_type": args.model_type,
        "model_name": args.model_name,
        "cartridge_path": args.cartridge_path,
        "dtype": args.dtype,
        "device": str(device),
        "mode": args.mode,
        "add_generation_prompt": args.add_generation_prompt,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    save_qkv = args.save_qkv and not args.summary_only
    save_attention_weights = args.save_attention_weights and not args.summary_only

    for idx, raw_messages in enumerate(iter_conversations(args)):
        chat_messages = to_chat_messages(raw_messages)
        result = capture_attention_for_messages(
            model=model,
            tokenizer=tokenizer,
            messages=chat_messages,
            conversation_index=idx,
            device=device,
            mode=args.mode,
            cache=cache,
            add_generation_prompt=args.add_generation_prompt,
            save_qkv=save_qkv,
            save_attention_weights=save_attention_weights,
        )
        write_conversation_outputs(
            output_dir,
            result,
            save_qkv=save_qkv or save_attention_weights,
        )

    print(f"Wrote attention summaries to {output_dir}")


if __name__ == "__main__":
    main()

