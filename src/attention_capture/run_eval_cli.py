from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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
    load_cartridges_from_ids,
    load_model,
    load_tokenizer,
    save_attention_results,
)

from src.agent.history_agent import HistoryAgentConfig


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run attention evaluation over validation prompts using a checkpointed history agent.",
    )
    parser.add_argument("--config", required=True, help="Path to run configuration YAML.")
    parser.add_argument(
        "--memory-snapshot",
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
        "--cartridge-ids",
        nargs="+",
        default=[],
        help="List of cartridge IDs to load from local directory.",
    )
    parser.add_argument(
        "--cartridge-dir",
        default=None,
        help="Base directory where cartridge files are stored.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of validation prompts to process.",
    )
    parser.add_argument(
        "--layer-idxs",
        nargs="+",
        type=int,
        default=[],
        help="Decoder layer indices for capturing attention.",
    )
    parser.add_argument(
        "--prompts-idxs",
        nargs="+",
        type=int,
        default=[],
        help="Validation prompts indices to process.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    timer = StepTimer()

    args = parse_args(argv)

    run_conf, raw_conf = load_run_configuration(args.config)

    agent_conf = HistoryAgentConfig(**run_conf.agent)
    system_prompt = resolve_system_prompt(agent_conf.system_prompt or "")
    history_memory = load_history_memory(agent_conf.memory_config, args.memory_snapshot)
    builder = HistoryPromptBuilder(agent_conf)

    dataset = build_validation_dataset(run_conf.validation_dataset or {})

    dataset_items = dataset.get_dataset()
    dataset_size = len(dataset_items)
    sample_limit = min(dataset_size, args.max_samples) if args.max_samples is not None else dataset_size

    assert torch.cuda.is_available(), "CUDA is not available"
    device = torch.device("cuda")
    model = load_model(model_name=args.model_name, device=device)
    tokenizer = load_tokenizer(args.model_name)

    cartridges = None
    if args.cartridge_ids and args.cartridge_dir:
        with timer.track("load_cartridges"):
            cartridges = load_cartridges_from_ids(
                cartridge_ids=args.cartridge_ids,
                cartridge_dir=args.cartridge_dir,
                device=device,
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_entries = list(history_memory.recall())

    with timer.track("process_validation_prompts"):
        if args.prompts_idxs:
            to_be_processed = [(idx, (env, obs)) for idx, (env, obs) in enumerate(iter_validation_prompts(dataset)) if idx in args.prompts_idxs]
        else:
            to_be_processed = [(idx, (env, obs)) for idx, (env, obs) in enumerate(iter_validation_prompts(dataset))]
        progress = tqdm(
            total=len(to_be_processed),
            initial=0,
            desc="conversations",
            unit="conv",
        )
        for idx, (env, obs) in to_be_processed:
            if idx >= sample_limit:
                break

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
                    cartridges=cartridges,
                    layer_idxs=args.layer_idxs,
                )
            
            with timer.track(f"{conversation_step}.save_results"):
                output_file = output_dir / f"conversation_{idx:04d}_attention.json"
                save_attention_results(
                    result=result,
                    output_path=output_file,
                    tokenizer=tokenizer,
                    model_name=args.model_name,
                )
                print(f"[run_eval] Saved results to {output_file}")

            progress.update(1)

        progress.close()

if __name__ == "__main__":
    main()

