from typing import List, Dict, Any, Tuple, Optional, Literal, cast
from pathlib import Path
import json
import yaml
from pydantic import BaseModel
from tqdm import tqdm
from datasets import Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from src.memory.distillation.strategies.exclude_current import ExcludeCurrentStrategy
from src.memory.distillation.strategies.full_memory import FullMemoryStrategy
from src.memory.distillation.strategies.base import MemoryFormationStrategy
from src.agent.history_agent import HistoryAgent, HistoryAgentConfig
from src.memory.history_list import HistoryList, HistoryListConfig
from src.lm.tokasaurus_client import TokasaurusConfig, TokasaurusClient


# ============================================================================
# Configuration Schemas
# ============================================================================

class StrategyConfig(BaseModel):
    name: Literal["full_memory", "exclude_current", "rolling_window", "failure_focus"]
    window_k: Optional[int] = None
    exclude_current: bool = False
    failure_focus: bool = False
    # Shuffling parameters for exclude_current strategy
    do_shuffle: bool = False
    num_shufflings: int = 1
    # Parameters for full_memory strategy
    memory_checkpoint_path: Optional[str] = None
    target_dataset_config: Optional[Dict[str, Any]] = None
    max_target_samples: Optional[int] = None
    # Optional slicing over target environments (applied before max_target_samples)
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None


class DataGenConfig(BaseModel):
    checkpoint_dir: str
    output_dir: str
    output_format: Literal["jsonl", "parquet"] = "jsonl"

    # Strategy
    strategy: StrategyConfig

    # Agent prompt rendering
    agent_type: Literal["history_agent"] = "history_agent"
    system_prompt_override: Optional[str] = None

    # LM client (Tokasaurus)
    lm_model: str
    lm_base_url: str
    lm_temperature: float = 0.0
    lm_max_output_tokens: int = 512
    top_logprobs: Optional[int] = None
    
    # Retry configuration
    lm_max_retries: int = 5
    lm_starting_delay: float = 1.0
    lm_backoff_factor: float = 2.0
    lm_max_delay: float = 10.0
    lm_timeout_s: float = 900.0
    
    # Full sequence tensor data for distillation training
    include_full_sequence_data: bool = False

    # Teacher messages text storage
    include_teacher_messages_texts: bool = True

    # Individual file saving for resume functionality
    save_individual_files: bool = True

    # Sampling
    max_samples: Optional[int] = None
    num_repeat_samples: int = 1

    # Parallelism
    num_threads: int = 1

    # Optional HF upload
    hf_repo_id: Optional[str] = None
    hf_private: bool = True

    # Extra metadata
    metadata: Dict[str, Any] = {}


# ============================================================================
# Helper Functions
# ============================================================================


def _make_strategy(cfg: DataGenConfig) -> MemoryFormationStrategy:
    if cfg.strategy.name == "exclude_current":
        return ExcludeCurrentStrategy(
            do_shuffle=cfg.strategy.do_shuffle,
            num_shufflings=cfg.strategy.num_shufflings
        )
    elif cfg.strategy.name == "full_memory":
        if not cfg.strategy.memory_checkpoint_path:
            raise ValueError("memory_checkpoint_path is required for full_memory strategy")
        if not cfg.strategy.target_dataset_config:
            raise ValueError("target_dataset_config is required for full_memory strategy")
        
        return FullMemoryStrategy(
            memory_checkpoint_path=cfg.strategy.memory_checkpoint_path,
            target_dataset_config=cfg.strategy.target_dataset_config,
            max_target_samples=cfg.strategy.max_target_samples,
            start_idx=cfg.strategy.start_idx,
            end_idx=cfg.strategy.end_idx,
            logger=None  # Will be set by the calling context
        )
    raise ValueError(f"Unsupported strategy: {cfg.strategy.name}")


def _make_teacher_agent(system_prompt_override: Optional[str]) -> HistoryAgent:
    agent_cfg = HistoryAgentConfig(
        memory_config=HistoryListConfig(),
        history_k=None,
        system_prompt=system_prompt_override,
        # Dummy LM to avoid external calls; we won't use agent.lm
        lm_config={"model": "toka:dummy", "base_url": "http://localhost:0"},
    )
    return HistoryAgent(agent_cfg)


def _find_latest_snapshot(checkpoint_dir: str) -> Path:
    base = Path(checkpoint_dir)
    candidates = sorted(base.glob("memory_*.jsonl"))
    return candidates[-1]


def _load_history(snapshot_path: Path) -> HistoryList:
    return HistoryList.load_snapshot(snapshot_path)


def _get_existing_sample_ids(output_dir: Path) -> "set[str]":
    """Get set of sample IDs that already exist as individual JSON files."""
    existing_ids = set()
    if output_dir.exists():
        for json_file in output_dir.glob("*.json"):
            if json_file.stem != "data_gen_config":  # Skip config file
                existing_ids.add(json_file.stem)
    return existing_ids


def _to_conversation_row(
    system_prompt: str, 
    chats: List[Dict[str, str]], 
    lm_response: Dict[str, Any], 
    meta: Dict[str, Any],
    include_teacher_messages_texts: bool = True
) -> Dict[str, Any]:

    # Base row data
    row_data = {
        "metadata": meta,
        "type": "memory_distillation",  # Dataset type label
    }
    
    if include_teacher_messages_texts:
        row_data["teacher_messages"] = [
            {"role": "system", "content": system_prompt},
        ] + chats
    
    row_data["input_messages"] = [
        {"role": "system", "content": system_prompt},
        chats[-1] # The last message in the chat which is the target task
    ]
    
    row_data["output_message"] = lm_response['text']
    row_data["output_ids"] = lm_response['output_ids']
    row_data["topk_logprobs"] = lm_response['topk_logprobs']
    row_data["topk_token_ids"] = lm_response['topk_token_ids']
    
    return row_data


def _process_single_sample(
    sample: Any,
    history_entries: List[Any],
    strategy: MemoryFormationStrategy,
    teacher_agent: HistoryAgent,
    client: TokasaurusClient,
    top_logprobs: Optional[int],
    include_teacher_messages_texts: bool = True,
    output_dir: Optional[Path] = None,
    save_individual_files: bool = True,
) -> Dict[str, Any]:

    memory_view = strategy.build_memory_for_sample(history_entries, sample)
    system_prompt = teacher_agent.build_system_prompt()
    chats = teacher_agent.build_user_prompt(
        sample.observation, 
        memory_view, 
        teacher_agent.config.history_k
    )

    messages = [
        {"role": "system", "content": system_prompt},
    ] + chats
    
    lm_response = client.call_with_full_sequence_data(
        messages,
        cartridges=None,
        top_logprobs=top_logprobs,  # Number of top logprobs to return
    )
    
    row = _to_conversation_row(
        system_prompt, 
        chats, 
        lm_response, 
        sample.meta or {},
        include_teacher_messages_texts
    )
    
    # Add sample ID to the row
    if sample.sample_id:
        row["sample_id"] = sample.sample_id
    
    # Save individual JSON file if output_dir is provided and enabled
    if output_dir and sample.sample_id and save_individual_files:
        individual_file = output_dir / f"{sample.sample_id}.json"
        try:
            with open(individual_file, "w") as f:
                json.dump(row, f, indent=2)
        except Exception as e:
            print(f"[DataGen] WARNING: Failed to save individual file {individual_file}: {e}")
    
    return row


def run_data_generation(cfg: DataGenConfig) -> str:
    """Generate distillation training data from a HistoryAgent checkpoint.
    
    Args:
        cfg: DataGenConfig with all generation parameters
        
    Returns:
        Path to the generated dataset file
    """
    print(f"[DataGen] Starting data generation from checkpoint: {cfg.checkpoint_dir}")
    
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DataGen] Output directory: {out_dir}")

    # Save config for reproducibility
    config_path = out_dir / "data_gen_config.json"
    with open(config_path, "w") as f:
        json.dump(cfg.model_dump(), f, indent=2)
    print(f"[DataGen] Config saved to: {config_path}")

    strategy = _make_strategy(cfg)
    print(f"[DataGen] Using strategy: {cfg.strategy.name}")
    
    # For full_memory strategy, the memory is loaded within the strategy itself
    if cfg.strategy.name == "full_memory":
        history_entries = []  # Not used for full_memory strategy
        print(f"[DataGen] Full memory strategy - memory loaded within strategy")
    else:
        # For other strategies, load from checkpoint_dir
        snapshot = _find_latest_snapshot(cfg.checkpoint_dir)
        print(f"[DataGen] Loading history from: {snapshot}")
        history = _load_history(snapshot)
        history_entries = history.recall()
        print(f"[DataGen] Loaded {len(history_entries)} history entries")
    
    teacher_agent = _make_teacher_agent(cfg.system_prompt_override)
    print(f"[DataGen] Teacher agent initialized")

    lm_cfg = TokasaurusConfig(
        model=cfg.lm_model,
        base_url=cfg.lm_base_url,
        temperature=cfg.lm_temperature,
        max_output_tokens=cfg.lm_max_output_tokens,
        max_retries=cfg.lm_max_retries,
        starting_delay=cfg.lm_starting_delay,
        backoff_factor=cfg.lm_backoff_factor,
        max_delay=cfg.lm_max_delay,
        timeout_s=cfg.lm_timeout_s,
    )
    client = TokasaurusClient(lm_cfg)
    print(f"[DataGen] LM client initialized: {cfg.lm_model} @ {cfg.lm_base_url}")

    # Check for existing samples to enable resume functionality
    existing_sample_ids = set()
    if cfg.save_individual_files:
        existing_sample_ids = _get_existing_sample_ids(out_dir)
        if existing_sample_ids:
            print(f"[DataGen] Found {len(existing_sample_ids)} existing samples, will skip them")
    
    # Determine total samples for progress bar
    if cfg.strategy.name == "full_memory":
        # For full_memory strategy, samples are based on target dataset size
        full_memory_strategy = cast(FullMemoryStrategy, strategy)
        total_samples = cfg.max_samples if cfg.max_samples is not None else len(full_memory_strategy.target_environments)
    else:
        total_samples = cfg.max_samples if cfg.max_samples is not None else len(history_entries)
    print(f"[DataGen] Starting sample generation (max_samples={cfg.max_samples}, num_threads={cfg.num_threads})...")
    
    # Collect all samples first to enable parallel processing, filtering out existing ones
    samples_to_process: List[Tuple[int, Any]] = []
    for count, sample in enumerate(strategy.iter_samples(history_entries)):
        if cfg.max_samples is not None and count >= cfg.max_samples:
            break
        
        if cfg.num_repeat_samples == -1:
            # No repetition - use original sample ID as-is
            if sample.sample_id and sample.sample_id in existing_sample_ids:
                print(f"[DataGen] Skipping existing sample: {sample.sample_id}")
                continue
            samples_to_process.append((count, sample))
        else:
            # Repeat each sample num_repeat_samples times
            for repeat_idx in range(cfg.num_repeat_samples):
                # Create a copy of the sample with updated ID
                repeated_sample = sample.model_copy()
                if repeated_sample.sample_id:
                    repeated_sample.sample_id = f"{sample.sample_id}_repeat_{repeat_idx}"
                
                # Skip if sample already exists
                if repeated_sample.sample_id and repeated_sample.sample_id in existing_sample_ids:
                    print(f"[DataGen] Skipping existing sample: {repeated_sample.sample_id}")
                    continue
                    
                samples_to_process.append((count * cfg.num_repeat_samples + repeat_idx, repeated_sample))
    
    print(f"[DataGen] Collected {len(samples_to_process)} new samples to process")
    
    # Process samples in parallel while preserving order
    rows: Dict[int, Dict[str, Any]] = {}  # index -> row
    
    # breakpoint()
    if cfg.num_threads == 1:
        # Single-threaded mode (original behavior)
        with tqdm(total=len(samples_to_process), desc="Generating samples", unit="sample") as pbar:
            for idx, sample in samples_to_process:
                row = _process_single_sample(
                    sample,
                    history_entries,
                    strategy,
                    teacher_agent,
                    client,
                    cfg.top_logprobs,
                    cfg.include_teacher_messages_texts,
                    out_dir,
                    cfg.save_individual_files,
                )
                rows[idx] = row
                pbar.update(1)
    else:
        # Multi-threaded mode
        with ThreadPoolExecutor(max_workers=cfg.num_threads) as executor:
            # Submit all tasks
            future_to_idx = {}
            for idx, sample in samples_to_process:
                future = executor.submit(
                    _process_single_sample,
                    sample,
                    history_entries,
                    strategy,
                    teacher_agent,
                    client,
                    cfg.top_logprobs,
                    cfg.include_teacher_messages_texts,
                    out_dir,
                    cfg.save_individual_files,
                )
                future_to_idx[future] = idx
            
            # Collect results with progress bar
            with tqdm(total=len(samples_to_process), desc="Generating samples", unit="sample") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        row = future.result()
                        rows[idx] = row
                    except Exception as e:
                        print(f"\n[DataGen] ERROR processing sample {idx}: {e}")
                        continue
                    pbar.update(1)
    
    # Convert to ordered list
    ordered_rows = [rows[i] for i in sorted(rows.keys())]
    print(f"[DataGen] Finished generating {len(ordered_rows)} new samples")
    
    # Load existing samples to include in final dataset
    if cfg.save_individual_files and existing_sample_ids:
        print(f"[DataGen] Loading {len(existing_sample_ids)} existing samples...")
        for sample_id in existing_sample_ids:
            existing_file = out_dir / f"{sample_id}.json"
            if existing_file.exists():
                with open(existing_file, "r") as f:
                    existing_row = json.load(f)
                    ordered_rows.append(existing_row)
        print(f"[DataGen] Total samples in final dataset: {len(ordered_rows)}")

    out_path = out_dir / f"dataset.{cfg.output_format}"
    
    if cfg.output_format == "jsonl":
        with open(out_path, "w") as f:
            for row in ordered_rows:
                f.write(json.dumps(row) + "\n")
        print(f"[DataGen] Dataset written to: {out_path}")
    elif cfg.output_format == "parquet":
        try:
            import pandas as pd
            df = pd.DataFrame(ordered_rows)
            df.to_parquet(out_path, index=False)
            print(f"[DataGen] Dataset written to: {out_path}")
        except ImportError:
            print("[DataGen] WARNING: pandas/pyarrow not available, falling back to JSONL")
            out_path = out_dir / "dataset.jsonl"
            with open(out_path, "w") as f:
                for row in ordered_rows:
                    f.write(json.dumps(row) + "\n")
            print(f"[DataGen] Dataset written to: {out_path}")

    if cfg.hf_repo_id:
        print(f"[DataGen] Uploading to Hugging Face: {cfg.hf_repo_id}")
        dataset = Dataset.from_list(ordered_rows)
        dataset.push_to_hub(cfg.hf_repo_id, private=cfg.hf_private)
        print(f"[DataGen] ✓ Uploaded to https://huggingface.co/datasets/{cfg.hf_repo_id}")

    return str(out_path)


def main():
    """CLI entry point for data generation."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Generate distillation training data from HistoryAgent checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python -m src.memory.distillation.data_generation --config configs/distillation/gen_example.yaml
  
  # Or with Python:
  python src/memory/distillation/data_generation.py --config configs/distillation/gen_example.yaml
"""
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file with DataGenConfig parameters"
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    cfg = DataGenConfig(**config_dict)
    
    # Run generation
    output_path = run_data_generation(cfg)
    print(f"\n[DataGen] ✓ SUCCESS! Dataset generated at: {output_path}")


if __name__ == "__main__":
    main()
