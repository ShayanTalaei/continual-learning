from typing import List, Dict, Any, Tuple, Optional, Literal
from pathlib import Path
import json
import yaml
from pydantic import BaseModel
from tqdm import tqdm
from datasets import Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from src.memory.distillation.strategies.exclude_current import ExcludeCurrentStrategy
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

    # Sampling
    max_samples: Optional[int] = None

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
        return ExcludeCurrentStrategy()
    raise ValueError(f"Unsupported strategy: {cfg.strategy.name}")


def _make_teacher_agent(system_prompt_override: str | None) -> HistoryAgent:
    agent_cfg = HistoryAgentConfig(
        memory_config=HistoryListConfig().model_dump(),
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


def _to_conversation_row(
    system_prompt: str, 
    user_prompt: str, 
    lm_response: Dict[str, Any], 
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a conversation row compatible with cartridges format.
    
    This format includes:
    - messages: list of message dicts with role, content, token_ids, logprobs
    - system_prompt: separate field (also included in messages[0] for completeness)
    - metadata: task-specific metadata
    - type: dataset type label for organization
    
    The logprobs are in OpenAI format and will be converted to FlatTopLogprobs
    by convert_to_cartridges.py for efficient knowledge distillation training.
    
    Args:
        system_prompt: System prompt text
        user_prompt: User prompt text
        lm_response: Full LM response dict with 'text', 'metrics', 'logprobs'
        meta: Additional metadata
        
    Returns:
        Dict with messages, system_prompt, metadata, type, and raw logprobs
    """
    assistant_text = lm_response.get("text", "")
    
    messages: List[Dict[str, Any]] = [
        {
            "role": "system", 
            "content": system_prompt,
            # System and user messages don't need logprobs or token_ids
            "token_ids": None,
            "logprobs": None,
        },
        {
            "role": "user", 
            "content": user_prompt,
            "token_ids": None,
            "logprobs": None,
        },
        {
            "role": "assistant", 
            "content": assistant_text,
            # Assistant message gets logprobs if available
            "token_ids": None,  # Will be filled by converter
            "logprobs": lm_response.get("logprobs"),  # OpenAI-style logprobs for distillation
        },
    ]

    
    return {
        "messages": messages,
        "system_prompt": system_prompt,  # Separate field for cartridges compatibility
        "metadata": meta,
        "type": "memory_distillation",  # Dataset type label
    }


def _process_single_sample(
    sample: Any,
    history_entries: List[Any],
    strategy: MemoryFormationStrategy,
    teacher_agent: HistoryAgent,
    client: TokasaurusClient,
    top_logprobs: Optional[int],
) -> Dict[str, Any]:
    """Process a single sample to generate a training row.
    
    This function is designed to be thread-safe and can be called in parallel.
    
    Args:
        sample: Sample from strategy.iter_samples()
        history_entries: Full history entries list
        strategy: Memory formation strategy
        teacher_agent: Teacher agent for prompt building
        client: LM client for inference
        top_logprobs: Number of top logprobs to return
        
    Returns:
        Conversation row dict
    """
    memory_view = strategy.build_memory_for_sample(history_entries, sample)
    system_prompt = teacher_agent.build_system_prompt()
    user_prompt = teacher_agent.build_user_prompt(
        sample.observation, 
        memory_view, 
        teacher_agent.config.history_k
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    lm_response = client.call(
        messages,
        cartridges=None,
        top_logprobs=top_logprobs,
    )
    
    row = _to_conversation_row(
        system_prompt, 
        user_prompt, 
        lm_response, 
        sample.meta or {}
    )
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

    snapshot = _find_latest_snapshot(cfg.checkpoint_dir)
    print(f"[DataGen] Loading history from: {snapshot}")
    history = _load_history(snapshot)
    print(f"[DataGen] Loaded {len(history.recall())} history entries")

    strategy = _make_strategy(cfg)
    print(f"[DataGen] Using strategy: {cfg.strategy.name}")
    
    teacher_agent = _make_teacher_agent(cfg.system_prompt_override)
    print(f"[DataGen] Teacher agent initialized")

    lm_cfg = TokasaurusConfig(
        model=cfg.lm_model,
        base_url=cfg.lm_base_url,
        temperature=cfg.lm_temperature,
        max_output_tokens=cfg.lm_max_output_tokens,
    )
    client = TokasaurusClient(lm_cfg)
    print(f"[DataGen] LM client initialized: {cfg.lm_model} @ {cfg.lm_base_url}")

    history_entries = history.recall()
    
    # Determine total samples for progress bar
    total_samples = cfg.max_samples if cfg.max_samples is not None else len(history_entries)
    print(f"[DataGen] Starting sample generation (max_samples={cfg.max_samples}, num_threads={cfg.num_threads})...")
    
    # Collect all samples first to enable parallel processing
    samples_to_process: List[Tuple[int, Any]] = []
    for count, sample in enumerate(strategy.iter_samples(history_entries)):
        if cfg.max_samples is not None and count >= cfg.max_samples:
            break
        samples_to_process.append((count, sample))
    
    print(f"[DataGen] Collected {len(samples_to_process)} samples to process")
    
    # Process samples in parallel while preserving order
    rows: Dict[int, Dict[str, Any]] = {}  # index -> row
    
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
                        raise
                    pbar.update(1)
    
    # Convert to ordered list
    ordered_rows = [rows[i] for i in sorted(rows.keys())]
    print(f"[DataGen] Finished generating {len(ordered_rows)} samples")

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
