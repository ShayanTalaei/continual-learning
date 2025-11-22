"""
Evaluate perplexity or QA accuracy of trained cartridges on eval questions.

This script:
1. Converts FinanceBench QA pairs to Conversation format (parquet)
2. Loads AMD and/or PepsiCo cartridges
3. Evaluates using the cartridge(s) on the eval questions:
   - "perplexity": loss-based evaluation
   - "qa": generation-based QA evaluation with EM/F1 scoring
4. If both cartridges are provided, also evaluates composition

Usage:
    python examples/10k/eval_perplexity_with_cartridge.py --config examples/10k/eval_perplexity_config.yaml
"""

import argparse
import json
import os
import re
import importlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

import yaml
import torch
import requests

from cartridges.train import evaluate_perplexity, evaluate_generations, CacheAndModel, GenerationEvalConfig, LossEvalConfig
from cartridges.cache import TrainableCache, AttnConfig
from cartridges.structs import Conversation, write_conversations
from cartridges import datasets
# Force reload to pick up latest changes (fixes import cache issues)
importlib.reload(datasets)
from cartridges.datasets import LossEvalDataset, GenerateEvalDataset, DataSource, DatasetBatch
from cartridges.models import HFModelConfig, FlexLlamaForCausalLM
from cartridges.utils import get_logger, seed_everything
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from cartridges.generation import flex_generate
from tqdm.auto import tqdm
from dotenv import load_dotenv
import pdb;

logger = get_logger(__name__)

CARTRIDGES_DIR = Path(__file__).parent.parent.parent
DATA_DIR = CARTRIDGES_DIR / "data" / "10k"

def _collate_fn_first(x):
    return x[0]


def normalize_answer(s: str) -> str:
    """Lowercase, strip, and remove extra whitespace + punctuation."""
    s = s.lower().strip()
    # Remove punctuation
    s = re.sub(r"[^\w\s]", " ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def em_and_f1(pred: str, gold: str) -> Tuple[float, float]:
    """Compute Exact Match and token-level F1 between two strings."""
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    if pred_norm == gold_norm:
        return 1.0, 1.0

    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return 0.0, f1


class FinanceBenchQADataset(GenerateEvalDataset):
    """GenerateEvalDataset with EM/F1 scoring for FinanceBench QA."""

    class Config(GenerateEvalDataset.Config):
        _pass_as_config = True

    def __getitem__(self, index: int):
        # Use the default GenerateEvalDataset behavior for building prompts
        # and gold answers from Conversation records.
        element = super().__getitem__(index)
        # Ensure input_ids is 1D for downstream generation code.
        if element.input_ids.dim() == 2 and element.input_ids.shape[0] == 1:
            element.input_ids = element.input_ids.squeeze(0)
        return element

    def score(self, pred: str, answer: str, convo_id: Optional[str] = None) -> Tuple[Dict[str, float], Dict]:
        """Score a prediction against the gold answer using EM and F1."""
        em, f1 = em_and_f1(pred, answer)
        return {"em": em, "f1": f1}, {}
    
    def batch_score_with_answers(self, preds: List[str], answers: List[str]) -> Dict[str, float]:
        """Compute batch EM and F1 scores."""
        assert len(preds) == len(answers)
        em_scores = []
        f1_scores = []
        for pred, answer in zip(preds, answers):
            em, f1 = em_and_f1(pred, answer)
            em_scores.append(em)
            f1_scores.append(f1)
        return {
            "em": sum(em_scores) / len(em_scores),
            "f1": sum(f1_scores) / len(f1_scores),
        }


def convert_qa_to_conversations(
    qa_json_path: str,
    output_parquet_path: str,
    system_prompt: Optional[str] = None,
) -> List[Conversation]:
    """
    Convert FinanceBench QA JSON to Conversation format (parquet).
    
    Args:
        qa_json_path: Path to FinanceBench QA JSON file
        output_parquet_path: Path to write the converted conversations parquet
        system_prompt: Optional system prompt to add to each conversation
    
    Returns:
        List of Conversation objects
    """
    logger.info(f"Loading QA pairs from {qa_json_path}")
    with open(qa_json_path, "r") as f:
        qa_list = json.load(f)
    
    logger.info(f"Converting {len(qa_list)} QA pairs to conversations")
    conversations = []
    
    for i, qa in enumerate(qa_list):
        question = qa.get("question", "")
        answer = qa.get("answer", "")
        
        messages = []
        if system_prompt:
            messages.append(
                Conversation.Message(
                    role="system",
                    content=system_prompt,
                    token_ids=None,
                )
            )
        messages.append(
            Conversation.Message(
                role="user",
                content=question,
                token_ids=None,
            )
        )
        messages.append(
            Conversation.Message(
                role="assistant",
                content=answer,
                token_ids=None,
            )
        )
        
        conversations.append(
            Conversation(
                messages=messages,
                system_prompt=system_prompt or "",
                metadata={
                    "question_id": qa.get("question_id", str(i)),
                    "source": "financebench",
                },
                type="qa",
            )
        )
    
    logger.info(f"Writing {len(conversations)} conversations to {output_parquet_path}")
    output_path = Path(output_parquet_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use write_conversations which handles parquet format
    write_conversations(conversations, str(output_path))
    
    logger.info(f"Successfully wrote conversations to {output_parquet_path}")
    return conversations


def load_cartridge(
    cartridge_path: str,
    model,  # The model instance (needed for AttnConfig)
    device: str = "cuda",
) -> TrainableCache:
    """
    Load a trained cartridge from disk.
    
    Args:
        cartridge_path: Path to the cartridge.pt file
        model: The model instance (used to get attention config)
        device: Device to load the cartridge onto
    
    Returns:
        Loaded TrainableCache
    """
    logger.info(f"Loading cartridge from {cartridge_path}")

    if not os.path.exists(cartridge_path):
        raise FileNotFoundError(
            f"Cartridge file not found: {cartridge_path}\n"
            f"Please check the cartridge_path in your config file."
        )
    
    # Load the cartridge
    cache = TrainableCache.from_pretrained(
        cartridge_path,
        device=device if device != "cuda" else None,
    )
    cache = cache.to(device).to(torch.bfloat16)
    cache.eval()
    
    logger.info(f"Successfully loaded cartridge with {cache._num_trainable_tokens} trainable tokens")
    return cache


def chat_with_tokasaurus(
    url: str,
    question: str,
    cartridges: list[dict[str, Any]],
    system_prompt: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """Send a single question to Tokasaurus and return the text reply."""
    payload = {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "messages": [],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "cartridges": cartridges,
    }
    if system_prompt:
        payload["messages"].append({"role": "system", "content": system_prompt})
    payload["messages"].append({"role": "user", "content": question})

    r = requests.post(url + "/custom/cartridge/chat/completions", json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def combine_cartridges(cache1: TrainableCache, cache2: TrainableCache) -> TrainableCache:
    """
    Combine two cartridges by concatenating their trainable tokens.
    
    This is the standard way to compose cartridges for evaluation, as described in the paper.
    The composition is done by concatenating the KV caches along the sequence dimension.
    This works because cartridges are designed to be composable - each cartridge's tokens
    have seq_id=-1 (CARTRIDGE_SEQ_ID), meaning they can attend to each other.
    
    For perplexity evaluation, we need local model access to compute full-sequence logprobs,
    so we combine the caches locally. For QA evaluation, Tokasaurus supports native composition
    by passing multiple cartridges in a list, but the current `evaluate_generations` infrastructure
    uses local HF generation, so we also combine here.
    
    Args:
        cache1: First cartridge to combine
        cache2: Second cartridge to combine
        
    Returns:
        A new TrainableCache containing both cartridges' tokens concatenated
    """
    logger.info("Combining cartridges for composition evaluation...")
    
    # Verify both caches have the same config
    assert cache1.config == cache2.config, "Cartridges must have the same attention config"
    
    # Concatenate keys and values for each layer using the full cartridge
    # (frozen + trainable tokens) from both caches
    combined_keys = []
    combined_values = []
    
    for layer_idx in range(cache1.config.n_layers):
        # Build full layer KV for cache1: [frozen, trainable]
        k1_parts = []
        v1_parts = []
        if getattr(cache1, "_num_frozen_tokens", 0) > 0 and getattr(cache1, "frozen_keys", None):
            k1_parts.append(cache1.frozen_keys[layer_idx])
            v1_parts.append(cache1.frozen_values[layer_idx])
        k1_parts.append(cache1.trainable_keys[layer_idx])
        v1_parts.append(cache1.trainable_values[layer_idx])
        keys1 = torch.cat(k1_parts, dim=2)
        values1 = torch.cat(v1_parts, dim=2)

        # Build full layer KV for cache2: [frozen, trainable]
        k2_parts = []
        v2_parts = []
        if getattr(cache2, "_num_frozen_tokens", 0) > 0 and getattr(cache2, "frozen_keys", None):
            k2_parts.append(cache2.frozen_keys[layer_idx])
            v2_parts.append(cache2.frozen_values[layer_idx])
        k2_parts.append(cache2.trainable_keys[layer_idx])
        v2_parts.append(cache2.trainable_values[layer_idx])
        keys2 = torch.cat(k2_parts, dim=2)
        values2 = torch.cat(v2_parts, dim=2)
        
        # Concatenate along the sequence dimension (dim=2)
        combined_key = torch.cat([keys1, keys2], dim=2)
        combined_value = torch.cat([values1, values2], dim=2)
        
        combined_keys.append(combined_key)
        combined_values.append(combined_value)
    
    combined_cache = TrainableCache(
        config=cache1.config,
        init_keys=combined_keys,
        init_values=combined_values,
        num_frozen_tokens=0,
    )
    
    # Move to the same device and dtype as cache1
    device = keys1.device
    combined_cache = combined_cache.to(device).to(torch.bfloat16)
    combined_cache.eval()
    
    logger.info(
        "Combined cartridge has %d trainable tokens (%d from cache1 + %d from cache2)",
        combined_cache._num_trainable_tokens,
        cache1._num_trainable_tokens,
        cache2._num_trainable_tokens,
    )
    
    return combined_cache


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def evaluate_standard_perplexity_base_model(
    model,
    eval_dataset: LossEvalDataset,
    local_rank: int,
    name: str,
):
    """Compute standard perplexity for the base model (no cartridge)."""
    import torch.nn.functional as F
    import math
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    logger.info("Evaluating standard perplexity (base model) `%s` (n=%d)", name, len(eval_dataset))
    
    model.eval()
    
    dataloader = DataLoader(
        eval_dataset,
        batch_size=1,
        collate_fn=_collate_fn_first,
        num_workers=0,
    )
    
    dataloader_pbar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"Base Model Perplexity Evaluation ({name})",
        leave=False,
    )
    
    with torch.no_grad():
        epoch_nll = 0.0
        epoch_denom = 0.0
        
        for batch in dataloader_pbar:
            batch: DatasetBatch
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=batch.input_ids.to(local_rank),
                    seq_ids=batch.element_ids.to(local_rank),
                    position_ids=batch.position_ids.to(local_rank),
                    logits_to_keep=batch.topk_token_idxs.to(local_rank),
                )

                topk_pred_logprobs = torch.gather(
                    F.log_softmax(outputs.logits, dim=-1)[0],
                    dim=-1,
                    index=batch.topk_token_ids.unsqueeze(-1).to(local_rank),
                ).squeeze(-1)
                nll = -topk_pred_logprobs.sum()
                epoch_nll += nll.item()
                epoch_denom += len(topk_pred_logprobs)
                
                # del outputs
        
        # Compute final metrics
        mean_nll = epoch_nll / epoch_denom if epoch_denom > 0 else float('inf')
        perplexity = math.exp(mean_nll)
        
        logger.info(f"Standard Perplexity (Base Model) ({name}): {perplexity:.4f} (mean NLL: {mean_nll:.4f}, tokens: {int(epoch_denom)})")
    
    return perplexity, mean_nll


def evaluate_standard_perplexity(
    cache: TrainableCache,
    model,
    eval_dataset: LossEvalDataset,
    local_rank: int,
    name: str,
):
    """Compute standard perplexity with a given cache."""
    import torch.nn.functional as F
    import math
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    logger.info("Evaluating standard perplexity `%s` (n=%d)", name, len(eval_dataset))
    logger.info("Cache has %d trainable tokens", cache._num_trainable_tokens)
    
    cache_and_model = CacheAndModel(cache, model)
    cache_and_model.eval()
    
    dataloader = DataLoader(
        eval_dataset,
        batch_size=1,
        collate_fn=_collate_fn_first,
        num_workers=0,
    )
    
    dataloader_pbar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"Standard Perplexity Evaluation ({name})",
        leave=False,
    )
    
    with torch.no_grad():
        epoch_nll = 0.0
        epoch_denom = 0.0
        
        for batch in dataloader_pbar:
            batch: DatasetBatch
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = cache_and_model(
                    input_ids=batch.input_ids.to(local_rank),
                    seq_ids=batch.element_ids.to(local_rank),
                    position_ids=batch.position_ids.to(local_rank),
                    logits_to_keep=batch.topk_token_idxs.to(local_rank),
                )

                topk_pred_logprobs = torch.gather(
                    F.log_softmax(outputs.logits, dim=-1)[0],
                    dim=-1,
                    index=batch.topk_token_ids.unsqueeze(-1).to(local_rank),
                ).squeeze(-1)

                nll = -topk_pred_logprobs.sum()
                epoch_nll += nll.item()
                epoch_denom += len(topk_pred_logprobs)
                
            cache.clear()
        
        # Compute final metrics
        mean_nll = epoch_nll / epoch_denom if epoch_denom > 0 else float('inf')
        perplexity = math.exp(mean_nll)
        
        logger.info(f"Standard Perplexity ({name}): {perplexity:.4f} (mean NLL: {mean_nll:.4f}, tokens: {int(epoch_denom)})")
    
    return perplexity, mean_nll


def run_perplexity_evaluation_base_model(
    name: str,
    model,
    eval_dataset: LossEvalDataset,
    system_prompt: Optional[str],
    local_rank: int,
    seed: int,
):
    """Run standard perplexity evaluation for the base model (no cartridge)."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {name} (Base Model - Standard Perplexity)")
    logger.info(f"{'='*60}")
    
    evaluate_standard_perplexity_base_model(
        model=model,
        eval_dataset=eval_dataset,
        local_rank=local_rank,
        name=name,
    )


def run_perplexity_evaluation(
    name: str,
    cache: TrainableCache,
    model,
    eval_dataset: LossEvalDataset,
    system_prompt: Optional[str],
    local_rank: int,
    seed: int,
):
    """Run standard perplexity evaluation with a given cache."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {name} (Standard Perplexity)")
    logger.info(f"{'='*60}")
    
    # Use standard perplexity calculation instead of teacher-weighted
    evaluate_standard_perplexity(
        cache=cache,
        model=model,
        eval_dataset=eval_dataset,
        local_rank=local_rank,
        name=name,
    )


def run_qa_evaluation(
    name: str,
    cache: TrainableCache,
    model,
    eval_dataset: FinanceBenchQADataset,
    system_prompt: Optional[str],
    local_rank: int,
    seed: int,
    model_config: HFModelConfig,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    batch_size: int = 4,
):
    """Run QA evaluation with a given cache and save per-example predictions.

    This implementation bypasses the generic evaluate_generations() helper and
    instead runs per-example generation using flex_generate, then writes a
    simple JSON file with question, gold answer, and model prediction.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {name} (QA - EM/F1)")
    logger.info(f"{'='*60}")
    
    results = []

    tokenizer = eval_dataset.tokenizer
    num_examples = len(eval_dataset)
    logger.info(f"Running QA generation on {num_examples} examples")

    for idx in tqdm(range(num_examples), desc=f"QA generation: {name}", leave=False):
        elem = eval_dataset[idx]

        # Ensure 1D tensor and move to device
        input_ids = elem.input_ids
        if input_ids.dim() == 2 and input_ids.shape[0] == 1:
            input_ids = input_ids.squeeze(0)
        input_ids = input_ids.to(local_rank)

        seq_ids = torch.zeros_like(input_ids, dtype=torch.long, device=local_rank)
        position_ids = torch.arange(input_ids.shape[0], device=local_rank)

        pred_ids = flex_generate(
            input_ids=input_ids,
            seq_ids=seq_ids,
            position_ids=position_ids,
            cache=cache,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            show_progress=False,
        )

        # Single sequence with id 0
        curr_pred_ids = pred_ids[0]
        pred_text = tokenizer.decode(curr_pred_ids, skip_special_tokens=True)

        # Compute EM/F1 using dataset scoring helper
        metrics, _ = eval_dataset.score(
            pred=pred_text, answer=elem.answer, convo_id=elem.convo_id
        )

        results.append(
            {
                "idx": idx,
                "prompt": elem.prompt,
                "answer": elem.answer,
                "pred": pred_text,
                "em": metrics.get("em"),
                "f1": metrics.get("f1"),
            }
        )

    # Aggregate metrics
    if results:
        avg_em = sum(r["em"] for r in results if r["em"] is not None) / len(results)
        avg_f1 = sum(r["f1"] for r in results if r["f1"] is not None) / len(results)
        logger.info(f"{name} QA results: EM={avg_em:.3f}, F1={avg_f1:.3f}")

    # Save per-example predictions to JSON
    slug = name.lower().replace(" ", "_").replace("/", "_")
    output_path = DATA_DIR / "eval" / f"{slug}_qa_predictions.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved QA predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate perplexity or QA accuracy with trained cartridges on eval questions."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the evaluation config YAML file",
    )
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Skip converting QA JSON to conversations (use existing parquet)",
    )
    parser.add_argument(
        "--tokasaurus",
        action="store_true",
        help="Use Tokasaurus for QA generation (bypasses local model generation).",
    )
    args = parser.parse_args()

    # Load .env values if present (for tokasaurus url, etc.)
    load_dotenv()
    
    # Load YAML config
    logger.info(f"Loading config from {args.config}")
    yaml_config = load_yaml_config(args.config)
    
    # Extract config values
    amd_cartridge_path = yaml_config.get("amd_cartridge_path")
    pepsi_cartridge_path = yaml_config.get("pepsi_cartridge_path")
    model_name = yaml_config["model_name"]
    eval_type = yaml_config.get("eval_type", "perplexity")
    system_prompt = yaml_config.get("system_prompt")
    device = yaml_config.get("device", "cuda")
    seed = yaml_config.get("seed", 42)
    
    # Validate that at least one cartridge is provided
    if not amd_cartridge_path and not pepsi_cartridge_path:
        raise ValueError(
            "At least one of 'amd_cartridge_path' or 'pepsi_cartridge_path' must be provided in the config."
        )
    
    # Validate eval_type
    if eval_type not in ["perplexity", "qa"]:
        raise ValueError(f"Unsupported eval_type: {eval_type}. Must be 'perplexity' or 'qa'.")
    
    # QA evaluation parameters
    max_new_tokens = yaml_config.get("max_new_tokens", 256)
    temperature = yaml_config.get("temperature", 0.0)
    batch_size = yaml_config.get("batch_size", 4)
    tokasaurus_url = yaml_config.get("tokasaurus_url", os.environ.get("TOKASAURUS_URL", "http://localhost:10210"))
    
    # Determine which eval QA files to use (allow overrides from config)
    amd_qa_json = yaml_config.get(
        "amd_qa_json",
        str(DATA_DIR / "eval" / "amd_qa_financebench.json"),
    )
    pepsi_qa_json = yaml_config.get(
        "pepsi_qa_json",
        str(DATA_DIR / "eval" / "pepsi_qa_financebench.json"),
    )
    
    # Load the model
    logger.info(f"Loading model: {model_name}")
    model_config = HFModelConfig(
        pretrained_model_name_or_path=model_name,
        model_cls=FlexLlamaForCausalLM,
        tuning_method="custom_prefix",
    )
    model = model_config.instantiate()
    
    # Handle device
    if isinstance(device, str):
        if device == "cuda":
            device_str = "cuda"
            local_rank = 0
        else:
            device_str = device
            local_rank = 0
    else:
        device_str = f"cuda:{device}"
        local_rank = device
    
    model = model.to(device_str).to(torch.bfloat16)
    model.eval()
    
    seed_everything(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load cartridges and prepare datasets
    amd_cache = None
    pepsi_cache = None
    amd_dataset = None
    pepsi_dataset = None
    mixed_dataset = None
    
    if amd_cartridge_path:
        amd_cache = load_cartridge(amd_cartridge_path, model, device_str)
        
        # Prepare AMD dataset
        amd_parquet = str(Path(amd_qa_json).with_suffix(".parquet"))
        if not args.skip_conversion:
            if not os.path.exists(amd_qa_json):
                raise FileNotFoundError(f"AMD QA JSON not found: {amd_qa_json}")
            convert_qa_to_conversations(amd_qa_json, amd_parquet, system_prompt)
        elif not os.path.exists(amd_parquet):
            raise FileNotFoundError(f"AMD conversations parquet not found: {amd_parquet}")
        
        if eval_type == "perplexity":
            amd_dataset = LossEvalDataset.Config(
                data_source=DataSource(path=amd_parquet, type="local"),
                packed_seq_length=2048,
                packing_mode="pad",  # LossEvalDataset only supports truncate/pad
                system_prompt=system_prompt,
            ).instantiate(tokenizer=tokenizer, seed=seed)
        else:  # qa
            amd_dataset = FinanceBenchQADataset.Config(
                data_source=DataSource(path=amd_parquet, type="local"),
                cot=False,
            ).instantiate(tokenizer=tokenizer, seed=seed)
    
    if pepsi_cartridge_path:
        pepsi_cache = load_cartridge(pepsi_cartridge_path, model, device_str)
        
        # Prepare Pepsi dataset
        pepsi_parquet = str(Path(pepsi_qa_json).with_suffix(".parquet"))
        if not args.skip_conversion:
            if not os.path.exists(pepsi_qa_json):
                raise FileNotFoundError(f"Pepsi QA JSON not found: {pepsi_qa_json}")
            convert_qa_to_conversations(pepsi_qa_json, pepsi_parquet, system_prompt)
        elif not os.path.exists(pepsi_parquet):
            raise FileNotFoundError(f"Pepsi conversations parquet not found: {pepsi_parquet}")
        
        if eval_type == "perplexity":
            pepsi_dataset = LossEvalDataset.Config(
                data_source=DataSource(path=pepsi_parquet, type="local"),
                packed_seq_length=2048,
                packing_mode="pad",  # LossEvalDataset only supports truncate/pad
                system_prompt=system_prompt,
            ).instantiate(tokenizer=tokenizer, seed=seed)
        else:  # qa
            pepsi_dataset = FinanceBenchQADataset.Config(
                data_source=DataSource(path=pepsi_parquet, type="local"),
                cot=False,
            ).instantiate(tokenizer=tokenizer, seed=seed)
    
    # Run base model evaluations (baseline) - only for perplexity
    if eval_type == "perplexity":
        if amd_dataset:
            run_perplexity_evaluation_base_model(
                name="Base model on AMD questions",
                model=model,
                eval_dataset=amd_dataset,
                system_prompt=system_prompt,
                local_rank=local_rank,
                seed=seed,
            )
        
        if pepsi_dataset:
            run_perplexity_evaluation_base_model(
                name="Base model on PepsiCo questions",
                model=model,
                eval_dataset=pepsi_dataset,
                system_prompt=system_prompt,
                local_rank=local_rank,
                seed=seed,
            )
    
    # Run cartridge evaluations
    if amd_cache and amd_dataset and not args.tokasaurus:
        if eval_type == "perplexity":
            run_perplexity_evaluation(
                name="AMD cartridge on AMD questions",
                cache=amd_cache,
                model=model,
                eval_dataset=amd_dataset,
                system_prompt=system_prompt,
                local_rank=local_rank,
                seed=seed,
            )
        else:  # qa
            run_qa_evaluation(
                name="AMD cartridge on AMD questions",
                cache=amd_cache,
                model=model,
                eval_dataset=amd_dataset,
                system_prompt=system_prompt,
                local_rank=local_rank,
                seed=seed,
                model_config=model_config,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                batch_size=batch_size,
            )
    
    if pepsi_cache and pepsi_dataset and not args.tokasaurus:
        if eval_type == "perplexity":
            run_perplexity_evaluation(
                name="PepsiCo cartridge on PepsiCo questions",
                cache=pepsi_cache,
                model=model,
                eval_dataset=pepsi_dataset,
                system_prompt=system_prompt,
                local_rank=local_rank,
                seed=seed,
            )
        else:  # qa
            run_qa_evaluation(
                name="PepsiCo cartridge on PepsiCo questions",
                cache=pepsi_cache,
                model=model,
                eval_dataset=pepsi_dataset,
                system_prompt=system_prompt,
                local_rank=local_rank,
                seed=seed,
                model_config=model_config,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                batch_size=batch_size,
            )
    
    # Composition evaluation if both cartridges are available (local path only)
    if amd_cache and pepsi_cache and amd_dataset and pepsi_dataset and not args.tokasaurus:
        logger.info("\nPreparing composition evaluation...")
        
        # For perplexity: manually combine cartridges (needed for local model access)
        combined_cache = combine_cartridges(amd_cache, pepsi_cache)
        
        # Create mixed dataset (AMD + Pepsi questions)
        mixed_parquet = str(DATA_DIR / "eval" / "mixed_qa_conversations.parquet")
        
        if not args.skip_conversion or not os.path.exists(mixed_parquet):
            # Load both QA JSONs and combine
            with open(amd_qa_json, "r") as f:
                amd_qa_list = json.load(f)
            with open(pepsi_qa_json, "r") as f:
                pepsi_qa_list = json.load(f)
            
            mixed_qa_list = amd_qa_list + pepsi_qa_list
            mixed_qa_json = str(DATA_DIR / "eval" / "mixed_qa_financebench.json")
            with open(mixed_qa_json, "w") as f:
                json.dump(mixed_qa_list, f)
            
            convert_qa_to_conversations(mixed_qa_json, mixed_parquet, system_prompt)
        
        if eval_type == "perplexity":
            mixed_dataset = LossEvalDataset.Config(
                data_source=DataSource(path=mixed_parquet, type="local"),
                packed_seq_length=2048,
                packing_mode="pad",  # LossEvalDataset only supports truncate/pad
                system_prompt=system_prompt,
            ).instantiate(tokenizer=tokenizer, seed=seed)
            
            # Evaluate base model on mixed dataset first
            run_perplexity_evaluation_base_model(
                name="Base model on mixed questions",
                model=model,
                eval_dataset=mixed_dataset,
                system_prompt=system_prompt,
                local_rank=local_rank,
                seed=seed,
            )
            
            # Then evaluate with composed cartridges
            run_perplexity_evaluation(
                name="Composed [AMD, PepsiCo] cartridges on mixed questions",
                cache=combined_cache,
                model=model,
                eval_dataset=mixed_dataset,
                system_prompt=system_prompt,
                local_rank=local_rank,
                seed=seed,
            )
        else:  # qa
            mixed_dataset = FinanceBenchQADataset.Config(
                data_source=DataSource(path=mixed_parquet, type="local"),
                cot=False,
            ).instantiate(tokenizer=tokenizer, seed=seed)
            
            run_qa_evaluation(
                name="Composed [AMD, PepsiCo] cartridges on mixed questions",
                cache=combined_cache,
                model=model,
                eval_dataset=mixed_dataset,
                system_prompt=system_prompt,
                local_rank=local_rank,
                seed=seed,
                model_config=model_config,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                batch_size=batch_size,
            )
    
    logger.info("\n" + "="*60)
    logger.info("All evaluations complete!")
    logger.info("="*60)

    # Tokasaurus QA generation path (Gemini QAs) if requested
    if args.tokasaurus and eval_type == "qa":
        logger.info("\nRunning Tokasaurus QA generations (Gemini QAs)...")
        qa_splits = []
        if amd_qa_json and os.path.exists(amd_qa_json):
            qa_splits.append(("amd", amd_qa_json))
        if pepsi_qa_json and os.path.exists(pepsi_qa_json):
            qa_splits.append(("pepsi", pepsi_qa_json))

        # Four configs: A) amd on amd, B) pepsi on pepsi, C) amd+pepsi on amd, D) amd+pepsi on pepsi
        cartridge_sets = {
            "amd_only": [{"id": "amd_gemini", "source": "local"}],
            "pepsi_only": [{"id": "pepsi_gemini", "source": "local"}],
            "both": [
                {"id": "amd_gemini", "source": "local"},
                {"id": "pepsi_gemini", "source": "local"},
            ],
        }

        results = []
        for split_name, qa_path in qa_splits:
            with open(qa_path, "r") as f:
                qa_list = json.load(f)

            for config_name, cartridges in [
                ("amd_only", cartridge_sets["amd_only"]),
                ("pepsi_only", cartridge_sets["pepsi_only"]),
                ("both", cartridge_sets["both"]),
            ]:
                # Skip mismatched cases (e.g., amd_only on pepsi split if you don't want it)
                if config_name == "amd_only" and split_name != "amd":
                    continue
                if config_name == "pepsi_only" and split_name != "pepsi":
                    continue

                logger.info(f"Tokasaurus eval: {config_name} on {split_name} split (n={len(qa_list)})")
                for idx, qa in enumerate(tqdm(qa_list, desc=f"{config_name} on {split_name}", leave=False)):
                    question = qa.get("question", "")
                    answer = qa.get("answer", "")
                    try:
                        pred = chat_with_tokasaurus(
                            url=tokasaurus_url,
                            question=question,
                            cartridges=cartridges,
                            system_prompt=system_prompt,
                            max_tokens=max_new_tokens,
                            temperature=temperature,
                        )
                    except Exception as e:
                        pred = f"[ERROR] {e}"
                    results.append(
                        {
                            "split": split_name,
                            "config": config_name,
                            "idx": idx,
                            "question": question,
                            "answer": answer,
                            "pred": pred,
                        }
                    )

        out_path = DATA_DIR / "eval" / "tokasaurus_qa_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Tokasaurus QA results written to {out_path}")


if __name__ == "__main__":
    main()
