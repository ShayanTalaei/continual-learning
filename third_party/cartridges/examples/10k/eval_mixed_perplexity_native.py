import argparse
import json
import os
import re
import math
import torch
import torch.nn.functional as F
import wandb
from pathlib import Path
from transformers import AutoTokenizer

from cartridges.train import evaluate_perplexity, CacheAndModel, TrainConfig, LossEvalConfig
from cartridges.cache import TrainableCache
from cartridges.datasets import LossEvalDataset, DataSource, TrainDataset
from cartridges.models import HFModelConfig, FlexLlamaForCausalLM
from cartridges.utils import get_logger, seed_everything
from cartridges.structs import Conversation, write_conversations

logger = get_logger(__name__)

def convert_qa_to_parquet(qa_json_path, output_parquet_path):
    """Converts QA JSON to Parquet format for LossEvalDataset."""
    print(f"Converting {qa_json_path} to {output_parquet_path}...")
    with open(qa_json_path, "r") as f:
        qa_list = json.load(f)
    
    conversations = []
    for i, qa in enumerate(qa_list):
        conversations.append(Conversation(
            messages=[
                Conversation.Message(role="user", content=qa.get("question", ""), token_ids=None),
                Conversation.Message(role="assistant", content=qa.get("answer", ""), token_ids=None)
            ],
            system_prompt="You are a helpful financial analyst assistant.", 
            metadata={"question_id": str(i)},
            type="qa",
        ))
    write_conversations(conversations, output_parquet_path)
    return output_parquet_path

def combine_cartridges(cache1: TrainableCache, cache2: TrainableCache) -> TrainableCache:
    """Combines two cartridges by concatenating their key/value tensors along the sequence dimension."""
    assert cache1.config == cache2.config, "Cartridges must have matching attention configurations"
    
    combined_keys = []
    combined_values = []
    
    for layer_idx in range(cache1.config.n_layers):
        # Helper to extract full tensor (frozen + trainable) for a specific layer
        def get_full_kv(c, idx):
            k_parts, v_parts = [], []
            # Check for frozen tokens (e.g. initial sink tokens)
            if getattr(c, "_num_frozen_tokens", 0) > 0:
                k_parts.append(c.frozen_keys[idx])
                v_parts.append(c.frozen_values[idx])
            # Add trainable tokens
            k_parts.append(c.trainable_keys[idx])
            v_parts.append(c.trainable_values[idx])
            return torch.cat(k_parts, dim=2), torch.cat(v_parts, dim=2)

        k1, v1 = get_full_kv(cache1, layer_idx)
        k2, v2 = get_full_kv(cache2, layer_idx)
        
        # Concatenate along sequence dimension (dim=2)
        combined_keys.append(torch.cat([k1, k2], dim=2))
        combined_values.append(torch.cat([v1, v2], dim=2))
        
    combined_cache = TrainableCache(
        config=cache1.config,
        init_keys=combined_keys,
        init_values=combined_values,
        num_frozen_tokens=0, # Treat the entire combined history as one block
    )
    return combined_cache

def load_cache_from_wandb(run_name_or_id, project="cdingg/cartridges", device="cuda", step=None):
    """Finds a W&B run by name/ID, downloads the latest cache checkpoint, and loads it."""
    print(f"Looking for run '{run_name_or_id}' in project '{project}'...")
    api = wandb.Api()
    
    # Handle project format: could be "entity/project" or just "project"
    if "/" not in project:
        # Try to get entity from logged-in user
        try:
            entity = api.viewer.username
            project = f"{entity}/{project}"
        except:
            raise ValueError(f"Could not determine entity. Please provide project as 'entity/project' format.")
    
    # 1. Try to find run
    run = None
    try:
        # First try as direct Run ID (e.g., 'uvju0o35')
        run = api.run(f"{project}/{run_name_or_id}")
        print(f"Found run by ID: {run.name} (ID: {run.id})")
    except Exception as e1:
        try:
            # Try as Display Name (e.g., 'amd-10k-cartridge-gemini-eval')
            runs = api.runs(path=project, filters={"display_name": {"$regex": run_name_or_id}})
            run_list = list(runs)
            if not run_list:
                # Try exact match
                runs = api.runs(path=project)
                run_list = [r for r in runs if r.name == run_name_or_id or run_name_or_id in r.name]
            
            if not run_list:
                raise ValueError(f"Could not find run with name or ID '{run_name_or_id}' in {project}")
            run = run_list[0]
            print(f"Found run by name: {run.name} (ID: {run.id})")
        except Exception as e2:
            raise ValueError(f"Could not find run '{run_name_or_id}' in {project}: {e2}")
    
    # 2. Find cache files
    all_files = list(run.files())
    files = [f.name for f in all_files if re.match(r"^cache-.*\.pt$", f.name)]
    if not files:
        # Fallback to generic names if step-based naming wasn't used
        files = [f.name for f in all_files if f.name.endswith("cartridge.pt") or f.name.endswith(".pt")]
    
    if not files:
        available = [f.name for f in all_files[:10]]
        raise ValueError(f"No cache files found in run {run.id}. Available files: {available}")
        
    # 3. Select the checkpoint
    def get_step(name):
        # Extract step number from filename like 'cache-step128.pt'
        match = re.search(r"cache-step(\d+)\.pt", name)
        if match:
            return int(match.group(1))
        # Fallback: look for any number
        match = re.search(r"(\d+)", name)
        return int(match.group(1)) if match else 0
    

    best_file = sorted(files, key=get_step)[-1]
    selected_step = get_step(best_file)
    print(f"Selected latest checkpoint: {best_file} (step {selected_step}, from {len(files)} available)")
    
    # 4. Download
    cache_dir = Path(f"artifacts/{run.id}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / best_file
    
    if not path.exists():
        print(f"Downloading to {path}...")
        run.file(best_file).download(root=cache_dir)
    else:
        print(f"Found cached file at {path}")
        
    return TrainableCache.from_pretrained(str(path), device=device)

def main():
    parser = argparse.ArgumentParser(description="Evaluate mixed perplexity on cartridges from WandB")
    parser.add_argument("--amd-run", default="amd-10k-cartridge-gemini-eval", help="WandB run name for AMD")
    parser.add_argument("--pepsi-run", default="pepsi-10k-cartridge-gemini-eval", help="WandB run name for Pepsi")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--data-dir", default="data/10k/eval")
    parser.add_argument("--project", default="cdingg/cartridges", help="WandB project name (entity/project)")
    parser.add_argument("--cache-step", type=int, default=172, help="Specific cache step to load (default: 172)")
    args = parser.parse_args()

    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Prepare All Datasets (AMD-only, Pepsi-only, Mixed)
    data_dir = Path(args.data_dir)
    # Verify inputs exist
    if not (data_dir / "amd_qa_gemini.json").exists():
         raise FileNotFoundError(f"Could not find amd_qa_gemini.json in {data_dir}")

    with open(data_dir / "amd_qa_gemini.json") as f: amd_data = json.load(f)
    with open(data_dir / "pepsi_qa_gemini.json") as f: pepsi_data = json.load(f)
    
    print(f"Loaded {len(amd_data)} AMD questions and {len(pepsi_data)} Pepsi questions")
    
    # Create AMD-only dataset
    amd_json_path = data_dir / "amd_only_qa_gemini_generated.json"
    with open(amd_json_path, "w") as f:
        json.dump(amd_data, f, indent=2)
    amd_parquet = str(amd_json_path).replace(".json", ".parquet")
    convert_qa_to_parquet(amd_json_path, amd_parquet)
    
    # Create Pepsi-only dataset
    pepsi_json_path = data_dir / "pepsi_only_qa_gemini_generated.json"
    with open(pepsi_json_path, "w") as f:
        json.dump(pepsi_data, f, indent=2)
    pepsi_parquet = str(pepsi_json_path).replace(".json", ".parquet")
    convert_qa_to_parquet(pepsi_json_path, pepsi_parquet)
    
    # Create Mixed dataset
    mixed_json_path = data_dir / "mixed_qa_gemini_generated.json"
    with open(mixed_json_path, "w") as f:
        json.dump(amd_data + pepsi_data, f, indent=2)
    mixed_parquet = str(mixed_json_path).replace(".json", ".parquet")
    convert_qa_to_parquet(mixed_json_path, mixed_parquet)

    # 2. Initialize Model & Tokenizer
    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    model_config = HFModelConfig(
        pretrained_model_name_or_path=args.model_name,
        model_cls=FlexLlamaForCausalLM, 
        tuning_method="custom_prefix",
    )
    model = model_config.instantiate().to(device).to(torch.bfloat16).eval()

    # 3. Load Cartridges from WandB
    print("Loading cartridges from WandB...")
    # Load specified cache step for both runs
    amd_cache = load_cache_from_wandb(args.amd_run, project=args.project, device=device, step=args.cache_step)
    amd_cache = amd_cache.to(device).to(torch.bfloat16).eval()
    # Ensure seq_ids buffer is on the correct device
    if amd_cache._seq_ids is not None:
        amd_cache._seq_ids = amd_cache._seq_ids.to(device)
    if amd_cache._init_seq_ids is not None:
        amd_cache._init_seq_ids = amd_cache._init_seq_ids.to(device)
    
    pepsi_cache = load_cache_from_wandb(args.pepsi_run, project=args.project, device=device, step=args.cache_step)
    pepsi_cache = pepsi_cache.to(device).to(torch.bfloat16).eval()
    # Ensure seq_ids buffer is on the correct device
    if pepsi_cache._seq_ids is not None:
        pepsi_cache._seq_ids = pepsi_cache._seq_ids.to(device)
    if pepsi_cache._init_seq_ids is not None:
        pepsi_cache._init_seq_ids = pepsi_cache._init_seq_ids.to(device)

    # 4. Prepare All Datasets
    print("\n" + "="*60)
    print("Preparing evaluation datasets...")
    print("="*60)
    
    def create_dataset_config(parquet_path, name):
        return LossEvalConfig(
            dataset=LossEvalDataset.Config(
                data_source=DataSource(path=parquet_path, type="local"),
                packed_seq_length=2048,
                packing_mode="pad",
            ),
            name_for_wandb=name
        )
    
    amd_ds_config = create_dataset_config(amd_parquet, "amd_only")
    pepsi_ds_config = create_dataset_config(pepsi_parquet, "pepsi_only")
    mixed_ds_config = create_dataset_config(mixed_parquet, "mixed")
    
    amd_dataset = amd_ds_config.dataset.instantiate(tokenizer=tokenizer, seed=42)
    pepsi_dataset = pepsi_ds_config.dataset.instantiate(tokenizer=tokenizer, seed=42)
    mixed_dataset = mixed_ds_config.dataset.instantiate(tokenizer=tokenizer, seed=42)
    
    print(f"AMD-only dataset: {len(amd_dataset)} batches, {len(amd_dataset.elements)} elements")
    print(f"Pepsi-only dataset: {len(pepsi_dataset)} batches, {len(pepsi_dataset.elements)} elements")
    print(f"Mixed dataset: {len(mixed_dataset)} batches, {len(mixed_dataset.elements)} elements")
    
    # Placeholder train config needed for evaluate_perplexity signature
    train_config = TrainConfig(
        model=model_config,
        dataset=TrainDataset.Config(data_sources=[DataSource(path="dummy", type="local")]),
        output_dir=".",
        wandb=None 
    )

    # Store results for summary (unused for now, kept for future extensions)
    results = {}
    
    def run_naive_perplexity(
        model_to_use,
        eval_dataset,
        dataset_name,
        local_rank: int = 0,
        cache: TrainableCache | None = None,
        max_elements: int | None = None,
    ):
        """
        Naive perplexity: run each conversation as its own batch (no cross-conversation packing),
        and compute perplexity directly from logits and ground-truth tokens.
        """
        model_to_use.eval()
        total_nll = 0.0
        total_tokens = 0
        
        elements = eval_dataset.elements
        if max_elements is not None:
            elements = elements[:max_elements]
        
        print(f"\n[Naive PPL] Evaluating {dataset_name} on {len(elements)} un-packed elements...")
        
        with torch.no_grad():
            for idx, elem in enumerate(elements):
                batch = eval_dataset.collate([elem])
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model_to_use(
                        input_ids=batch.input_ids.to(local_rank),
                        seq_ids=batch.element_ids.to(local_rank),
                        position_ids=batch.position_ids.to(local_rank),
                    )
                    # Same indexing as train/evaluate_perplexity
                    log_probs = F.log_softmax(outputs.logits, dim=-1)[
                        0,
                        batch.topk_token_idxs.to(local_rank) - 1,
                        batch.topk_token_ids.to(local_rank),
                    ]
                    nll = -log_probs.sum().item()
                    total_nll += nll
                    total_tokens += log_probs.shape[0]
                
                # Clear cache between elements if using a TrainableCache
                if cache is not None:
                    cache.clear()
        
        mean_nll = total_nll / total_tokens if total_tokens > 0 else float("inf")
        ppl = math.exp(mean_nll)
        print(f"[Naive PPL] {dataset_name}: perplexity={ppl:.4f}, mean NLL={mean_nll:.4f}, tokens={total_tokens}")
        return ppl, mean_nll
    
    def run_base_model_eval(dataset, ds_config, dataset_name):
        """Evaluate the base model (no cartridge) on a dataset using naive perplexity."""
        print(f"\n>>> Evaluating Base Model on {dataset_name} Dataset (naive, no packing) <<<")
        run_naive_perplexity(
            model_to_use=model,
            eval_dataset=dataset,
            dataset_name=f"{dataset_name} (Base Model)",
            local_rank=0,
            cache=None,
        )
    
    def run_single_eval(cartridge_name, cache, dataset, ds_config, dataset_name):
        """Evaluate a cartridge on a dataset using naive perplexity."""
        print(f"\n>>> Evaluating {cartridge_name} on {dataset_name} Dataset (naive, no packing) <<<")
        wrapped_model = CacheAndModel(cache, model)
        run_naive_perplexity(
            model_to_use=wrapped_model,
            eval_dataset=dataset,
            dataset_name=f"{dataset_name} ({cartridge_name})",
            local_rank=0,
            cache=cache,
        )

    # 5. Run Base Model Evaluations (no cartridges)
    print("\n" + "="*60)
    print("Running Base Model Evaluations (no cartridges)")
    print("="*60)
    for ds_name, ds, ds_config in [
        ("AMD-only", amd_dataset, amd_ds_config),
        ("Pepsi-only", pepsi_dataset, pepsi_ds_config),
        ("Mixed", mixed_dataset, mixed_ds_config),
    ]:
        run_base_model_eval(ds, ds_config, ds_name)

    # 6. Run All 9 Cartridge Evaluations
    print("\n" + "="*60)
    print("Running All Evaluations (3 cartridges × 3 datasets = 9 total)")
    print("="*60)
    
    # Prepare combined cache
    print("\n>>> Combining Cartridges... <<<")
    combined_cache = combine_cartridges(pepsi_cache, amd_cache)
    combined_cache = combined_cache.to(device).to(torch.bfloat16).eval()
    if combined_cache._seq_ids is not None:
        combined_cache._seq_ids = combined_cache._seq_ids.to(device)
    if combined_cache._init_seq_ids is not None:
        combined_cache._init_seq_ids = combined_cache._init_seq_ids.to(device)
    
    # Evaluation matrix: cartridge × dataset
    cartridges = [
        ("AMD Cartridge", amd_cache),
        ("Pepsi Cartridge", pepsi_cache),
        ("Combined Cartridge", combined_cache),
    ]
    
    datasets = [
        ("AMD-only", amd_dataset, amd_ds_config),
        ("Pepsi-only", pepsi_dataset, pepsi_ds_config),
        ("Mixed", mixed_dataset, mixed_ds_config),
    ]
    
    for cart_name, cart_cache in cartridges:
        for ds_name, ds, ds_config in datasets:
            run_single_eval(cart_name, cart_cache, ds, ds_config, ds_name)
    
    print("\n" + "="*60)
    print("All evaluations complete!")
    print("="*60)

if __name__ == "__main__":
    main()

