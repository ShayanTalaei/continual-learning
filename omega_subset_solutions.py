import os
import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset, get_dataset_config_names
from tqdm import tqdm
import json

# Reuse in-repo LM client and evaluator
from src.lm.language_model import LMConfig
from src.lm.gemini_client import GeminiConfig, GeminiClient
from src.data.envs.omega_math_env import evaluate_answer, OmegaMathEnv
from src.utils import logger as jsonlogger


def extract_user_prompt(messages: Any) -> str:
    if isinstance(messages, list):
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content", ""))
    return ""


def build_prompt(system_prompt: Optional[str], user_prompt: str) -> Tuple[str, str]:
    sys = system_prompt or (
        "You are a careful math assistant. It is absolutely essential that you put your final "
        "answer in the \\boxed{{}} format—do not forget this, as your answer will not be parsed otherwise. "
        "Always clearly present your final answer in \\boxed{{}}. "
        "Respond strictly as JSON with keys 'final_answer' and 'rationale' (both required). "
        "Place the LaTeX-boxed final answer string in the 'final_answer' field and a concise explanation in 'rationale'."
    )
    return sys, user_prompt


def run_gemini_calls(
    lm: GeminiClient,
    system_prompt: str,
    user_prompt: str,
    repeats: int,
    expect_json: bool,
    context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    # Optionally enable JSON schema via json logger context if needed in future
    outputs: List[str] = []
    schema = OmegaMathEnv.OmegaAnswerSchema.model_json_schema()
    # Enforce rationale as required in schema
    try:
        req = set(schema.get("required", []))
        req.update(["final_answer", "rationale"])
        schema["required"] = list(req)
    except Exception:
        pass
    
    for _ in range(repeats):
        with jsonlogger.json_log_context(response_schema=schema):
            txt = lm.call(system_prompt, user_prompt)
        outputs.append(txt)
    return outputs


def evaluate_outputs(
    outputs: List[str],
    ground_truth: str,
    eval_mode: str,
    eval_tol: float,
    expect_boxed: bool,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for out in outputs:
        pred_for_eval: str = out
        rationale: Optional[str] = None
        # Try to parse Omega JSON schema to extract final_answer
        try:
            data = json.loads(out)
            parsed = OmegaMathEnv.OmegaAnswerSchema(**data)
            pred_for_eval = parsed.final_answer
            rationale = getattr(parsed, "rationale", None)
        except Exception:
            pass
        res = evaluate_answer(pred_for_eval, ground_truth, mode=eval_mode, tol=eval_tol, expect_boxed=expect_boxed)
        results.append({
            "model_output": out,
            "final_answer": pred_for_eval,
            "rationale": rationale,
            "is_correct": bool(res.get("score", 0) == 1),
            "evaluation": res,
        })
    return results


def process_example(
    ex: Dict[str, Any],
    lm: GeminiClient,
    repeats: int,
    expect_boxed: bool,
    eval_mode: str,
    eval_tol: float,
    system_prompt: Optional[str],
) -> Dict[str, Any]:
    user_prompt = extract_user_prompt(ex.get("messages"))
    sys_prompt, usr_prompt = build_prompt(system_prompt, user_prompt)

    outputs = run_gemini_calls(
        lm=lm,
        system_prompt=sys_prompt,
        user_prompt=usr_prompt,
        repeats=repeats,
        expect_json=False,
        context=None,
    )
    evals = evaluate_outputs(outputs, ex.get("ground_truth", ""), eval_mode, eval_tol, expect_boxed)

    # ground_truth_solution can be taken as any correct model_output if exists, else empty
    correct_outputs = [e["model_output"] for e in evals if e.get("is_correct")]
    ground_truth_solution = correct_outputs[0] if correct_outputs else ""

    return {
        **ex,
        "ground_truth_solution": ground_truth_solution,
    }


def sample_per_subset(
    ds_name: str,
    subsets: Optional[List[str]],
    split: str,
    per_subset: int,
    seed: int,
    num_subsets: Optional[int] = None,
) -> List[Dict[str, Any]]:
    # If subsets not specified, use all config names
    if subsets is None:
        from datasets import get_dataset_config_names
        subsets = get_dataset_config_names(ds_name)

    rng = random.Random(seed)
    # Optionally limit number of subsets (random, seed-controlled)
    if num_subsets is not None and num_subsets > 0 and len(subsets) > num_subsets:
        tmp = list(subsets)
        rng.shuffle(tmp)
        subsets = tmp[:num_subsets]
    examples: List[Dict[str, Any]] = []
    for subset in subsets:
        sub = load_dataset(ds_name, subset, split=split)
        idxs = list(range(len(sub)))
        rng.shuffle(idxs)
        take = idxs[: min(per_subset, len(idxs))]
        for i in take:
            ex = dict(sub[i])
            ex["subset"] = subset
            examples.append(ex)
    return examples


def push_to_hub(
    records: List[Dict[str, Any]],
    repo_id: str,
    private: bool,
) -> None:
    train_ds = Dataset.from_list(records)
    dsd = DatasetDict({"train": train_ds})
    dsd.push_to_hub(repo_id, private=private)


def upload_periodically(
    results: List[Dict[str, Any]],
    repo_id: str,
    private: bool,
    upload_interval: int,
    last_upload_count: int,
) -> int:
    """Upload to HF if we have enough new records since last upload."""
    current_count = len(results)
    if current_count - last_upload_count >= upload_interval:
        print(f"Uploading {current_count} records to {repo_id}...")
        push_to_hub(results, repo_id, private)
        print(f"Upload completed. Total records: {current_count}")
        return current_count
    return last_upload_count


def main():
    parser = argparse.ArgumentParser(description="Create OMEGA subset with Gemini solutions and correctness flags")
    parser.add_argument("--hf_dataset", default="allenai/omega-explorative", help="Source HF dataset")
    parser.add_argument("--split", default="train", help="Split to sample from")
    parser.add_argument("--subsets", nargs="*", default=None, help="Specific subsets (configs). Default: all")
    parser.add_argument("--subset_name", default=None, help="Process only this single subset/config name")
    parser.add_argument("--per_subset", type=int, default=20, help="Num examples to sample per subset")
    parser.add_argument("--target_correct_per_subset", type=int, default=10, help="Target number of correct solutions per subset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_subsets", type=int, default=None, help="Max number of subsets to process (randomly chosen with seed)")

    parser.add_argument("--model", default="gemini-2.5-pro", help="Model name for Gemini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_output_tokens", type=int, default=65536)
    parser.add_argument("--repeats", type=int, default=4, help="How many calls per example")
    parser.add_argument("--threads", type=int, default=50, help="Parallel threads for inference")

    parser.add_argument("--eval_mode", default="numeric_tol", help="Evaluation mode (auto, numeric_tol, ...)")
    parser.add_argument("--eval_tol", type=float, default=1e-6)
    parser.add_argument("--expect_boxed", action="store_true", help="Expect answers in \\boxed{}")
    parser.add_argument("--system_prompt", default=None, help="Override system prompt")

    parser.add_argument("--hub_repo", required=True, help="Target HF repo id, e.g. user/omega-explorative-with-solutions")
    parser.add_argument("--private", action="store_true", help="Create/update as private dataset")
    parser.add_argument("--upload_interval", type=int, default=20, help="Upload to HF every N rows (default: 20)")

    parser.add_argument("--log_calls", action="store_true", help="Enable JSON call logging to outputs directory")
    parser.add_argument("--calls_dir", default="outputs/omega_subset_calls", help="Directory for call logs if enabled")

    args = parser.parse_args()

    # Build LM client
    lm_cfg = GeminiConfig(
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )
    lm = GeminiClient(config=lm_cfg)

    # Determine subsets to process (single subset overrides list)
    if args.subset_name:
        subsets_to_use = [args.subset_name]
    elif args.subsets:
        subsets_to_use = args.subsets
    else:
        # If no subsets specified, get all available subsets
        subsets_to_use = get_dataset_config_names(args.hf_dataset)
        if args.num_subsets:
            subsets_to_use = subsets_to_use[:args.num_subsets]

    # Track correct solutions per subset
    subset_correct_counts = {}
    subset_total_counts = {}
    results: List[Dict[str, Any]] = []
    last_upload_count = 0
    start = time.time()
    
    # Initialize counters for all subsets
    for subset in subsets_to_use:
        subset_correct_counts[subset] = 0
        subset_total_counts[subset] = 0
    
    # Process examples dynamically, stopping when target is reached per subset
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        # Load examples for each subset
        all_examples = []
        for subset in subsets_to_use:
            sub = load_dataset(args.hf_dataset, subset, split=args.split)
            idxs = list(range(len(sub)))
            random.Random(args.seed).shuffle(idxs)
            take = idxs[:min(args.per_subset, len(idxs))]
            for i in take:
                ex = dict(sub[i])
                ex["subset"] = subset
                all_examples.append(ex)
        
        # Shuffle all examples
        random.Random(args.seed).shuffle(all_examples)
        
        futures = []
        processed_count = 0
        example_index = 0
        
        with tqdm(total=len(all_examples), desc="Processing examples", unit="ex") as pbar:
            while example_index < len(all_examples) or futures:
                # Process completed futures first
                completed_futures = [f for f in futures if f.done()]
                for f in completed_futures:
                    try:
                        result = f.result()
                        subset = result["subset"]
                        
                        # Update total count for this subset
                        subset_total_counts[subset] += 1
                        
                        # Check if this is a correct solution
                        if result.get("ground_truth_solution"):
                            subset_correct_counts[subset] += 1
                        
                        results.append(result)
                        
                        # Check for periodic upload
                        last_upload_count = upload_periodically(
                            results, args.hub_repo, args.private, 
                            args.upload_interval, last_upload_count
                        )
                        
                    except Exception as e:
                        # Skip on failure, but keep going
                        print(f"Skipping example because of failure: {e}")
                        pass
                    finally:
                        futures.remove(f)
                        pbar.update(1)
                
                # Submit new jobs if we have available workers and more examples to process
                while (len(futures) < args.threads and 
                       example_index < len(all_examples)):
                    
                    ex = all_examples[example_index]
                    subset = ex["subset"]
                    example_index += 1
                    
                    # Skip if we already have enough correct solutions for this subset
                    if subset_correct_counts[subset] >= args.target_correct_per_subset:
                        print(f"Skipping subset {subset} because we already have enough correct solutions")
                        pbar.update(1)
                        continue
                    
                    # Skip if we already processed enough examples for this subset
                    if subset_total_counts[subset] >= args.per_subset:
                        print(f"Skipping subset {subset} because we already processed enough examples")
                        pbar.update(1)
                        continue
                    
                    # Submit this example for processing
                    future = pool.submit(
                        process_example,
                        ex,
                        lm,
                        args.repeats,
                        args.expect_boxed,
                        args.eval_mode,
                        args.eval_tol,
                        args.system_prompt,
                    )
                    futures.append(future)
                    processed_count += 1
                
                # If no futures are running and we have no more examples, break
                if not futures and example_index >= len(all_examples):
                    break
    
    duration = time.time() - start
    print(f"Processed {len(results)} examples in {duration:.2f}s")
    
    # Print summary of correct solutions per subset
    print("\nCorrect solutions per subset:")
    for subset in subsets_to_use:
        print(f"  {subset}: {subset_correct_counts[subset]}/{args.target_correct_per_subset} (processed {subset_total_counts[subset]} total)")

    # Final upload to hub
    print(f"Final upload: {len(results)} records to {args.hub_repo}")
    push_to_hub(results, args.hub_repo, private=args.private)
    print(f"Uploaded to Hugging Face Hub: {args.hub_repo}")


if __name__ == "__main__":
    main()



