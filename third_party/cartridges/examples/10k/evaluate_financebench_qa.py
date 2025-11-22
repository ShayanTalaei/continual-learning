"""
Evaluate AMD and Pepsi cartridges on FinanceBench QA with EM and F1.

Runs three modes:
  - Run A: AMD cartridge only on AMD questions.
  - Run B: Pepsi cartridge only on Pepsi questions.
  - Run C: [AMD, Pepsi] composed on mixed AMD+Pepsi questions.

This script assumes:
  - `setup_financebench.py` has been run, so that:
      data/10k/eval/amd_qa_financebench.json
      data/10k/eval/pepsi_qa_financebench.json
    exist.
  - Tokasaurus is running and reachable at `TOKASAURUS_URL`
    (default: http://localhost:10210).
  - The trained cartridges are visible to Tokasaurus and referenced
    via cartridge specs passed in env vars:
      AMD_CARTRIDGES_JSON   e.g. '[{"id": "amd_10k", "source": "local"}]'
      PEPSI_CARTRIDGES_JSON e.g. '[{"id": "pepsi_10k", "source": "local"}]'
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import requests
import yaml

import pandas as pd  # For inspecting synthesized datasets and building synthetic QA splits
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math


CARTRIDGES_DIR = Path(__file__).parent.parent.parent
DATA_DIR = CARTRIDGES_DIR / "data" / "10k"
HF_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

_hf_model = None
_hf_tokenizer = None


# ---------------------- QA loading & metrics ---------------------- #

def load_qa(company: str) -> List[Dict[str, Any]]:
    """Load FinanceBench QA JSON for 'amd' or 'pepsi'."""
    fname = DATA_DIR / "eval" / f"{company}_qa_financebench.json"
    with open(fname, "r") as f:
        return json.load(f)


def normalize_answer(s: str) -> str:
    """Lowercase, strip, and remove extra whitespace + punctuation."""
    import re
    s = s.lower().strip()
    # Remove punctuation
    s = re.sub(r"[^\w\s]", " ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def em_and_f1(pred: str, gold: str) -> Tuple[float, float]:
    """Compute Exact Match and token-level F1 between two strings."""
    from collections import Counter

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


# ---------------------- Tokasaurus helper ---------------------- #

def get_tokasaurus_url() -> str:
    base = os.environ.get("TOKASAURUS_URL", "http://localhost:10212")
    return base.rstrip("/")


def get_hf_model():
    """Lazy-load the base HF model used for perplexity checks."""
    global _hf_model, _hf_tokenizer
    if _hf_model is None or _hf_tokenizer is None:
        print(f"\n[HF] Loading {HF_MODEL_NAME} for perplexity checks...")
        _hf_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        _hf_model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _hf_model.to(device)
        _hf_model.eval()
        print(f"[HF] Loaded on device: {device}")
    return _hf_model, _hf_tokenizer


def compute_ppl_for_pair(question: str, answer: str) -> float:
    """
    Approximate perplexity of the base model on (question, answer).

    We build a simple Q/A prompt and compute the mean per-token NLL over the
    entire sequence (question + answer), then exponentiate.
    """
    model, tokenizer = get_hf_model()
    device = next(model.parameters()).device

    prompt = f"Question: {question}\nAnswer: {answer}"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
    return math.exp(loss)


def ask_with_cartridges(question: str, cartridges: List[Dict[str, Any]], max_tokens: int = 256) -> str:
    """Send a single question to Tokasaurus with the given cartridge specs."""
    url = get_tokasaurus_url() + "/custom/cartridge/chat/completions"
    payload = {
        "model": "default",
        "messages": [
            {"role": "system", "content": "You are a helpful financial analyst assistant."},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens,
    }
    if cartridges:
        payload["cartridges"] = cartridges

    resp = requests.post(url, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    # Tokasaurus is OpenAI-compatible; expect choices[0].message.content
    return data["choices"][0]["message"]["content"]


def parse_cartridges_env(var_name: str) -> List[Dict[str, Any]]:
    """Parse a JSON-encoded list of cartridge specs from an env var."""
    raw = os.environ.get(var_name)
    if not raw:
        return []
    return json.loads(raw)


def print_cartridge_info():
    """Print basic info about the local cartridges (size p, model name)."""
    print("\n=== Cartridge Info (from local config.yaml, if available) ===")
    for name in ["amd_10k", "pepsi_10k"]:
        cfg_path = CARTRIDGES_DIR / "cartridges" / name / "config.yaml"
        if not cfg_path.exists():
            print(f"- {name}: config.yaml not found at {cfg_path}")
            continue
        try:
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            print(f"- {name}: error reading {cfg_path}: {e}")
            continue

        max_tokens = cfg.get("kv_cache_initializer", {}).get("max_tokens", None)
        model_name = cfg.get("model", {}).get("pretrained_model_name_or_path", "unknown")
        print(f"- {name}: max_tokens={max_tokens}, model={model_name}")


def print_synth_examples():
    """Print a couple of synthesized training examples for AMD and Pepsi (if paths are set)."""
    print("\n=== Synthesized Dataset Sanity Check (first 2 rows) ===")
    for env_var, label in [("AMD_SYNTH_DATASET_PATH", "AMD"), ("PEPSI_SYNTH_DATASET_PATH", "Pepsi")]:
        path = os.environ.get(env_var)
        if not path:
            print(f"- {label}: {env_var} not set; skipping.")
            continue
        p = Path(path)
        if not p.exists():
            print(f"- {label}: {p} does not exist; skipping.")
            continue
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"- {label}: error reading {p}: {e}")
            continue

        print(f"\n- {label} ({p}): {len(df)} rows total")
        cols_to_show = [c for c in ["conversation_id", "messages"] if c in df.columns]
        if cols_to_show:
            print(df.head(2)[cols_to_show])
        else:
            print(df.head(2))


def extract_qa_from_messages(messages: list[Dict[str, Any]]) -> Tuple[str, str] | None:
    """
    Extract a (question, answer) pair from a synthesized conversation.

    Heuristic:
      - Take the last 'user' message as the question.
      - Take the last 'assistant' message as the answer.
    """
    last_user = None
    last_assistant = None
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            last_user = content
        elif role == "assistant":
            last_assistant = content
    if last_user and last_assistant:
        return last_user, last_assistant
    return None


def load_synth_qa_from_parquet(path: str, label: str) -> List[Dict[str, Any]]:
    """
    Load a synthetic dataset parquet and convert it into a QA list
    using `extract_qa_from_messages`.
    """
    p = Path(path)
    if not p.exists():
        print(f"- {label} synth QA: {p} does not exist; skipping.")
        return []
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        print(f"- {label} synth QA: error reading {p}: {e}")
        return []

    if "messages" not in df.columns:
        print(f"- {label} synth QA: no 'messages' column in {p}; skipping.")
        return []

    qa_list: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        qa = extract_qa_from_messages(row["messages"])
        if qa is None:
            continue
        q, a = qa
        qa_list.append({"question": q, "answer": a})

    print(f"- {label} synth QA: {len(qa_list)} QA pairs extracted from {p}")
    return qa_list


def compute_split_perplexity(
    name: str,
    qa_list: List[Dict[str, Any]],
    max_questions: int | None = None,
) -> None:
    """
    Compute base-model perplexity on a QA list (no cartridges) using the HF model.

    This is useful to see how well the underlying Llama-3.2-3B matches the
    synthetic / FinanceBench answers, independently of cartridges.
    """
    if not qa_list:
        print(f"\n{name} PPL — no questions to evaluate.")
        return

    n = len(qa_list) if max_questions is None else min(len(qa_list), max_questions)
    ppl_sum = 0.0

    print(f"\n=== Perplexity (base HF model) on {name} ({n} questions) ===")
    for i, qa in enumerate(qa_list[:n], start=1):
        q = qa["question"]
        gold = qa["answer"]
        try:
            ppl = compute_ppl_for_pair(q, gold)
        except Exception as e:
            print(f"[{i}/{n}] ERROR computing perplexity: {e}")
            continue
        ppl_sum += ppl
        print(f"[{i}/{n}] PPL={ppl:.3f}")

    if n > 0:
        print(f"\n{name} — mean PPL={ppl_sum / n:.3f}")


# ---------------------- Evaluation loops ---------------------- #

def evaluate_split(
    name: str,
    qa_list: List[Dict[str, Any]],
    cartridges: List[Dict[str, Any]],
    max_questions: int | None = None,
) -> None:
    """Run EM/F1 evaluation on a QA list with a given cartridge config."""
    em_sum = 0.0
    f1_sum = 0.0
    n = len(qa_list) if max_questions is None else min(len(qa_list), max_questions)

    print(f"\n=== Evaluating {name} on {n} questions ===")

    for i, qa in enumerate(qa_list[:n], start=1):
        q = qa["question"]
        gold = qa["answer"]

        try:
            pred = ask_with_cartridges(q, cartridges)
        except Exception as e:
            print(f"[{i}/{n}] ERROR calling Tokasaurus: {e}")
            continue

        em, f1 = em_and_f1(pred, gold)
        em_sum += em
        f1_sum += f1

        print(f"\n[{i}/{n}] EM={em:.3f} F1={f1:.3f}")
        print("Q:", q)
        print("PRED:", pred.strip())
        print("GOLD:", gold.strip())

    if n > 0:
        print(f"\n{name} — EM={em_sum / n:.3f}, F1={f1_sum / n:.3f}")
    else:
        print(f"\n{name} — no questions evaluated.")


def evaluate_baseline_no_cartridge(
    name: str,
    qa_list: List[Dict[str, Any]],
    max_questions: int | None = None,
) -> None:
    """
    Sanity check: run the same QA split with **no cartridges** to see the base
    model's behavior.
    """
    evaluate_split(f"{name} (no cartridge baseline)", qa_list, cartridges=[], max_questions=max_questions)


def main():
    parser = argparse.ArgumentParser(description="Evaluate AMD/Pepsi cartridges on FinanceBench QA.")
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Optional cap on number of questions per split (for quick sanity checks).",
    )
    args = parser.parse_args()

    # Load FinanceBench QA splits
    amd_qa = load_qa("amd")
    pepsi_qa = load_qa("pepsi")

    # Cartridge specs (JSON from env)
    amd_cartridges = parse_cartridges_env("AMD_CARTRIDGES_JSON")
    pepsi_cartridges = parse_cartridges_env("PEPSI_CARTRIDGES_JSON")

    if not amd_cartridges:
        print("⚠ AMD_CARTRIDGES_JSON not set or empty; AMD runs will use no cartridges.")
    if not pepsi_cartridges:
        print("⚠ PEPSI_CARTRIDGES_JSON not set or empty; Pepsi runs will use no cartridges.")

    # Print cartridge sizes / model names and show a few synth examples
    print_cartridge_info()
    print_synth_examples()

    # Build synthetic QA splits from the synthesized datasets (if paths are set)
    amd_synth_path = os.environ.get("AMD_SYNTH_DATASET_PATH")
    pepsi_synth_path = os.environ.get("PEPSI_SYNTH_DATASET_PATH")
    amd_synth_qa: List[Dict[str, Any]] = []
    pepsi_synth_qa: List[Dict[str, Any]] = []

    if amd_synth_path:
        amd_synth_qa = load_synth_qa_from_parquet(amd_synth_path, "AMD")
    else:
        print("- AMD synth QA: AMD_SYNTH_DATASET_PATH not set; skipping.")

    if pepsi_synth_path:
        pepsi_synth_qa = load_synth_qa_from_parquet(pepsi_synth_path, "Pepsi")
    else:
        print("- Pepsi synth QA: PEPSI_SYNTH_DATASET_PATH not set; skipping.")

    # Perplexity of base HF model on synthetic QA (per-pair + mean)
    if amd_synth_qa:
        compute_split_perplexity("AMD synthetic QA (base model)", amd_synth_qa, args.max_questions)
    if pepsi_synth_qa:
        compute_split_perplexity("Pepsi synthetic QA (base model)", pepsi_synth_qa, args.max_questions)

    # Baseline sanity: model without cartridges on each FinanceBench split
    # evaluate_baseline_no_cartridge("Run A baseline: base model on AMD QA", amd_qa, args.max_questions)
    # evaluate_baseline_no_cartridge("Run B baseline: base model on Pepsi QA", pepsi_qa, args.max_questions)

    # Baseline sanity on synthetic QA splits (if available)
    # if amd_synth_qa:
    #     evaluate_baseline_no_cartridge("AMD synthetic baseline (base model)", amd_synth_qa, args.max_questions)
    # if pepsi_synth_qa:
    #     evaluate_baseline_no_cartridge("Pepsi synthetic baseline (base model)", pepsi_synth_qa, args.max_questions)

    # Run A: AMD cartridge only on AMD FinanceBench questions
    evaluate_split("Run A: AMD-only on AMD QA", amd_qa, amd_cartridges, args.max_questions)

    # Run B: Pepsi cartridge only on Pepsi FinanceBench questions
    evaluate_split("Run B: Pepsi-only on Pepsi QA", pepsi_qa, pepsi_cartridges, args.max_questions)

    # Synthetic evals with their respective cartridges
    if amd_synth_qa and amd_cartridges:
        evaluate_split("AMD synthetic QA with AMD cartridge", amd_synth_qa, amd_cartridges, args.max_questions)
    if pepsi_synth_qa and pepsi_cartridges:
        evaluate_split("Pepsi synthetic QA with Pepsi cartridge", pepsi_synth_qa, pepsi_cartridges, args.max_questions)

    # Run C: composed [AMD, Pepsi] on mixed QA
    mixed_qa = amd_qa + pepsi_qa
    composed_cartridges = amd_cartridges + pepsi_cartridges
    evaluate_split("Run C: [AMD, Pepsi] composition on mixed QA", mixed_qa, composed_cartridges, args.max_questions)


if __name__ == "__main__":
    main()


