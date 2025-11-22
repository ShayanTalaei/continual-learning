"""
Evaluate AMD / PepsiCo cartridges on the FinanceBench QA dataset.

This script does **not** run any generation yet – it is focused on:
  1. Verifying that the FinanceBench HuggingFace dataset has QA rows
     corresponding to the AMD and PepsiCo 10-K documents we train on.
  2. Producing filtered splits you can later plug into a generation loop.

Dataset: https://huggingface.co/datasets/PatronusAI/financebench
PDFs:    https://github.com/patronus-ai/financebench/tree/main/pdfs
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset  # pip install datasets


CARTRIDGES_DIR = Path(__file__).parent.parent.parent
FINANCEBENCH_LOCAL_DIR = CARTRIDGES_DIR / "data" / "financebench"


def load_local_doc_info():
    """
    Load financebench_document_information.jsonl from the local clone
    to see which 10-K doc_name we used for AMD / PepsiCo.
    """
    info_path = FINANCEBENCH_LOCAL_DIR / "data" / "financebench_document_information.jsonl"
    if not info_path.exists():
        print(f"[WARN] Local document info not found at {info_path}")
        return []

    docs = []
    with info_path.open("r") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def pick_latest_10k(docs, company_substr: str):
    """
    From local docs, pick the latest 10-K for a given company substring.
    """
    filtered = [
        d
        for d in docs
        if d.get("doc_type") == "10-K"
        and company_substr.lower() in d.get("company", "").lower()
    ]
    if not filtered:
        return None

    # Sort by fiscal_year if present, descending
    filtered.sort(key=lambda d: d.get("fiscal_year", 0), reverse=True)
    return filtered[0]


def summarize_hf_financebench():
    """
    Load the PatronusAI/financebench dataset from HuggingFace and print:
      - counts by company
      - doc_name distribution for AMD / PepsiCo
    """
    print("[INFO] Loading PatronusAI/financebench from HuggingFace...")
    ds = load_dataset("PatronusAI/financebench", split="train")
    print(f"[INFO] Loaded {len(ds)} rows.")

    companies = Counter(row["company"] for row in ds)
    print("\n[INFO] Companies in HF FinanceBench:")
    for company, count in companies.most_common():
        print(f"  - {company}: {count} rows")

    # Collect doc_name distribution for AMD / Pepsi
    by_company_doc = defaultdict(Counter)
    for row in ds:
        company = row["company"]
        doc_name = row.get("doc_name", "")
        by_company_doc[company][doc_name] += 1

    print("\n[INFO] AMD-related doc_name counts:")
    for company, counter in by_company_doc.items():
        if "amd" in company.lower():
            print(f"  Company={company}")
            for doc_name, count in counter.most_common():
                print(f"    - {doc_name}: {count} rows")

    print("\n[INFO] Pepsi-related doc_name counts:")
    for company, counter in by_company_doc.items():
        if "pepsi" in company.lower():
            print(f"  Company={company}")
            for doc_name, count in counter.most_common():
                print(f"    - {doc_name}: {count} rows")

    return ds, by_company_doc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only_print",
        action="store_true",
        help="Only print doc_name summaries and exit.",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("FinanceBench QA – Doc Name Sanity Check")
    print("=" * 80)

    # 1) Inspect local document info (what did we train on?)
    local_docs = load_local_doc_info()
    if local_docs:
        print("\n[INFO] Local financebench_document_information.jsonl loaded.")
        amd_doc = pick_latest_10k(local_docs, "AMD")
        pepsi_doc = pick_latest_10k(local_docs, "Pepsi")

        if amd_doc:
            print("\n[LOCAL] AMD 10-K used for training (latest):")
            print(f"  company:    {amd_doc.get('company')}")
            print(f"  doc_name:   {amd_doc.get('doc_name')}")
            print(f"  fiscal_year:{amd_doc.get('fiscal_year')}")
        else:
            print("\n[WARN] No AMD 10-K found in local doc info.")

        if pepsi_doc:
            print("\n[LOCAL] PepsiCo 10-K used for training (latest):")
            print(f"  company:    {pepsi_doc.get('company')}")
            print(f"  doc_name:   {pepsi_doc.get('doc_name')}")
            print(f"  fiscal_year:{pepsi_doc.get('fiscal_year')}")
        else:
            print("\n[WARN] No PepsiCo 10-K found in local doc info.")
    else:
        print("\n[WARN] Could not load local financebench_document_information.jsonl")
        amd_doc = pepsi_doc = None

    # 2) Inspect HuggingFace FinanceBench
    ds, by_company_doc = summarize_hf_financebench()

    # 3) Cross-check: ensure HF has QA rows for the same doc_name we train on
    def check_match(label: str, doc):
        if not doc:
            return
        target_doc_name = doc.get("doc_name")
        target_company = doc.get("company")
        if not target_doc_name:
            print(f"[WARN] {label} has no doc_name in local info.")
            return

        # Look for exact company match first
        hf_counter = by_company_doc.get(target_company, Counter())
        total_for_doc = hf_counter.get(target_doc_name, 0)

        print(f"\n[CHECK] {label}:")
        print(f"  company:   {target_company}")
        print(f"  doc_name:  {target_doc_name}")
        print(f"  HF rows with this (company, doc_name): {total_for_doc}")

        if total_for_doc == 0:
            # Fallback: search across all companies for this doc_name
            global_count = sum(
                counter.get(target_doc_name, 0)
                for counter in by_company_doc.values()
            )
            print(f"  HF rows with this doc_name (any company): {global_count}")
            if global_count == 0:
                print("  => No exact match found in HF FinanceBench.")
            else:
                print("  => doc_name exists but company label may differ.")
        else:
            print("  => Good: HF has QA for the exact doc we trained on.")

    check_match("AMD 10-K", amd_doc if local_docs else None)
    check_match("PepsiCo 10-K", pepsi_doc if local_docs else None)

    print("\nDone. Next step: wire this into a generation loop using your trained cartridges.")


if __name__ == "__main__":
    main()


