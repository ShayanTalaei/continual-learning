"""
Quick inspector for synthesized dataset.parquet files.

Usage:
  python examples/10k/inspect_synth_dataset.py \
    --path outputs/8192-pepsi_synthesize/pepsi_10k_synthesize_meta-llama/Llama-3.2-3B-Instruct_n8192-0/artifact/dataset.parquet \
    --rows 3

Prints basic info (num rows, columns) and a few sample rows with messages truncated.
"""

import argparse
from pathlib import Path
import pandas as pd
import pdb


def truncate(text, max_len=120):
    text = str(text)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def format_messages(msgs, max_len=200):
    if not isinstance(msgs, (list, tuple)):
        return truncate(msgs, max_len)
    parts = []
    for m in msgs:
        role = m.get("role", "?")
        content = truncate(m.get("content", ""), max_len=80)
        parts.append(f"{role}: {content}")
        if len(parts) * 2 > 10:  # avoid too many
            break
    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Inspect a synthesized dataset.parquet")
    parser.add_argument("--path", type=str, required=True, help="Path to dataset.parquet")
    parser.add_argument("--rows", type=int, default=3, help="Number of rows to show")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} rows from {path}")
    print("Columns:", list(df.columns))
    pdb.set_trace()
    to_show = df.head(args.rows)
    for idx, row in to_show.iterrows():
        print(f"\nRow {idx}:")
        for col in df.columns:
            val = row[col]
            if col == "messages":
                val = format_messages(val)
            else:
                val = truncate(val)
            print(f"  {col}: {val}")


if __name__ == "__main__":
    main()

