#!/usr/bin/env python3

import pydra
from pathlib import Path
from typing import List, Optional
import random
import json


class MergeJsonlConfig(pydra.Config):
    """Configuration for merging multiple JSONL files into a single destination.

    - input_files: list of paths to source .jsonl files
    - destination_file: output .jsonl path (will be overwritten)
    - max_per_jsonl: maximum number of rows to take from each source; if <= 0, take all
    - shuffle_each: shuffle lines within each source before selecting
    - seed: RNG seed used when shuffle_each is True
    """

    def __init__(self):
        super().__init__()
        self.input_files: List[str] = []
        self.destination_file: str = pydra.REQUIRED
        self.max_per_jsonl: int = pydra.REQUIRED
        self.shuffle_each: bool = False
        self.seed: int = 42
        self.filter_none_output_ids: bool = False


def _reservoir_sample_lines(path: Path, k: int, rng: random.Random) -> List[str]:
    """Return up to k uniformly sampled lines from a file using reservoir sampling.

    If k <= 0, returns all lines.
    """
    if k <= 0:
        with path.open("r") as f:
            return f.readlines()

    reservoir: List[str] = []
    with path.open("r") as f:
        for idx, line in enumerate(f):
            if idx < k:
                reservoir.append(line)
            else:
                j = rng.randint(0, idx)
                if j < k:
                    reservoir[j] = line
    rng.shuffle(reservoir)
    return reservoir


def _head_lines(path: Path, k: int) -> List[str]:
    """Return first k lines from file; if k <= 0, return all lines."""
    if k <= 0:
        with path.open("r") as f:
            return f.readlines()
    out: List[str] = []
    with path.open("r") as f:
        for i, line in enumerate(f):
            if i >= k:
                break
            out.append(line)
    return out


def merge_jsonl_files(config: MergeJsonlConfig) -> None:
    input_paths = [Path(p) for p in config.input_files]
    dest_path = Path(config.destination_file)

    print(f"[MergeJSONL] Starting merge")
    print(f"[MergeJSONL] Destination: {dest_path}")
    print(f"[MergeJSONL] Sources ({len(input_paths)}):")
    for p in input_paths:
        print(f"  - {p}")
    print(f"[MergeJSONL] max_per_jsonl={config.max_per_jsonl} | shuffle_each={config.shuffle_each} | seed={config.seed}")

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.seed)

    total_written = 0
    with dest_path.open("w") as out_f:
        for idx, src in enumerate(input_paths):
            print(f"[MergeJSONL] Processing ({idx+1}/{len(input_paths)}): {src}")
            if config.shuffle_each:
                lines = _reservoir_sample_lines(src, config.max_per_jsonl, rng)
            else:
                lines = _head_lines(src, config.max_per_jsonl)
            if config.filter_none_output_ids:
                lines = [line for line in lines if json.loads(line)["output_ids"] is not None]
            out_f.writelines(lines)
            total_written += len(lines)
            print(f"[MergeJSONL]   Selected {len(lines)} lines")

    print(f"[MergeJSONL] Done. Wrote {total_written} total lines to {dest_path}")


@pydra.main(MergeJsonlConfig)
def main(config: MergeJsonlConfig):
    merge_jsonl_files(config)


if __name__ == "__main__":
    main()


