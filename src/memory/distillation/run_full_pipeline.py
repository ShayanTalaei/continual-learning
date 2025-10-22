#!/usr/bin/env python3
"""
Convenience script to run the full dataset generation and conversion pipeline.

This combines both stages:
1. Generate intermediate JSONL dataset from HistoryAgent checkpoint
2. Convert to cartridges-compatible parquet format

Usage:
    python src/memory/distillation/run_full_pipeline.py \\
        --config configs/distillation/example_gen.yaml \\
        --convert
"""

import sys
import argparse
from pathlib import Path
import yaml
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Run full dataset generation and conversion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to data generation YAML config"
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Also convert to cartridges format after generation"
    )
    parser.add_argument(
        "--min-prob-mass",
        type=float,
        default=0.99,
        help="Minimum probability mass for conversion (default: 0.99)"
    )
    
    args = parser.parse_args()
    
    # Load config to get output paths and model
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(config["output_dir"])
    output_format = config.get("output_format", "jsonl")
    model_name = config["lm_model"]
    
    print("=" * 80)
    print("STAGE 1: GENERATING INTERMEDIATE DATASET")
    print("=" * 80)
    
    # Run data generation
    cmd = [
        sys.executable,
        "-m", "src.memory.distillation.data_generation",
        "--config", str(config_path)
    ]
    
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"ERROR: Data generation failed with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    
    intermediate_path = output_dir / f"dataset.{output_format}"
    
    if not intermediate_path.exists():
        print(f"ERROR: Expected output not found: {intermediate_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"\n✓ Intermediate dataset generated: {intermediate_path}")
    
    if args.convert:
        print("\n" + "=" * 80)
        print("STAGE 2: CONVERTING TO CARTRIDGES FORMAT")
        print("=" * 80)
        
        cartridges_path = output_dir / "dataset.parquet"
        
        cmd = [
            sys.executable,
            "src/memory/distillation/convert_to_cartridges.py",
            "--input", str(intermediate_path),
            "--output", str(cartridges_path),
            "--model", model_name,
            "--min-prob-mass", str(args.min_prob_mass),
        ]
        
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"ERROR: Conversion failed with code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)
        
        print(f"\n✓ Cartridges dataset created: {cartridges_path}")
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print(f"\nIntermediate format: {intermediate_path}")
        print(f"Cartridges format:   {cartridges_path}")
        print("\nYou can now use the parquet file for cartridges training!")
    else:
        print("\n" + "=" * 80)
        print("GENERATION COMPLETE")
        print("=" * 80)
        print(f"\nIntermediate dataset: {intermediate_path}")
        print("\nTo convert to cartridges format, run:")
        print(f"  python src/memory/distillation/convert_to_cartridges.py \\")
        print(f"    --input {intermediate_path} \\")
        print(f"    --output {output_dir / 'dataset.parquet'} \\")
        print(f"    --model {model_name}")


if __name__ == "__main__":
    main()

