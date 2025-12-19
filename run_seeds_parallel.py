#!/usr/bin/env python3
"""
Run training with multiple seeds in parallel with resume capability.

Usage:
    python run_seeds_parallel.py --config configs/cities/l8b_cities_v2.yaml \
                                  --start-seed 0 --end-seed 4999 \
                                  --parallel 50 \
                                  --status-file .seed_status.txt
"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Set

import yaml
from tqdm import tqdm


def load_completed_seeds(status_file: Path) -> Set[int]:
    """Load set of completed seed numbers from status file (thread-safe)."""
    if not status_file.exists():
        return set()
    try:
        with open(status_file, "r") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
            seeds = {int(line.strip()) for line in f if line.strip().isdigit()}
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Release lock
            return seeds
    except Exception as e:
        print(f"Warning: Could not read status file {status_file}: {e}", file=sys.stderr)
        return set()


def save_completed_seed(status_file: Path, seed: int) -> None:
    """Append a completed seed number to the status file (thread-safe)."""
    status_file.parent.mkdir(parents=True, exist_ok=True)
    with open(status_file, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock
        f.write(f"{seed}\n")
        f.flush()
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Release lock


def check_seed_complete(config_path: Path, seed: int) -> bool:
    """Check if a seed run is already complete by looking for metrics.json."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Get base results_dir from config
        base_results_dir = config.get("output", {}).get("results_dir", "")
        if not base_results_dir:
            return False
        
        # Construct expected results dir with seed suffix
        results_dir_base = Path(base_results_dir)
        if str(seed) not in str(results_dir_base):
            # If seed not in path, append it
            results_dir_base = Path(f"{base_results_dir}_seed_{seed}")
        
        # Check if any timestamped subdirectory contains metrics.json
        if results_dir_base.exists():
            for subdir in results_dir_base.iterdir():
                if subdir.is_dir():
                    metrics_file = subdir / "metrics.json"
                    if metrics_file.exists():
                        return True
        
        # Also check if LOGS_DIR is set and prepended
        logs_dir = os.getenv("LOGS_DIR")
        if logs_dir:
            full_results_dir = Path(logs_dir) / results_dir_base
            if full_results_dir.exists():
                for subdir in full_results_dir.iterdir():
                    if subdir.is_dir():
                        metrics_file = subdir / "metrics.json"
                        if metrics_file.exists():
                            return True
        
        return False
    except Exception as e:
        print(f"Warning: Could not check completion for seed {seed}: {e}", file=sys.stderr)
        return False


def create_seed_config(base_config: Path, seed: int, temp_dir: Path) -> Path:
    """Create a temporary config file with the seed set."""
    with open(base_config, "r") as f:
        config = yaml.safe_load(f)
    
    # Set train_dataset.seed
    if "train_dataset" not in config:
        config["train_dataset"] = {}
    config["train_dataset"]["seed"] = seed
    
    # Modify results_dir to include seed
    if "output" in config and isinstance(config["output"], dict):
        if "results_dir" in config["output"]:
            base_dir = config["output"]["results_dir"]
            # Only append seed if not already present
            if f"_seed_{seed}" not in base_dir:
                config["output"]["results_dir"] = f"{base_dir}_seed_{seed}"
    
    # Write temporary config
    temp_config = temp_dir / f"config_seed_{seed}.yaml"
    with open(temp_config, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    
    return temp_config


def stream_output(pipe, log_file, prefix, is_stderr=False):
    """Stream output from a pipe to both a log file and stdout/stderr."""
    output_file = sys.stderr if is_stderr else sys.stdout
    with open(log_file, "w") as f:
        for line in iter(pipe.readline, ''):
            if not line:
                break
            # Write to log file
            f.write(line)
            f.flush()
            # Print to console with prefix
            print(f"[{prefix}] {line.rstrip()}", file=output_file, flush=True)
    pipe.close()


def run_seed(config_path: Path, seed: int, temp_dir: Path, status_file: Path, log_dir: Path, show_output: bool = False, stream_output_flag: bool = False) -> bool:
    """Run training for a single seed. Returns True if successful."""
    try:
        # Create seed-specific config
        seed_config = create_seed_config(config_path, seed, temp_dir)
        
        # Set up log files
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = log_dir / f"seed_{seed}_stdout.log"
        stderr_log = log_dir / f"seed_{seed}_stderr.log"
        
        if stream_output_flag:
            # Stream output in real-time
            process = subprocess.Popen(
                [sys.executable, "-m", "src.main", "--config", str(seed_config)],
                cwd=Path(__file__).parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
            )
            
            # Start threads to stream stdout and stderr
            prefix = f"Seed-{seed}"
            stdout_thread = threading.Thread(
                target=stream_output,
                args=(process.stdout, stdout_log, prefix, False),
                daemon=True
            )
            stderr_thread = threading.Thread(
                target=stream_output,
                args=(process.stderr, stderr_log, prefix, True),
                daemon=True
            )
            
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for process to complete
            returncode = process.wait()
            
            # Wait for threads to finish
            stdout_thread.join()
            stderr_thread.join()
            
            result = subprocess.CompletedProcess(
                process.args, returncode, None, None
            )
        else:
            # Run the training (original behavior)
            with open(stdout_log, "w") as stdout_file, open(stderr_log, "w") as stderr_file:
                result = subprocess.run(
                    [sys.executable, "-m", "src.main", "--config", str(seed_config)],
                    cwd=Path(__file__).parent,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True,
                    timeout=None,
                )
        
        # Read output for display if requested (only if not streaming)
        if not stream_output_flag and (show_output or result.returncode != 0):
            stdout_content = ""
            stderr_content = ""
            try:
                with open(stdout_log, "r") as f:
                    stdout_content = f.read()
                with open(stderr_log, "r") as f:
                    stderr_content = f.read()
            except Exception:
                pass
            
            if result.returncode != 0:
                print(f"\n{'='*80}", file=sys.stderr)
                print(f"Seed {seed} FAILED (return code {result.returncode})", file=sys.stderr)
                print(f"{'='*80}", file=sys.stderr)
                if stdout_content:
                    print(f"STDOUT (last 50 lines):", file=sys.stderr)
                    print("\n".join(stdout_content.split("\n")[-50:]), file=sys.stderr)
                if stderr_content:
                    print(f"\nSTDERR (last 50 lines):", file=sys.stderr)
                    print("\n".join(stderr_content.split("\n")[-50:]), file=sys.stderr)
                print(f"{'='*80}\n", file=sys.stderr)
            elif show_output:
                print(f"\n[Seed {seed} output - see {stdout_log} for full log]")
                # Show last few lines
                if stdout_content:
                    lines = stdout_content.strip().split("\n")
                    if len(lines) > 10:
                        print("\n".join(lines[-10:]))
                    else:
                        print(stdout_content)
        
        if result.returncode == 0:
            save_completed_seed(status_file, seed)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error running seed {seed}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run training with multiple seeds in parallel with resume capability"
    )
    parser.add_argument("--config", required=True, type=Path, help="Base config YAML file")
    parser.add_argument("--start-seed", type=int, default=0, help="Starting seed (inclusive)")
    parser.add_argument("--end-seed", type=int, required=True, help="Ending seed (inclusive)")
    parser.add_argument(
        "--parallel", type=int, default=50, help="Maximum number of parallel processes"
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=Path(".seed_status.txt"),
        help="File to track completed seeds",
    )
    parser.add_argument(
        "--check-existing",
        action="store_true",
        help="Check for existing results to mark seeds as complete",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("/tmp/seed_configs"),
        help="Directory for temporary config files",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(".seed_logs"),
        help="Directory for seed output logs",
    )
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="Show output from each seed process after completion (may be verbose)",
    )
    parser.add_argument(
        "--stream-output",
        action="store_true",
        help="Stream output from each seed process in real-time (outputs are prefixed with seed number)",
    )
    
    args = parser.parse_args()
    
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    
    # Create temp and log directories
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    
    # Load completed seeds
    completed_seeds = load_completed_seeds(args.status_file)
    print(f"Loaded {len(completed_seeds)} completed seeds from {args.status_file}")
    
    # Check for existing results if requested
    if args.check_existing:
        print("Checking for existing results...")
        all_seeds = set(range(args.start_seed, args.end_seed + 1))
        for seed in tqdm(all_seeds - completed_seeds, desc="Checking seeds"):
            if check_seed_complete(args.config, seed):
                completed_seeds.add(seed)
                save_completed_seed(args.status_file, seed)
        print(f"Found {len(completed_seeds)} total completed seeds")
    
    # Determine seeds to run
    all_seeds = set(range(args.start_seed, args.end_seed + 1))
    seeds_to_run = sorted(all_seeds - completed_seeds)
    
    if not seeds_to_run:
        print("All seeds are already complete!")
        sys.exit(0)
    
    total = len(seeds_to_run)
    total_all = args.end_seed - args.start_seed + 1
    print(f"Running {total} seeds (seeds {args.start_seed} to {args.end_seed}, {len(completed_seeds)} already complete)")
    
    # Run seeds in parallel with progress bar
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    completed_count = len(completed_seeds)
    
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        # Submit all jobs
        future_to_seed = {
            executor.submit(run_seed, args.config, seed, args.temp_dir, args.status_file, args.log_dir, args.show_output, args.stream_output): seed
            for seed in seeds_to_run
        }
        
        # Track progress
        with tqdm(total=total_all, initial=completed_count, desc="Running seeds", unit="seed") as pbar:
            for future in as_completed(future_to_seed):
                seed = future_to_seed[future]
                try:
                    success = future.result()
                    if success:
                        completed_count += 1
                        pbar.set_postfix({"completed": completed_count, "total": total_all})
                except Exception as e:
                    print(f"\nError processing seed {seed}: {e}", file=sys.stderr)
                finally:
                    pbar.update(1)
    
    # Final status
    final_completed = load_completed_seeds(args.status_file)
    total_all = args.end_seed - args.start_seed + 1
    print(f"\nCompleted: {len(final_completed)}/{total_all} seeds")
    print(f"Status file: {args.status_file}")
    print(f"Log files: {args.log_dir}/seed_*_stdout.log and seed_*_stderr.log")


if __name__ == "__main__":
    main()

