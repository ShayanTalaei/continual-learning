import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml
from datetime import datetime

from src.run_config import RunConfig
from src.agent.registry import AGENT_REGISTRY
from src.data.dataset_factory import build_dataset
from src.run_time import RunTime
from src.utils.logger import setup_logger, child, add_console


def resolve_paths(conf: Dict[str, Any]) -> Dict[str, Any]:
    base = Path.cwd()
    d = dict(conf)
    dataset = d.get("dataset")
    if dataset and dataset.get("dataset_path") and not os.path.isabs(dataset["dataset_path"]):
        dataset["dataset_path"] = str((base / dataset["dataset_path"]).resolve())
    output = d.get("output")
    if output:
        if output.get("results_dir") and not os.path.isabs(output["results_dir"]):
            # Resolve output paths relative to the current working directory (project root)
            output["results_dir"] = str((Path.cwd() / output["results_dir"]).resolve())
    return d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        raw = yaml.safe_load(f)

    raw = resolve_paths(raw)
    run_conf = RunConfig(**raw)

    # Logger
    out = run_conf.output
    log_level = (out.log_level if out else "INFO")
    # Suffix results_dir with timestamp
    results_dir = None
    if out and out.results_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = str(Path(out.results_dir) / ts)
    log_path = str(Path(results_dir) / "run.log") if results_dir else ""
    # Always log to file; add console based on component verbosity later
    root_logger = setup_logger("run", log_level, log_path, add_stream=False)
    root_logger.info("Loaded config from %s", args.config)

    agent_type = run_conf.agent.get("type", "history_agent")
    ConfigCls, AgentCls = AGENT_REGISTRY[agent_type]
    agent_conf = ConfigCls(**run_conf.agent)
    agent_logger = child(root_logger, "agent")
    if run_conf.agent.get("verbose", True):
        add_console(agent_logger, log_level)
    agent = AgentCls(agent_conf, logger=agent_logger)
    root_logger.info("Instantiated agent: %s", agent_type)
    # Enable LM call logging if requested
    try:
        lm = getattr(agent, "lm", None)
        lm_conf = getattr(agent_conf, "lm_config", None)
        if lm and lm_conf and getattr(lm_conf, "log_calls", False) and results_dir:
            calls_dir = str(Path(results_dir) / "llm_calls")
            lm.enable_call_logging(calls_dir)
            root_logger.info("Enabled LM call logging to %s", calls_dir)
    except Exception as e:
        root_logger.warning("Failed to enable LM call logging: %s", str(e))

    dataset_logger = child(root_logger, "dataset")
    if run_conf.train_dataset and run_conf.train_dataset.get("verbose", True):
        add_console(dataset_logger, log_level)
    train_dataset = build_dataset(run_conf.train_dataset, logger=dataset_logger)
    val_dataset = build_dataset(run_conf.validation_dataset, logger=dataset_logger) if run_conf.validation_dataset else None
    root_logger.info(
        "Datasets ready: train=%d%s",
        len(train_dataset.get_dataset()),
        f", val={len(val_dataset.get_dataset())}" if val_dataset else "",
    )
    runtime_logger = child(root_logger, "runtime")
    if run_conf.runtime.verbose:
        add_console(runtime_logger, log_level)
    # If scores_path provided but not absolute, resolve relative to results_dir
    if run_conf.runtime.scores_path and results_dir and not os.path.isabs(run_conf.runtime.scores_path):
        run_conf.runtime.scores_path = str(Path(results_dir) / run_conf.runtime.scores_path)
    runtime = RunTime(run_conf.runtime, train_dataset, agent, logger=runtime_logger, validation_dataset=val_dataset)
    root_logger.info("Starting runtime")
    results = runtime.run()
    root_logger.info("Runtime finished")

    if results_dir:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        # Save metrics
        with open(Path(results_dir) / "metrics.json", "w") as f:
            json.dump(results, f, indent=2)
        root_logger.info("Saved metrics to %s", str(Path(results_dir) / "metrics.json"))
        # Copy effective config for reproducibility
        try:
            with open(Path(results_dir) / "config.yaml", "w") as f:
                yaml.safe_dump(raw, f, sort_keys=False)
            root_logger.info("Saved config snapshot to %s", str(Path(results_dir) / "config.yaml"))
        except Exception as e:
            root_logger.warning("Failed to save config snapshot: %s", str(e))
        # Save memory into run folder using snapshot API if agent has memory
        mem = getattr(agent, "memory", None)
        if mem:
            snapshot_id = results.get("train_episodes", 0)
            try:
                saved_path = mem.save_snapshot(str(Path(results_dir) / "memories"), snapshot_id=snapshot_id)
                root_logger.info("Saved memory snapshot to %s", saved_path)
            except Exception as e:
                root_logger.warning("Failed to save memory snapshot: %s", str(e))


if __name__ == "__main__":
    main()


