import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml

from src.run_config import RunConfig
from src.agent.registry import AGENT_REGISTRY
from src.data.qa_dataset import QAEnvDataset
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
        for k in ["results_dir", "save_memory_path"]:
            if output.get(k) and not os.path.isabs(output[k]):
                # Resolve output paths relative to the current working directory (project root)
                output[k] = str((Path.cwd() / output[k]).resolve())
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
    log_path = str(Path(out.results_dir) / "run.log") if out and out.results_dir else ""
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

    dataset_logger = child(root_logger, "dataset")
    if run_conf.dataset.verbose:
        add_console(dataset_logger, log_level)
    dataset = QAEnvDataset(run_conf.dataset, logger=dataset_logger)
    root_logger.info("Dataset ready with %d environments", len(dataset.get_dataset()))
    runtime_logger = child(root_logger, "runtime")
    if run_conf.runtime.verbose:
        add_console(runtime_logger, log_level)
    runtime = RunTime(run_conf.runtime, dataset, agent, logger=runtime_logger)
    root_logger.info("Starting runtime")
    results = runtime.run()
    root_logger.info("Runtime finished")

    if out and out.results_dir:
        Path(out.results_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(out.results_dir) / "metrics.json", "w") as f:
            json.dump(results, f, indent=2)
        root_logger.info("Saved metrics to %s", str(Path(out.results_dir) / "metrics.json"))
        if out.save_memory_path:
            mem = getattr(agent, "memory", None)
            if mem:
                entries = [e.model_dump() for e in mem.recall()]
                Path(out.save_memory_path).parent.mkdir(parents=True, exist_ok=True)
                with open(out.save_memory_path, "w") as f:
                    for rec in entries:
                        f.write(json.dumps(rec) + "\n")
                root_logger.info("Saved memory to %s", out.save_memory_path)


if __name__ == "__main__":
    main()


