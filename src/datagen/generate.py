from pathlib import Path
import argparse
import sys
import yaml
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
from pydantic import BaseModel
from typing import Literal, Optional
from logging import Logger


from src.datagen.strategies.factory import build_strategy
from src.datagen.io import (
    ensure_output_dir,
    save_config,
    get_existing_sample_ids,
    write_per_sample_json,
    load_existing_rows,
    write_dataset_jsonl,
)
from src.lm.lm_factory import get_lm_client
from src.lm.language_model import LMConfig, LanguageModel
from src.datagen.strategies.strategy import StrategyConfig, Strategy
from src.datagen.types import GenerationItem

class DataGenConfig(BaseModel):
    output_dir: str
    output_format: Literal["jsonl", "parquet"] = "jsonl"
    num_threads: int = 1
    max_samples: Optional[int] = None
    num_repeat_samples: int = 1
    save_individual_files: bool = True
    include_teacher_messages_texts: bool = True
    llm: Dict[str, Any]
    strategy: Dict[str, Any]
    cartridges: Optional[List[Dict[str, Any]]] = None


class DataGenerationOrchestrator:
    def __init__(self, cfg: "DataGenConfig", logger: Optional[Logger] = None) -> None:
        self.cfg = cfg
        self.logger = logger
        self.out_dir = ensure_output_dir(cfg.output_dir)
        # Persist the effective config once output_dir is known
        save_config(self.out_dir, cfg.model_dump())

        # Initialize core components
        self.strategy: Strategy = build_strategy(cfg.strategy, logger=logger)
        self.lm_client: LanguageModel = get_lm_client(cfg.llm, logger=logger)

    def run(self) -> Path:
        cfg = self.cfg

        generation_items = self.strategy.generate()

        existing_ids = (
            get_existing_sample_ids(self.out_dir) if cfg.save_individual_files else set()
        )

        items_to_process: List[GenerationItem] = []
        for idx, generation_item in enumerate(generation_items):
            if cfg.max_samples is not None and idx >= cfg.max_samples:
                break
            if cfg.num_repeat_samples == -1:
                items_to_process.append(generation_item)
            else:
                for rep in range(cfg.num_repeat_samples):
                    new_generation_item = generation_item.model_copy()
                    new_generation_item.id = f"{generation_item.id}_repeat_{rep}"
                    items_to_process.append(new_generation_item)

        if cfg.save_individual_files:
            items_to_process = [it for it in items_to_process if it.id not in existing_ids]

        def _row_from_item(item: GenerationItem, lm_response: Dict[str, Any]) -> Dict[str, Any]:
            row: Dict[str, Any] = {
                "metadata": item.metadata,
                "type": "memory_distillation",
                "input_messages": [m.model_dump() for m in item.student_messages],
                "output_message": lm_response.get("text"),
                "output_ids": lm_response.get("output_ids"),
                "topk_logprobs": lm_response.get("topk_logprobs"),
                "topk_token_ids": lm_response.get("topk_token_ids"),
                "sample_id": item.id,
            }
            if cfg.include_teacher_messages_texts:
                row["teacher_messages"] = [m.model_dump() for m in item.teacher_messages]
            return row

        def _call_lm(teacher_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
            return self.lm_client.call(teacher_messages)

        rows_by_id: Dict[str, Dict[str, Any]] = {}

        if cfg.num_threads == 1:
            with tqdm(total=len(items_to_process), desc="Generating samples", unit="sample") as pbar:
                for it in items_to_process:
                    lm_resp = _call_lm([m.model_dump() for m in it.teacher_messages])
                    row = _row_from_item(it, lm_resp)
                    if cfg.save_individual_files:
                        write_per_sample_json(self.out_dir, it.id, row)
                    rows_by_id[it.id] = row
                    pbar.update(1)
        else:
            lock = Lock()
            with ThreadPoolExecutor(max_workers=cfg.num_threads) as ex:
                future_to_item = {}
                for it in items_to_process:
                    fut = ex.submit(_call_lm, [m.model_dump() for m in it.teacher_messages])
                    future_to_item[fut] = it
                with tqdm(total=len(items_to_process), desc="Generating samples", unit="sample") as pbar:
                    for fut in as_completed(future_to_item):
                        it = future_to_item[fut]
                        lm_resp = fut.result()
                        row = _row_from_item(it, lm_resp)
                        with lock:
                            if cfg.save_individual_files:
                                write_per_sample_json(self.out_dir, it.id, row)
                            rows_by_id[it.id] = row
                        pbar.update(1)

        ordered_rows: List[Dict[str, Any]] = [rows_by_id[k] for k in sorted(rows_by_id.keys())]
        if cfg.save_individual_files and existing_ids:
            ordered_rows.extend(load_existing_rows(self.out_dir, existing_ids))

        filename = f"dataset.jsonl"
        out_path = write_dataset_jsonl(self.out_dir, filename, ordered_rows)
        print(f"\n[DataGen] ✓ SUCCESS! Dataset generated at: {out_path}")
        return out_path

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate distillation data using the new datagen pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (matches src.datagen.config.DataGenConfig)",
    )

    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = DataGenConfig(**cfg_dict)
    orchestrator = DataGenerationOrchestrator(cfg, logger=None)
    orchestrator.run()


if __name__ == "__main__":
    main()


