from typing import Dict, Any, Optional, Tuple, List, Literal
import json
import random
from pydantic import BaseModel
from datasets import load_dataset, load_from_disk

from src.data.env import Environment, EnvDataset, EnvDatasetConfig
from src.data.envs.qa_env import QAEnv
from src.data.envs.math_qa_env import MathQAEnv
from src.data.envs.mcq_env import MCQEnv
from logging import Logger, getLogger
from typing import cast


class QAEnvDatasetConfig(EnvDatasetConfig):
    question_field: str = "question"
    answer_field: str = "answer"
    instruction_template: Optional[str] = None
    # Optional HF dataset identifiers
    hf_name: Optional[str] = None
    hf_config: Optional[str] = None
    split: Optional[str] = None
    # Optional field mappings
    choices_field: Optional[str] = None
    id_field: Optional[str] = None
    meta_fields: List[str] = []
    # Routing
    task_type: Literal["exact", "numeric", "mcq", "custom"] = "exact"
    env_class: Optional[str] = None
    # Backward-compat in-memory items (rarely used once HF is set up)
    items: List[Dict[str, Any]] = []
    verbose: bool = True
    idx_range: Optional[Tuple[int, int]] = None
    # Sampling/shuffle (optional)
    max_samples: Optional[int] = None
    shuffle: bool = False
    seed: int = 42


class QAEnvDataset(EnvDataset):
    def __init__(self, config: QAEnvDatasetConfig, logger: Optional[Logger] = None):
        self.config = config
        self.logger = logger or getLogger("qa_dataset")
        self.dataset = self.load_dataset()

    def load_dataset(self) -> List[Environment]:
        dataset: List[Environment] = []
        qf = self.config.question_field
        af = self.config.answer_field
        # Load rows from HF datasets or disk or fallback to in-memory items
        rows: List[Dict[str, Any]]
        if self.config.hf_name:
            ds = load_dataset(self.config.hf_name, self.config.hf_config, split=self.config.split)
            rows = cast(List[Dict[str, Any]], list(ds))
        elif self.config.dataset_path:
            # Support both HF disk datasets and plain JSONL
            p = str(self.config.dataset_path)
            if p.endswith(".jsonl"):
                with open(p, "r") as f:
                    rows = [json.loads(line) for line in f]
            else:
                data = load_from_disk(self.config.dataset_path)
                rows = cast(List[Dict[str, Any]], list(data))
        else:
            rows = list(self.config.items)

        if self.config.idx_range is not None:
            rows = rows[self.config.idx_range[0]:self.config.idx_range[1]]
        # Optional shuffle and sample
        if self.config.shuffle and len(rows) > 1:
            rng = random.Random(self.config.seed)
            rng.shuffle(rows)
        if self.config.max_samples is not None:
            rows = rows[: max(0, int(self.config.max_samples))]

        # Env class routing map
        env_map = {
            "QAEnv": QAEnv,
            "MathQAEnv": MathQAEnv,
            "MCQEnv": MCQEnv,
        }

        for row in rows:
            question = row[qf]
            answer = row[af]
            metadata = {k: v for k, v in row.items() if k not in {qf, af}}
            # Inject choices/id/meta fields when available
            if self.config.choices_field and self.config.choices_field in row:
                metadata["choices"] = row[self.config.choices_field]
            if self.config.id_field and self.config.id_field in row:
                metadata["id"] = row[self.config.id_field]
            for mf in self.config.meta_fields:
                if mf in row:
                    metadata[mf] = row[mf]

            # Select env class
            if self.config.env_class and self.config.env_class in env_map:
                EnvCls = env_map[self.config.env_class]
            else:
                ttype = self.config.task_type
                if ttype == "numeric":
                    EnvCls = MathQAEnv
                elif ttype == "mcq":
                    EnvCls = MCQEnv
                else:
                    EnvCls = QAEnv

            env_logger = None
            if self.config.verbose:
                env_logger = self.logger.getChild(str(metadata.get("id", "env")))

            dataset.append(EnvCls(
                env_id=str(metadata.get("id", "qa")),
                env_type=str(metadata.get("dataset", metadata.get("split", "qa"))),
                question=question,
                answer=answer,
                metadata=metadata,
                instruction_template=self.config.instruction_template,
                logger=env_logger,
            ))
        return dataset