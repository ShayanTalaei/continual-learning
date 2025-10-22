from typing import Dict, Any, Optional, Tuple, List, Literal
import json
from pydantic import BaseModel
from datasets import load_dataset, load_from_disk

from src.data.env import Environment, EnvDataset, EnvDatasetConfig
from src.data.envs.qa_env import QAEnv
from src.data.envs.math_qa_env import MathQAEnv
from src.data.envs.mcq_env import MCQEnv
from logging import Logger


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


class QAEnvDataset(EnvDataset):
    def __init__(self, config: QAEnvDatasetConfig, logger: Optional[Logger] = None):
        super().__init__(config, logger)

    def load_dataset(self) -> List[Environment]:
        dataset: List[Environment] = []
        qf = self.config.question_field
        af = self.config.answer_field
        # Load rows from HF datasets or disk or fallback to in-memory items
        rows: List[Dict[str, Any]]
        if self.config.hf_name:
            ds = load_dataset(self.config.hf_name, self.config.hf_config, split=self.config.split)
            rows = list(ds)
        elif self.config.dataset_path:
            # Support both HF disk datasets and plain JSONL
            p = str(self.config.dataset_path)
            if p.endswith(".jsonl"):
                with open(p, "r") as f:
                    rows = [json.loads(line) for line in f]
            else:
                data = load_from_disk(self.config.dataset_path)
                rows = list(data)
        else:
            rows = list(self.config.items)

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
                question=question,
                answer=answer,
                metadata=metadata,
                instruction_template=self.config.instruction_template,
                logger=env_logger,
            ))
        return dataset