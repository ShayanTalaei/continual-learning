from typing import List, Optional, Tuple, Dict, Any
import random
import torch

from pydrantic import ObjectConfig
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

from src.data.envs.finer_env import is_correct_finer

from cartridges.datasets import GenerateEvalDataset, GenerateEvalDatasetElement, LLAMA_CARTRIDGE_TEMPLATE
from cartridges.structs import read_conversations


class SyntheticCitiesGenerateDataset(GenerateEvalDataset):
    class Config(GenerateEvalDataset.Config):
        num_problems: int = 1000
        system_prompt_path: str = "/scratch/m000122/bcabrown/continual-learning/src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt"
        dataset_split: str = "val"
        in_context_examples_path: str | None = None
        max_incontext_examples: int = 0
        min_incontext_examples: int = 0
        num_repeats: int = 1
        
        # Generic: Support multiple eval configurations for grouped evaluation
        # Each dict should have keys that the dataset understands (e.g., problems_range, num_incontext_examples)
        # The dataset will add metadata to track which config each sample belongs to
        eval_configs: List[Dict[str, Any]] = []
        
        # Backward compatibility: single config fields
        num_incontext_examples: int = 0
        problems_range: tuple[int, int] = (0, 0)

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast, seed: int):
        self.config = config
        self.tokenizer = tokenizer
        
        # Load full dataset
        full_dataset = [
            instance for instance in load_dataset("stalaei/easy_synth_cities_40_25")[self.config.dataset_split]
        ]
        
        self.system_prompt = open(self.config.system_prompt_path).read()
        
        # Load in-context examples if needed
        if self.config.in_context_examples_path:
            self.incontext_examples = read_conversations(self.config.in_context_examples_path)
        else:
            self.incontext_examples = []
        
        # Build dataset with all combinations if eval_configs provided, otherwise use single config
        self.dataset = []
        self.eval_group_mapping = []  # Maps dataset index to eval group info
        
        if self.config.eval_configs:
            # Multi-config mode: create samples for all eval configs
            eval_configs = self.config.eval_configs
        else:
            # Single config mode (backward compatibility)
            eval_configs = [{
                "problems_range": self.config.problems_range,
                "num_incontext_examples": self.config.num_incontext_examples,
            }]
        
        for eval_idx, eval_config in enumerate(eval_configs):
            problems_range = eval_config.get("problems_range", (0, 0))
            num_incontext_examples = eval_config.get("num_incontext_examples", 0)
            data_tags = eval_config.get("data_tags", "")
            
            # Get subset of problems
            if problems_range[1] > 0:
                subset = full_dataset[problems_range[0]:problems_range[1]]
            else:
                subset = full_dataset[problems_range[0]:]
            
            if self.config.num_problems > 0:
                subset = subset[:self.config.num_problems]
            
            # Repeat for num_repeats
            subset = subset * self.config.num_repeats
            
            # Add in-context examples to each row
            # Convert to dict if needed to allow modification
            processed_subset = []
            for row in subset:
                if isinstance(row, dict):
                    row_dict = dict(row)  # Create a mutable copy
                else:
                    # Handle case where row might not be a dict
                    # type: ignore - row from HuggingFace dataset can be dict-like
                    row_dict = {"question": getattr(row, "question", ""), "name": getattr(row, "name", "")}
                
                if num_incontext_examples > 0:
                    row_dict["incontext_examples"] = self.incontext_examples[:num_incontext_examples]
                else:
                    row_dict["incontext_examples"] = []
                
                processed_subset.append(row_dict)
            
            self.dataset.extend(processed_subset)
            
            # Track which indices belong to this eval group
            start_idx = len(self.dataset) - len(subset)
            for i in range(len(subset)):
                self.eval_group_mapping.append({
                    "group_id": f"{data_tags}_{num_incontext_examples}" if data_tags else "",
                    "data_tags": data_tags,
                    "num_incontext_examples": num_incontext_examples,
                    "eval_config": eval_config,
                })

    def __getitem__(
        self, index: int
    ) -> GenerateEvalDatasetElement:

        row = self.dataset[index]
        group_info = self.eval_group_mapping[index]

        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]

        if "incontext_examples" in row and row["incontext_examples"]:
            for incontext_example in row["incontext_examples"]:
                messages.extend(incontext_example)
        
        messages.append({
            "role": "user",
            "content": row["question"]
        })

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            chat_template=LLAMA_CARTRIDGE_TEMPLATE,
            add_special_tokens=False,
        )
        
        # input_ids from apply_chat_template should be a list[int]
        # Ensure it's a list for decoding and tensor conversion
        if not isinstance(input_ids, list):
            input_ids = list(input_ids)
        
        # Type cast for type checker - apply_chat_template returns list[int] when add_generation_prompt=True
        input_ids_list: List[int] = input_ids if isinstance(input_ids, list) else list(input_ids)
        
        input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)
        # type: ignore - tokenizer.decode accepts list[int] at runtime
        prompt_text = self.tokenizer.decode(input_ids_list)  # type: ignore

        return GenerateEvalDatasetElement(
            input_ids=input_ids_tensor,
            prompt=prompt_text,
            answer=row["name"],
            metadata={
                "data_tags": group_info["data_tags"],
                "num_incontext_examples": group_info["num_incontext_examples"],
                "eval_group_id": group_info["group_id"],
            }
        )

    def __len__(self):
        return len(self.dataset)

    def score(
        self,
        pred: str,
        answer: str,
        convo_id: str
    ) -> Tuple[bool, Dict[str, Optional[str]]]:
        return is_correct_finer(pred, answer), {}