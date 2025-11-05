from typing import List, Optional, Tuple, Dict
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

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast, seed: int):
        self.config = config
        self.tokenizer = tokenizer
        
        self.dataset = [
            instance for instance in load_dataset("stalaei/easy_synth_cities_40_25")[self.config.dataset_split]
        ][:self.config.num_problems]

        self.dataset = self.dataset * self.config.num_repeats

        self.system_prompt = open(self.config.system_prompt_path).read()

        if self.config.max_incontext_examples > 0:
            assert self.config.in_context_examples_path is not None
            self.incontext_examples = read_conversations(self.config.in_context_examples_path)

            for row in self.dataset:
                num_incontext_examples = random.randint(self.config.min_incontext_examples, self.config.max_incontext_examples)
                incontext_examples = random.sample(self.incontext_examples, num_incontext_examples)
                row["incontext_examples"] = incontext_examples


    def __getitem__(
        self, index: int
    ) -> GenerateEvalDatasetElement:

        row = self.dataset[index]

        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]

        if "incontext_examples" in row:
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

        return GenerateEvalDatasetElement(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            prompt=self.tokenizer.decode(input_ids),
            answer=row["name"],
            metadata={}
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