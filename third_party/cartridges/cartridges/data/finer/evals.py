from typing import List, Optional, Tuple, Dict
import random
import torch

from pydrantic import ObjectConfig
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

from src.data.envs.finer_env import is_correct_finer

from cartridges.datasets import GenerateEvalDataset, GenerateEvalDatasetElement


# TODO: centralize in util file
LLAMA_CARTRIDGE_TEMPLATE = """\
{%- for message in messages %}
    {%- if  (message.role == 'assistant') %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}{% generation %}{{- message['content'] | trim + '<|eot_id|>' }}{% endgeneration %}

    {%- else %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
        
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"""
class FinerGenerateDataset(GenerateEvalDataset):
    class Config(GenerateEvalDataset.Config):
        num_problems: int = 1000
        system_prompt_path: str = "/scratch/m000122/bcabrown/continual-learning/src/memory/distillation/prompts/system_prompt.txt"
        dataset_split: str = "val"

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast, seed: int):
        self.config = config
        self.tokenizer = tokenizer
        
        self.dataset = [
            instance for instance in load_dataset("stalaei/finer_v1")[self.config.dataset_split]
        ][:self.config.num_problems]

        self.system_prompt = open(self.config.system_prompt_path).read()

    def __getitem__(
        self, index: int
    ) -> GenerateEvalDatasetElement:

        row = self.dataset[index]

        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": row["context"]
            }
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            chat_template=LLAMA_CARTRIDGE_TEMPLATE,
            add_special_tokens=False,
        )

        return GenerateEvalDatasetElement(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            prompt=self.tokenizer.decode(input_ids),
            answer=row["target"],
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