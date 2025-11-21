from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
import asyncio
import itertools
import math
import os
import time
from typing import Callable, Dict, List, Literal, Optional, Union, Tuple

# from kvpress import DuoAttentionPress, ExpectedAttentionPress
import pandas as pd
from pydantic import Field
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer
import wandb
import pydrantic

from cartridges.cache import AttnConfig, KVCacheFactory, TrainableCache
from cartridges.data.resources import Resource
from cartridges.datasets import GenerateEvalDatasetElement, GenerateEvalDataset
from cartridges.clients.base import ClientConfig, ClientResponse

from cartridges.models.config import ModelConfig
from cartridges.train import CacheAndModel, GenerationEvalConfig, LossEvalConfig, evaluate_perplexity
from cartridges.utils import get_logger, seed_everything
from cartridges.utils.wandb import WandBConfig, prepare_wandb



logger = get_logger(__name__)

class LossEvalRunConfig(pydrantic.RunConfig):
    eval: LossEvalConfig
    model: ModelConfig
    kv_cache_initializer: Optional[KVCacheFactory.Config] = None

    batch_size: int
    device: str = "cuda"
    seed: int = 42
    
    name: str = "default"  # A name for the run for wandb
    wandb: Optional[WandBConfig] = Field(default_factory=WandBConfig)
    
    def run(self):
        return evaluate_loss(self)


def evaluate_loss(config: LossEvalRunConfig):
    seed_everything(config.seed)
    
    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)


        logger.info(f"[Rank {dist.get_rank()}] initialized.")
        is_rank_zero = dist.get_rank() == 0
        num_devices = dist.get_world_size()
    else:
        local_rank = config.device
        is_rank_zero = True
        num_devices = 1
    
    model = config.model.instantiate().to(local_rank).to(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name_or_path)    

    attn_config = AttnConfig(
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_key_value_heads,
        head_dim=(
            model.config.head_dim
            if hasattr(model.config, "head_dim")
            else model.config.hidden_size // model.config.num_attention_heads
        )
    )

    if config.kv_cache_initializer is not None:
        initializer = config.kv_cache_initializer.instantiate()
        cache: TrainableCache = initializer.initialize_kv_cache(
            tokenizer=tokenizer, model=model, attn_config=attn_config,
        )
        cache = cache.to(local_rank).to(torch.bfloat16)
    else:
        cache = None

    eval_dataset = config.eval.dataset.instantiate(tokenizer=tokenizer, seed=config.seed)

    # Only set up W&B if rank 0 or running single-process
    if config.wandb is not None and is_rank_zero:
        config.wandb.name = config.name
        prepare_wandb(config.wandb, config.to_dict())

        wandb_log_dict = {
            "num_cache_tokens": cache._num_trainable_tokens if cache is not None else 0
        }

        wandb.log(
            wandb_log_dict,
            # SE (03/10): by setting commit=False, we avoid incrementing the step count
            # to 1. Without this, the first evaluation at step 0 will not be logged
            commit=False,
        )

    evaluate_perplexity(
        config=config,
        model=CacheAndModel(cache, model),
        cache=cache,
        eval_dataset=eval_dataset,
        ds_config=config.eval,
        optimizer_step=0,
        epoch=0,
        local_rank=local_rank,
        cache_tuning=cache is not None,
    )
    
    if config.wandb is not None:
        wandb.finish()



class GenerationEvalRunConfig(pydrantic.RunConfig):
    # dataset for actually producing generations
    eval: GenerationEvalConfig
    generator: BaselineGenerator.Config

    batch_size: int
    max_num_batches_in_parallel: int = 1
    max_tokens_per_batch: Optional[int] = None  # If set, use token-based dynamic batching instead of fixed batch size

    tokenizer: str = "meta-llama/Llama-3.2-1B-Instruct"

    seed: int = 42
    
    name: str = "default"  # A name for the run for wandb
    wandb: Optional[WandBConfig] = None
    output_dir: str = os.environ.get("CARTRIDGES_OUTPUT_DIR", ".")

    def run(self):
        return asyncio.run(evaluate_generation(self))


async def evaluate_generation(config: GenerationEvalRunConfig):
    seed_everything(config.seed)

    logger.info(f"ICL will be saved to {config.run_dir}")
    logger.info("Initializing tokenizer and dataset")
    # download_wandb_artifacts(config)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    
    dataset=config.eval.dataset.instantiate(tokenizer=tokenizer, seed=config.seed)
    generator = config.generator.instantiate()

    if config.wandb is not None:
        config.wandb.name = config.name
        prepare_wandb(config.wandb, config.to_dict())

    # Determine batching strategy: token-based dynamic batching or fixed batch size
    if config.max_tokens_per_batch is not None:
        # Use token-based dynamic batching
        batches = _create_token_based_batches(
            dataset=dataset,
            max_tokens_per_batch=config.max_tokens_per_batch,
            num_samples=config.eval.num_samples,
        )
        logger.info(f"Using token-based dynamic batching with max_tokens_per_batch={config.max_tokens_per_batch}, created {len(batches)} batches")
    else:
        # Use fixed batch size (existing behavior)
        dataset_batch_size = config.batch_size // config.eval.num_samples
        total_batches = math.ceil(len(dataset) / dataset_batch_size)
        batches = [
            (batch_idx * dataset_batch_size, min((batch_idx + 1) * dataset_batch_size, len(dataset)))
            for batch_idx in range(total_batches)
        ]
        logger.info(f"Using fixed batch size: {dataset_batch_size} examples per batch, {len(batches)} total batches")
    
    all_rows = []

    # TODO: the generate dataset actually determines the temperature for training
    # so to ensure fair comparison we assert that the temperature is the same
    print("config.eval.temperature: ", config.eval.temperature)
    print("config.generator.temperature: ", config.generator.temperature)
    assert config.eval.temperature == config.generator.temperature

    # Use asyncio for concurrent execution
    tasks = [
        _process_batch(
            batch_start=batch_start,
            batch_end=batch_end,
            generator=generator,
            dataset=dataset,
            eval_config=config.eval,
        )
        for batch_start, batch_end in batches
    ]
    
    # Process in chunks to limit concurrency
    chunk_size = config.max_num_batches_in_parallel
    for i in tqdm(range(0, len(tasks), chunk_size), desc="Processing batch chunks"):
        chunk_tasks = tasks[i:i + chunk_size]
        batch_results = await asyncio.gather(*chunk_tasks)
        for batch_rows in batch_results:
            all_rows += batch_rows

    prefix = f"generate_{config.eval.name_for_wandb}"
    df = pd.DataFrame(all_rows)
    score_cols = [col for col in df.columns if col.endswith("score")]
    avg_scores = {f"{prefix}/{col}": df[col].mean() for col in score_cols}

    if hasattr(dataset, "batch_score_with_answers"):
        batch_score = dataset.batch_score_with_answers(
            df["pred"].tolist(), df["answer"].tolist()
        )
        if isinstance(batch_score, dict):
            batch_score = {f"{prefix}/{k}": v for k, v in batch_score.items()}
        else:
            batch_score = {f"{prefix}/batch_score": batch_score}
        avg_scores.update(batch_score)

    if hasattr(generator, "kv_cache_size_bytes"):
        avg_scores[f"{prefix}/kv_cache_size_bytes"] = generator.kv_cache_size_bytes

    if config.wandb is not None:
        wandb.log(
            {
                **avg_scores,
                f"{prefix}/num_system_and_user_tokens": df[
                    "num_system_and_user_tokens"
                ].mean(),
                f"{prefix}/num_assistant_tokens": df["num_assistant_tokens"].mean(),
                f"{prefix}/table": df,
                f"{prefix}/num_samples": len(df),
            },
            step=0,
        )
        wandb.finish()


async def _process_batch(
    batch_start: int,
    batch_end: int,
    generator: BaselineGenerator,
    dataset: GenerateEvalDataset,
    eval_config: GenerationEvalConfig,
):
    num_samples = eval_config.num_samples
    has_score = hasattr(dataset, "score")
    elems = [dataset[elem_idx] for elem_idx in range(batch_start, batch_end)]
    sample_idxs = sum([[i] * len(elems) for i in range(num_samples)], [])
    elems = elems * num_samples
    responses: List[GenerateBaselineResponse] = await generator.generate(elems)

    results = []
    for response, element, sample_idx in zip(responses, elems, sample_idxs):
        if has_score:
            metrics, extras = dataset.score(
                pred=response.text, answer=element.answer, convo_id=element.convo_id
            )
        else:
            metrics, extras = None, {}

        if not isinstance(metrics, dict):
            # TODO: Support for older datasets that return a single bool or float as metrics
            metrics = {"score": metrics}
        else:
            metrics = {f"{k}_score": v for k, v in metrics.items()}

        results.append(
            {
                "prompt": element.prompt,
                "answer": element.answer,
                "pred": response.text,
                "num_system_and_user_tokens": response.num_system_and_user_tokens,
                "num_assistant_tokens": response.num_assistant_tokens,
                "prompt_messages": response.prompt_messages,
                "convo_id": element.convo_id,
                "sample_idx": sample_idx,
                **metrics,
                **element.metadata,
                **extras,
            }
        )
    return results


def _create_token_based_batches(
    dataset: GenerateEvalDataset,
    max_tokens_per_batch: int,
    num_samples: int,
    indexes: Optional[List[int]] = None,
) -> List[Tuple[int, int]]:
    """
    Create batches based on token count rather than example count.
    
    This function groups examples into batches such that the total number of tokens
    (accounting for num_samples multiplication) does not exceed max_tokens_per_batch.
    This is useful when sequences have highly variable lengths (e.g., 0-500 ICL examples).
    
    Args:
        dataset: The evaluation dataset
        max_tokens_per_batch: Maximum number of tokens per batch (before num_samples multiplication)
        num_samples: Number of samples per example (each element is processed this many times)
        indexes: Optional list of dataset indices to process. If None, processes entire dataset.
        
    Returns:
        List of (batch_start, batch_end) tuples representing batch boundaries
        (within indexes if provided, otherwise within full dataset)
    """
    if indexes is None:
        indexes = list(range(len(dataset)))
    
    if len(indexes) == 0:
        return []
    
    batches = []
    current_batch_start = 0
    current_batch_tokens = 0
    
    for idx_pos in range(len(indexes)):
        dataset_idx = indexes[idx_pos]
        element = dataset[dataset_idx]
        # Get sequence length (input_ids length)
        seq_length = len(element.input_ids)
        
        # Account for num_samples multiplication since each element is processed multiple times
        tokens_needed = seq_length * num_samples
        
        # If a single element exceeds the limit, put it in its own batch
        if tokens_needed > max_tokens_per_batch:
            # Finalize current batch if it has any elements
            if current_batch_start < idx_pos:
                batches.append((current_batch_start, idx_pos))
            # Create a batch with just this element
            batches.append((idx_pos, idx_pos + 1))
            current_batch_start = idx_pos + 1
            current_batch_tokens = 0
        # If adding this example would exceed the limit, start a new batch
        elif current_batch_tokens > 0 and current_batch_tokens + tokens_needed > max_tokens_per_batch:
            batches.append((current_batch_start, idx_pos))
            current_batch_start = idx_pos
            current_batch_tokens = tokens_needed
        else:
            # Add to current batch
            current_batch_tokens += tokens_needed
    
    # Add the final batch if there are remaining elements
    if current_batch_start < len(indexes):
        batches.append((current_batch_start, len(indexes)))
    
    return batches


@dataclass
class GenerateBaselineResponse:
    prompt_messages: List[Dict[str, str]]
    num_system_and_user_tokens: int
    num_assistant_tokens: int
    text: str


class BaselineGenerator(ABC):

    class Config(pydrantic.ObjectConfig):
        _pass_as_config: bool = True

    def __init__(self, config: Config):
        self.config = config

    async def generate(
        self, elements: List[GenerateEvalDatasetElement]
    ) -> List[GenerateBaselineResponse]:
        raise NotImplementedError()


class ICLBaseline(BaselineGenerator):

    class Config(BaselineGenerator.Config):
        client: ClientConfig
        temperature: float = 0.0
        # used to count number of tokens in the prompt
        tokenizer: str = "meta-llama/Llama-3.2-3B-Instruct"

        # The user prompt template which should contain {content}
        user_prompt_template: str = "{content}"

        context: Union[str, Resource.Config]

        # The system prompt template which should contain {title} and {content}
        # variables.
        system_prompt_template: str = "{title}\n\n{content}"
        max_completion_tokens: int = 384
        max_context_tokens: Optional[int] = None  # will truncate if longer
        enable_thinking: Optional[bool] = None

        log_system_prompt: bool = False

    def __init__(self, config: Config):
        self.config = config
        self.client = config.client.instantiate()

        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)


        if isinstance(self.config.context, str):
            ctx_text = self.config.context
        else:
            resource = self.config.context.instantiate()
            # TODO (SE): Need to properly call the resource setup!
            ctx_text = resource.to_string()


        if self.config.max_context_tokens is not None:
            ctx_text = self.tokenizer.decode(
                self.tokenizer.encode(ctx_text)[: self.config.max_context_tokens],
                add_special_tokens=False,

                # suppresses the truncation error 
                max_length=999_999_999, 
                truncation=True
                
            )

        if config.system_prompt_template is not None:
            system_prompt = config.system_prompt_template.format(
                content=ctx_text,
            )

            self.system_prompt = self.post_process_system_prompt(
                system_prompt,
            )
        else:
            self.system_prompt = None

       
        self.metadata = {}

    def post_process_system_prompt(self, system_prompt: str) -> str:
        return system_prompt

    async def generate(
        self, elements: List[GenerateEvalDatasetElement]
    ) -> List[GenerateBaselineResponse]:

        chats = []
        for element in elements:
            messages = []
            if self.system_prompt is not None:
                messages.append(
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                )
            messages.append(
                {
                    "role": "user",
                    "content": self.config.user_prompt_template.format(
                        content=element.prompt
                    ),
                },
            )
            chats.append(messages)


        response: ClientResponse = await self.client.chat(
            chats=chats,
            max_completion_tokens=self.config.max_completion_tokens,
            temperature=self.config.temperature,
            enable_thinking=self.config.enable_thinking,
        )
        assert len(response.samples) == len(chats)

        results = []
        for sample, messages, element in zip(response.samples, chats, elements):
            num_prompt_tokens = len(
                self.tokenizer.apply_chat_template(
                    messages,
                )
            )
            num_assistant_tokens = len(self.tokenizer.encode(sample.text))
            log_messages = [
                msg
                for msg in messages
                if (msg["role"] != "system" or self.config.log_system_prompt)
            ]

            results.append(
                GenerateBaselineResponse(
                    prompt_messages=log_messages,
                    num_system_and_user_tokens=num_prompt_tokens,
                    num_assistant_tokens=num_assistant_tokens,
                    text=sample.text,
                )
            )
        return results
