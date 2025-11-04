from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
import random

from src.datagen.strategies.strategy import StrategyConfig
from src.datagen.strategies.strategy import Strategy
from src.datagen.types import GenerationItem, Message
from src.datagen.memory_adapters.memory_adapter import MemoryAdapterConfig
from src.datagen.memory_adapters.factory import build_memory_adapter
from src.lm.lm_factory import get_lm_client
from src.datagen.memory_adapters.history_list_adapter import HistoryAdapter

class SyntheticGenStrategyConfig(StrategyConfig):
    synthetic_gen_system_prompt_path: str
    synthetic_gen_user_prompt_path: str
    task_system_prompt_path: str
    num_generations: int
    num_threads: int = 1
    chat_mode: bool = True
    num_experiences_subsample_per_task_generation: Optional[int] = None
    use_task_generation_experiences_for_solving: bool = False
    shuffle_experiences_when_solving: bool = False
    memory_adapter: MemoryAdapterConfig
    lm_config: Dict[str, Any]
    
    
class SyntheticGenStrategy(Strategy):
    def __init__(self, config: SyntheticGenStrategyConfig, logger=None):
        super().__init__(config, logger)
        self.config: SyntheticGenStrategyConfig = config
        self.resolve_prompts_paths()
        self.memory_adapter: HistoryAdapter = build_memory_adapter(config.memory_adapter) # type: ignore[assignment]
        assert isinstance(self.memory_adapter, HistoryAdapter), "Memory adapter must be a HistoryAdapter"
        self.lm_client = get_lm_client(config.lm_config)
    
    def resolve_prompts_paths(self):
        with open(self.config.synthetic_gen_system_prompt_path, "r") as f:
            self.synthetic_gen_system_prompt = f.read()
        with open(self.config.synthetic_gen_user_prompt_path, "r") as f:
            self.synthetic_gen_user_prompt = f.read()
        with open(self.config.task_system_prompt_path, "r") as f:
            self.task_system_prompt = f.read()
    
    def triplets_to_strings(self, triplets: List[Dict[str, Any]]) -> str:
        result = []
        for triplet in triplets:
            question = triplet["Observation"]
            answer = triplet["Action"]
            feedback = triplet["Feedback"]
            result.append(f"Question: {question}\n Generated Answer: {answer}\nFeedback: {feedback}\n\n")
        return "".join(result)
    
    def generate_synthetic_task(self) -> Any:
        experiences = list(self.memory_adapter.to_triplets())
        
        # breakpoint()
        random.shuffle(experiences)
        if self.config.num_experiences_subsample_per_task_generation is not None:
            experiences = experiences[:self.config.num_experiences_subsample_per_task_generation]
        experiences_string = self.triplets_to_strings(experiences)
        messages = [
            {"role": "system", "content": self.synthetic_gen_system_prompt},
            {"role": "user", "content": self.synthetic_gen_user_prompt.format(experiences=experiences_string)}
        ]
        response = self.lm_client.call(messages=messages)
        return response["text"], experiences
    
    def generate(self) -> List[GenerationItem]:
        all_memory_triplets = self.memory_adapter.to_triplets()

        def build_messages_from_triplets(triplets: List[Dict[str, Any]]) -> List[Message]:
            messages: List[Message] = []
            for triplet in triplets:
                messages.append(Message(role="user", content=triplet["Observation"]))
                messages.append(Message(role="assistant", content=triplet["Action"]))
                messages.append(Message(role="user", content=triplet["Feedback"]))
            return messages

        def build_collapsed_prefix(triplets: List[Dict[str, Any]]) -> str:
            exp_blocks: List[str] = []
            for triplet in triplets:
                lines: List[str] = []
                if "Observation" in triplet:
                    lines.append(f"Observation: {triplet['Observation']}")
                if "Action" in triplet:
                    lines.append(f"Action: {triplet['Action']}")
                if "Feedback" in triplet:
                    lines.append(f"Feedback: {triplet['Feedback']}")
                if lines:
                    exp_blocks.append("\n".join(lines))
            user_content = "Here are the previous experiences you've had and their feedback:\n\n"
            user_content += "\n\n".join(exp_blocks)
            user_content += "\n\nHere is the current observation: "
            return user_content
        
        def _create_item(idx: int, synthetic_task: str, used_experiences: List[Dict[str, Any]]) -> GenerationItem:
            triplets_to_use = used_experiences if self.config.use_task_generation_experiences_for_solving else all_memory_triplets
            if self.config.shuffle_experiences_when_solving:
                random.shuffle(triplets_to_use)
            if self.config.chat_mode:
                teacher_messages = [Message(role="system", content=self.task_system_prompt)] + build_messages_from_triplets(triplets_to_use) + [Message(role="user", content=synthetic_task)]
            else:
                combined_user = build_collapsed_prefix(triplets_to_use) + synthetic_task
                teacher_messages = [Message(role="system", content=self.task_system_prompt), Message(role="user", content=combined_user)]

            return GenerationItem(
                id=f"synthetic_task_{idx}",
                teacher_messages=teacher_messages,
                student_messages=[Message(role="system", content=self.task_system_prompt), Message(role="user", content=synthetic_task)]
            )
        
        items: List[GenerationItem] = []
        
        if self.config.num_threads == 1:
            for i in range(self.config.num_generations):
                synthetic_task, used_experiences = self.generate_synthetic_task()
                items.append(_create_item(i, synthetic_task, used_experiences))
        else:
            lock = Lock()
            with ThreadPoolExecutor(max_workers=self.config.num_threads) as ex:
                future_to_idx = {}
                for i in range(self.config.num_generations):
                    fut = ex.submit(self.generate_synthetic_task)
                    future_to_idx[fut] = i
                
                with tqdm(total=self.config.num_generations, desc="Generating synthetic tasks", unit="task") as pbar:
                    for fut in as_completed(future_to_idx):
                        idx = future_to_idx[fut]
                        synthetic_task, used_experiences = fut.result()
                        with lock:
                            items.append(_create_item(idx, synthetic_task, used_experiences))
                        pbar.update(1)
        
        return items
