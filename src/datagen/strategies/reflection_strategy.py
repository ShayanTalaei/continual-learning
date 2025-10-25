from typing import List, Dict, Any

from src.datagen.strategies.strategy import StrategyConfig
from src.datagen.strategies.strategy import Strategy
from src.datagen.types import GenerationItem
from src.datagen.memory_adapters.memory_adapter import MemoryAdapterConfig
from src.datagen.memory_adapters.factory import build_memory_adapter
from src.datagen.types import Message

class ReflectionStrategyConfig(StrategyConfig):
    reflection_system_prompt_path: str
    reflection_user_prompt_path: str
    student_system_prompt_path: str
    student_user_prompt_path: str
    memory_adapter: MemoryAdapterConfig
    
class ReflectionStrategy(Strategy):
    def __init__(self, config: ReflectionStrategyConfig, logger=None):
        super().__init__(config, logger)
        self.memory_adapter = build_memory_adapter(config.memory_adapter)
        self.resolve_prompts_paths()
        
    def resolve_prompts_paths(self):
        # Type narrowing: assert that config is ReflectionStrategyConfig
        assert isinstance(self.config, ReflectionStrategyConfig), \
            f"Expected ReflectionStrategyConfig, got {type(self.config)}"
        
        with open(self.config.reflection_system_prompt_path, "r") as f:
            self.reflection_system_prompt = f.read()
        with open(self.config.reflection_user_prompt_path, "r") as f:
            self.reflection_user_prompt = f.read()
        with open(self.config.student_system_prompt_path, "r") as f:
            self.student_system_prompt = f.read()
        with open(self.config.student_user_prompt_path, "r") as f:
            self.student_user_prompt = f.read()
    
    def triplets_to_strings(self, triplets: List[Dict[str, Any]]) -> str:
        result = []
        for triplet in triplets:
            question = triplet["Observation"]
            answer = triplet["Action"]
            feedback = triplet["Feedback"]
            result.append(f"Question: {question}\n Generated Answer: {answer}\nFeedback: {feedback}\n\n")
        return "".join(result)
    
    def generate(self) -> List[GenerationItem]:

        experience_triplets = self.memory_adapter.to_triplets()
        experience_strings = self.triplets_to_strings(experience_triplets)
        items = [
            GenerationItem(
                id=f"reflection",
                teacher_messages=[
                    Message(
                        role="system",
                        content=self.reflection_system_prompt
                    ),
                    Message(
                        role="user",
                        content=self.reflection_user_prompt.format(experiences=experience_strings)
                    )
                ],
                student_messages=[
                    Message(
                        role="system",
                        content=self.student_system_prompt
                    ),
                    Message(
                        role="user",
                        content=self.student_user_prompt
                    )
                ]
            )
        ]
        return items