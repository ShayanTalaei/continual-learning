from pydantic import BaseModel

class LMConfig(BaseModel):
    model: str
    temperature: float = 0.2
    max_output_tokens: int = 2048
    
class LLMResponseMetrics(BaseModel):
    duration: float
    input_tokens: int
    thinking_tokens: int
    output_tokens: int
    total_tokens: int


class LanguageModel:
    def __init__(self, config: LMConfig):
        self.config = config

    def call(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError