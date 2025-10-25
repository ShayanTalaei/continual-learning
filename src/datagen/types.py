from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class DataSourceItem(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any] = {}


class GenerationItem(BaseModel):
    id: str
    teacher_messages: List[Message]
    student_messages: List[Message]
    metadata: Dict[str, Any] = {}


class LMResponse(BaseModel):
    text: str
    output_ids: Optional[List[int]] = None
    topk_logprobs: Optional[List[List[float]]] = None
    topk_token_ids: Optional[List[List[int]]] = None


