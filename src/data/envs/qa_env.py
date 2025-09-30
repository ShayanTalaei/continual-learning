from typing import Dict, Any, Optional, Tuple
from logging import Logger, getLogger

from src.data.env import Environment


class QAEnv(Environment):
    def __init__(self, question: str, answer: str, metadata: Dict[str, Any], instruction_template: Optional[str] = None, logger: Optional[Logger] = None):
        self.question = question
        self.answer = answer
        self.metadata = metadata
        self.instruction_template = instruction_template
        self.logger = logger or getLogger("env")
        if self.instruction_template:
            assert "{question}" in self.instruction_template, "instruction_template must contain {question}"
        self._done = False

    def reset(self) -> str:
        self._done = False
        if self.instruction_template:
            return self.instruction_template.format(question=self.question)
        return self.question

    def step(self, action: str) -> Tuple[Optional[str], Dict[str, Any], bool, Dict[str, Any]]:
        self._done = True
        feedback = self.evaluate(action)
        info: Dict[str, Any] = {**self.metadata}
        return None, feedback, True, info

    def evaluate(self, action: str) -> Dict[str, Any]:
        if "boxed" in action:
            predicted_answer = action.split("\\boxed{")[1].split("}")[0]
        else:
            predicted_answer = action
        correct = (predicted_answer == self.answer)
        message = "Correct!" if correct else f"Incorrect! The correct answer is {self.answer}."
        return {"correct": correct, "target": self.answer, "message": message}


