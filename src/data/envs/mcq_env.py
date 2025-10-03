from typing import Dict, Any, List

from src.data.envs.qa_env import QAEnv


class MCQEnv(QAEnv):
    def evaluate(self, action: str) -> Dict[str, Any]:
        choices: List[str] = self.metadata.get("choices", [])
        target = str(self.answer).strip()
        act = str(action).strip()

        def normalize(x: str) -> str:
            return x.strip().lower()

        idx_map = {chr(ord('a') + i): i for i in range(len(choices))}
        target_text = None
        if choices and len(target) == 1 and target.lower() in idx_map:
            target_text = choices[idx_map[target.lower()]]

        correct = False
        msg = "mcq"
        if choices:
            if len(act) == 1 and act.lower() in idx_map and len(target) == 1 and target.lower() in idx_map:
                correct = act.lower() == target.lower()
                msg = "mcq-letter"
            elif target_text is not None:
                correct = normalize(act) == normalize(target_text)
                msg = "mcq-text"
            else:
                correct = normalize(act) == normalize(target)
                msg = "mcq-text"
        else:
            correct = normalize(act) == normalize(target)
            msg = "mcq-fallback"

        score = 1 if correct else 0
        return {"score": score, "target": self.answer, "message": msg, "extra": {"choices": choices}}


