from typing import Dict, Any, Optional

from src.data.envs.qa_env import QAEnv


class MathQAEnv(QAEnv):
    def _normalize(self, s: str) -> str:
        s = s.strip()
        if s.startswith("\\boxed{") and s.endswith("}"):
            s = s[len("\\boxed{"):-1]
        if s.startswith("$") and s.endswith("$"):
            s = s[1:-1]
        return s.strip()

    def _to_float(self, s: str) -> Optional[float]:
        try:
            return float(s.replace(",", ""))
        except Exception:
            return None

    def evaluate(self, action: str) -> Dict[str, Any]:
        act_n = self._normalize(action)
        tgt_n = self._normalize(self.answer)
        a_f = self._to_float(act_n)
        t_f = self._to_float(tgt_n)
        if a_f is not None and t_f is not None:
            tol = 1e-6
            correct = abs(a_f - t_f) <= tol
            return {"correct": correct, "target": self.answer, "message": "numeric-compare", "extra": {"normalized_action": act_n, "normalized_target": tgt_n}}
        correct = act_n == tgt_n
        return {"correct": correct, "target": self.answer, "message": "normalized-exact", "extra": {"normalized_action": act_n, "normalized_target": tgt_n}}


