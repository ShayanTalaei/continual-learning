from __future__ import annotations

from typing import List, Any, Dict, Optional, Union
import json
from dataclasses import dataclass

from src.agent.memory_agent import MemoryAgent, MemoryAgentConfig
from src.memory.ace_playbook import AceMemory, AceMemoryConfig, Playbook
from src.memory.history_list import Entry
from src.utils import logger as jsonlogger


class AceAgentConfig(MemoryAgentConfig):
    memory_config: AceMemoryConfig  # type: ignore[assignment]
    # Prompt file paths
    generator_prompt: str
    reflector_prompt: str
    curator_prompt: str
    # Curator settings
    max_playbook_tokens: int = 4000
    refine_mode: str = "lazy"  # or "eager"
    dedupe_threshold: Optional[float] = None  # reserved for future use


@dataclass
class _GeneratorResult:
    reasoning: str
    bullet_ids: List[str]
    final_answer: str


class AceAgent(MemoryAgent):
    def __init__(self, config: AceAgentConfig, logger=None):
        super().__init__(config, logger=logger)
        self.generator_template = self._read_file(config.generator_prompt)
        self.reflector_template = self._read_file(config.reflector_prompt)
        self.curator_template = self._read_file(config.curator_prompt)
        self._last_question: Optional[str] = None
        self._last_generator_payload: Optional[Dict[str, Any]] = None

    @property
    def mem(self) -> AceMemory:
        return self.memory  # type: ignore[return-value]

    def _read_file(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # MemoryAgent abstract methods (not used by act/observe, kept for interface)
    def build_user_prompt(self, obs: str, history: List[Any], k: Union[int, None]) -> List[Dict[str, str]]:
        # ACE does not use generic history-to-chat messages; it renders role-specific templates.
        return [{"role": "user", "content": obs}]

    # -----------------------------
    # Core ACE loop
    # -----------------------------
    def act(self, obs: str) -> str:
        self._last_question = obs
        playbook_text = self.mem.playbook.to_text()
        reflection_text = self.mem.last_reflection_text()

        rendered = self.generator_template.format(
            playbook=playbook_text,
            reflection=reflection_text,
            question=obs,
            # context=obs,
        )

        # Enforce strict JSON via Gemini response_schema
        action_schema = {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "bullet_ids": {"type": "array", "items": {"type": "string"}},
                "final_answer": {"type": "string"},
            },
            "required": ["reasoning", "bullet_ids", "final_answer"],
            "additionalProperties": False,
        }
        with jsonlogger.json_log_context(call_type="action", response_schema=action_schema):
            messages = [
                {"role": "system", "content": self.build_system_prompt()},
                {"role": "user", "content": rendered},
            ]
            resp = self._lm_call(messages)
        text = (resp.get("text") or "").strip()
        payload = self._parse_json(text)
        self._last_generator_payload = payload

        final_answer = str(payload.get("final_answer", "")).strip()

        return final_answer

    def observe(self, obs: Optional[str], feedback: dict, done: bool) -> None:
        if self._last_generator_payload is None or self._last_question is None:
            return
        # Skip reflection/curation during evaluation
        if not self.training:
            self.logger.debug("ACE observe: skipping reflection/curation in eval mode")
            return
        # No memory logging for observations in ACE memory

        playbook_text = self.mem.playbook.to_text()
        gen_reasoning = self._last_generator_payload.get("reasoning", "")
        gen_answer = self._last_generator_payload.get("final_answer", "")

        reflection = self._run_reflection(
            question=self._last_question,
            gen_reasoning=gen_reasoning,
            gen_answer=gen_answer,
            feedback=feedback,
            playbook_text=playbook_text,
        )
        self._run_curation(
            reflection=reflection,
            feedback=feedback,
            playbook_text=playbook_text,
        )

        # No memory logging for feedback in ACE memory

    def _playbook_json(self) -> Dict[str, Any]:
        # Serialize playbook to plain dict
        data: Dict[str, List[Dict[str, Any]]] = {}
        for s, bullets in self.mem.playbook.sections.items():
            data[s] = [b.model_dump() for b in bullets]
        return {"sections": data}

    def _parse_json(self, text: str) -> Dict[str, Any]:
        # Simple JSON parser: attempt direct loads, then extract first {...} block
        try:
            return json.loads(text)
        except Exception as e:
            self.logger.error(f"Error parsing JSON: {e}")
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        return {}

    # -----------------------------
    # Helpers: reflection and curation
    # -----------------------------
    def _run_reflection(
        self,
        *,
        question: str,
        gen_reasoning: str,
        gen_answer: str,
        feedback: Dict[str, Any],
        playbook_text: str,
    ) -> Dict[str, Any]:
        ground_truth = feedback.get("target")
        env_feedback = feedback.get("message")

        reflection_rendered = self.reflector_template.format(
            question=question,
            model_reasoning_trace=gen_reasoning,
            model_predicted_answer=gen_answer,
            ground_truth_answer=ground_truth,
            environment_feedback=env_feedback,
            playbook=playbook_text,
        )

        reflection_schema = {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "error_identification": {"type": "string"},
                "root_cause_analysis": {"type": "string"},
                "correct_approach": {"type": "string"},
                "key_insight": {"type": "string"},
                "bullet_tags": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "tag": {"type": "string", "enum": ["helpful", "harmful", "neutral"]},
                        },
                        "required": ["id", "tag"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": [
                "reasoning",
                "error_identification",
                "root_cause_analysis",
                "correct_approach",
                "key_insight",
                "bullet_tags",
            ],
            "additionalProperties": False,
        }
        with jsonlogger.json_log_context(call_type="reflection", response_schema=reflection_schema):
            messages = [
                {"role": "user", "content": reflection_rendered},
            ]
            r_resp = self._lm_call(messages)
        r_text = (r_resp.get("text") or "").strip()
        reflection = self._parse_json(r_text)
        self.mem.reflections.append(reflection)
        self.mem.playbook.tag_bullets(reflection.get("bullet_tags") or [])
        return reflection

    def _run_curation(
        self,
        *,
        reflection: Dict[str, Any],
        feedback: Dict[str, Any],
        playbook_text: str,
    ) -> Dict[str, Any]:
        playbook_stats = self.mem.playbook.stats()
        model_limit = getattr(self.lm.config, "max_output_tokens", 8192)
        safety_margin = 512
        token_budget = max(512, min(self.config.max_playbook_tokens, model_limit - safety_margin))  # type: ignore[attr-defined]
        curator_rendered = self.curator_template.format(
            token_budget=token_budget,
            current_step=0,
            total_samples=0,
            playbook_stats=json.dumps(playbook_stats, ensure_ascii=False, indent=2),
            recent_reflection=json.dumps(reflection, ensure_ascii=False, indent=2),
            current_playbook=json.dumps(self._playbook_json(), ensure_ascii=False, indent=2),
            question_context=json.dumps({
                "question": self._last_question,
                "feedback": feedback,
            }, ensure_ascii=False, indent=2),
        )

        curation_schema = {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "operations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["ADD"]},
                            "section": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["type", "section", "content"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["reasoning", "operations"],
            "additionalProperties": False,
        }
        with jsonlogger.json_log_context(call_type="curation", response_schema=curation_schema):
            messages = [
                {"role": "system", "content": self.build_system_prompt()},
                {"role": "user", "content": curator_rendered},
            ]
            c_resp = self._lm_call(messages)
        c_text = (c_resp.get("text") or "").strip()
        curator = self._parse_json(c_text)
        operations = curator.get("operations") or []
        self.mem.playbook.apply_curator_ops(operations)
        return curator


