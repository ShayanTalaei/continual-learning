from asyncio import base_events
from typing import List, Any, Union, Dict

from src.agent.memory_agent import MemoryAgent, MemoryAgentConfig
from src.memory.history_list import HistoryListConfig, Entry
from src.utils import logger as jsonlogger


class SystemSummaryAgentConfig(MemoryAgentConfig):
    memory_config: HistoryListConfig  # type: ignore[assignment]
    history_k: Union[int, None] = None


# NOTE: only works for synth cities
class SystemSummaryAgent(MemoryAgent):
    def __init__(self, config: SystemSummaryAgentConfig, logger=None):
        super().__init__(config, logger=logger)
    
    def build_user_prompt(self, obs: str, history: List[Any], k: Union[int, None], merge_feedback_and_observation: bool = False) -> List[Dict[str, str]]:
        return [{"role": "user", "content": obs}]
    
    def create_observation_event(self, obs: str) -> Any:
        # Stores the description of the city
        return Entry(type="Observation", content=obs.replace("What is the name of the city with the following description: ", ""))

    def create_action_event(self, action: str) -> Any:
        return Entry(type="Action", content=action)

    def create_feedback_event(self, feedback: dict) -> Any:
        # Stores the correct answer
        feedback_message = feedback.get("message", "")
        if "did not output any answer" in feedback_message:
            correct_answer = " ".join(feedback_message.split(" ")[-2:]).split("\\boxed{")[1].split("}")[0]
        elif "correct!" in feedback_message.lower():
            history = self.memory.recall()
            last_message = history[-1].content
            if "boxed" in last_message:
                correct_answer = last_message.split("\\boxed{")[1].split("}")[0]
            else:
                correct_answer = ""
        else:
            correct_answer = " ".join(feedback_message.split(" ")[-2:])
        return Entry(type="Feedback", content=correct_answer.replace(".", ""))
    
    def add_history_to_system_prompt(self, system_prompt: str, history: List[Any]) -> str:
        desc_name_pairs = []
        history_idx = 0
        while history_idx < len(history):
            assert history[history_idx].type == "Observation"
            assert history[history_idx+1].type == "Action" and history[history_idx+2].type == "Feedback"
            desc_name_pairs.append((history[history_idx].content, history[history_idx+2].content))
            history_idx += 3
        if len(desc_name_pairs) > 0:
            system_prompt += "\n\nCities seen so far:\n\n" + "\n".join([f"- {name}: {desc}" for desc, name in desc_name_pairs])
        else:
            system_prompt += "\n\nCities seen so far:\n\nNo cities seen so far."
        return system_prompt

    def act(self, obs: str) -> str:
        self.logger.info("Act: obs_len=%d", len(obs))
        history = self.memory.recall()
        base_system_prompt = self.build_system_prompt()
        system_prompt = self.add_history_to_system_prompt(base_system_prompt, history)

        obs_event = self.create_observation_event(obs)
        if obs_event is not None:
            self.memory.update(obs_event)
            self._trajectory.append(obs_event)
        self.logger.info("Logged Observation")
        
        with jsonlogger.json_log_context(call_type="action"):
            messages = [
                {"role": "system", "content": system_prompt}
            ] + self.build_user_prompt(obs, history, self.config.history_k, self.config.merge_feedback_and_observation)
            resp = self._lm_call(messages)
        action = (resp.get("text") or "").strip()
        if not action:
            self.logger.warning("No action returned from LM")
        self.logger.info(f"Action generated: {action[:25] + '...' + action[-25:] if len(action) > 50 else action}")
        
        self._last_action = action
        action_event = self.create_action_event(self._last_action)
        if action_event is not None:
            self.memory.update(action_event)
            self._trajectory.append(action_event)
        self.logger.info("Logged Action")
        
        return action

