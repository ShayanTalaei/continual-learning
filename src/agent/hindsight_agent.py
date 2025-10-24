from typing import List, Any, Union, Dict, Optional, Tuple

from src.agent.memory_agent import MemoryAgent, MemoryAgentConfig
from src.memory.history_list import HistoryListConfig, Entry
from src.utils import logger as jsonlogger


class HindsightAgentConfig(MemoryAgentConfig):
    memory_config: HistoryListConfig  # type: ignore[assignment]
    history_k: Union[int, None] = None
    enable_hindsight: bool = True


class HindsightAgent(MemoryAgent):
    def __init__(self, config: HindsightAgentConfig, logger=None):
        super().__init__(config, logger=logger)
        self.config: HindsightAgentConfig
        # Keep last step state for one-step episodes
        self._pending_observation: Optional[str] = None
        self._pending_action: Optional[str] = None
        self._pending_feedback: Optional[dict] = None

    # -----------------------------
    # Prompt construction
    # -----------------------------
    def build_system_prompt(self) -> str:
        return self.system_prompt

    def build_user_prompt(self, obs: str, history: List[Any]) -> List[Dict[str, str]]:
        messages: List[dict] = []

        # Use the same alternating format as HistoryAgent to leverage chat conditioning
        for entry in history:
            if entry.type.lower() == "observation":
                messages.append({"role": "user", "content": str(entry.content)})
            elif entry.type.lower() == "action":
                messages.append({"role": "assistant", "content": str(entry.content)})
            elif entry.type.lower() == "feedback":
                messages.append({"role": "user", "content": f"Feedback: {str(entry.content["message"])}"})

        # Current observation as final user message
        messages.append({"role": "user", "content": f"{obs}"})
        return messages

    # -----------------------------
    # Event constructors
    # -----------------------------
    def create_observation_event(self, obs: str) -> Any:
        return Entry(type="Observation", content=obs)

    def create_action_event(self, action: str) -> Any:
        return Entry(type="Action", content=action)

    def create_feedback_event(self, feedback: dict) -> Any:
        # Store full feedback in trajectory only; not persisted in memory for this agent
        return Entry(type="Feedback", content=feedback)

    # -----------------------------
    # Core loop overrides
    # -----------------------------
    def act(self, obs: str) -> str:
        # Regular action generation identical to MemoryAgent, but we defer persisting the action
        self.logger.info("Act (hindsight): obs_len=%d", len(obs))

        history = self.memory.recall()
        system_prompt = self.build_system_prompt()
        memory_messages = self.build_user_prompt(obs, history)

        # Persist observation immediately
        obs_event = self.create_observation_event(obs)
        self._trajectory.append(obs_event)
        self.logger.info("Logged Observation")

        with jsonlogger.json_log_context(call_type="action"):
            messages = [
                {"role": "system", "content": system_prompt}
            ] + memory_messages
            resp = self.lm.call(messages)
        action = (resp.get("text") or "").strip()
        self.logger.info(f"Action generated: {action[:25] + '...' + action[-25:] if len(action) > 50 else action}")
        action_event = self.create_action_event(action)
        self._trajectory.append(action_event)
        return action

    def observe(self, obs: str | None, feedback: dict, done: bool) -> None:
        # For one-step envs, we only track feedback in the trajectory and hold it for end_episode
        if obs is not None:
            obs_event = self.create_observation_event(obs)
            self._trajectory.append(obs_event)
            self.logger.info("Logged Observation")

        if feedback is not None:
            fb_event = self.create_feedback_event(feedback)
            self._trajectory.append(fb_event)
            self.logger.info("Logged Feedback correct=%s", str(feedback.get("correct")))

    # -----------------------------
    # Hindsight logic
    # -----------------------------
    def _should_generate_hindsight(self) -> bool:
        if not self.training:
            return False
        if not getattr(self.config, "enable_hindsight", True):
            return False
        assert self._trajectory[-1].type.lower() == "feedback", "Last trajectory entry must be a feedback event"
        score = self._trajectory[-1].content.get("score")
        return score < 1.0

    def _compose_hindsight_instruction(self, prev_obs: str) -> str:
        # Mirror the notebook style: append a new user message that instructs rethinking, includes feedback and the question
        hindsight_generation_prompt = """
Given that your previous action was incorrect, let's rethink the question and answer it again. You shouldn't refer to the previous action nor the feedback you received directly in your response.

Now, let's think step by step and answer the question again.

Question: {question}
"""
        return hindsight_generation_prompt.format(question=prev_obs)
    
    def _build_hindsight_messages(self) -> List[Dict[str, str]]:
        # Build base conversation (system + history-formatted messages + current obs as last user)
        history = self.memory.recall()
        obs = self._pending_observation or ""
        system_prompt = self.build_system_prompt()
        base_messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ] + self.build_user_prompt(obs, history)
        
        for entry in self._trajectory:
            if entry.type.lower() == "observation":
                base_messages.append({"role": "user", "content": str(entry.content)})
            elif entry.type.lower() == "action":
                base_messages.append({"role": "assistant", "content": str(entry.content)})
            elif entry.type.lower() == "feedback":
                base_messages.append({"role": "user", "content": f"Feedback: {str(entry.content["message"])}"})
    
        base_messages.append({"role": "user", "content": self._compose_hindsight_instruction(obs)})
        return base_messages

    def _generate_hindsight_action(self) -> Optional[str]:
        try:
            with jsonlogger.json_log_context(call_type="hindsight"):
                messages = self._build_hindsight_messages()
                resp = self.lm.call(messages)
            text = (resp.get("text") or "").strip()
            generated_action = self.create_action_event(text)
            return generated_action
        except Exception as e:
            self.logger.error("Failed to generate hindsight action: %s", str(e))
            return None

    def end_episode(self) -> None:
        # Decide what to persist based on correctness and training mode
        try:

            if self._should_generate_hindsight():
                self.logger.info("End episode: generating hindsight action")
                hindsight_action = self._generate_hindsight_action()
                action_event = hindsight_action
                feedback_event = Entry(type="Feedback", content={"message": "Correct!", "score": 0.0})
            else:
                action_event = self._trajectory[-2]
                feedback_event = Entry(type="Feedback", content={"message": "Correct!", "score": 1.0})
            observation_event = self._trajectory[-3]
            assert observation_event.type.lower() == "observation", "Last trajectory entry must be an observation event"
            self.memory.update(observation_event)

            self.memory.update(action_event)
            
            self.memory.update(feedback_event)
        finally:
            # Clear episode state and trajectory
            self._trajectory = []

    def clone_for_episode(self, training: bool, share_memory: bool = True) -> "HindsightAgent":
        clone = self.__class__(self.config, logger=self.logger)
        # Share LM to save resources
        clone.lm = self.lm
        # Optionally share memory
        if share_memory:
            clone.memory = self.memory
        clone.training = training
        self.memory.training = training
        clone._trajectory = []
        clone._last_action = None
        clone._pending_action = None
        clone._pending_observation = None
        clone._pending_feedback = None
        return clone


