"""
ReflexionAgent: Self-reflective agent inspired by "Reflexion: Language Agents with 
Verbal Reinforcement Learning" (Shinn et al., 2023).

After each episode, the agent analyzes its trajectory using an LM to generate verbal
reflections that are stored in memory and used to improve future performance.
"""

from typing import List, Any
from src.agent.memory_agent import MemoryAgent, MemoryAgentConfig
from src.memory.history_list import Entry, HistoryListConfig


# Default prompts (task-agnostic, based on original Reflexion paper)
DEFAULT_REFLECTION_SYSTEM_PROMPT = """You are a self-reflective AI assistant. You will be given a trajectory of your attempt to solve a task, including your observations, actions, and feedback received. Your task failed or was suboptimal. Analyze what went wrong and provide a concise reflection (2-4 sentences) that explains:
1) What strategy or approach you took
2) Why it didn't work or what mistake you made
3) What you should do differently next time

Be specific and actionable. Focus on the reasoning or approach, not just the surface-level error."""


class ReflexionAgentConfig(MemoryAgentConfig):
    """Configuration for ReflexionAgent with reflection-specific settings."""
    
    # Memory configuration - override to use HistoryListConfig like HistoryAgent
    memory_config: HistoryListConfig
    history_k: int | None = None
    
    # Reflection control
    enable_reflection: bool = True
    reflect_on_failure_only: bool = False
    failure_threshold: float = 1.0  # Score below this is considered failure
    
    # Prompt configuration
    max_reflections_in_prompt: int | None = 5  # None = all reflections
    reflection_system_prompt: str | None = None  # Override default
    reflection_few_shot_examples: str | None = None  # Optional examples
    
    # Agent system prompt override
    agent_system_prompt: str | None = None


class ReflexionAgent(MemoryAgent):
    """
    Agent that generates self-reflections after episodes to improve future performance.
    
    Inherits from MemoryAgent and adds:
    - Reflection generation at episode end (when training)
    - Reflection-aware prompt formatting
    - Configurable reflection triggers (all episodes vs failures only)
    """
    
    def __init__(self, config: ReflexionAgentConfig, logger=None):
        super().__init__(config, logger=logger)
        self.config: ReflexionAgentConfig  # Type hint for IDE
        self._episode_index: int = 0  # Track episode for reflection logging
        self.logger.info(
            "ReflexionAgent init: enable_reflection=%s, reflect_on_failure_only=%s, failure_threshold=%.2f",
            self.config.enable_reflection,
            self.config.reflect_on_failure_only,
            self.config.failure_threshold,
        )
    
    def build_system_prompt(self) -> str:
        """Build system prompt with reflection awareness."""
        if self.config.agent_system_prompt:
            return self.config.agent_system_prompt
        if self.config.system_prompt:
            return self.config.system_prompt
        return "You are a helpful assistant. Learn from your past experiences and reflections to improve performance."
    
    def build_user_prompt(self, obs: str, history: List[Any], k: int | None) -> str:
        """
        Build user prompt with reflections prominently featured.
        
        Structure:
        1. Previous Reflections (if any)
        2. Recent History (non-reflection entries)
        3. Current Task
        """
        lines: List[str] = []
        
        # Filter reflections and experiences from history
        reflections: List[Entry] = []
        experiences: List[Entry] = []
        
        for entry in history:
            if entry.type == "Reflection":
                reflections.append(entry)
            else:
                experiences.append(entry)
        
        # Limit reflections if configured
        if self.config.max_reflections_in_prompt is not None:
            reflections = reflections[-self.config.max_reflections_in_prompt:]
        
        # Add reflections section
        if reflections:
            lines.append("=== Previous Reflections (lessons learned from past attempts) ===")
            for i, reflection in enumerate(reflections, 1):
                lines.append(f"Reflection {i}: {reflection.content}")
            lines.append("")
        
        # Add recent history section (limited by k)
        if experiences:
            recent_experiences: List[Entry] = experiences[-k:] if k is not None else experiences
            if recent_experiences:
                lines.append("=== Recent History ===")
                for entry in recent_experiences:
                    entry_type = entry.type.upper()
                    lines.append(f"{entry_type}: {entry.content}")
                lines.append("")
        
        # Add current task
        lines.append("=== Current Task ===")
        lines.append(obs)
        
        return "\n".join(lines)
    
    def should_reflect(self) -> bool:
        """Determine if reflection should be generated after current episode."""
        # Don't reflect during evaluation
        if not self.training:
            self.logger.debug("Skip reflection: eval mode")
            return False
        
        # Check if reflection is enabled
        if not self.config.enable_reflection:
            self.logger.debug("Skip reflection: disabled in config")
            return False
        
        # If reflect_on_failure_only, check episode outcome
        if self.config.reflect_on_failure_only:
            episode_score = self._compute_episode_score()
            is_failure = episode_score < self.config.failure_threshold
            self.logger.debug(
                "Reflect on failure check: episode_score=%.3f, threshold=%.3f, is_failure=%s",
                episode_score,
                self.config.failure_threshold,
                is_failure,
            )
            return is_failure
        
        # Default: reflect on every episode
        return True
    
    def _compute_episode_score(self) -> float:
        """Compute total score from episode trajectory."""
        total_score = 0.0
        for event in self._trajectory:
            if event.type == "Feedback":
                content = event.content
                if isinstance(content, dict):
                    if 'score' in content:
                        total_score += float(content['score'])
                    elif 'correct' in content:
                        total_score += 1.0 if content['correct'] else 0.0
                elif isinstance(content, str):
                    if "correct" in content.lower() or "score" in content.lower():
                        if "true" in content.lower() or "1" in content:
                            total_score += 1.0
        
        return total_score
    
    def build_reflection_prompt(self) -> tuple[str, str]:
        """
        Build (system_prompt, user_prompt) for reflection generation.
        
        Returns task-agnostic prompts based on episode trajectory.
        """
        # System prompt
        system_prompt = (
            self.config.reflection_system_prompt 
            if self.config.reflection_system_prompt 
            else DEFAULT_REFLECTION_SYSTEM_PROMPT
        )
        
        # Build trajectory summary
        trajectory_lines: List[str] = []
        trajectory_lines.append("Trajectory:")
        
        step_num = 0
        current_step: dict[str, Any] = {}
        
        for event in self._trajectory:
            if event.type == "Observation":
                # Start new step
                if current_step:
                    trajectory_lines.append(self._format_step(step_num, current_step))
                    step_num += 1
                    current_step = {}
                current_step['observation'] = event.content
            elif event.type == "Action":
                current_step['action'] = event.content
            elif event.type == "Feedback":
                current_step['feedback'] = event.content
        
        # Add final step
        if current_step:
            trajectory_lines.append(self._format_step(step_num, current_step))
        
        # Add few-shot examples if provided
        user_prompt_parts: List[str] = []
        if self.config.reflection_few_shot_examples:
            user_prompt_parts.append("=== Examples ===")
            user_prompt_parts.append(self.config.reflection_few_shot_examples)
            user_prompt_parts.append("")
        
        user_prompt_parts.append("\n".join(trajectory_lines))
        user_prompt_parts.append("")
        user_prompt_parts.append("Reflection:")
        
        user_prompt = "\n".join(user_prompt_parts)
        
        return system_prompt, user_prompt
    
    def _format_step(self, step_num: int, step_data: dict[str, Any]) -> str:
        """Format a single step for the trajectory."""
        lines = [f"\nStep {step_num + 1}:"]
        
        if 'observation' in step_data:
            obs_str = str(step_data['observation'])
            lines.append(f"  Observation: {obs_str}")
        
        if 'action' in step_data:
            action_str = str(step_data['action'])
            lines.append(f"  Action: {action_str}")
        
        if 'feedback' in step_data:
            feedback_str = str(step_data['feedback'])
            lines.append(f"  Feedback: {feedback_str}")
        
        return "\n".join(lines)
    
    def generate_reflection(self) -> str | None:
        """
        Generate a self-reflection using the LM.
        
        Returns the reflection text or None if generation fails.
        """
        try:
            # Import here to avoid circular dependency
            from src.utils import logger as jsonlogger
            
            system_prompt, user_prompt = self.build_reflection_prompt()
            self.logger.info("Generating reflection: trajectory_len=%d", len(self._trajectory))
            
            # Add reflection-specific context for better logging organization
            # This will place reflection calls in a "reflections/" subdirectory
            with jsonlogger.json_log_context(call_type="reflection", episode_index=self._episode_index):
                reflection = self.lm.call(system_prompt, user_prompt)
            
            if reflection:
                reflection = reflection.strip()
                self.logger.info("Reflection generated: len=%d", len(reflection))
                return reflection
            else:
                self.logger.warning("Reflection generation returned empty response")
                return None
                
        except Exception as e:
            self.logger.error("Failed to generate reflection: %s", str(e))
            return None
    
    def create_observation_event(self, obs: str) -> Any:
        """Create an observation entry for storage in memory."""
        return Entry(type="Observation", content=obs)
    
    def create_action_event(self, action: str) -> Any:
        """Create an action entry for storage in memory."""
        return Entry(type="Action", content=action)
    
    def create_feedback_event(self, feedback: dict) -> Any:
        """Create a feedback entry for storage in memory."""
        return Entry(type="Feedback", content=feedback.get("message", ""))
    
    def create_reflection_event(self, reflection: str) -> Any:
        """Create a reflection entry for storage in memory."""
        return Entry(type="Reflection", content=reflection)
    
    def act(self, obs: str) -> str:
        """Override to track episode index for reflection logging."""
        # Import here to avoid circular dependency
        from src.utils import logger as jsonlogger
        
        # Get episode index from context for reflection logging
        ctx = jsonlogger.json_get_context()
        episode_idx = ctx.get("episode_index")
        if episode_idx is not None:
            self._episode_index = episode_idx
        
        # Call parent act method
        return super().act(obs)
    
    def end_episode(self) -> None:
        """
        Override to generate and store reflection after episode completion.
        
        Reflections are only generated during training and when conditions are met.
        """
        # Check if we should reflect
        if self.should_reflect():
            self.logger.info("End episode: generating reflection")
            reflection = self.generate_reflection()
            
            if reflection:
                # Create and store reflection event
                reflection_event = self.create_reflection_event(reflection)
                if reflection_event is not None:
                    self.memory.update(reflection_event)
                    self.logger.info("Reflection stored in memory")
            else:
                self.logger.warning("End episode: reflection generation failed")
        else:
            self.logger.info("End episode: reflection skipped")
        
        # Clear trajectory (inherited behavior)
        self._trajectory = []
    
    def clone_for_episode(self, training: bool, share_memory: bool = True) -> "ReflexionAgent":
        """Create a per-episode clone with independent trajectory but optionally shared memory."""
        clone = self.__class__(self.config, logger=self.logger)
        # Share LM to save resources
        clone.lm = self.lm
        # Optionally share memory (safe for eval when training=False)
        if share_memory:
            clone.memory = self.memory
        clone.training = training
        self.memory.training = training
        clone._trajectory = []
        clone._last_action = None
        clone._episode_index = self._episode_index
        return clone

