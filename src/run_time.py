from typing import Optional, List, Dict, Any, Tuple
from logging import Logger, getLogger
from pydantic import BaseModel
from tqdm.auto import tqdm

from src.data.env import EnvDataset, Environment
from src.agent.agent import Agent


class RunTimeConfig(BaseModel):
    max_envs_to_visit: Optional[int] = None
    max_steps_per_episode: Optional[int] = None
    verbose: bool = True


class StepResult(BaseModel):
    obs: Optional[str]
    action: str
    feedback: Dict[str, Any]
    done: bool


class RunTime:
    def __init__(self, config: RunTimeConfig, env_dataset: EnvDataset, agent: Agent, logger: Optional[Logger] = None):
        self.config = config
        self.env_dataset = env_dataset
        self.agent = agent
        self.logger = logger or getLogger("runtime")

    def run(self) -> Dict[str, Any]:
        environments = self.env_dataset.get_dataset()
        if self.config.max_envs_to_visit is not None:
            environments = environments[: self.config.max_envs_to_visit]

        all_steps: List[List[StepResult]] = []
        self.logger.info("Run: environments=%d", len(environments))
        for idx, environment in enumerate(tqdm(environments, desc="Episodes", total=len(environments)), start=1):
            self.logger.info("Episode %d: start", idx)
            steps = self.run_episode(environment)
            all_steps.append(steps)
            ep_correct = sum(1 for s in steps if s.feedback.get("correct"))
            self.logger.info("Episode %d: end steps=%d correct=%d", idx, len(steps), ep_correct)

        # Minimal aggregate: overall accuracy (expects env feedback to include 'correct')
        total = 0
        correct = 0
        for episode in all_steps:
            for step in episode:
                total += 1
                if step.feedback.get("correct"):
                    correct += 1
        accuracy = (correct / total) if total > 0 else 0.0
        self.logger.info("Run finished: accuracy=%.3f total=%d correct=%d", accuracy, total, correct)
        episodes_serialized = [[s.model_dump() for s in episode] for episode in all_steps]
        return {"accuracy": accuracy, "episodes": episodes_serialized}

    def run_episode(self, environment: Environment) -> List[StepResult]:
        steps: List[StepResult] = []
        obs = environment.reset()
        self.agent.reset()
        done = False
        step_counter = 0
        while not done:
            action = self.agent.act(obs)
            next_obs, feedback, done, info = environment.step(action)
            self.agent.observe(next_obs, feedback, done)
            steps.append(StepResult(obs=next_obs, action=action, feedback=feedback, done=done))
            self.logger.info(
                "Step %d: done=%s correct=%s",
                step_counter + 1,
                str(done),
                str(feedback.get("correct")),
            )
            obs = next_obs if next_obs is not None else ""
            step_counter += 1
            if self.config.max_steps_per_episode is not None and step_counter >= self.config.max_steps_per_episode:
                break
        self.agent.end_episode()
        return steps