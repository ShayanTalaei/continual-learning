from typing import Optional, List, Dict, Any, Tuple
from logging import Logger, getLogger
from pydantic import BaseModel
from tqdm.auto import tqdm
from pathlib import Path
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from src.data.env import EnvDataset, Environment
from src.utils import logger as jsonlogger
from src.agent.agent import Agent


class RunTimeConfig(BaseModel):
    max_envs_to_visit: Optional[int] = None
    max_steps_per_episode: Optional[int] = None
    verbose: bool = True
    scores_path: Optional[str] = None  # if provided, write scores.jsonl and scores.json snapshot
    validation_freq: Optional[int] = None
    validation_num_workers: Optional[int] = None
    run_validation_at_start: bool = False


class StepResult(BaseModel):
    obs: Optional[str]
    action: str
    feedback: Dict[str, Any]
    done: bool


class RunTime:
    def __init__(self, config: RunTimeConfig, train_dataset: EnvDataset, agent: Agent, logger: Optional[Logger] = None, validation_dataset: Optional[EnvDataset] = None):
        self.config = config
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.agent = agent
        self.logger = logger or getLogger("runtime")
        self._scores_buffer: List[Dict[str, Any]] = []
        self._scores_io_lock: Lock = Lock()
        # Training progress counters
        self.num_seen_episodes: int = 0
        self.num_seen_episodes = 0

    def run(self) -> Dict[str, Any]:
        environments = self.train_dataset.get_dataset()
        if self.config.max_envs_to_visit is not None:
            environments = environments[: self.config.max_envs_to_visit]

        all_steps: List[List[StepResult]] = []
        self.logger.info("Run: environments=%d", len(environments))

        if self.config.run_validation_at_start:
            self._run_validation()

        train_steps_total = 0
        for idx, environment in enumerate(tqdm(environments, desc="Episodes", total=len(environments)), start=1):
            self.logger.info("Episode %d: start", idx)
            steps = self._run_episode_with_agent(self.agent, environment, idx, mode="train")
            all_steps.append(steps)
            train_steps_total += len(steps)
            # Update counters visible to logger contexts for subsequent validations
            self.num_seen_episodes += 1
            ep_score = sum(self._get_score(s.feedback) for s in steps)
            self.logger.info("Episode %d: end steps=%d score_sum=%.3f", idx, len(steps), ep_score)


            if self.num_seen_episodes % self.config.validation_freq == 0:
                self._run_validation()

        # Aggregates: mean score across steps
        total = 0
        score_sum = 0.0
        for episode in all_steps:
            for step in episode:
                total += 1
                score_sum += float(self._get_score(step.feedback))
        mean_score = (score_sum / total) if total > 0 else 0.0
        self.logger.info("Run finished: mean_score=%.3f total=%d", mean_score, total)
        episodes_serialized = [[s.model_dump() for s in episode] for episode in all_steps]
        return {"mean_score": mean_score, "episodes": episodes_serialized, 
                "train_steps": train_steps_total, "train_episodes": self.num_seen_episodes}

    def _run_episode_with_agent(self, running_agent: Agent, environment: Environment, episode_index: int, mode: str) -> List[StepResult]:
        steps: List[StepResult] = []
        obs = environment.reset()
        running_agent.reset()
        done = False
        step_counter = 0
        episode_cum_score = 0.0
        # Per-episode logging context (include training progress)
        with jsonlogger.json_log_context(
            mode=mode,
            episode_index=episode_index,
            num_seen_episodes=self.num_seen_episodes,
        ):
            while not done:
                # Per-step logging context (includes step index)
                with jsonlogger.json_log_context(step_index=step_counter + 1):
                    action = running_agent.act(obs)
                next_obs, feedback, done, info = environment.step(action)
                # Only observe when training
                if getattr(running_agent, "training", True):
                    with jsonlogger.json_log_context(step_index=step_counter + 1):
                        running_agent.observe(next_obs, feedback, done)
                steps.append(StepResult(obs=next_obs, action=action, feedback=feedback, done=done))
                score = float(self._get_score(feedback))
                episode_cum_score += score
                self.logger.info(
                    "Step %d: done=%s score=%.3f episode_cum_score=%.3f",
                    step_counter + 1,
                    str(done),
                    score,
                    episode_cum_score,
                )
                self._write_score_line(mode, episode_index, step_counter + 1, score, episode_cum_score)
                obs = next_obs if next_obs is not None else ""
                step_counter += 1
                if self.config.max_steps_per_episode is not None and step_counter >= self.config.max_steps_per_episode:
                    break
        running_agent.end_episode()
        return steps

    def _run_validation(self) -> None:
        if self.validation_dataset is None:
            self.logger.warning("Validation requested but no validation dataset available")
            return
        val_ds: List[Environment] = self.validation_dataset.get_dataset()
        self.logger.info("Starting validation at train episode %d on %d examples", self.num_seen_episodes, len(val_ds))
        max_workers = self.config.validation_num_workers
        results: List[List[StepResult]] = []
        with self.agent.eval_mode():
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for j, env in enumerate(val_ds, start=1):
                    agent_clone = self.agent.clone_for_episode(training=False, share_memory=True)
                    futures.append(executor.submit(self._run_episode_with_agent, agent_clone, env, j, "val"))
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Validation"):
                    results.append(fut.result())
        # Aggregate
        total = sum(len(ep) for ep in results)
        score_sum = sum(self._get_score(s.feedback) for ep in results for s in ep)
        mean_score_val = (score_sum / total) if total > 0 else 0.0
        self.logger.info("Validation finished: mean_score_val=%.3f total=%d", mean_score_val, total)

    def _get_score(self, feedback: Dict[str, Any]) -> float:
        if "score" in feedback:
            try:
                return float(feedback.get("score", 0))
            except Exception:
                return 0.0
        # Backward compatibility
        if feedback.get("correct") is not None:
            self.logger.warning("Deprecated 'correct' key detected; please migrate to 'score'.")
            return 1.0 if bool(feedback.get("correct")) else 0.0
        if feedback.get("is_correct") is not None:
            self.logger.warning("Deprecated 'is_correct' key detected; please migrate to 'score'.")
            return 1.0 if bool(feedback.get("is_correct")) else 0.0
        return 0.0

    def _write_score_line(self, mode: str, episode_index: int, step_index: int, 
                          score: float, episode_cum_score: float) -> None:
        if not self.config.scores_path:
            return
        base = Path(self.config.scores_path)
        scores_dir = base if base.is_dir() or base.suffix == "" else base.parent
        scores_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mode-specific subdirectory
        mode_dir = scores_dir / "scores" / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        
        # For validation, include step number in filename; for training, use single file
        if mode == "train":
            filename = "scores.jsonl"
        else:  # validation
            filename = f"{self.num_seen_episodes}_seen_episodes_scores.jsonl"
        
        scores_jsonl = mode_dir / filename
        rec = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "episode_index": episode_index,
            "step_index": step_index,
            "score": score,
            "episode_cum_score": episode_cum_score,
        }
        jsonlogger.jsonl_append(scores_jsonl, rec)