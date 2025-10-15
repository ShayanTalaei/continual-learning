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
from src.utils.checkpoint import CheckpointManager


class RunTimeConfig(BaseModel):
    max_envs_to_visit: Optional[int] = None
    max_steps_per_episode: Optional[int] = None
    verbose: bool = True
    scores_path: Optional[str] = None  # if provided, write scores.jsonl and scores.json snapshot
    validation_freq: Optional[int] = None
    validation_num_workers: Optional[int] = None
    run_validation_at_start: bool = False
    verbose_score_logging: bool = True
    # Checkpointing
    checkpoint_dir: Optional[str] = None
    checkpoint_every_episodes: Optional[int] = None
    checkpoint_keep_last: Optional[int] = None
    checkpoint_on_start: bool = False
    # Strategy (optional): "last_n" (default) or "top_k_val" when combined with keep_last
    checkpoint_strategy: Optional[str] = None
    resume_from: Optional[str] = None
    start_episode_index: int = 0


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
        # Buffer for validation logs to preserve deterministic ordering (episode, step)
        self._val_logs_buffer = jsonlogger.ValidationLogsBuffer()
        # Checkpoint manager (optional)
        self._cp_manager: Optional[CheckpointManager] = None
        if self.config.checkpoint_dir:
            strategy = (self.config.checkpoint_strategy or "last_n").lower()
            keep_count = int(self.config.checkpoint_keep_last or 0)
            self._cp_manager = CheckpointManager(
                base_dir=self.config.checkpoint_dir,
                strategy=strategy,
                keep_count=keep_count,
                every_episodes=self.config.checkpoint_every_episodes,
                logger=self.logger,
            )

    def run(self) -> Dict[str, Any]:
        environments = self.train_dataset.get_dataset()
        if self.config.max_envs_to_visit is not None:
            environments = environments[: self.config.max_envs_to_visit]

        all_steps: List[List[StepResult]] = []
        self.logger.info("Run: environments=%d", len(environments))

        # Optional checkpoint at start
        if self._cp_manager and self.config.checkpoint_on_start:
            self._cp_manager.maybe_checkpoint_on_start(self.agent, episode_index=self.config.start_episode_index)

        train_steps_total = 0
        # Initialize counters if resuming
        if self.config.start_episode_index > 0:
            self.num_seen_episodes = self.config.start_episode_index
        
        if self.config.run_validation_at_start:
            with jsonlogger.json_log_context(mode="val"):
                self._run_validation()

        for idx, environment in enumerate(tqdm(environments, desc="Episodes", total=len(environments)), start=1):
            # Skip already-processed episodes on resume
            if idx <= self.config.start_episode_index:
                continue
            with jsonlogger.json_log_context(
                mode="train",
            ):
                self.logger.info("Episode %d: start", idx)
                steps = self._run_episode_with_agent(self.agent, environment, idx, mode="train")
                all_steps.append(steps)
                train_steps_total += len(steps)
                # Update counters visible to logger contexts for subsequent validations
                self.num_seen_episodes += 1
                ep_score = sum(self._get_score(s.feedback) for s in steps)
                self.logger.info("Episode %d: end steps=%d score_sum=%.3f", idx, len(steps), ep_score)

            # Checkpoint after each episode if configured (delegated)
            if self._cp_manager:
                self._cp_manager.on_episode_end(self.agent, episode_index=self.num_seen_episodes, train_steps_total=train_steps_total)

            if self.config.validation_freq is not None and self.num_seen_episodes % self.config.validation_freq == 0:
                with jsonlogger.json_log_context(mode="val"):
                    mean_val = self._run_validation()
                if self._cp_manager and mean_val is not None:
                    self._cp_manager.on_validation_complete(self.agent, episode_index=self.num_seen_episodes, train_steps_total=train_steps_total, mean_val_score=mean_val)

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
                step_start = datetime.utcnow()
                # Per-step logging context (includes step index)
                # Attach response schema to context if env provides it
                schema = None
                try:
                    schema = getattr(environment, "response_schema")()  # type: ignore[attr-defined]
                except Exception:
                    schema = None
                with jsonlogger.json_log_context(step_index=step_counter + 1, response_schema=schema):
                    action = running_agent.act(obs)
                next_obs, feedback, done, info = environment.step(action)
                # Only observe when training
                with jsonlogger.json_log_context(step_index=step_counter + 1):
                    running_agent.observe(next_obs, feedback, done)
                steps.append(StepResult(obs=next_obs, action=action, feedback=feedback, done=done))
                score = float(self._get_score(feedback))
                episode_cum_score += score
                step_end = datetime.utcnow()
                duration_ms = (step_end - step_start).total_seconds() * 1000.0
                self.logger.info(
                    "Step %d: done=%s score=%.3f episode_cum_score=%.3f",
                    step_counter + 1,
                    str(done),
                    score,
                    episode_cum_score,
                )
                self._write_score_line(
                    mode,
                    environment,
                    episode_index,
                    step_counter + 1,
                    score,
                    episode_cum_score,
                    observation=next_obs,
                    action=action,
                    feedback=feedback,
                    info=info,
                    lm_model=running_agent.lm.config.model if running_agent.lm.config is not None else None,
                    agent_type=running_agent.__class__.__name__,
                    step_start=step_start,
                    step_end=step_end,
                    duration_ms=duration_ms,
                )
                obs = next_obs if next_obs is not None else ""
                step_counter += 1
                if self.config.max_steps_per_episode is not None and step_counter >= self.config.max_steps_per_episode:
                    break
            running_agent.end_episode()
        return steps

    def _run_validation(self) -> Optional[float]:
        if self.validation_dataset is None:
            self.logger.warning("Validation requested but no validation dataset available")
            return None
        val_ds: List[Environment] = self.validation_dataset.get_dataset()
        self.logger.info("Starting validation at train episode %d on %d examples", self.num_seen_episodes, len(val_ds))
        max_workers = self.config.validation_num_workers
        results: List[List[StepResult]] = []
        # Clear validation buffer before run
        self._val_logs_buffer.clear()
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
        # Flush validation logs in deterministic order
        self._val_logs_buffer.flush()
        return mean_score_val

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

    def _write_score_line(self, mode: str, 
                          environment: Environment,
                          episode_index: int, step_index: int, 
                          score: float, episode_cum_score: float,
                          observation: Optional[str] = None,
                          action: Optional[str] = None,
                          feedback: Optional[Dict[str, Any]] = None,
                          info: Optional[Dict[str, Any]] = None,
                          lm_model: Optional[str] = None,
                          agent_type: Optional[str] = None,
                          step_start: Optional[datetime] = None,
                          step_end: Optional[datetime] = None,
                          duration_ms: Optional[float] = None,) -> None:
        if not self.config.scores_path:
            return
        scores_jsonl = jsonlogger.score_file_for_mode(self.config.scores_path, mode, self.num_seen_episodes)
        rec: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "mode": mode,
            "episode_index": episode_index,
            "step_index": step_index,
            "score": score,
            "episode_cum_score": episode_cum_score,
            "env_id": environment.env_id,
            "env_type": environment.env_type,
        }
        if self.config.verbose_score_logging:
            rec = jsonlogger.build_score_record(
                mode=mode,
                environment=environment,
                episode_index=episode_index,
                step_index=step_index,
                score=score,
                episode_cum_score=episode_cum_score,
                observation=observation,
                action=action,
                feedback=feedback,
                info=info,
                lm_model=lm_model,
                agent_type=agent_type,
                step_start=step_start,
                step_end=step_end,
                duration_ms=duration_ms,
                verbose_score_logging=self.config.verbose_score_logging,
            )
        if mode == "val":
            # Buffer validation logs to preserve order; flush after validation completes
            self._val_logs_buffer.add(episode_index, step_index, scores_jsonl, rec)
            return
        jsonlogger.write_score_record(scores_jsonl, rec)

    # (Checkpoint utilities removed; delegated to CheckpointManager)