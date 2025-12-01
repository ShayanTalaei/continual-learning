from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from logging import getLogger
from logging import getLogger, Logger

from pydantic import BaseModel

from src.data.env import Environment, EnvDataset, EnvDatasetConfig
from appworld.common.code_tools import extract_code_from_text


def extract_python_code_loose(text: str) -> str:
    """Extract Python code from markdown fences without requiring strict newlines.

    Supports variants like:
    - ```python\n...\n```
    - ```python ... ``` (inline, no newlines required)
    - ```python ... (no closing fence) -> captures to end of string
    Returns the first matched code block content or an empty string if none.
    """
    try:
        import re
        # Prefer standard fenced block with optional newline after header
        pattern_block = re.compile(r"```\s*python\b[^\n]*\n?([\s\S]*?)\n?```", re.IGNORECASE)
        m = pattern_block.search(text)
        if m and m.group(1) is not None:
            return m.group(1).strip()
        # Fallback: capture until end if closing fence is missing
        pattern_open_ended = re.compile(r"```\s*python\b[^\n]*\n?([\s\S]*)$", re.IGNORECASE)
        m2 = pattern_open_ended.search(text)
        if m2 and m2.group(1) is not None:
            return m2.group(1).strip()
        return ""
    except Exception:
        return ""


class AppWorldEnvConfig(BaseModel):
    env_id: str = "appworld_env"
    dataset_name: str
    task_id: Optional[str] = None
    max_steps: Optional[int] = 50
    experiment_name: str = "continual_learning"
    inject_examples: bool = True
    prompts_path: Optional[str] = None


class AppWorldEnv(Environment):
    def __init__(self, cfg: AppWorldEnvConfig, logger=None, prompts: Optional[Dict[str, str]] = None):
        super().__init__(env_id=cfg.env_id, env_type="appworld")
        self.cfg = cfg
        self._logger = logger or getLogger("appworld_env")
        self._step_count = 0
        self._app: Any = None
        self._task_id: Optional[str] = None
        self._prompts: Dict[str, str] = prompts or {}

        try:
            import appworld 
        except Exception as e:
            raise RuntimeError("appworld package is required for AppWorldEnv") from e

    # Task selection is handled by AppWorldEnvDataset. Require explicit task_id here.
    def _select_task_id(self) -> str:
        if not self.cfg.task_id:
            raise RuntimeError(
                "AppWorldEnv requires an explicit task_id. Provide one via AppWorldEnvConfig or use AppWorldEnvDataset to construct environments."
            )
        return self.cfg.task_id
    
    def close(self) -> None:
        """Close the AppWorld environment and release database connections."""
        if self._app is not None:
            try:
                self._app.close()
                self._logger.debug("Closed AppWorld instance for task %s", self._task_id)
            except Exception as e:
                self._logger.warning("Failed to close AppWorld instance: %s", e)
            finally:
                self._app = None

    def _build_initial_prompt(self) -> str:
        # AppWorld provides task specification; use instruction as observation header
        try:
            from appworld.task import Task  # type: ignore
            from appworld.task import task_id_to_generator_id  # type: ignore
        except Exception as e:
            raise RuntimeError("Failed to import appworld.task.Task") from e

        assert self._task_id is not None
        task = Task.load(task_id=self._task_id, load_ground_truth=False)

        # If a general text template is provided, render simple placeholders without Jinja
        general_tmpl = self._prompts.get("general") if self._prompts else None
        if isinstance(general_tmpl, str) and len(general_tmpl.strip()) > 0:
            # Replace {{ instruction }}
            rendered = general_tmpl.replace("{{ instruction }}", str(task.instruction))
            rendered = rendered.replace("{{instruction}}", str(task.instruction))
            # Replace supervisor fields of form {{ supervisor.first_name }}
            sup = task.supervisor
            replacements = {
                "{{ supervisor.first_name }}": getattr(sup, "first_name", ""),
                "{{ supervisor.last_name }}": getattr(sup, "last_name", ""),
                "{{ supervisor.email }}": getattr(sup, "email", ""),
                "{{ supervisor.phone_number }}": getattr(sup, "phone_number", ""),
                # Add main_user aliases (same as supervisor for AppWorld)
                "{{ main_user.first_name }}": getattr(sup, "first_name", ""),
                "{{ main_user.last_name }}": getattr(sup, "last_name", ""),
                "{{ main_user.email }}": getattr(sup, "email", ""),
                "{{ main_user.phone_number }}": getattr(sup, "phone_number", ""),
            }
            for k, v in replacements.items():
                rendered = rendered.replace(k, str(v))
            # Also handle variants without spaces inside braces
            replacements_no_space = {
                "{{supervisor.first_name}}": getattr(sup, "first_name", ""),
                "{{supervisor.last_name}}": getattr(sup, "last_name", ""),
                "{{supervisor.email}}": getattr(sup, "email", ""),
                "{{supervisor.phone_number}}": getattr(sup, "phone_number", ""),
                # Add main_user aliases (same as supervisor for AppWorld)
                "{{main_user.first_name}}": getattr(sup, "first_name", ""),
                "{{main_user.last_name}}": getattr(sup, "last_name", ""),
                "{{main_user.email}}": getattr(sup, "email", ""),
                "{{main_user.phone_number}}": getattr(sup, "phone_number", ""),
            }
            for k, v in replacements_no_space.items():
                rendered = rendered.replace(k, str(v))
            return rendered
        
        return str(task.instruction)

    def reset(self) -> str:
        try:
            from appworld.environment import AppWorld  # type: ignore
        except Exception as e:
            raise RuntimeError("Failed to import appworld.environment.AppWorld") from e

        self._step_count = 0
        self._task_id = self._select_task_id()
        # Initialize a local in-process AppWorld environment
        self._app = AppWorld(
            task_id=self._task_id,
            experiment_name=self.cfg.experiment_name,
        )
        obs = self._build_initial_prompt()
        return obs

    def step(self, action: str) -> Tuple[Optional[str], Dict[str, Any], bool, Dict[str, Any]]:
        self._step_count += 1
        act = (action or "").strip()
        # Handle deliberate thinking without advancing the environment
        if act.lower().startswith("think:"):
            feedback = {
                "score": 0.0,
                "message": "OK.",
                "won": False,
                "done": False,
            }
            done = False
            if self.cfg.max_steps is not None and self._step_count >= self.cfg.max_steps:
                done = True
                feedback["done"] = True
            return "OK.", feedback, done, {}

        # AppWorld expects Python code in execute(). Prefer AppWorld extractor first.
        code = extract_code_from_text(act)
        if not code.strip():
            code = extract_python_code_loose(act)
            if not code.strip():
                code = extract_code_from_text(act)
        if not code.strip():
            code = act

        try:
            message: str = self._app.execute(code)
        except Exception as e:
            message = f"Execution failed. Traceback:\n{e}"

        done = False
        won = False
        # Determine terminal condition first (either task completed or max steps reached)
        task_complete: bool = False
        try:
            task_complete = bool(self._app.task_completed())
        except Exception:
            task_complete = False
        if task_complete:
            done = True
        if self.cfg.max_steps is not None and self._step_count >= self.cfg.max_steps:
            done = True

        # Default binary score; will refine using AppWorld's evaluate() when possible
        score: float = 1.0 if won else 0.0
        evaluation_dict: Dict[str, Any] = {}
        # Use in-Python evaluation when we reach a terminal condition to derive a more informative score
        if done:
            tracker = self._app.evaluate(suppress_errors=True)
            # tracker is a TestTracker; convert to dict for portability
            evaluation_dict = (
                tracker.to_dict(stats_only=False) if hasattr(tracker, "to_dict") else {}
            )
            # Align won to TestTracker.success when available
            if isinstance(evaluation_dict.get("success"), bool):
                won = bool(evaluation_dict.get("success"))
            # Derive score primarily from passes/num_tests in TestTracker
            try:
                num_tests = evaluation_dict.get("num_tests")
                passes_list = evaluation_dict.get("passes")
                if isinstance(num_tests, int) and num_tests > 0 and isinstance(passes_list, list):
                    score = float(len(passes_list)) / float(num_tests)
                else:
                    # Fallbacks for older schemas
                    if isinstance(evaluation_dict.get("score"), (int, float)):
                        score = float(evaluation_dict["score"])  
                    else:
                        passed = evaluation_dict.get("num_passed") or evaluation_dict.get("passed")
                        total = evaluation_dict.get("num_tests") or evaluation_dict.get("total")
                        if passed is not None and total:
                            score = float(passed) / float(total)
            except Exception:
                pass

        feedback = {
            "score": score,
            "message": "OK.",
            "done": done,
            "won": won,
        }
        if evaluation_dict:
            feedback["evaluation"] = evaluation_dict
        info: Dict[str, Any] = {}
        return message, feedback, done, info


class AppWorldEnvDatasetConfig(EnvDatasetConfig):
    type: str = "appworld"
    dataset_name: str
    root: Optional[str] = None
    task_ids: Optional[List[str]] = None
    num_episodes: int = 1
    verbose: bool = True
    max_steps: Optional[int] = 50
    experiment_name: str = "continual_learning"
    prompts_path: Optional[str] = None


class AppWorldEnvDataset(EnvDataset):
    def __init__(self, config: AppWorldEnvDatasetConfig, logger=None):
        self.config = config
        self.logger = logger or getLogger("appworld_dataset")
        self.dataset = self.load_dataset()

    def load_dataset(self) -> List[Environment]:
        try:
            import appworld  # type: ignore  # noqa: F401
        except Exception as e:
            raise RuntimeError("appworld package is required for AppWorldEnvDataset") from e

        # Ensure AppWorld knows where to find its data (datasets/tasks)
        if self.config.root:
            import os
            os.environ["APPWORLD_ROOT"] = self.config.root

        # If explicit task_ids are not provided, sample first N from the dataset
        task_ids = list(self.config.task_ids or [])
        if not task_ids:
            try:
                from appworld.task import load_task_ids  # type: ignore
            except Exception as e:
                raise RuntimeError("Failed to import appworld.task.load_task_ids") from e
            task_ids = load_task_ids(dataset_name=self.config.dataset_name)
        if not task_ids:
            raise RuntimeError(f"No AppWorld task IDs found for dataset: {self.config.dataset_name}")

        # Load prompts if provided (JSON mapping or single text file as "general")
        prompts: Dict[str, str] = {}
        if self.config.prompts_path:
            try:
                import json, os
                if self.config.prompts_path.lower().endswith(".json"):
                    with open(self.config.prompts_path, "r") as f:
                        prompts = json.load(f)
                else:
                    with open(self.config.prompts_path, "r") as f:
                        text_prompt = f.read()
                    prompts = {"general": text_prompt}
                if self.logger:
                    self.logger.debug("Loaded AppWorld prompts: %d keys", len(prompts))
            except Exception:
                prompts = {}

        dataset: List[Environment] = []
        for i, tid in enumerate(task_ids[: self.config.num_episodes]):
            env_logger = None
            if self.config.verbose and self.logger is not None:
                env_logger = self.logger.getChild(f"episode_{i+1}")
            env_cfg = AppWorldEnvConfig(
                env_id=f"appworld_env_{i+1}",
                dataset_name=self.config.dataset_name,
                task_id=tid,
                max_steps=self.config.max_steps,
                experiment_name=self.config.experiment_name,
                inject_examples=True,
                prompts_path=self.config.prompts_path,
            )
            dataset.append(AppWorldEnv(env_cfg, logger=env_logger, prompts=prompts))
        return dataset
    
    def evaluate_appworld_metrics(self, environments: Optional[List[Environment]] = None) -> Optional[Dict[str, Any]]:
        """Compute TGC and SGC metrics using AppWorld's evaluation tools.
        
        This reads the saved database snapshots from the experiment output directory
        and runs AppWorld's unit tests to compute official metrics.
        
        Args:
            environments: Optional list of environments that were actually run. 
                         If None, evaluates all environments in the dataset.
        
        Returns:
            Dictionary with 'tgc', 'sgc', and full 'evaluation' results, or None if evaluation fails.
        """
        try:
            from appworld.evaluator import evaluate_tasks
        except ImportError:
            if self.logger:
                self.logger.warning("AppWorld evaluator not available")
            return None
        
        # Use provided environments or fall back to full dataset
        envs_to_evaluate = environments if environments is not None else self.dataset
        
        # Extract task_ids from the environments that were actually run
        task_ids = []
        for env in envs_to_evaluate:
            if hasattr(env, '_task_id') and env._task_id:
                task_ids.append(env._task_id)
            elif hasattr(env, 'cfg') and hasattr(env.cfg, 'task_id') and env.cfg.task_id:
                task_ids.append(env.cfg.task_id)
        
        if not task_ids:
            if self.logger:
                self.logger.warning("No AppWorld task_ids found to evaluate")
            return None
        
        try:
            if self.logger:
                self.logger.info("Running AppWorld evaluation on %d tasks", len(task_ids))
            
            # Run AppWorld's evaluation
            evaluation = evaluate_tasks(
                task_ids=task_ids,
                experiment_name=self.config.experiment_name,
                suppress_errors=True,
                include_details=True,  # Include individual task details
                save_reports=True
            )
            
            # Extract TGC and SGC
            aggregate = evaluation.get("aggregate", {})
            tgc = aggregate.get("task_goal_completion")
            sgc = aggregate.get("scenario_goal_completion")
            
            if tgc is not None and sgc is not None:
                if self.logger:
                    self.logger.info("AppWorld evaluation: TGC=%.1f%% SGC=%.1f%%", tgc, sgc)
                return {
                    "tgc": tgc,
                    "sgc": sgc,
                    "evaluation": evaluation
                }
            else:
                if self.logger:
                    self.logger.warning("AppWorld evaluation returned no TGC/SGC metrics")
                return None
            
        except Exception as e:
            if self.logger:
                self.logger.warning("Failed to compute AppWorld metrics: %s", e)
            return None


