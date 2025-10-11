from typing import Optional, Tuple, Dict, Any, List
from logging import getLogger, Logger

from pydantic import BaseModel

from src.data.env import Environment, EnvDataset, EnvDatasetConfig


def _process_observation(ob: str) -> str:
    # Normalize initial location banner as upstream does
    if ob.startswith('You arrive at loc '):
        try:
            return ob[ob.find('. ') + 2:]
        except Exception:
            return ob
    return ob


_PREFIXES = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo',
}


class ALFWorldEnvConfig(BaseModel):
    env_id: str = "alfworld_env"
    base_config_path: str
    split: str = "eval_out_of_distribution"
    max_steps: Optional[int] = 50
    inject_examples: bool = True
    prompts_path: Optional[str] = None  # Path to alfworld_3prompts.json


class ALFWorldEnv(Environment):
    def __init__(self, cfg: ALFWorldEnvConfig, env, prompts: Optional[Dict[str, str]] = None, logger: Optional[Logger] = None):
        super().__init__(env_id=cfg.env_id, env_type="alfworld")
        self.cfg = cfg
        self._env = env  # Underlying ALFWorld vectorized env (batch_size=1)
        self._prompts = prompts or {}
        self._step_count: int = 0
        self._logger = logger or getLogger("alfworld_env")

    # Optional JSON schema describing an action format to help LM frontends
    def response_schema(self) -> Dict[str, Any]:
        return {
            "type": "string",
            "description": "Text command like 'go to X', 'open Y', 'take Z from W', 'use A B', or 'think: ...'",
        }

    def _build_initial_prompt(self, ob: str, info: Dict[str, Any]) -> str:
        if not self.cfg.inject_examples or not self._prompts:
            return ob
        try:
            name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        except Exception:
            return ob
        prefix_key = None
        for k, v in _PREFIXES.items():
            if name.startswith(k):
                prefix_key = v
                break
        if prefix_key is None:
            return ob
        # Choose two examples react_{prefix}_1 then react_{prefix}_0 if available; fallback to act_*
        keys: List[str] = [
            f"react_{prefix_key}_1",
            f"react_{prefix_key}_0",
            f"act_{prefix_key}_1",
            f"act_{prefix_key}_0",
        ]
        examples: List[str] = []
        for k in keys:
            v = self._prompts.get(k)
            if v:
                examples.append(v)
        if not examples:
            return ob
        header = "Interact with a household to solve a task. Here are two examples.\n"
        return header + "".join(examples[:2]) + ob

    def reset(self) -> str:
        self._step_count = 0
        ob_list, info = self._env.reset()
        # env returns lists when batch_size=1
        raw = ob_list[0] if isinstance(ob_list, (list, tuple)) else ob_list
        # Drop the initial map header
        try:
            raw = "\n".join(raw.split("\n\n")[1:])
        except Exception:
            pass
        ob = self._build_initial_prompt(raw, info)
        return ob

    def step(self, action: str) -> Tuple[Optional[str], Dict[str, Any], bool, Dict[str, Any]]:
        act = (action or "").strip()
        # Handle deliberate thinking without advancing the environment
        if act.lower().startswith("think:"):
            self._step_count += 1
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

        ob_list, reward, done_flags, info = self._env.step([act])
        # Normalize shapes
        ob0 = ob_list[0] if isinstance(ob_list, (list, tuple)) else ob_list
        ob0 = _process_observation(ob0)
        won = False
        try:
            won = bool(info.get('won', [False])[0])
        except Exception:
            won = False
        done = bool(done_flags[0]) if isinstance(done_flags, (list, tuple)) else bool(done_flags)
        self._step_count += 1

        # Local max-steps guard (runtime also caps steps)
        if self.cfg.max_steps is not None and self._step_count >= self.cfg.max_steps:
            done = True

        feedback = {
            "score": 1.0 if won else 0.0,
            "message": ob0,  # Keep message textual for current memory serializer
            "won": won,
            "done": done,
        }
        return ob0, feedback, done, info


class ALFWorldEnvDatasetConfig(EnvDatasetConfig):
    base_config_path: str
    split: str = "eval_out_of_distribution"
    num_episodes: int = 1
    prompts_path: Optional[str] = None
    verbose: bool = True


class ALFWorldEnvDataset(EnvDataset):
    def __init__(self, config: ALFWorldEnvDatasetConfig, logger: Optional[Logger] = None):
        self._env_mgr = None
        self._env = None
        super().__init__(config, logger)

    @property
    def config(self) -> ALFWorldEnvDatasetConfig:  # type: ignore[override]
        return super().config  # type: ignore[return-value]

    def load_dataset(self) -> List[Environment]:
        log = self.logger
        try:
            import yaml  # lazy import
            import importlib
            import alfworld  # type: ignore
            import alfworld.agents.environment as alfw_env  # type: ignore
        except Exception as e:
            raise RuntimeError("alfworld package is required for ALFWorldEnvDataset") from e

        with open(self.config.base_config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Initialize manager and env (batch_size=1)
        env_type = cfg.get("env", {}).get("type")
        if not env_type:
            raise ValueError("Missing env.type in ALFWorld base config")
        # Reload modules to avoid stale global state across runs (mirrors upstream script)
        importlib.reload(alfworld)
        importlib.reload(alfw_env)
        self._env_mgr = getattr(alfw_env, env_type)(cfg, train_eval=self.config.split)
        self._env = self._env_mgr.init_env(batch_size=1)

        # Load prompts JSON if provided
        prompts: Dict[str, str] = {}
        if self.config.prompts_path:
            try:
                import json
                with open(self.config.prompts_path, "r") as f:
                    prompts = json.load(f)
                log.debug("Loaded ALFWorld prompts: %d keys", len(prompts))
            except Exception:
                prompts = {}

        # Build episode wrappers
        dataset: List[Environment] = []
        for i in range(self.config.num_episodes):
            env_logger = None
            if self.config.verbose:
                env_logger = log.getChild(f"episode_{i+1}")
            env_cfg = ALFWorldEnvConfig(
                env_id=f"alfworld_env_{i+1}",
                base_config_path=self.config.base_config_path,
                split=self.config.split,
                prompts_path=self.config.prompts_path,
            )
            dataset.append(ALFWorldEnv(env_cfg, self._env, prompts=prompts, logger=env_logger))
        return dataset


