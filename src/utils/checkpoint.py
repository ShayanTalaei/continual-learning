from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from logging import Logger, getLogger
import json
import shutil


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def create_checkpoint_dir(base_dir: str | Path, episode_index: int) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    ep_dir = base / f"ep_{episode_index:06d}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    return ep_dir


def write_runtime_manifest(ep_dir: str | Path, manifest: Dict[str, Any]) -> Path:
    p = Path(ep_dir) / "runtime.json"
    payload = dict(manifest)
    payload.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
    _atomic_write_json(p, payload)
    return p


def read_runtime_manifest(cp_dir: str | Path) -> Dict[str, Any]:
    p = Path(cp_dir) / "runtime.json"
    if not p.exists():
        return {}
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def write_agent_manifest(ep_dir: str | Path, manifest: Dict[str, Any]) -> Path:
    p = Path(ep_dir) / "agent.json"
    _atomic_write_json(p, manifest)
    return p


def read_agent_manifest(cp_dir: str | Path) -> Dict[str, Any]:
    p = Path(cp_dir) / "agent.json"
    if not p.exists():
        return {}
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def update_latest_pointer(base_dir: str | Path, target_dir: str | Path) -> Tuple[Optional[Path], Path]:
    base = Path(base_dir)
    target = Path(target_dir)
    # Prefer symlink if available
    latest_link = base / "latest"
    latest_json = base / "latest.json"
    try:
        if latest_link.exists() or latest_link.is_symlink():
            try:
                latest_link.unlink()
            except Exception:
                pass
        latest_link.symlink_to(target.name)
        # Also write json pointer for portability
        _atomic_write_json(latest_json, {"path": str(target.name)})
        return latest_link, latest_json
    except Exception:
        # Fallback to JSON pointer only
        _atomic_write_json(latest_json, {"path": str(target.name)})
        return None, latest_json


def resolve_checkpoint_path(path: str | Path) -> Path:
    p = Path(path)
    # If points directly to ep_* dir, return
    if p.name.startswith("ep_") and p.is_dir():
        return p
    # If 'latest' symlink/dir exists
    if p.name == "latest" and p.exists():
        if p.is_symlink():
            try:
                resolved = p.resolve()
                return resolved
            except Exception:
                pass
        # If latest is a directory
        if p.is_dir():
            return p
        # If latest.json present
        j = p.parent / "latest.json"
        if j.exists():
            try:
                with open(j, "r") as f:
                    rec = json.load(f)
                name = rec.get("path")
                if name:
                    cand = p.parent / name
                    if cand.exists():
                        return cand
            except Exception:
                pass
    # If directory contains latest pointer
    latest = p / "latest"
    if latest.exists():
        return resolve_checkpoint_path(latest)
    latest_json = p / "latest.json"
    if latest_json.exists():
        try:
            with open(latest_json, "r") as f:
                rec = json.load(f)
            name = rec.get("path")
            if name:
                cand = p / name
                if cand.exists():
                    return cand
        except Exception:
            pass
    # Otherwise return path as-is
    return p


def list_checkpoints(base_dir: str | Path) -> List[Path]:
    base = Path(base_dir)
    if not base.exists():
        return []
    eps = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("ep_")]
    eps.sort()
    return eps


def prune_checkpoints(base_dir: str | Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    eps = list_checkpoints(base_dir)
    if len(eps) <= keep_last:
        return
    to_delete = eps[: len(eps) - keep_last]
    for d in to_delete:
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass



# -----------------------------
# High-level checkpoint manager
# -----------------------------
class CheckpointManager:
    """Encapsulates checkpoint saving and pruning strategies.

    Strategies:
      - "last_n": keep the last N checkpoints, saving every `every_episodes`.
      - "top_k_val": keep top-K checkpoints by validation score; save on validation.
    """

    def __init__(
        self,
        *,
        base_dir: str | Path,
        strategy: str = "last_n",
        keep_count: int = 2,
        every_episodes: Optional[int] = None,
        logger: Optional[Logger] = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.strategy = strategy
        self.keep_count = max(int(keep_count or 0), 0)
        self.every_episodes = every_episodes
        self.logger = logger or getLogger("checkpoint_manager")

        self._episode_to_val_score: Dict[int, float] = {}
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing_scores()

    def _load_existing_scores(self) -> None:
        for ep_dir in list_checkpoints(self.base_dir):
            manifest = read_runtime_manifest(ep_dir)
            ep_idx = manifest.get("episode_index")
            val = manifest.get("val_score")
            if isinstance(ep_idx, int) and isinstance(val, (int, float)):
                self._episode_to_val_score[ep_idx] = float(val)

    def maybe_checkpoint_on_start(self, agent: Any, *, episode_index: int) -> None:
        try:
            ep_dir = create_checkpoint_dir(self.base_dir, episode_index)
            manifest_agent = agent.save_checkpoint(str(ep_dir), episode_index)
            manifest_runtime = {
                "episode_index": episode_index,
                "train_steps_total": 0,
                "agent_type": agent.__class__.__name__,
            }
            if isinstance(manifest_agent, dict) and manifest_agent.get("memory_snapshot_path"):
                manifest_runtime["memory_snapshot_path"] = manifest_agent["memory_snapshot_path"]
            write_runtime_manifest(ep_dir, manifest_runtime)
            update_latest_pointer(self.base_dir, ep_dir)
            if self.strategy == "last_n" and self.keep_count > 0:
                prune_checkpoints(self.base_dir, self.keep_count)
            self.logger.info("Saved checkpoint at start: %s", str(ep_dir))
        except Exception as e:
            self.logger.warning("Failed to save start checkpoint: %s", str(e))

    def on_episode_end(self, agent: Any, *, episode_index: int, train_steps_total: int) -> None:
        if self.strategy != "last_n":
            return
        if not self.every_episodes or self.every_episodes <= 0:
            return
        if episode_index % self.every_episodes != 0:
            return
        self._save(agent=agent, episode_index=episode_index, train_steps_total=train_steps_total, val_score=None)
        if self.keep_count > 0:
            prune_checkpoints(self.base_dir, self.keep_count)

    def on_validation_complete(self, agent: Any, *, episode_index: int, train_steps_total: int, mean_val_score: float) -> None:
        if self.strategy != "top_k_val":
            return
        # Always save on validation; prune to keep top-K by score
        self._save(agent=agent, episode_index=episode_index, train_steps_total=train_steps_total, val_score=mean_val_score)
        if self.keep_count > 0:
            self._prune_top_k_by_score(top_k=self.keep_count)

    def _save(self, *, agent: Any, episode_index: int, train_steps_total: int, val_score: Optional[float]) -> None:
        try:
            ep_dir = create_checkpoint_dir(self.base_dir, episode_index)
            manifest_agent = agent.save_checkpoint(str(ep_dir), episode_index)
            manifest_runtime: Dict[str, Any] = {
                "episode_index": episode_index,
                "train_steps_total": train_steps_total,
                "agent_type": agent.__class__.__name__,
            }
            if val_score is not None:
                manifest_runtime["val_score"] = float(val_score)
                self._episode_to_val_score[episode_index] = float(val_score)
            if isinstance(manifest_agent, dict) and manifest_agent.get("memory_snapshot_path"):
                manifest_runtime["memory_snapshot_path"] = manifest_agent["memory_snapshot_path"]
            write_runtime_manifest(ep_dir, manifest_runtime)
            update_latest_pointer(self.base_dir, ep_dir)
            self.logger.info("Saved checkpoint: %s", str(ep_dir))
        except Exception as e:
            self.logger.warning("Failed to save checkpoint: %s", str(e))

    def _prune_top_k_by_score(self, *, top_k: int) -> None:
        if top_k <= 0:
            return
        eps = list_checkpoints(self.base_dir)
        if len(eps) <= top_k:
            return
        scored: List[Tuple[Path, float, int]] = []
        for ep in eps:
            man = read_runtime_manifest(ep)
            ep_idx = man.get("episode_index")
            score = man.get("val_score")
            if not isinstance(score, (int, float)):
                score = self._episode_to_val_score.get(int(ep_idx) if isinstance(ep_idx, int) else -1, float("-inf"))
            if isinstance(ep_idx, int):
                scored.append((ep, float(score), ep_idx))
        scored.sort(key=lambda t: (t[1], t[2]), reverse=True)
        to_delete = [p for (p, _, _) in scored[top_k:]]
        for d in to_delete:
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass

