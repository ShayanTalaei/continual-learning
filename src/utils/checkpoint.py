from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
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


