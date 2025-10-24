import logging
from logging import Logger
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union
import contextvars
import contextlib
from threading import Lock
from datetime import datetime
import json


def setup_logger(name: str, level: str, log_path: str, extra: Optional[Dict[str, str]] = None, add_stream: bool = False) -> Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    if add_stream:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    if extra:
        return logging.LoggerAdapter(logger, extra)  # type: ignore[return-value]
    return logger


def child(logger: Logger, name: str, extra: Optional[Dict[str, str]] = None) -> Logger:
    lg = logger.getChild(name)
    if extra:
        return logging.LoggerAdapter(lg, extra)  # type: ignore[return-value]
    return lg


def add_console(logger: Logger, level: str) -> None:
    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setLevel(getattr(logging, level.upper(), logging.INFO))
    sh.setFormatter(fmt)
    logger.addHandler(sh)



# ------------------------
# Generic JSON file logger
# ------------------------

_json_base_dir_global: Optional[Path] = None
_json_ctx: contextvars.ContextVar = contextvars.ContextVar("json_log_context", default={})
_json_lock: Lock = Lock()
_json_counter: int = 0
_jsonl_locks: Dict[str, Lock] = {}
_jsonl_locks_guard: Lock = Lock()


def enable_json_logging(base_dir: str) -> None:
    global _json_base_dir_global
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    _json_base_dir_global = base


@contextlib.contextmanager
def json_log_context(**kwargs: Any):
    # Only update keys present in kwargs (and not None)
    current = dict(_json_ctx.get()) if _json_ctx.get() else {}
    update_keys = {k: v for k, v in kwargs.items() if v is not None}
    # Save previous values for only the keys being replaced
    prev_values = {k: current[k] for k in update_keys if k in current}
    merged = {**current, **update_keys}
    token = _json_ctx.set(merged)
    try:
        yield
    finally:
        # Restore only the keys that were replaced to their previous values
        ctx = dict(_json_ctx.get()) if _json_ctx.get() else {}
        for k in update_keys:
            if k in prev_values:
                ctx[k] = prev_values[k]
            else:
                ctx.pop(k, None)
        _json_ctx.set(ctx)


def json_get_context() -> Dict[str, Any]:
    return dict(_json_ctx.get()) if _json_ctx.get() else {}


def json_next_id() -> str:
    global _json_counter
    with _json_lock:
        _json_counter += 1
        idx = _json_counter
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    return f"{idx:06d}_{ts}"


def _ensure_base_dir() -> Path:
    global _json_base_dir_global
    base = _json_base_dir_global
    if base is None:
        raise RuntimeError("JSON logging base directory is not set. Call enable_json_logging().")
    return base


def json_ensure_dir(subdirs: Optional[List[str]] = None) -> Path:
    base = _ensure_base_dir()
    path = base
    if subdirs:
        for part in subdirs:
            if part:
                path = path / str(part)
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_record_begin(subdirs: Optional[List[str]], filename: Optional[str], payload: Dict[str, Any], supplied_id: Optional[str] = None) -> Tuple[str, Path]:
    call_id = supplied_id or json_next_id()
    directory = json_ensure_dir(subdirs)
    name = filename or f"{call_id}.json"
    path = directory / name
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass
    return call_id, path


def json_record_update(path: Path, updates: Dict[str, Any]) -> None:
    try:
        if path.exists():
            with open(path, "r") as f:
                rec = json.load(f)
        else:
            rec = {}
        rec.update(updates)
        with open(path, "w") as f:
            json.dump(rec, f, indent=2)
    except Exception:
        pass


def _get_file_lock(path: Union[str, Path]) -> Lock:
    key = str(path)
    with _jsonl_locks_guard:
        if key not in _jsonl_locks:
            _jsonl_locks[key] = Lock()
        return _jsonl_locks[key]


def jsonl_append(path: Union[str, Path], record: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lock = _get_file_lock(p)
    line = json.dumps(record) + "\n"
    with lock:
        with open(p, "a") as f:
            f.write(line)
            f.flush()


# ------------------------------------
# Scores logging helper utilities (API)
# ------------------------------------

def score_file_for_mode(scores_path: str, mode: str, num_seen_episodes: int) -> Path:
    """Compute the JSONL file path for a given mode.

    For training: scores.jsonl
    For validation: {num_seen_episodes}_seen_episodes_scores.jsonl
    """
    base = Path(scores_path)
    scores_dir = base if base.is_dir() or base.suffix == "" else base.parent
    scores_dir.mkdir(parents=True, exist_ok=True)
    mode_dir = scores_dir / "scores" / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    if mode == "train":
        filename = "scores.jsonl"
    else:
        filename = f"{num_seen_episodes}_seen_episodes_scores.jsonl"
    return mode_dir / filename


def build_score_record(
    *,
    mode: str,
    environment: Any,
    episode_index: int,
    step_index: int,
    score: float,
    episode_cum_score: float,
    observation: Optional[str],
    action: Optional[str],
    feedback: Optional[Dict[str, Any]],
    info: Optional[Dict[str, Any]],
    lm_model: Optional[str],
    agent_type: Optional[str],
    step_start: Optional[datetime],
    step_end: Optional[datetime],
    duration_ms: Optional[float],
    verbose_score_logging: bool,
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "mode": mode,
        "episode_index": episode_index,
        "step_index": step_index,
        "score": score,
        "episode_cum_score": episode_cum_score,
        "env_id": getattr(environment, "env_id", None),
        "env_type": getattr(environment, "env_type", None),
    }
    if verbose_score_logging:
        if observation is not None:
            rec["observation"] = observation
        if action is not None:
            rec["action"] = action
        if feedback is not None:
            rec["feedback_message"] = feedback.get("message")
            rec["target"] = feedback.get("target")
            rec["feedback_extra"] = feedback.get("extra")
            msg = feedback.get("message") or ""
            if isinstance(msg, str) and "Incorrect due to " in msg and "!" in msg:
                try:
                    rec["error_type"] = msg.split("Incorrect due to ", 1)[1].split("!", 1)[0]
                except Exception:
                    pass
        if info is not None:
            rec["env_info"] = info
        if agent_type is not None:
            rec["agent_type"] = agent_type
        if lm_model is not None:
            rec["lm_model"] = lm_model
        if step_start is not None and step_end is not None and duration_ms is not None:
            rec["timing"] = {
                "step_start": step_start.isoformat() + "Z",
                "step_end": step_end.isoformat() + "Z",
                "duration_ms": duration_ms,
            }
    return rec


def write_score_record(path: Union[str, Path], record: Dict[str, Any]) -> None:
    jsonl_append(path, record)


class ValidationLogsBuffer:
    """Buffers validation logs and flushes them sorted by (episode_index, step_index)."""

    def __init__(self) -> None:
        self._entries: List[Tuple[int, int, Path, Dict[str, Any]]] = []
        self._lock: Lock = Lock()

    def clear(self) -> None:
        with self._lock:
            self._entries = []

    def add(self, episode_index: int, step_index: int, path: Union[str, Path], record: Dict[str, Any]) -> None:
        p = Path(path)
        with self._lock:
            self._entries.append((episode_index, step_index, p, record))

    def flush(self) -> None:
        with self._lock:
            entries = list(self._entries)
            self._entries = []
        entries.sort(key=lambda t: (t[0], t[1]))
        for _, _, p, rec in entries:
            jsonl_append(p, rec)

