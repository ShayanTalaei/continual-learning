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
    current = dict(_json_ctx.get()) if _json_ctx.get() else {}
    merged = {**current, **{k: v for k, v in kwargs.items() if v is not None}}
    token = _json_ctx.set(merged)
    try:
        yield
    finally:
        _json_ctx.reset(token)


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

