import logging
from logging import Logger
from pathlib import Path
from typing import Optional, Dict


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


