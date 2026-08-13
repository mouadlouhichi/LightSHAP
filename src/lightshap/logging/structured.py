"""Structured run logging. print() is not the logging mechanism."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, TextIO

from lightshap.utils.io import ensure_dir
from lightshap.utils.provenance import utc_now


class JsonlHandler(logging.Handler):
    def __init__(self, path: Path) -> None:
        super().__init__()
        ensure_dir(path.parent)
        self.path = path
        self._fp: TextIO = path.open("a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        payload = {
            "timestamp": utc_now(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "event", None)
        if isinstance(extra, dict):
            payload.update(extra)
        try:
            self._fp.write(json.dumps(payload, default=str) + "\n")
            self._fp.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        try:
            self._fp.close()
        finally:
            super().close()


def setup_run_logging(run_dir: Path, run_id: str) -> logging.Logger:
    log_dir = ensure_dir(run_dir / "logs")
    logger = logging.getLogger("lightshap")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    fh = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    eh = logging.FileHandler(log_dir / "errors.log", encoding="utf-8")
    eh.setLevel(logging.WARNING)
    eh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    jh = JsonlHandler(log_dir / "events.jsonl")
    jh.setLevel(logging.DEBUG)

    logger.addHandler(fh)
    logger.addHandler(eh)
    logger.addHandler(sh)
    logger.addHandler(jh)

    metrics = logging.getLogger("lightshap.metrics")
    metrics.setLevel(logging.INFO)
    metrics.handlers.clear()
    metrics.propagate = False
    mh = JsonlHandler(log_dir / "metrics.jsonl")
    metrics.addHandler(mh)

    logger.info("logging initialised", extra={"event": {"run_id": run_id, "event": "log_init"}})
    return logger


def emit(
    logger: logging.Logger,
    event: str,
    *,
    stage: str | None = None,
    status: str | None = None,
    seed: int | None = None,
    run_id: str | None = None,
    **fields: Any,
) -> None:
    payload = {
        "event": event,
        "stage": stage,
        "status": status,
        "seed": seed,
        "run_id": run_id,
        **fields,
    }
    logger.info(event, extra={"event": payload})


def emit_metric(run_id: str, stage: str, name: str, value: Any, **fields: Any) -> None:
    logging.getLogger("lightshap.metrics").info(
        name,
        extra={
            "event": {
                "run_id": run_id,
                "stage": stage,
                "metric": name,
                "value": value,
                **fields,
            }
        },
    )
