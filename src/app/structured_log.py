"""Stable JSON-lines logging contract for long-running robot processes."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TextIO

LOG_SCHEMA = "go2agent.log.v1"


def emit_log(
    *,
    owner: str,
    event_type: str,
    data: Mapping[str, object] | None = None,
    level: str = "info",
    stream: TextIO | None = None,
) -> None:
    if not owner.strip():
        raise ValueError("log owner must not be empty")
    if not event_type.strip():
        raise ValueError("log event type must not be empty")
    payload = {
        "schema": LOG_SCHEMA,
        "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        ),
        "level": level,
        "type": event_type,
        "owner": owner,
        "data": dict(data or {}),
    }
    print(
        json.dumps(payload, ensure_ascii=False, default=str),
        file=stream or sys.stdout,
        flush=True,
    )


class _StructuredLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            data: dict[str, object] = {"message": record.getMessage()}
            if record.exc_info is not None:
                data["exception"] = self.formatException(record.exc_info)
            emit_log(
                owner=record.name or "python.root",
                event_type="python_log",
                level=record.levelname.lower(),
                data=data,
                stream=sys.stderr,
            )
        except Exception:  # noqa: BLE001 - logging must never crash the process
            self.handleError(record)

    @staticmethod
    def formatException(exception_info: object) -> str:
        formatter = logging.Formatter()
        return formatter.formatException(exception_info)  # type: ignore[arg-type]


def configure_structured_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(_StructuredLogHandler())
    root.setLevel(level)
    # Preserve transport warnings/errors without emitting every successful POST.
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(max(level, logging.WARNING))
