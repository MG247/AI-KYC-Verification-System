"""Centralized logging config for the KYC service.

Why this file
-------------
* One place to control format / level / handler wiring so production and
  local runs don't drift.
* Structured (JSON-line) logs in production - parseable by Loki / ELK /
  Datadog without grok. Pretty console logs locally.
* ``configure_logging`` is idempotent and safe to call from FastAPI
  startup, the e2e test script, and ad-hoc tools.

Env knobs
---------
* ``KYC_LOG_LEVEL``  - DEBUG | INFO (default) | WARNING | ERROR
* ``KYC_LOG_FORMAT`` - ``console`` (default) | ``json``
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    """One JSON object per record. Stable key order for grep-ability."""

    _STD_ATTRS = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message",
    }

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Promote any extra=... attributes set by the caller.
        for k, v in record.__dict__.items():
            if k in self._STD_ATTRS or k.startswith("_"):
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except TypeError:
                payload[k] = str(v)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    level = os.getenv("KYC_LOG_LEVEL", "INFO").upper()
    fmt = os.getenv("KYC_LOG_FORMAT", "console").lower()

    root = logging.getLogger()
    # Idempotent: only configure once.
    if getattr(root, "_kyc_configured", False):
        root.setLevel(level)
        return

    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        ))
    root.addHandler(handler)
    root.setLevel(level)

    # Tame chatty third-party loggers.
    for noisy in ("httpx", "httpcore", "urllib3", "PIL", "paddle"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root._kyc_configured = True  # type: ignore[attr-defined]
