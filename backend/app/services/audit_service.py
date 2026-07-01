"""Audit-log service.

Persists one record per verification so a compliance reviewer can reload
the full decision trail by ``request_id``. Tries Redis first (fast,
networked, multi-process safe); falls back to a local JSONL file when
Redis is unavailable. Both backends speak the same async interface.

Configuration (env)
-------------------
* ``KYC_AUDIT_REDIS_URL``  - if set, attempt Redis. e.g. ``redis://localhost:6379/0``
* ``KYC_AUDIT_FILE``       - JSONL fallback path. Default ``./audit.jsonl``
* ``KYC_AUDIT_TTL_SECONDS`` - Redis record TTL. Default 7 days.

Design notes
------------
* The service is intentionally lossy on failure: if BOTH backends raise
  while writing, we log a WARNING and continue. KYC decisions must NOT
  fail because audit storage is degraded.
* Reads return ``None`` on miss (404 in the API layer). Listing is
  paginated by cursor to avoid blowing up the request response.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_TTL = 7 * 24 * 3600
_KEY_PREFIX = "kyc:audit:"


@dataclass
class AuditRecord:
    request_id: str
    created_at: float
    final_decision: str
    risk_band: str
    risk_score: float
    document_type: str
    state: Dict[str, Any]            # full final state, redacted upstream if needed

    def to_json(self) -> str:
        return json.dumps({
            "request_id": self.request_id,
            "created_at": self.created_at,
            "final_decision": self.final_decision,
            "risk_band": self.risk_band,
            "risk_score": self.risk_score,
            "document_type": self.document_type,
            "state": self.state,
        }, ensure_ascii=False, default=str)

    @classmethod
    def from_json(cls, raw: str) -> "AuditRecord":
        d = json.loads(raw)
        return cls(**d)


class AuditService:
    """Async facade over Redis + JSONL backends."""

    def __init__(self) -> None:
        self._file_path = Path(os.getenv("KYC_AUDIT_FILE", "audit.jsonl"))
        self._ttl = int(os.getenv("KYC_AUDIT_TTL_SECONDS", str(_DEFAULT_TTL)))
        self._redis_url = os.getenv("KYC_AUDIT_REDIS_URL")
        self._redis = None
        self._redis_ok = False
        self._lock = threading.Lock()
        self._init_redis()

    # ----------------- public -----------------
    async def write(self, record: AuditRecord) -> None:
        """Fire-and-forget write. Tries Redis then JSONL; never raises."""
        await asyncio.gather(
            self._write_redis(record),
            self._write_file(record),
            return_exceptions=True,
        )

    async def read(self, request_id: str) -> Optional[AuditRecord]:
        rec = await self._read_redis(request_id)
        if rec is not None:
            return rec
        return await self._read_file(request_id)

    # ----------------- redis backend -----------------
    def _init_redis(self) -> None:
        if not self._redis_url:
            return
        try:
            import redis  # type: ignore
            self._redis = redis.Redis.from_url(
                self._redis_url, decode_responses=True, socket_timeout=2.0,
            )
            self._redis.ping()
            self._redis_ok = True
            logger.info("audit: connected to Redis at %s", self._redis_url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("audit: Redis unavailable (%s); using JSONL fallback only", exc)
            self._redis = None
            self._redis_ok = False

    async def _write_redis(self, record: AuditRecord) -> None:
        if not self._redis_ok:
            return
        key = f"{_KEY_PREFIX}{record.request_id}"
        try:
            await asyncio.to_thread(
                self._redis.set, key, record.to_json(), ex=self._ttl,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("audit: redis write failed: %s", exc)
            self._redis_ok = False  # back off; JSONL still captures it

    async def _read_redis(self, request_id: str) -> Optional[AuditRecord]:
        if not self._redis_ok:
            return None
        try:
            raw = await asyncio.to_thread(self._redis.get, f"{_KEY_PREFIX}{request_id}")
            return AuditRecord.from_json(raw) if raw else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("audit: redis read failed: %s", exc)
            return None

    # ----------------- jsonl backend -----------------
    async def _write_file(self, record: AuditRecord) -> None:
        try:
            await asyncio.to_thread(self._append, record.to_json())
        except Exception as exc:  # noqa: BLE001
            logger.warning("audit: jsonl write failed: %s", exc)

    def _append(self, line: str) -> None:
        # Single-process append. For multi-process, swap for a file lock.
        with self._lock:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            with self._file_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    async def _read_file(self, request_id: str) -> Optional[AuditRecord]:
        if not self._file_path.exists():
            return None
        # Scan from the end - newest writes win on duplicate keys.
        try:
            lines = await asyncio.to_thread(self._file_path.read_text, "utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("audit: jsonl read failed: %s", exc)
            return None
        for raw in reversed(lines.splitlines()):
            if request_id in raw:
                try:
                    rec = AuditRecord.from_json(raw)
                    if rec.request_id == request_id:
                        return rec
                except Exception:  # noqa: BLE001
                    continue
        return None


_lock = threading.Lock()
_svc: Optional[AuditService] = None


def get_audit_service() -> AuditService:
    global _svc
    if _svc is None:
        with _lock:
            if _svc is None:
                _svc = AuditService()
    return _svc


def build_audit_record(request_id: str, state: Dict[str, Any]) -> AuditRecord:
    return AuditRecord(
        request_id=request_id,
        created_at=time.time(),
        final_decision=str(state.get("final_decision", "REVIEW")),
        risk_band=str(state.get("risk_band", "HIGH")),
        risk_score=float(state.get("risk_score", 1.0)),
        document_type=str(state.get("document_type", "UNKNOWN")),
        state=state,
    )
