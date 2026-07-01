"""Node 3 — fraud / AI-synthesis detection."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from app.graph.state import KYCState
from app.services.fraud_service import get_fraud_service

logger = logging.getLogger(__name__)


def detect_fraud(state: KYCState) -> Dict[str, Any]:
    started = time.perf_counter()
    path = state.get("document_path") or ""
    doc_type = state.get("document_type")
    try:
        report = get_fraud_service().assess(path, doc_type=doc_type)
    except Exception as exc:  # noqa: BLE001
        logger.exception("fraud detection failed")
        return {
            "fraud_score": 0.0,
            "fraud_reasons": [f"fraud: {exc}"],
            "fraud_verdict": "INSUFFICIENT_DATA",
            "errors": [f"fraud: {exc}"],
            "timings": {"fraud": time.perf_counter() - started},
        }

    return {
        "fraud_score": report["score"],
        "fraud_reasons": report["reasons"],
        "fraud_verdict": report["verdict"],
        "timings": {"fraud": time.perf_counter() - started},
    }
