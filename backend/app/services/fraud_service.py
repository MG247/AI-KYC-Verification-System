"""Fraud-signal aggregation service.

Today this service is a thin facade over the forensic AI-generated
document detector. The signature is intentionally extensible — future
detectors (face tamper, template mismatch, copy-move) plug in here and
the orchestrator stays unchanged.

The caller-facing contract is a flat dict — easy to drop into
LangGraph state without leaking Pydantic types into the graph layer.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from app.services.ai_generator_detector import (
    AIDetectionResult,
    AIVerdict,
    get_ai_detector,
)

logger = logging.getLogger(__name__)


class FraudService:
    def assess(self, file_path: str, doc_type: Optional[str] = None) -> Dict[str, Any]:
        ai: AIDetectionResult = get_ai_detector().detect(file_path, doc_type=doc_type)
        reasons = self._summarize(ai)
        return {
            "score": float(ai.confidence),
            "verdict": ai.verdict.value,
            "reasons": reasons,
            "signals": [s.model_dump() for s in ai.signals],
        }

    @staticmethod
    def _summarize(ai: AIDetectionResult) -> List[str]:
        if ai.verdict == AIVerdict.LIKELY_AI:
            head = "document looks AI-generated"
        elif ai.verdict == AIVerdict.SUSPICIOUS:
            head = "document has suspicious forensic signals"
        elif ai.verdict == AIVerdict.INSUFFICIENT_DATA:
            return [f"fraud check skipped: {ai.explanation}"]
        else:
            return []
        triggered = [s for s in ai.signals if s.triggered]
        if not triggered:
            return [head]
        detail = ", ".join(f"{s.code.value}({s.score:.2f})" for s in triggered)
        return [f"{head}: {detail}"]


_lock = threading.Lock()
_svc: Optional[FraudService] = None


def get_fraud_service() -> FraudService:
    global _svc
    if _svc is None:
        with _lock:
            if _svc is None:
                _svc = FraudService()
    return _svc
