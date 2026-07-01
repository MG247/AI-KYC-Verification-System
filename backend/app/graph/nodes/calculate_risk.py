"""Node - risk scoring.

Combines field-match, fraud, OCR-confidence, hallucination, and metadata
signals into a single 0..1 risk number plus a LOW / MEDIUM / HIGH /
CRITICAL band. Pure delegation to ``RiskService`` so the weighting
policy lives in one place and can be re-tuned without touching the
graph.

Runs AFTER ``rectify_fields`` so any LLM-applied field corrections are
already baked into ``field_match_score``.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from app.graph.state import KYCState
from app.services.risk_service import get_risk_service


def calculate_risk(state: KYCState) -> Dict[str, Any]:
    started = time.perf_counter()
    md = state.get("metadata_results") or {}
    metadata_suspicion = float(md.get("suspicion", 0.0)) if md.get("available") else 0.0

    score, band, reasons = get_risk_service().score(
        field_match_score=float(state.get("field_match_score", 0.0)),
        fraud_score=float(state.get("fraud_score", 0.0)),
        ocr_confidence=float(state.get("ocr_confidence", 0.0)),
        validation_overall=str(state.get("validation_overall", "MISSING")),
        fraud_verdict=str(state.get("fraud_verdict", "INSUFFICIENT_DATA")),
        hallucination_score=float(state.get("hallucination_score", 0.0)),
        metadata_suspicion=metadata_suspicion,
    )
    return {
        "risk_score": score,
        "risk_band": band,
        "decision_reasons": reasons,
        "audit_logs": [{
            "node": "calculate_risk",
            "score": score, "band": band,
            "metadata_suspicion": metadata_suspicion,
        }],
        "timings": {"risk": time.perf_counter() - started},
    }
