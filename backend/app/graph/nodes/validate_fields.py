"""Node 2 — document-aware field validation.

Re-runs ``ValidationService`` against the OCR response so the orchestrator
gets a structured per-field report (which fields PASS/FAIL/MISSING/N-A),
plus a normalized ``field_match_score`` in [0, 1] for the risk node.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from app.graph.state import KYCState
from app.schemas.ocr_schema import (
    DocumentType,
    ExtractedFields,
    OCRMetadata,
    OCRResponse,
)
from app.services.validation_service import get_validation_service

logger = logging.getLogger(__name__)


def _rebuild_ocr_response(state: KYCState) -> OCRResponse:
    """The validator works off an ``OCRResponse``; we synthesize a minimal
    one from the state so this node can also be invoked in isolation
    (tests, replay) without re-running the OCR engine."""
    try:
        doc_type = DocumentType(state.get("document_type", "UNKNOWN"))
    except ValueError:
        doc_type = DocumentType.UNKNOWN
    fields = ExtractedFields(**(state.get("extracted_data") or {}))
    return OCRResponse(
        success=bool(state.get("ocr_success", False)),
        document_type=doc_type,
        raw_text=state.get("raw_ocr_text", ""),
        fields=fields,
        confidence_score=float(state.get("ocr_confidence", 0.0)),
        metadata=OCRMetadata(
            request_id=state.get("request_id", "n/a"),
            file_name=str(state.get("document_path", "")),
            file_size_bytes=0,
        ),
    )


def validate_fields(state: KYCState) -> Dict[str, Any]:
    started = time.perf_counter()
    if not state.get("ocr_success"):
        return {
            "validation_results": {"overall": "FAIL", "checks": [], "reason": "OCR failed"},
            "field_match_score": 0.0,
            "validation_overall": "FAIL",
            "timings": {"validate": time.perf_counter() - started},
        }

    truth = state.get("user_data") or {}
    response = _rebuild_ocr_response(state)
    report = get_validation_service().validate(response, truth)

    return {
        "validation_results": report.model_dump(mode="json"),
        "field_match_score": float(report.score),
        "validation_overall": report.overall.value,
        "timings": {"validate": time.perf_counter() - started},
    }
