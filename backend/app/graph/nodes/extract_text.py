"""Node 1 — OCR extraction.

Runs the OCR service over ``state['document_path']`` and writes
``raw_ocr_text``, ``extracted_data``, ``document_type``, and
``ocr_confidence`` back to the state. Failures land in ``errors``;
downstream nodes guard on ``ocr_success``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from app.graph.state import KYCState
from app.schemas.ocr_schema import DocumentType, OCREngine, OCRRequest
from app.services.ocr_service import get_ocr_service

logger = logging.getLogger(__name__)


def _coerce_doc_hint(hint: Any) -> DocumentType | None:
    if not hint:
        return None


def _coerce_ocr_engine(engine: Any) -> OCREngine:
    if not engine:
        return OCREngine.PADDLE
    try:
        return OCREngine(str(engine).lower())
    except ValueError:
        return OCREngine.PADDLE
    try:
        return DocumentType(str(hint).upper())
    except ValueError:
        return None


def extract_text(state: KYCState) -> Dict[str, Any]:
    started = time.perf_counter()
    svc = get_ocr_service()
    path = state.get("document_path") or ""
    hint = _coerce_doc_hint(state.get("document_type_hint"))
    ocr_engine = _coerce_ocr_engine(state.get("ocr_engine"))
    request = OCRRequest(file_path=path, document_type=hint, ocr_engine=ocr_engine, extract_qr=True)

    try:
        response = svc.recognize(request)
    except Exception as exc:  # noqa: BLE001
        logger.exception("OCR node crashed")
        return {
            "ocr_success": False,
            "raw_ocr_text": "",
            "extracted_data": {},
            "ocr_confidence": 0.0,
            "document_type": "UNKNOWN",
            "document_type_label": "Unknown Document",
            "ocr_engine": ocr_engine.value,
            "dynamic_extracted_data": {},
            "generic_document_analysis": {},
            "errors": [f"ocr: {exc}"],
            "timings": {"ocr": time.perf_counter() - started},
        }

    if not response.success:
        err = response.error.message if response.error else "OCR failed"
        return {
            "ocr_success": False,
            "raw_ocr_text": response.raw_text or "",
            "extracted_data": {},
            "ocr_confidence": 0.0,
            "document_type": response.document_type.value,
            "document_type_label": response.document_type_label or response.document_type.value,
            "ocr_engine": response.ocr_engine.value,
            "dynamic_extracted_data": response.dynamic_fields or {},
            "generic_document_analysis": response.generic_analysis.model_dump(mode="json") if response.generic_analysis else {},
            "errors": [f"ocr: {err}"],
            "timings": {"ocr": time.perf_counter() - started},
        }

    fields = response.fields.model_dump(exclude_none=True)
    fields.pop("tables", None)
    # Authoritative per-doc-type payload — only the fields valid for the
    # classified document type. The graph state keeps the legacy `fields`
    # for back-compat; downstreams should prefer `typed_extracted_data`.
    typed = response.typed_fields.model_dump(exclude_none=True) if response.typed_fields else {}
    typed.pop("tables", None)
    return {
        "ocr_success": True,
        "raw_ocr_text": response.raw_text,
        "extracted_data": fields,
        "typed_extracted_data": typed,
        "dynamic_extracted_data": response.dynamic_fields or {},
        "generic_document_analysis": response.generic_analysis.model_dump(mode="json") if response.generic_analysis else {},
        "ocr_confidence": float(response.confidence_score),
        "ocr_engine": response.ocr_engine.value,
        "document_type": response.document_type.value,
        "document_type_label": response.document_type_label or response.document_type.value,
        "classification_confidence": float(response.classification_confidence),
        "timings": {"ocr": time.perf_counter() - started},
    }
