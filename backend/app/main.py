"""FastAPI entrypoint for the KYC AI system.

Endpoints
---------
GET  /healthz                 - engine readiness + version
GET  /metrics                 - pipeline counters + p50/p95 latency (in-process)
POST /v1/ocr                  - OCR only (multipart file)
POST /v1/detect-ai            - AI-generated document forensic check
POST /v1/kyc/verify           - full pipeline (legacy alias of /verify)
POST /verify                  - full pipeline (multipart or JSON)
POST /batch-verify            - run the pipeline over up to 20 files concurrently
GET  /audit/{request_id}      - retrieve a prior decision's full state

Run locally
-----------
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
import time
import uuid
from collections import deque
from pathlib import Path
from statistics import median
from typing import Any, Deque, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.graph.workflow import graph as kyc_graph
from app.logging_config import configure_logging
from app.schemas.request_schema import AIDetectRequest, KYCVerifyRequest
from app.schemas.response_schema import (
    ErrorResponse,
    HealthResponse,
    KYCVerifyResponse,
)
from app.services.ai_generator_detector import get_ai_detector
from app.services.audit_service import build_audit_record, get_audit_service
from app.services.llm_reasoner import get_llm_reasoner
from app.services.ocr_service import get_ocr_service, startup_banner

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KYC AI System",
    description="Document OCR + validation + AI-synthesis detection + LLM second-opinion pipeline.",
    version="1.1.0",
)


# ---------------------------------------------------------------------------
# In-process metrics. For multi-pod deployments replace with prometheus_client.
# ---------------------------------------------------------------------------
class _Metrics:
    def __init__(self) -> None:
        self.requests_total = 0
        self.decisions: Dict[str, int] = {"APPROVE": 0, "REVIEW": 0, "REJECT": 0}
        self.errors_total = 0
        self.latencies_ms: Deque[float] = deque(maxlen=512)

    def record(self, decision: str, elapsed_s: float, errored: bool) -> None:
        self.requests_total += 1
        if errored:
            self.errors_total += 1
        self.decisions[decision] = self.decisions.get(decision, 0) + 1
        self.latencies_ms.append(elapsed_s * 1000.0)

    def snapshot(self) -> Dict[str, Any]:
        lats = sorted(self.latencies_ms)
        n = len(lats)
        p50 = median(lats) if lats else 0.0
        p95 = lats[int(0.95 * (n - 1))] if lats else 0.0
        return {
            "requests_total": self.requests_total,
            "errors_total": self.errors_total,
            "decisions": dict(self.decisions),
            "latency_ms": {"p50": round(p50, 1), "p95": round(p95, 1), "samples": n},
        }


_METRICS = _Metrics()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
@app.on_event("startup")
def _warmup() -> None:
    startup_banner()
    # Touch the audit service so Redis init (if any) happens at boot, not on first write.
    get_audit_service()


# ---------------------------------------------------------------------------
# Health + metrics
# ---------------------------------------------------------------------------
@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    llm = get_llm_reasoner()
    return HealthResponse(
        status="ok",
        ocr=get_ocr_service().health(),
        llm={"enabled": llm.enabled, "configured": llm._settings.is_configured},
    )


@app.get("/metrics")
def metrics() -> JSONResponse:
    return JSONResponse(_METRICS.snapshot())


# ---------------------------------------------------------------------------
# OCR (file upload)
# ---------------------------------------------------------------------------
@app.post("/v1/ocr")
async def ocr(
    file: UploadFile = File(...),
    document_type_hint: Optional[str] = Form(default=None),
    ocr_engine: str = Form(default="paddle"),
) -> JSONResponse:
    tmp = _stash_upload(file)
    try:
        from app.schemas.ocr_schema import DocumentType, OCREngine, OCRRequest
        hint = None
        if document_type_hint:
            try:
                hint = DocumentType(document_type_hint.upper())
            except ValueError:
                hint = None
        try:
            engine = OCREngine(ocr_engine.lower())
        except ValueError:
            engine = OCREngine.PADDLE
        resp = get_ocr_service().recognize(
            OCRRequest(file_path=str(tmp), document_type=hint, ocr_engine=engine, extract_qr=True),
        )
        return JSONResponse(resp.model_dump(mode="json"))
    finally:
        _safe_unlink(tmp)


# ---------------------------------------------------------------------------
# AI-synth detection
# ---------------------------------------------------------------------------
@app.post("/v1/detect-ai")
async def detect_ai(
    file: Optional[UploadFile] = File(default=None),
    document_path: Optional[str] = Form(default=None),
) -> JSONResponse:
    if not file and not document_path:
        raise HTTPException(400, "Provide either a multipart 'file' or 'document_path' form field")
    if file:
        tmp = _stash_upload(file)
        try:
            result = get_ai_detector().detect(tmp)
            return JSONResponse(result.model_dump(mode="json"))
        finally:
            _safe_unlink(tmp)
    AIDetectRequest(document_path=document_path or "")
    result = get_ai_detector().detect(document_path or "")
    return JSONResponse(result.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
async def _run_pipeline(
    *, request_id: str, doc_path: str, truth: Dict[str, Any], hint: Optional[str], ocr_engine: str = "paddle",
) -> KYCVerifyResponse:
    initial: Dict[str, Any] = {
        "request_id": request_id,
        "user_data": truth,
        "document_path": doc_path,
        "document_type_hint": hint,
        "ocr_engine": ocr_engine,
        "errors": [],
        "timings": {},
        "decision_reasons": [],
        "audit_logs": [],
        "retry_count": 0,
    }
    t0 = time.perf_counter()
    errored = False
    try:
        # LangGraph's checkpointer requires a thread_id per invocation.
        config = {"configurable": {"thread_id": request_id}}
        final = await asyncio.to_thread(kyc_graph.invoke, initial, config)
    except Exception as exc:  # noqa: BLE001
        errored = True
        logger.exception("pipeline crashed for request_id=%s", request_id)
        raise HTTPException(500, f"pipeline failure: {exc}")
    elapsed = time.perf_counter() - t0
    final["processing_time"] = elapsed
    _METRICS.record(str(final.get("final_decision", "REVIEW")), elapsed, errored)

    # Best-effort audit write; never blocks the response.
    try:
        await get_audit_service().write(build_audit_record(request_id, final))
    except Exception:  # noqa: BLE001
        logger.warning("audit write failed for %s", request_id, exc_info=True)

    return _as_response(request_id, final)


@app.post("/verify", response_model=KYCVerifyResponse, responses={400: {"model": ErrorResponse}})
@app.post("/v1/kyc/verify", response_model=KYCVerifyResponse, responses={400: {"model": ErrorResponse}})
async def kyc_verify(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    user_data: Optional[str] = Form(default=None),
    document_type_hint: Optional[str] = Form(default=None),
    ocr_engine: str = Form(default="paddle"),
    payload: Optional[KYCVerifyRequest] = None,
) -> KYCVerifyResponse:
    """Runs the LangGraph pipeline end-to-end. Accepts multipart upload
    (browser uploads) or JSON body with a server-local ``document_path``
    (trusted internal callers / tests)."""
    req_id = str(uuid.uuid4())
    tmp_to_delete: Optional[Path] = None
    try:
        if file is not None:
            tmp_to_delete = _stash_upload(file)
            doc_path = str(tmp_to_delete)
            truth = json.loads(user_data) if user_data else {}
            hint = document_type_hint
            engine = ocr_engine
        elif payload is not None:
            doc_path = payload.document_path
            truth = payload.user_data or {}
            hint = payload.document_type_hint
            engine = payload.ocr_engine.value
            req_id = payload.request_id or req_id
        else:
            content_type = (request.headers.get("content-type") or "").lower()
            if "application/json" not in content_type:
                raise HTTPException(400, "Provide either multipart file+user_data or JSON payload")
            try:
                payload = KYCVerifyRequest(**(await request.json()))
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(400, f"invalid JSON payload: {exc}") from exc
            doc_path = payload.document_path
            truth = payload.user_data or {}
            hint = payload.document_type_hint
            engine = payload.ocr_engine.value
            req_id = payload.request_id or req_id

        if not Path(doc_path).exists():
            raise HTTPException(400, f"document not found: {doc_path}")

        return await _run_pipeline(request_id=req_id, doc_path=doc_path, truth=truth, hint=hint, ocr_engine=engine)
    finally:
        if tmp_to_delete is not None:
            _safe_unlink(tmp_to_delete)


@app.post("/batch-verify")
async def batch_verify(payload: Dict[str, Any]) -> JSONResponse:
    """Run the pipeline over multiple server-local documents concurrently.

    Body shape::
        {
          "user_data": {...},
          "requests": [{"document_path": "...", "document_type_hint": "PAN"}, ...]
        }
    """
    requests = payload.get("requests") or []
    if not isinstance(requests, list) or not requests:
        raise HTTPException(400, "requests[] must be a non-empty array")
    if len(requests) > 20:
        raise HTTPException(400, "batch size capped at 20")
    truth_default = payload.get("user_data") or {}
    engine_default = str(payload.get("ocr_engine") or "paddle")

    async def _one(item: Dict[str, Any]) -> Dict[str, Any]:
        doc_path = item.get("document_path") or ""
        if not Path(doc_path).exists():
            return {"document_path": doc_path, "error": "not found"}
        rid = str(uuid.uuid4())
        try:
            resp = await _run_pipeline(
                request_id=rid,
                doc_path=doc_path,
                truth=item.get("user_data") or truth_default,
                hint=item.get("document_type_hint"),
                ocr_engine=str(item.get("ocr_engine") or engine_default),
            )
            return resp.model_dump(mode="json")
        except HTTPException as exc:
            return {"document_path": doc_path, "error": exc.detail, "request_id": rid}

    results = await asyncio.gather(*(_one(it) for it in requests))
    return JSONResponse({"count": len(results), "results": results})


@app.get("/audit/{request_id}")
async def audit_get(request_id: str) -> JSONResponse:
    rec = await get_audit_service().read(request_id)
    if rec is None:
        raise HTTPException(404, f"no audit record for {request_id}")
    return JSONResponse(json.loads(rec.to_json()))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _stash_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "upload").suffix or ".bin"
    fd = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(upload.file, fd)
    finally:
        fd.close()
    return Path(fd.name)


def _safe_unlink(p: Path) -> None:
    try:
        p.unlink(missing_ok=True)
    except OSError:
        logger.warning("could not unlink tempfile %s", p)


def _as_response(req_id: str, state: Dict[str, Any]) -> KYCVerifyResponse:
    extracted = state.get("extracted_data") or {}
    typed = state.get("typed_extracted_data") or {}
    corrected = state.get("corrected_fields") or {}
    dynamic = state.get("dynamic_extracted_data") or {}
    generic_analysis = state.get("generic_document_analysis") or {}
    # `document_data` is the authoritative "what's on the document" view.
    # Build from typed payload (strict per-doc fields) with rectifications
    # applied on top, falling back to the legacy union when no typed
    # payload was produced (e.g. UNKNOWN doc type).
    base = typed if _has_meaningful_fields(typed) else extracted if _has_meaningful_fields(extracted) else dynamic
    document_data = {**base, **{k: v for k, v in corrected.items() if k in base or typed}}
    return KYCVerifyResponse(
        request_id=req_id,
        document_type=str(state.get("document_type", "UNKNOWN")),
        document_type_label=state.get("document_type_label"),
        ocr_engine=str(state.get("ocr_engine", "paddle")),
        final_decision=str(state.get("final_decision", "REVIEW")),
        risk_band=str(state.get("risk_band", "HIGH")),
        risk_score=float(state.get("risk_score", 1.0)),
        validation_overall=str(state.get("validation_overall", "MISSING")),
        field_match_score=float(state.get("field_match_score", 0.0)),
        fraud_verdict=str(state.get("fraud_verdict", "INSUFFICIENT_DATA")),
        fraud_score=float(state.get("fraud_score", 0.0)),
        ocr_confidence=float(state.get("ocr_confidence", 0.0)),
        extracted_data=extracted,
        typed_extracted_data=typed,
        corrected_fields=corrected,
        dynamic_extracted_data=dynamic,
        generic_document_analysis=generic_analysis,
        document_data=document_data,
        classification_confidence=float(state.get("classification_confidence", 0.0)),
        validation_results=state.get("validation_results") or {},
        fraud_reasons=state.get("fraud_reasons") or [],
        metadata_results=state.get("metadata_results") or {},
        preprocess_report=state.get("preprocess_report") or {},
        decision_reasons=state.get("decision_reasons") or [],
        errors=state.get("errors") or [],
        timings=state.get("timings") or {},
        processing_time=float(state.get("processing_time", 0.0)),
        audit_logs=state.get("audit_logs") or [],
        hallucination_score=float(state.get("hallucination_score", 0.0)),
        document_status=str(state.get("document_status", "NEEDS_REVIEW")),
        document_feedback=state.get("document_feedback") or [],
        llm_enabled=bool(state.get("llm_enabled", False)),
        llm_recommendation=state.get("llm_recommendation"),
        llm_confidence=state.get("llm_confidence"),
        llm_reasoning=state.get("llm_reasoning"),
        llm_concerns=state.get("llm_concerns") or [],
        llm_rectifications=state.get("llm_rectifications") or [],
    )


def _has_meaningful_fields(payload: Dict[str, Any]) -> bool:
    ignored = {"schema_type", "qr_codes", "tables"}
    for key, value in (payload or {}).items():
        if key in ignored:
            continue
        if value not in (None, "", [], {}):
            return True
    return False
