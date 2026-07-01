"""Pydantic schemas for OCR requests, responses, and intermediate models.

These schemas form the contract between the OCR engine and any upstream
service (FastAPI endpoint, Celery worker, LangGraph node). They are designed
to be JSON-serializable end-to-end.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class DocumentType(str, Enum):
    PAN = "PAN"
    AADHAAR = "AADHAAR"
    PASSPORT = "PASSPORT"
    DRIVING_LICENSE = "DRIVING_LICENSE"
    GST_CERTIFICATE = "GST_CERTIFICATE"
    BANK_STATEMENT = "BANK_STATEMENT"
    SALARY_SLIP = "SALARY_SLIP"
    UNKNOWN = "UNKNOWN"


class OCREngine(str, Enum):
    PADDLE = "paddle"
    GPT_VISION = "gpt_vision"


class QualityVerdict(str, Enum):
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    POOR = "POOR"
    REJECTED = "REJECTED"


class OCRErrorCode(str, Enum):
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    CORRUPT_FILE = "CORRUPT_FILE"
    BLANK_IMAGE = "BLANK_IMAGE"
    LOW_QUALITY = "LOW_QUALITY"
    OCR_FAILED = "OCR_FAILED"
    PDF_RENDER_FAILED = "PDF_RENDER_FAILED"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------
BBox = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]


class OCRWord(BaseModel):
    """A single recognized text token with its polygon and confidence."""

    model_config = ConfigDict(frozen=True)

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BBox
    page: int = 0


class OCRLine(BaseModel):
    """A line aggregates spatially close words on the same page."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BBox
    page: int = 0
    words: List[OCRWord] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Quality & confidence
# ---------------------------------------------------------------------------
class QualityReport(BaseModel):
    blur_score: float
    is_blurry: bool
    brightness: float
    contrast: float
    width: int
    height: int
    quality_score: float = Field(ge=0.0, le=1.0)
    verdict: QualityVerdict


class ConfidenceReport(BaseModel):
    average: float = Field(ge=0.0, le=1.0)
    median: float = Field(ge=0.0, le=1.0)
    min: float = Field(ge=0.0, le=1.0)
    low_confidence_count: int = 0
    suspicious_count: int = 0
    low_confidence_lines: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Field extraction — per-doc-type strict schemas
# ---------------------------------------------------------------------------
# Each doc type emits ONLY the fields that exist on that document. The shared
# `BaseFields` carries a `schema_type` discriminator so the API response can
# union the variants without ambiguity. Side-channel containers `qr_codes` and
# `tables` are kept on the base because they apply regardless of doc type.


class BaseFields(BaseModel):
    """Common to every extracted-fields payload."""

    schema_type: str = "UNKNOWN"
    qr_codes: List[str] = Field(default_factory=list)
    tables: List[List[List[str]]] = Field(default_factory=list)


class PanFields(BaseFields):
    schema_type: str = "PAN"
    pan_number: Optional[str] = None
    name: Optional[str] = None
    father_name: Optional[str] = None
    dob: Optional[str] = None  # ISO YYYY-MM-DD


class AadhaarFields(BaseFields):
    schema_type: str = "AADHAAR"
    # Full 12-digit number is never returned — UIDAI masking policy.
    masked_aadhaar: Optional[str] = None      # XXXX XXXX 1234
    aadhaar_last4: Optional[str] = None
    name: Optional[str] = None
    dob: Optional[str] = None
    gender: Optional[str] = None


class PassportFields(BaseFields):
    schema_type: str = "PASSPORT"
    passport_number: Optional[str] = None
    surname: Optional[str] = None
    given_name: Optional[str] = None
    nationality: Optional[str] = None
    dob: Optional[str] = None
    expiry_date: Optional[str] = None


class DLFields(BaseFields):
    schema_type: str = "DRIVING_LICENSE"
    dl_number: Optional[str] = None
    name: Optional[str] = None
    dob: Optional[str] = None
    validity: Optional[str] = None  # expiry date, ISO


class GSTFields(BaseFields):
    schema_type: str = "GST_CERTIFICATE"
    gstin: Optional[str] = None
    legal_name: Optional[str] = None
    trade_name: Optional[str] = None


class UnknownFields(BaseFields):
    """Catch-all when classification fails; intentionally has no fields."""
    schema_type: str = "UNKNOWN"


class GenericDocumentAnalysis(BaseModel):
    """Dynamic extraction payload for documents outside fixed KYC schemas."""

    document_type_label: str = "Unknown Document"
    document_category: str = "OTHER"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    dynamic_fields: Dict[str, Any] = Field(default_factory=dict)
    raw_text: str = ""
    summary: str = ""
    visual_concerns: List[str] = Field(default_factory=list)
    extraction_notes: List[str] = Field(default_factory=list)
    source: str = "heuristic"


# Back-compat shim: legacy callers that import `ExtractedFields` continue to
# work but get the union of all known fields. New code should consume the
# per-doc subclasses via the dispatcher in `app.services.extractors`.
class ExtractedFields(BaseFields):
    schema_type: str = "LEGACY"
    pan_number: Optional[str] = None
    aadhaar_number: Optional[str] = None
    aadhaar_last4: Optional[str] = None
    passport_number: Optional[str] = None
    driving_license_number: Optional[str] = None
    gstin: Optional[str] = None
    name: Optional[str] = None
    father_name: Optional[str] = None
    dob: Optional[str] = None
    gender: Optional[str] = None
    address: Optional[str] = None


# ---------------------------------------------------------------------------
# Request / Response
# ---------------------------------------------------------------------------
class OCRRequest(BaseModel):
    file_path: str
    document_type: Optional[DocumentType] = None
    ocr_engine: OCREngine = OCREngine.PADDLE
    request_id: Optional[str] = None
    extract_tables: bool = False
    extract_qr: bool = True


class OCRMetadata(BaseModel):
    request_id: str
    file_name: str
    file_size_bytes: int
    mime_type: Optional[str] = None
    page_count: int = 1
    engine: str = "paddleocr"
    engine_version: Optional[str] = None
    preprocessing_applied: List[str] = Field(default_factory=list)
    quality: Optional[QualityReport] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None
    cache_hit: bool = False


class OCRError(BaseModel):
    code: OCRErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None


class OCRResponse(BaseModel):
    success: bool
    ocr_engine: OCREngine = OCREngine.PADDLE
    document_type: DocumentType = DocumentType.UNKNOWN
    document_type_label: Optional[str] = None
    classification_confidence: float = 0.0
    raw_text: str = ""
    lines: List[OCRLine] = Field(default_factory=list)
    words: List[OCRWord] = Field(default_factory=list)
    fields: ExtractedFields = Field(default_factory=ExtractedFields)
    # Typed per-doc-type payload. `fields` is the legacy union surface kept
    # for back-compat; `typed_fields` is the authoritative schema downstream
    # services should prefer.
    typed_fields: Optional[BaseFields] = None
    dynamic_fields: Dict[str, Any] = Field(default_factory=dict)
    generic_analysis: Optional[GenericDocumentAnalysis] = None
    confidence_score: float = 0.0
    confidence_report: Optional[ConfidenceReport] = None
    processing_time: float = 0.0
    metadata: OCRMetadata
    error: Optional[OCRError] = None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class ValidationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    MISSING = "MISSING"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class FieldValidation(BaseModel):
    field: str
    status: ValidationStatus
    expected: Optional[str] = None
    actual: Optional[str] = None
    reason: Optional[str] = None
    required: bool = True


class ValidationReport(BaseModel):
    document_type: DocumentType
    overall: ValidationStatus
    score: float = Field(ge=0.0, le=1.0)
    checks: List[FieldValidation] = Field(default_factory=list)
    required_passed: int = 0
    required_total: int = 0
    optional_passed: int = 0
    optional_total: int = 0
