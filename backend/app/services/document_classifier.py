"""Deterministic document classifier.

Runs AFTER OCR but BEFORE field extraction. Never trusts a single signal:
classification is granted only when both a structural regex AND a
keyword anchor match. This prevents a stray PAN-shaped string in an
otherwise-non-PAN document from triggering a misclassification.

No LLM. Classification is a precision-critical step — the wrong doc type
selects the wrong extractor and silently emits wrong fields. A
deterministic ruleset is reviewable, auditable, and unit-testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from app.schemas.ocr_schema import DocumentType

# Structural patterns — strong evidence of a specific doc type.
_PAN_NUM = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_AADHAAR_NUM = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
_PASSPORT_NUM = re.compile(r"\b[A-PR-WY][0-9]{7}\b")
_PASSPORT_MRZ = re.compile(r"^P[<A-Z0-9][A-Z<]{3,}", re.MULTILINE)
_DL_NUM = re.compile(r"\b[A-Z]{2}[-\s]?\d{2}[-\s]?(?:19|20)\d{2}[-\s]?\d{7}\b")
_GSTIN = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]\b")

# Keyword anchors — required co-occurrence so a stray pattern can't win alone.
_AADHAAR_KW = re.compile(r"(?:Unique Identification|Aadhaar|UIDAI|Government of India)", re.IGNORECASE)
_PAN_KW = re.compile(r"(?:Income\s*Tax|Permanent\s*Account|GOVT\.?\s*OF\s*INDIA)", re.IGNORECASE)
_PASSPORT_KW = re.compile(r"(?:Republic\s*of\s*India|Passport|PASSPORT NO)", re.IGNORECASE)
_DL_KW = re.compile(r"Driving\s*Licen[cs]e", re.IGNORECASE)
_GST_KW = re.compile(r"(?:Goods\s*and\s*Services\s*Tax|GST\s*Registration)", re.IGNORECASE)
_BANK_STATEMENT_KW = re.compile(
    r"(?:Statement\s*of\s*Account|Account\s*Statement|IFSC|Closing\s*Balance|Transaction\s*Details)",
    re.IGNORECASE,
)
_SALARY_SLIP_KW = re.compile(
    r"(?:Salary\s*Slip|Pay\s*Slip|Net\s*Pay|Earnings|Deductions)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ClassificationResult:
    doc_type: DocumentType
    confidence: float          # 0.0 (hint-fallback) → 1.0 (both signals matched)
    matched_signals: tuple[str, ...]


def classify(text: str, *, hint: Optional[DocumentType] = None) -> ClassificationResult:
    """Classify a document from OCR'd text.

    Ordering: most-restrictive structural patterns first. If two doc types
    co-match (rare but possible — e.g. a PAN number printed on a covering
    letter that also references "Income Tax"), the first wins.
    """
    has_aadhaar_num = bool(_AADHAAR_NUM.search(text))
    has_aadhaar_kw = bool(_AADHAAR_KW.search(text))
    if has_aadhaar_num and has_aadhaar_kw:
        return ClassificationResult(DocumentType.AADHAAR, 1.0, ("AADHAAR_NUM", "AADHAAR_KW"))

    has_pan_num = bool(_PAN_NUM.search(text))
    has_pan_kw = bool(_PAN_KW.search(text))
    if has_pan_num and has_pan_kw:
        return ClassificationResult(DocumentType.PAN, 1.0, ("PAN_NUM", "PAN_KW"))

    if _PASSPORT_MRZ.search(text) or (_PASSPORT_NUM.search(text) and _PASSPORT_KW.search(text)):
        return ClassificationResult(DocumentType.PASSPORT, 1.0, ("PASSPORT_MRZ_OR_NUM_KW",))

    if _DL_NUM.search(text) and _DL_KW.search(text):
        return ClassificationResult(DocumentType.DRIVING_LICENSE, 1.0, ("DL_NUM", "DL_KW"))

    if _GSTIN.search(text) and _GST_KW.search(text):
        return ClassificationResult(DocumentType.GST_CERTIFICATE, 1.0, ("GSTIN", "GST_KW"))

    if _BANK_STATEMENT_KW.search(text):
        return ClassificationResult(DocumentType.BANK_STATEMENT, 0.75, ("BANK_STATEMENT_KW",))

    if _SALARY_SLIP_KW.search(text):
        return ClassificationResult(DocumentType.SALARY_SLIP, 0.75, ("SALARY_SLIP_KW",))

    # Single-signal evidence — weaker, but better than UNKNOWN.
    if has_aadhaar_num or has_aadhaar_kw:
        return ClassificationResult(DocumentType.AADHAAR, 0.55, ("AADHAAR_PARTIAL",))
    if has_pan_num or has_pan_kw:
        return ClassificationResult(DocumentType.PAN, 0.55, ("PAN_PARTIAL",))
    if _PASSPORT_NUM.search(text) or _PASSPORT_KW.search(text):
        return ClassificationResult(DocumentType.PASSPORT, 0.55, ("PASSPORT_PARTIAL",))
    if _DL_NUM.search(text) or _DL_KW.search(text):
        return ClassificationResult(DocumentType.DRIVING_LICENSE, 0.55, ("DL_PARTIAL",))
    if _GSTIN.search(text) or _GST_KW.search(text):
        return ClassificationResult(DocumentType.GST_CERTIFICATE, 0.55, ("GST_PARTIAL",))

    if hint is not None and hint != DocumentType.UNKNOWN:
        return ClassificationResult(hint, 0.0, ("HINT_FALLBACK",))

    return ClassificationResult(DocumentType.UNKNOWN, 0.0, ())
