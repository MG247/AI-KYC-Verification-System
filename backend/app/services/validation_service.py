"""Document-aware KYC validation service.

Each KYC document declares a small set of REQUIRED and OPTIONAL fields.
The validator only checks fields that apply to the actual document type,
which fixes the historical bug where a PAN card was rejected for not
containing an Aadhaar number (and vice versa).

The decision rule is simple and auditable:
  * Overall PASS  → every REQUIRED check passed.
  * Overall FAIL  → at least one REQUIRED check failed.
  * Overall MISSING → at least one REQUIRED check is MISSING (no FAIL).
  * Score = required_passed / max(1, required_total)

Comparison semantics
--------------------
* PAN, Passport, GSTIN, DL — case-insensitive, whitespace-stripped equality.
* Aadhaar — only the masked last 4 digits are compared, per UIDAI guidance.
* DOB     — both sides normalized to ISO ``YYYY-MM-DD`` before equality.
* Name    — fuzzy match using token-set ratio (≥85 = PASS) so middle-name
            order / OCR jitter doesn't cause false negatives.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from app.schemas.ocr_schema import (
    DocumentType,
    ExtractedFields,
    FieldValidation,
    OCRResponse,
    ValidationReport,
    ValidationStatus,
)

logger = logging.getLogger(__name__)

# --- Optional fuzzy-match dep; degrade gracefully if missing. -------------
try:
    from rapidfuzz import fuzz as _fuzz  # type: ignore
    _HAS_FUZZ = True
except ImportError:  # pragma: no cover
    _HAS_FUZZ = False


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldRule:
    """Declarative description of one validation check.

    ``getter`` pulls the actual value out of ``ExtractedFields``; ``compare``
    is the equality function. Keeping both as callables means we can plug
    in custom normalizers per-field without an ``if-elif`` ladder.
    """
    name: str
    truth_key: str                                    # key inside sample_user.json
    getter: Callable[[ExtractedFields], Optional[str]]
    compare: Callable[[str, str], bool]
    required: bool = True


# ---------------------------------------------------------------------------
# Comparators
# ---------------------------------------------------------------------------
def _norm(s: Optional[str]) -> str:
    return re.sub(r"\s+", "", (s or "")).upper()


def eq_alnum(expected: str, actual: str) -> bool:
    """Case-insensitive, whitespace-stripped equality (IDs)."""
    return bool(actual) and _norm(expected) == _norm(actual)


def eq_aadhaar_last4(expected: str, actual: str) -> bool:
    """Compare only the last 4 digits — the full Aadhaar is masked."""
    e = re.sub(r"\D", "", expected or "")[-4:]
    a = re.sub(r"\D", "", actual or "")[-4:]
    return bool(e) and bool(a) and e == a


def eq_iso_date(expected: str, actual: str) -> bool:
    """Both sides normalized to ISO before equality."""
    return _iso(expected) == _iso(actual) and bool(_iso(expected))


def _iso(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.strip().replace(".", "-").replace("/", "-")
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d-%m-%y", "%d-%b-%Y", "%d-%B-%Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return ""


def eq_name(expected: str, actual: str, threshold: int = 85) -> bool:
    """Fuzzy name equality. Falls back to substring containment when
    rapidfuzz is not installed."""
    e = (expected or "").strip().upper()
    a = (actual or "").strip().upper()
    if not e or not a:
        return False
    if _HAS_FUZZ:
        return int(_fuzz.token_set_ratio(e, a)) >= threshold
    return e in a or a in e


# ---------------------------------------------------------------------------
# Per-document rule sets
# ---------------------------------------------------------------------------
def _pan_rules() -> List[FieldRule]:
    return [
        FieldRule("pan_number", "pan", lambda f: f.pan_number, eq_alnum, required=True),
        FieldRule("name", "name", lambda f: f.name, eq_name, required=True),
        FieldRule("dob", "dob", lambda f: f.dob, eq_iso_date, required=True),
        FieldRule("father_name", "father_name", lambda f: f.father_name, eq_name, required=False),
    ]


def _aadhaar_rules() -> List[FieldRule]:
    return [
        FieldRule("aadhaar_last4", "aadhaar", lambda f: f.aadhaar_last4, eq_aadhaar_last4, required=True),
        FieldRule("name", "name", lambda f: f.name, eq_name, required=True),
        FieldRule("dob", "dob", lambda f: f.dob, eq_iso_date, required=True),
        FieldRule("gender", "gender", lambda f: f.gender, eq_alnum, required=False),
    ]


def _passport_rules() -> List[FieldRule]:
    return [
        FieldRule("passport_number", "passport", lambda f: f.passport_number, eq_alnum, required=True),
        FieldRule("name", "name", lambda f: f.name, eq_name, required=True),
        FieldRule("dob", "dob", lambda f: f.dob, eq_iso_date, required=True),
    ]


def _driving_license_rules() -> List[FieldRule]:
    return [
        FieldRule("driving_license_number", "driving_license",
                  lambda f: f.driving_license_number, eq_alnum, required=True),
        FieldRule("name", "name", lambda f: f.name, eq_name, required=True),
        FieldRule("dob", "dob", lambda f: f.dob, eq_iso_date, required=True),
    ]


def _gst_rules() -> List[FieldRule]:
    return [
        FieldRule("gstin", "gstin", lambda f: f.gstin, eq_alnum, required=True),
        FieldRule("name", "name", lambda f: f.name, eq_name, required=False),
    ]


_RULES: Dict[DocumentType, Callable[[], List[FieldRule]]] = {
    DocumentType.PAN: _pan_rules,
    DocumentType.AADHAAR: _aadhaar_rules,
    DocumentType.PASSPORT: _passport_rules,
    DocumentType.DRIVING_LICENSE: _driving_license_rules,
    DocumentType.GST_CERTIFICATE: _gst_rules,
}


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class ValidationService:
    """Validate an OCR response against a ground-truth dict.

    The ground-truth dict comes from whatever upstream system holds the
    user's KYC profile (a database row, a form payload, sample_user.json
    in tests). Only keys relevant to the detected document type are
    consulted — everything else is reported as ``NOT_APPLICABLE``.
    """

    def __init__(self, *, name_fuzzy_threshold: int = 85) -> None:
        self._name_threshold = name_fuzzy_threshold

    def validate(
        self,
        ocr: OCRResponse,
        truth: Dict[str, str],
    ) -> ValidationReport:
        if not ocr.success:
            return ValidationReport(
                document_type=ocr.document_type,
                overall=ValidationStatus.FAIL,
                score=0.0,
                checks=[FieldValidation(
                    field="ocr",
                    status=ValidationStatus.FAIL,
                    reason=(ocr.error.message if ocr.error else "OCR failed"),
                )],
                required_total=1,
            )

        rules_fn = _RULES.get(ocr.document_type)
        if rules_fn is None:
            logger.warning("No validation rules for document_type=%s", ocr.document_type)
            return ValidationReport(
                document_type=ocr.document_type,
                overall=ValidationStatus.NOT_APPLICABLE,
                score=0.0,
            )

        rules = rules_fn()
        checks: List[FieldValidation] = []
        req_pass = req_total = opt_pass = opt_total = 0

        for rule in rules:
            expected = (truth or {}).get(rule.truth_key)
            actual = rule.getter(ocr.fields)

            if rule.required:
                req_total += 1
            else:
                opt_total += 1

            if not expected:
                # No truth provided → don't penalize, report NOT_APPLICABLE.
                checks.append(FieldValidation(
                    field=rule.name,
                    status=ValidationStatus.NOT_APPLICABLE,
                    expected=None,
                    actual=actual,
                    required=rule.required,
                    reason="no ground-truth provided",
                ))
                # An optional with no truth doesn't count; a required with no
                # truth ALSO doesn't count (we can't fail what we can't check).
                if rule.required:
                    req_total -= 1
                else:
                    opt_total -= 1
                continue

            if not actual:
                status = ValidationStatus.MISSING
                reason = "field not extracted"
            elif rule.compare(expected, actual):
                status = ValidationStatus.PASS
                reason = None
                if rule.required:
                    req_pass += 1
                else:
                    opt_pass += 1
            else:
                status = ValidationStatus.FAIL
                reason = "value mismatch"

            checks.append(FieldValidation(
                field=rule.name, status=status,
                expected=expected, actual=actual,
                required=rule.required, reason=reason,
            ))

        overall = self._decide(checks, req_total)
        score = (req_pass / req_total) if req_total else 1.0
        report = ValidationReport(
            document_type=ocr.document_type,
            overall=overall,
            score=score,
            checks=checks,
            required_passed=req_pass,
            required_total=req_total,
            optional_passed=opt_pass,
            optional_total=opt_total,
        )
        logger.info(
            "validation %s: %s score=%.2f (%d/%d required, %d/%d optional)",
            ocr.document_type.value, overall.value, score,
            req_pass, req_total, opt_pass, opt_total,
        )
        return report

    @staticmethod
    def _decide(checks: List[FieldValidation], required_total: int) -> ValidationStatus:
        required = [c for c in checks if c.required]
        if not required or required_total == 0:
            # All required checks were NOT_APPLICABLE → treat as MISSING so
            # the caller knows we couldn't make a real determination.
            return ValidationStatus.MISSING
        if any(c.status == ValidationStatus.FAIL for c in required):
            return ValidationStatus.FAIL
        if any(c.status == ValidationStatus.MISSING for c in required):
            return ValidationStatus.MISSING
        return ValidationStatus.PASS


# Module singleton
_service: Optional[ValidationService] = None


def get_validation_service() -> ValidationService:
    global _service
    if _service is None:
        _service = ValidationService()
    return _service
