"""Aadhaar extractor tests.

Two non-negotiables: (1) the 12-digit number must never appear in the
emitted payload, (2) Verhoeff failures drop the number entirely rather
than masking a corrupt value.
"""

from __future__ import annotations

from app.schemas.ocr_schema import DocumentType
from app.services.extractors.aadhaar import _verhoeff_valid, extract_aadhaar

from .conftest import ctx, line


# A real Verhoeff-valid Aadhaar (publicly documented test number).
_VALID_AADHAAR = "234123412346"  # Verhoeff-valid


def test_verhoeff_helper_accepts_known_valid():
    assert _verhoeff_valid(_VALID_AADHAAR) is True


def test_verhoeff_helper_rejects_typo():
    # Flip one digit — Verhoeff must catch it.
    assert _verhoeff_valid("234123412345") is False


def test_aadhaar_never_emits_full_number():
    lines = [
        line("Government of India", y=10),
        line("ROHAN GAUTAM", y=40),
        line("DOB: 12/03/1985", y=70),
        line(f"{_VALID_AADHAAR[:4]} {_VALID_AADHAAR[4:8]} {_VALID_AADHAAR[8:]}", y=100),
    ]
    out = extract_aadhaar(ctx(lines, DocumentType.AADHAAR))
    payload = out.model_dump()
    # The raw 12-digit string must not appear anywhere in the response.
    flat = " ".join(str(v) for v in payload.values() if v is not None)
    assert _VALID_AADHAAR not in flat
    assert out.aadhaar_last4 == _VALID_AADHAAR[-4:]
    assert out.masked_aadhaar == f"XXXX XXXX {_VALID_AADHAAR[-4:]}"


def test_aadhaar_drops_number_on_verhoeff_failure():
    """OCR garbled the digits — safer to drop than to emit a wrong number."""
    bad = "234123412345"  # one-digit-off, fails Verhoeff
    lines = [
        line("Government of India", y=10),
        line("ROHAN GAUTAM", y=40),
        line(f"{bad[:4]} {bad[4:8]} {bad[8:]}", y=70),
    ]
    out = extract_aadhaar(ctx(lines, DocumentType.AADHAAR))
    assert out.aadhaar_last4 is None
    assert out.masked_aadhaar is None


def test_aadhaar_gender_extraction():
    lines = [
        line("Government of India", y=10),
        line("ROHAN GAUTAM", y=40),
        line("DOB: 12/03/1985", y=70),
        line("Gender: MALE", y=100),
        line(f"{_VALID_AADHAAR[:4]} {_VALID_AADHAAR[4:8]} {_VALID_AADHAAR[8:]}", y=130),
    ]
    out = extract_aadhaar(ctx(lines, DocumentType.AADHAAR))
    assert out.gender == "MALE"


def test_aadhaar_name_skips_blacklisted_lines():
    """Name must not be 'Government of India'."""
    lines = [
        line("Government of India", y=10),
        line("Unique Identification Authority of India", y=30),
        line("ROHAN GAUTAM", y=60),
        line("DOB: 12/03/1985", y=90),
    ]
    out = extract_aadhaar(ctx(lines, DocumentType.AADHAAR))
    assert out.name == "Rohan Gautam"
