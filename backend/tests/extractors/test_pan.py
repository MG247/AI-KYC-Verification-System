"""PAN extractor tests.

The flagship test is `test_father_name_rejects_label_spillover` — it locks
in the fix for the production bug where `father_name` absorbed the next
labelled line ("Date Of Birth").
"""

from __future__ import annotations

from app.schemas.ocr_schema import DocumentType
from app.services.extractors.pan import extract_pan

from .conftest import ctx, line


def test_father_name_rejects_label_spillover():
    """Production failure case: must NOT capture "Date Of Birth"."""
    lines = [
        line("INCOME TAX DEPARTMENT", y=10),
        line("GOVT. OF INDIA", y=30),
        line("Permanent Account Number Card", y=60),
        line("ABCDE1234F", y=90),
        line("Name", y=120),
        line("ROHAN GAUTAM", y=140),
        line("Father's Name", y=170),
        line("POORAN SINGH GAUTAM", y=190),
        line("Date Of Birth", y=220),
        line("12/03/1985", y=240),
    ]
    out = extract_pan(ctx(lines, DocumentType.PAN))
    assert out.pan_number == "ABCDE1234F"
    assert out.father_name == "Pooran Singh Gautam"
    assert "Date" not in (out.father_name or "")
    assert "Birth" not in (out.father_name or "")
    assert out.dob == "1985-03-12"


def test_father_name_on_same_line():
    lines = [
        line("ABCDE1234F", y=10),
        line("Name: ROHAN GAUTAM", y=40),
        line("Father's Name: POORAN SINGH GAUTAM", y=70),
        line("Date Of Birth: 12/03/1985", y=100),
    ]
    out = extract_pan(ctx(lines, DocumentType.PAN))
    assert out.father_name == "Pooran Singh Gautam"


def test_father_name_label_only_returns_none():
    """If the line below the label is itself another label, we MUST NOT
    invent a value — anti-hallucination."""
    lines = [
        line("ABCDE1234F", y=10),
        line("Father's Name", y=40),
        line("Date Of Birth", y=70),  # label, not a value
        line("12/03/1985", y=100),
    ]
    out = extract_pan(ctx(lines, DocumentType.PAN))
    assert out.father_name is None


def test_father_name_rejects_digit_leak():
    """A father-name field with digits is a leak signal — must be dropped."""
    lines = [
        line("ABCDE1234F", y=10),
        line("Father's Name", y=40),
        line("POORAN 1985", y=70),
    ]
    out = extract_pan(ctx(lines, DocumentType.PAN))
    assert out.father_name is None


def test_pan_number_required_regex():
    lines = [line("Some random text", y=10), line("not a PAN", y=40)]
    out = extract_pan(ctx(lines, DocumentType.PAN))
    assert out.pan_number is None


def test_dob_normalization_variants():
    for raw, expected in [
        ("12/03/1985", "1985-03-12"),
        ("12-03-1985", "1985-03-12"),
        ("12.03.1985", "1985-03-12"),
    ]:
        lines = [
            line("ABCDE1234F", y=10),
            line("Date Of Birth", y=40),
            line(raw, y=70),
        ]
        out = extract_pan(ctx(lines, DocumentType.PAN))
        assert out.dob == expected, f"failed for {raw!r}"


def test_dob_rejects_future_year():
    lines = [
        line("ABCDE1234F", y=10),
        line("Date Of Birth", y=40),
        line("12/03/2099", y=70),
    ]
    out = extract_pan(ctx(lines, DocumentType.PAN))
    assert out.dob is None


def test_name_picks_line_above_father_label():
    lines = [
        line("ABCDE1234F", y=10),
        line("ROHAN GAUTAM", y=40),
        line("Father's Name", y=70),
        line("POORAN SINGH GAUTAM", y=100),
    ]
    out = extract_pan(ctx(lines, DocumentType.PAN))
    assert out.name == "Rohan Gautam"
