"""Document classifier tests.

Classification is precision-critical — wrong type silently selects the
wrong extractor. Each test pins one of the co-occurrence rules.
"""

from __future__ import annotations

from app.schemas.ocr_schema import DocumentType
from app.services.document_classifier import classify


def test_pan_requires_both_number_and_keyword():
    # Number alone → PAN_PARTIAL, not full confidence.
    res = classify("ABCDE1234F printed on something")
    assert res.doc_type == DocumentType.PAN
    assert res.confidence == 0.55

    # Number + keyword → full confidence.
    res = classify("INCOME TAX DEPARTMENT\nABCDE1234F")
    assert res.doc_type == DocumentType.PAN
    assert res.confidence == 1.0


def test_aadhaar_requires_both_number_and_keyword():
    res = classify("Government of India\nUnique Identification\n2341 2341 2346")
    assert res.doc_type == DocumentType.AADHAAR
    assert res.confidence == 1.0


def test_aadhaar_wins_over_pan_when_both_present():
    """Order matters: an Aadhaar that mentions PAN must still classify as
    Aadhaar (rare but happens on covering letters)."""
    text = (
        "Government of India\n"
        "Unique Identification Authority\n"
        "2341 2341 2346\n"
        "Also see Income Tax PAN ABCDE1234F"
    )
    assert classify(text).doc_type == DocumentType.AADHAAR


def test_gstin_requires_both():
    text = "Goods and Services Tax\n29ABCDE1234F1Z5"
    assert classify(text).doc_type == DocumentType.GST_CERTIFICATE


def test_unknown_returns_hint_at_zero_confidence():
    res = classify("nothing useful here", hint=DocumentType.PAN)
    assert res.doc_type == DocumentType.PAN
    assert res.confidence == 0.0
    assert res.matched_signals == ("HINT_FALLBACK",)


def test_unknown_with_no_hint():
    res = classify("blank slate")
    assert res.doc_type == DocumentType.UNKNOWN
    assert res.confidence == 0.0


def test_bank_statement_is_named_for_dynamic_extraction():
    res = classify("Statement of Account\nIFSC HDFC0001234\nClosing Balance 1200.00")
    assert res.doc_type == DocumentType.BANK_STATEMENT
    assert res.confidence == 0.75


def test_salary_slip_is_named_for_dynamic_extraction():
    res = classify("Salary Slip\nEarnings\nDeductions\nNet Pay 50000")
    assert res.doc_type == DocumentType.SALARY_SLIP
    assert res.confidence == 0.75
