"""Shared fixtures for extractor tests.

`line()` builds a synthetic `OCRLine` from raw text — sufficient for testing
boundary-aware extraction since the helpers only consume `.text`. Real OCR
output also populates `.bbox` / `.words`; those are exercised by E2E.
"""

from __future__ import annotations

from typing import List

from app.schemas.ocr_schema import DocumentType, OCRLine
from app.services.extractors.base import ExtractorContext


def line(text: str, y: float = 0.0) -> OCRLine:
    bbox = ((0.0, y), (100.0, y), (100.0, y + 20.0), (0.0, y + 20.0))
    return OCRLine(text=text, confidence=0.99, bbox=bbox, page=0, words=[])


def ctx(lines: List[OCRLine], doc_type: DocumentType) -> ExtractorContext:
    return ExtractorContext(
        lines=lines,
        words=[],
        raw_text="\n".join(l.text for l in lines),
        doc_type=doc_type,
    )
