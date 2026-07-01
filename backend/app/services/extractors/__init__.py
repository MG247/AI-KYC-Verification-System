"""Per-doc-type field extractor registry.

Replaces the monolithic `FieldExtractor` that previously lived in
`ocr_service`. Each registered extractor:

* takes an `ExtractorContext` carrying bbox-aware OCR lines, and
* returns a typed Pydantic model derived from `BaseFields` that exposes
  ONLY the fields valid for that document type.

Why this matters
----------------
The legacy extractor accepted `doc_type` but ignored it (`_ = doc_type`),
emitting the same union dict for every input. That broke two production
guarantees:

1. Downstreams could not infer doc type from field presence (everything
   was always present-or-None).
2. Greedy regexes spanning newlines absorbed the next labelled value
   into the previous field (PAN `father_name` → "... Date Of Birth").
"""

from __future__ import annotations

from typing import Callable, Dict

from app.schemas.ocr_schema import (
    AadhaarFields,
    BaseFields,
    DLFields,
    DocumentType,
    GSTFields,
    PanFields,
    PassportFields,
    UnknownFields,
)

from .aadhaar import extract_aadhaar
from .base import ExtractorContext
from .driving_license import extract_dl
from .gst import extract_gst
from .pan import extract_pan
from .passport import extract_passport

__all__ = [
    "ExtractorContext",
    "extract",
    "EXTRACTORS",
]

EXTRACTORS: Dict[DocumentType, Callable[[ExtractorContext], BaseFields]] = {
    DocumentType.PAN: extract_pan,
    DocumentType.AADHAAR: extract_aadhaar,
    DocumentType.PASSPORT: extract_passport,
    DocumentType.DRIVING_LICENSE: extract_dl,
    DocumentType.GST_CERTIFICATE: extract_gst,
}


def extract(ctx: ExtractorContext) -> BaseFields:
    """Dispatch to the per-doc-type extractor. UNKNOWN returns an empty payload."""
    fn = EXTRACTORS.get(ctx.doc_type)
    if fn is None:
        return UnknownFields()
    return fn(ctx)
