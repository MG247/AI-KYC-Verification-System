"""GST certificate extractor."""

from __future__ import annotations

import re

from app.schemas.ocr_schema import GSTFields

from .base import (
    GSTIN_RE,
    ExtractorContext,
    clean_name,
    find_label_line,
    reject_if_label_leak,
    value_after_label,
)

_LEGAL_LABEL = re.compile(r"Legal\s*Name", re.IGNORECASE)
_TRADE_LABEL = re.compile(r"Trade\s*Name", re.IGNORECASE)


def extract_gst(ctx: ExtractorContext) -> GSTFields:
    out = GSTFields()
    if m := GSTIN_RE.search(ctx.raw_text.upper()):
        out.gstin = m.group(1)
    if i := find_label_line(ctx.lines, [_LEGAL_LABEL]):
        out.legal_name = clean_name(reject_if_label_leak(
            value_after_label(ctx.lines, i, _LEGAL_LABEL), allow_digits=False,
        ))
    if i := find_label_line(ctx.lines, [_TRADE_LABEL]):
        out.trade_name = clean_name(reject_if_label_leak(
            value_after_label(ctx.lines, i, _TRADE_LABEL), allow_digits=False,
        ))
    return out
