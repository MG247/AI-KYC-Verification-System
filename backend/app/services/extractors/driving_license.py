"""Driving Licence extractor."""

from __future__ import annotations

import re
from typing import Optional

from app.schemas.ocr_schema import DLFields

from .base import (
    DATE_RE,
    DL_RE,
    ExtractorContext,
    clean_name,
    find_label_line,
    normalize_date,
    reject_if_label_leak,
    value_after_label,
)

_DOB_LABEL = re.compile(r"(?:Date\s*of\s*Birth|DOB)", re.IGNORECASE)
_NAME_LABEL = re.compile(r"\bName\b", re.IGNORECASE)
_VALIDITY_LABEL = re.compile(r"(?:Valid\s*Till|Validity|Valid\s*Until|Expiry)", re.IGNORECASE)


def extract_dl(ctx: ExtractorContext) -> DLFields:
    out = DLFields()

    if m := DL_RE.search(ctx.raw_text.upper()):
        out.dl_number = re.sub(r"[-\s]", "", m.group(1))

    if i := find_label_line(ctx.lines, [_NAME_LABEL]):
        raw = value_after_label(ctx.lines, i, _NAME_LABEL)
        out.name = clean_name(reject_if_label_leak(raw, allow_digits=False))

    out.dob = _date_after_label(ctx, _DOB_LABEL)
    out.validity = _date_after_label(ctx, _VALIDITY_LABEL)
    return out


def _date_after_label(ctx: ExtractorContext, label: re.Pattern[str]) -> Optional[str]:
    idx = find_label_line(ctx.lines, [label])
    if idx is not None:
        raw = value_after_label(ctx.lines, idx, label)
        if raw and (m := DATE_RE.search(raw)):
            return normalize_date(m.group(1))
    return None
