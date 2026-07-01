"""PAN extractor.

Production failure this fixes: PAN `father_name` captured
``"Pooran Singh Gautam Date Of Birth"`` because the legacy regex extended
40 chars past the label and crossed the next labelled line. Here we:

1. Locate the `Father's Name` label LINE (not regex window).
2. Read the value from the same line OR the one immediately below.
3. Stop at the next label-boundary token.
4. Reject the captured value if any boundary token or digit slipped through.
"""

from __future__ import annotations

import re
from typing import Optional

from app.schemas.ocr_schema import OCRLine, PanFields

from .base import (
    DATE_RE,
    PAN_RE,
    ExtractorContext,
    clean_name,
    find_label_line,
    normalize_date,
    reject_if_label_leak,
    value_after_label,
)

_FATHER_LABEL = re.compile(r"\bFather(?:'s)?\s*Name\b", re.IGNORECASE)
_NAME_LABEL   = re.compile(r"\bName\b", re.IGNORECASE)
_DOB_LABEL    = re.compile(
    r"(?:D[\.\s]?[O0Q][\.\s]?B|Date\s*of\s*Birth|Birth\s*Date|जन्म)",
    re.IGNORECASE,
)


def extract_pan(ctx: ExtractorContext) -> PanFields:
    out = PanFields()

    # 1. PAN number — regex on flat text is safe (no ambiguity in the format).
    if m := PAN_RE.search(ctx.raw_text.upper()):
        out.pan_number = m.group(1)

    # 2. Father's name — boundary-aware, single-line only.
    father_idx = find_label_line(ctx.lines, [_FATHER_LABEL])
    if father_idx is not None:
        raw = value_after_label(ctx.lines, father_idx, _FATHER_LABEL)
        out.father_name = clean_name(reject_if_label_leak(raw, allow_digits=False))

    # 3. Cardholder name — appears ABOVE the father-name label on a PAN card.
    #    Fall back to the first labelled "Name:" line.
    out.name = _extract_name(ctx.lines, father_idx)

    # 4. DOB — prefer label-anchored, else the first standalone date.
    dob_idx = find_label_line(ctx.lines, [_DOB_LABEL])
    dob_raw: Optional[str] = None
    if dob_idx is not None:
        dob_raw = value_after_label(ctx.lines, dob_idx, _DOB_LABEL)
    if not dob_raw:
        if m := DATE_RE.search(ctx.raw_text):
            dob_raw = m.group(1)
    out.dob = normalize_date(dob_raw)

    return out


def _extract_name(lines: list[OCRLine], father_idx: Optional[int]) -> Optional[str]:
    """Pick the cardholder name. PAN layout puts it directly above the
    father-name label; if that line is itself a label, walk upward until
    we find a plausible person-name."""
    if father_idx is not None and father_idx > 0:
        for i in range(father_idx - 1, max(-1, father_idx - 4), -1):
            cand = lines[i].text.strip()
            if not cand:
                continue
            if _looks_like_label(cand):
                continue
            cleaned = clean_name(reject_if_label_leak(cand, allow_digits=False))
            if cleaned:
                return cleaned

    # Fallback: labelled "Name: X"
    name_idx = find_label_line(lines, [_NAME_LABEL])
    if name_idx is not None:
        raw = value_after_label(lines, name_idx, _NAME_LABEL)
        return clean_name(reject_if_label_leak(raw, allow_digits=False))
    return None


def _looks_like_label(text: str) -> bool:
    lower = text.lower().strip(" :-,\t")
    return lower in {
        "name", "father's name", "fathers name", "father name",
        "income tax department", "govt. of india", "government of india",
        "permanent account number", "permanent account number card",
        "signature", "date of birth", "dob",
    }
