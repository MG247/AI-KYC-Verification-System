"""Passport extractor.

The MRZ (machine-readable zone, last two lines) is the source of truth
for surname/given names/nationality/expiry. We parse it strictly per
ICAO 9303 — if the MRZ is unreadable, return whatever a label scan can
recover but do NOT guess.
"""

from __future__ import annotations

import re
from typing import Optional

from app.schemas.ocr_schema import PassportFields

from .base import (
    DATE_RE,
    PASSPORT_RE,
    ExtractorContext,
    clean_name,
    find_label_line,
    normalize_date,
    reject_if_label_leak,
    value_after_label,
)

_DOB_LABEL = re.compile(r"(?:Date\s*of\s*Birth|DOB)", re.IGNORECASE)
_EXPIRY_LABEL = re.compile(r"(?:Date\s*of\s*Expiry|Valid\s*Until|Expiry)", re.IGNORECASE)
_SURNAME_LABEL = re.compile(r"(?:Surname|Last\s*Name)", re.IGNORECASE)
_GIVEN_LABEL = re.compile(r"(?:Given\s*Name|First\s*Name)", re.IGNORECASE)
_NATIONALITY_LABEL = re.compile(r"Nationality", re.IGNORECASE)

# MRZ line 1 example: P<IND<SURNAME<<GIVEN<NAME<<<<<<<<<<<<<<<<<<<<<<
_MRZ_L1 = re.compile(r"^P[<A-Z0-9]{1}([A-Z]{3})([A-Z<]+)$")


def extract_passport(ctx: ExtractorContext) -> PassportFields:
    out = PassportFields()

    if m := PASSPORT_RE.search(ctx.raw_text):
        out.passport_number = m.group(1).upper()

    # ---- MRZ path ----
    for line in ctx.lines:
        txt = line.text.replace(" ", "")
        m = _MRZ_L1.match(txt)
        if m:
            out.nationality = m.group(1)
            name_part = m.group(2).strip("<")
            if "<<" in name_part:
                surname, _, given = name_part.partition("<<")
                out.surname = clean_name(surname.replace("<", " ").strip())
                out.given_name = clean_name(given.replace("<", " ").strip())
            break

    # ---- Label fallbacks ----
    if not out.surname:
        if i := find_label_line(ctx.lines, [_SURNAME_LABEL]):
            out.surname = clean_name(reject_if_label_leak(
                value_after_label(ctx.lines, i, _SURNAME_LABEL), allow_digits=False,
            ))
    if not out.given_name:
        if i := find_label_line(ctx.lines, [_GIVEN_LABEL]):
            out.given_name = clean_name(reject_if_label_leak(
                value_after_label(ctx.lines, i, _GIVEN_LABEL), allow_digits=False,
            ))
    if not out.nationality:
        if i := find_label_line(ctx.lines, [_NATIONALITY_LABEL]):
            v = value_after_label(ctx.lines, i, _NATIONALITY_LABEL)
            if v:
                out.nationality = v.upper()[:3]

    out.dob = _date_after_label(ctx, _DOB_LABEL)
    out.expiry_date = _date_after_label(ctx, _EXPIRY_LABEL)

    return out


def _date_after_label(ctx: ExtractorContext, label: re.Pattern[str]) -> Optional[str]:
    idx = find_label_line(ctx.lines, [label])
    if idx is not None:
        raw = value_after_label(ctx.lines, idx, label)
        if raw and (m := DATE_RE.search(raw)):
            return normalize_date(m.group(1))
    return None
