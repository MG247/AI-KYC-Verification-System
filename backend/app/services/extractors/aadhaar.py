"""Aadhaar extractor.

Compliance note: the 12-digit Aadhaar is NEVER emitted in the response.
We Verhoeff-validate first; if validation fails the whole number is
dropped (OCR corruption is far more likely than UIDAI issuing an
invalid number). On success we store only `aadhaar_last4` and a masked
`XXXX XXXX 1234` form for display.
"""

from __future__ import annotations

import re
from typing import Optional

from app.schemas.ocr_schema import AadhaarFields

from .base import (
    AADHAAR_RE,
    DATE_RE,
    ExtractorContext,
    clean_name,
    find_label_line,
    normalize_date,
    reject_if_label_leak,
    value_after_label,
)

_DOB_LABEL = re.compile(
    r"(?:D[\.\s]?[O0Q][\.\s]?B|Date\s*of\s*Birth|Year\s*of\s*Birth|जन्म)",
    re.IGNORECASE,
)
_GENDER_LABEL = re.compile(r"(?:GENDER|SEX|लिंग)", re.IGNORECASE)
_GENDER_VALUE = re.compile(r"\b(FEMALE|MALE|TRANSGENDER|F|M)\b", re.IGNORECASE)
_NAME_BLACKLIST = {
    "government of india", "unique identification authority",
    "uidai", "aadhaar", "address", "year of birth",
    "date of birth", "dob",
}

# Words that never appear inside a real person's name on an Aadhaar card.
# Used to reject the Aadhaar slogan ("Aadhaar Is Proof Of Identity, Not Of
# Citizenship") even when OCR garbles it past the blacklist — e.g.
# "Asdhaar Is Prool D Identty, Not Ol Dt" still trips on `is`/`not`.
_NAME_STOPWORDS = {
    "is", "of", "not", "and", "or", "the", "a", "an", "to", "for",
    "proof", "identity", "citizenship", "card", "my", "your",
    "mera", "meri", "pehchaan", "pehchan", "aam", "aadmi",
}
# Fuzzy substrings: garbled OCR of slogan words. Lowercased substring match.
_NAME_FUZZY_REJECT = (
    "sdhaar", "dhaar",       # "Aadhaar"
    "prool", "proof",
    "identt", "identi",
    "citizen",
)


# ---- Verhoeff checksum -----------------------------------------------------
# UIDAI uses Verhoeff. Implemented from the canonical multiplication +
# permutation tables. Rejects every typo a single OCR misread would cause.
_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def _verhoeff_valid(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if len(digits) != 12:
        return False
    c = 0
    for i, n in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][n]]
    return c == 0


def extract_aadhaar(ctx: ExtractorContext) -> AadhaarFields:
    out = AadhaarFields()

    # 1. Aadhaar number — Verhoeff-gated. If it fails, drop the entire field.
    for m in AADHAAR_RE.finditer(ctx.raw_text):
        candidate = m.group(1) + m.group(2) + m.group(3)
        if _verhoeff_valid(candidate):
            out.aadhaar_last4 = m.group(3)
            out.masked_aadhaar = f"XXXX XXXX {m.group(3)}"
            break

    # 2. DOB / YOB
    dob_idx = find_label_line(ctx.lines, [_DOB_LABEL])
    dob_raw: Optional[str] = None
    if dob_idx is not None:
        dob_raw = value_after_label(ctx.lines, dob_idx, _DOB_LABEL)
    if not dob_raw:
        if m := DATE_RE.search(ctx.raw_text):
            dob_raw = m.group(1)
    out.dob = normalize_date(dob_raw)

    # 3. Gender — labelled value only; refuse lone "M"/"F" from "F/Father".
    if g_idx := find_label_line(ctx.lines, [_GENDER_LABEL]):
        g_raw = value_after_label(ctx.lines, g_idx, _GENDER_LABEL)
        if g_raw:
            gm = _GENDER_VALUE.search(g_raw)
            if gm:
                tok = gm.group(1).upper()
                out.gender = {"M": "MALE", "F": "FEMALE"}.get(tok, tok)

    # 4. Name — Aadhaar layout: name is the line directly above DOB. If we
    #    have a confirmed DOB line, look upward; otherwise pick the first
    #    title-cased non-blacklisted line.
    out.name = _extract_name(ctx, dob_idx)

    return out


def _extract_name(ctx: ExtractorContext, dob_idx: Optional[int]) -> Optional[str]:
    lines = ctx.lines
    search_above = range(dob_idx - 1, max(-1, dob_idx - 4), -1) if dob_idx else range(min(8, len(lines)))
    for i in search_above:
        if i < 0 or i >= len(lines):
            continue
        cand = lines[i].text.strip()
        if not cand:
            continue
        low = cand.lower()
        if any(b in low for b in _NAME_BLACKLIST):
            continue
        if _looks_like_slogan(cand, low):
            continue
        cleaned = clean_name(reject_if_label_leak(cand, allow_digits=False))
        if cleaned and len(cleaned.split()) >= 2:
            return cleaned
    return None


def _looks_like_slogan(cand: str, low: str) -> bool:
    """True if `cand` is sentence-like text rather than a person name.

    Person names on Aadhaar cards: 2-5 title-cased tokens, no commas, no
    English connector words. Anything else (e.g. the garbled UIDAI slogan
    "Asdhaar Is Prool D Identty, Not Ol Dt") is rejected here so it never
    reaches the `name` field.
    """
    if "," in cand or ";" in cand:
        return True
    tokens = [t for t in re.split(r"\s+", low) if t]
    if len(tokens) > 5:
        return True
    if any(t.strip(".") in _NAME_STOPWORDS for t in tokens):
        return True
    if any(frag in low for frag in _NAME_FUZZY_REJECT):
        return True
    return False
