"""Shared primitives for per-doc-type field extractors.

The legacy extractor in `ocr_service.FieldExtractor` walked a flattened text
blob and ran greedy regexes. That caused production failure #1 — the PAN
`father_name` value absorbed the next label ("Date Of Birth") because regex
match windows ignore OCR line boundaries.

This module gives every extractor:

* a bbox-aware `ExtractorContext` (already-parsed `OCRLine` objects, sorted
  top-to-bottom and left-to-right by the OCR post-filter);
* label-locating + value-after-label helpers that stop at the next labelled
  line so a value can never bleed into a sibling field;
* a leak-rejection helper that drops any captured value that contains a
  banned substring or digits where digits are forbidden.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Pattern, Sequence

from app.schemas.ocr_schema import DocumentType, OCRLine, OCRWord


# Tokens that almost always start a new labelled field on KYC documents.
# Anything matching one of these is treated as a hard boundary — a value
# capture must stop BEFORE consuming it.
LABEL_BOUNDARY_TOKENS: tuple[str, ...] = (
    "date of birth", "dob", "d.o.b", "d o b", "जन्म",
    "father", "mother", "spouse", "s/o", "d/o", "w/o", "c/o",
    "permanent account", "income tax", "account number",
    "signature", "address", "pin", "pincode",
    "gender", "sex", "लिंग",
    "issue date", "expiry", "valid till", "validity",
    "nationality", "place of birth", "country code",
    "year of birth", "yob",
)

_BOUNDARY_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in LABEL_BOUNDARY_TOKENS) + r")\b",
    re.IGNORECASE,
)

_DIGIT_RE = re.compile(r"\d")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ExtractorContext:
    """Bbox-aware input handed to every per-doc extractor."""

    lines: List[OCRLine]
    words: List[OCRWord]
    raw_text: str
    doc_type: DocumentType


# ---------------------------------------------------------------------------
# Label / line navigation
# ---------------------------------------------------------------------------
def find_label_line(lines: Sequence[OCRLine], patterns: Iterable[Pattern[str]]) -> Optional[int]:
    """Index of the first line whose text matches any of `patterns`."""
    for i, line in enumerate(lines):
        for pat in patterns:
            if pat.search(line.text):
                return i
    return None


def value_after_label(
    lines: Sequence[OCRLine],
    label_idx: int,
    label_regex: Pattern[str],
    *,
    max_lookahead: int = 2,
) -> Optional[str]:
    """Return the text immediately following a label.

    Search order:
      1. Same line, the substring after the label match.
      2. Next ``max_lookahead`` lines, whichever first looks like a value.

    Stops AT (does not consume) the next labelled token so values cannot
    bleed across field boundaries — this is the fix for the
    `"Pooran Singh Gautam Date Of Birth"` failure.
    """
    if label_idx < 0 or label_idx >= len(lines):
        return None

    same = lines[label_idx].text
    m = label_regex.search(same)
    if m:
        tail = same[m.end():].strip(" :-\t")
        tail = _trim_at_boundary(tail)
        if tail and not _is_label_only(tail):
            return tail

    for j in range(label_idx + 1, min(label_idx + 1 + max_lookahead, len(lines))):
        cand = lines[j].text.strip()
        if not cand:
            continue
        if _is_label_only(cand):
            # The next line is itself a label — stop, the value is missing.
            return None
        return _trim_at_boundary(cand)
    return None


def _trim_at_boundary(text: str) -> str:
    """Cut `text` at the first label-boundary token encountered."""
    m = _BOUNDARY_RE.search(text)
    if not m:
        return _WS_RE.sub(" ", text).strip(" :-,\t")
    return _WS_RE.sub(" ", text[:m.start()]).strip(" :-,\t")


def _is_label_only(text: str) -> bool:
    """True if the whole stripped text is a known label and nothing else."""
    bare = _WS_RE.sub(" ", text).strip(" :-,\t").lower()
    if not bare:
        return True
    return bool(_BOUNDARY_RE.fullmatch(bare))


# ---------------------------------------------------------------------------
# Value sanitisation
# ---------------------------------------------------------------------------
def reject_if_label_leak(value: Optional[str], *, allow_digits: bool = False) -> Optional[str]:
    """Drop `value` if it contains a label-boundary token or — when
    `allow_digits=False` — any digit. Used as a final gate on name-shaped
    fields so leakage from a sibling label can never escape the extractor.
    """
    if value is None:
        return None
    if _BOUNDARY_RE.search(value):
        return None
    if not allow_digits and _DIGIT_RE.search(value):
        return None
    cleaned = _WS_RE.sub(" ", value).strip(" :-,.\t")
    if len(cleaned) < 2:
        return None
    return cleaned


def clean_name(value: Optional[str]) -> Optional[str]:
    """Normalize an extracted person-name."""
    if value is None:
        return None
    v = _WS_RE.sub(" ", value).strip(" :-,.\t")
    if not v:
        return None
    # Reject single-token noise like "MR" or "NAME".
    if len(v.split()) == 1 and len(v) < 4:
        return None
    return v.title()


# ---------------------------------------------------------------------------
# Date normalization
# ---------------------------------------------------------------------------
_DOB_FORMATS = (
    "%d/%m/%Y", "%d/%m/%y",
    "%d-%m-%Y", "%d-%m-%y",
    "%d.%m.%Y", "%d.%m.%y",
    "%d %b %Y", "%d %B %Y",
    "%d/%b/%Y", "%d/%B/%Y",
)


def normalize_date(raw: Optional[str]) -> Optional[str]:
    """Coerce a date-shaped string to ISO ``YYYY-MM-DD``.

    Returns None for unparseable input or implausible years (<1900 or in
    the future). Implausible years are a strong OCR-corruption signal —
    safer to return None and let validation flag the missing field.
    """
    if not raw:
        return None
    s = _WS_RE.sub(" ", raw).strip()
    today = datetime.utcnow().date()
    for fmt in _DOB_FORMATS:
        try:
            d = datetime.strptime(s, fmt).date()
        except ValueError:
            continue
        if d.year < 1900 or d > today:
            return None
        return d.isoformat()
    return None


# ---------------------------------------------------------------------------
# Compiled regexes shared across extractors
# ---------------------------------------------------------------------------
DATE_RE = re.compile(
    r"\b(\d{1,2}[\/\-\.\s](?:0?[1-9]|1[0-2]|[A-Za-z]{3,9})[\/\-\.\s]\d{2,4})\b"
)
PAN_RE = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")
AADHAAR_RE = re.compile(r"\b(\d{4})[ \t-]?(\d{4})[ \t-]?(\d{4})\b")
PASSPORT_RE = re.compile(r"\b([A-PR-WYa-pr-wy][0-9]{7})\b")
DL_RE = re.compile(r"\b([A-Z]{2}[-\s]?\d{2}[-\s]?(?:19|20)\d{2}[-\s]?\d{7})\b")
GSTIN_RE = re.compile(r"\b(\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d])\b")
