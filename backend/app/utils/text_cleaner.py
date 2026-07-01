"""Enterprise OCR text-cleaning and validation utilities.

A PaddleOCR pass over a glossy KYC card yields three classes of output:

  1. **Signal**  — real fields ("Permanent Account Number", "ERHPG7459D",
     "Mohit Gautam", "10/08/2006").
  2. **Hologram / watermark noise** — fragments like "EaRHIST", "3TTU",
     "fatrT", Devanagari ligatures misread as Latin.
  3. **Symbol artifacts** — punctuation soup ("$$##", "....."), single
     characters, glare specks reported as 1–2 char tokens.

The functions in this module sit between PaddleOCR and the field
extractor. Each is pure, side-effect free, and individually testable so
the cleaning policy is auditable for compliance.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

# Whitelisted high-value patterns — text matching ANY of these is always kept
# regardless of confidence or length heuristics, because losing a real PAN is
# far worse than retaining a marginally noisy line.
_KYC_KEEP_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),                       # PAN
    re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),                       # Aadhaar
    re.compile(r"\b[A-PR-WY][0-9]{7}\b"),                           # Passport
    re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]\b"),      # GSTIN
    re.compile(r"\b\d{2}[\/\-\.](?:0[1-9]|1[0-2]|[A-Za-z]{3})[\/\-\.]\d{2,4}\b"),  # DOB
    re.compile(r"\b(?:MALE|FEMALE|TRANSGENDER)\b", re.I),
)

# Keyword anchors that mark a line as KYC-relevant context even if short.
_KYC_KEYWORDS: Tuple[str, ...] = (
    "PAN", "AADHAAR", "AADHAR", "PERMANENT", "ACCOUNT", "NAME", "FATHER",
    "DATE OF BIRTH", "DOB", "GENDER", "ADDRESS", "GOVT", "GOVERNMENT",
    "INDIA", "INCOME TAX", "DEPARTMENT", "UNIQUE IDENTIFICATION",
    "REPUBLIC", "PASSPORT", "DRIVING", "LICENCE", "LICENSE",
    "GSTIN", "ENROLMENT", "ENROLLMENT", "ISSUED",
)

# Devanagari range — anything mostly in this block on an English run is
# almost certainly a glyph misread (Hindi labels on PAN/Aadhaar cards).
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
_DIGIT_RE = re.compile(r"\d")
_VOWEL_RE = re.compile(r"[AEIOUaeiou]")
# Mojibake/devanagari latinizations PaddleOCR is known to emit on Indian IDs.
_GIBBERISH_RE = re.compile(
    r"^(?:[A-Za-z]*[0-9]+[A-Za-z]*[0-9]+[A-Za-z]*|[a-z]{2,4}[A-Z]{1,2}[a-z]{1,3})$"
)


# ---------------------------------------------------------------------------
# Reasons (for audit logging)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RejectionReason:
    code: str
    detail: str


class RejectCodes:
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    TOO_SHORT = "TOO_SHORT"
    SYMBOL_HEAVY = "SYMBOL_HEAVY"
    NO_LETTERS = "NO_LETTERS"
    NO_VOWELS = "NO_VOWELS"
    DEVANAGARI = "DEVANAGARI"
    GIBBERISH = "GIBBERISH"
    DUPLICATE = "DUPLICATE"


# ---------------------------------------------------------------------------
# Cleaning primitives
# ---------------------------------------------------------------------------
def clean_ocr_text(raw: str) -> str:
    """Normalize a single OCR token.

    * NFKC unicode normalization (full-width digits → ASCII).
    * Strip control characters and zero-width spaces.
    * Collapse repeated whitespace.
    * Fix common OCR substitutions inside obvious numeric runs (O→0, l→1)
      ONLY when surrounded by digits, to avoid corrupting real words.
    """
    if not raw:
        return ""
    s = unicodedata.normalize("NFKC", raw)
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")
    s = re.sub(r"\s+", " ", s).strip()
    # In a digit-dominant run, fold common confusions (5-line context):
    s = re.sub(r"(?<=\d)[Oo](?=\d)", "0", s)
    s = re.sub(r"(?<=\d)[lI](?=\d)", "1", s)
    s = re.sub(r"(?<=\d)[B](?=\d)", "8", s)
    return s


def matches_kyc_pattern(text: str) -> bool:
    """True if the line contains a regex-confirmed KYC identifier."""
    return any(p.search(text) for p in _KYC_KEEP_PATTERNS)


def has_kyc_keyword(text: str) -> bool:
    up = text.upper()
    return any(k in up for k in _KYC_KEYWORDS)


def validate_ocr_line(
    text: str,
    confidence: float,
    *,
    min_confidence: float = 0.50,
    min_length: int = 3,
    max_symbol_ratio: float = 0.40,
) -> Optional[RejectionReason]:
    """Return ``None`` when the line should be kept, else a structured reason.

    KYC identifiers (PAN/Aadhaar/DOB) ALWAYS pass regardless of heuristics —
    they're the only thing that matters and losing them to over-filtering is
    a compliance incident.
    """
    if matches_kyc_pattern(text):
        return None  # never reject a confirmed identifier

    # Confidence floor — but be lenient for keyword-bearing context.
    keyword_anchor = has_kyc_keyword(text)
    floor = min_confidence * (0.75 if keyword_anchor else 1.0)
    if confidence < floor:
        return RejectionReason(RejectCodes.LOW_CONFIDENCE, f"{confidence:.2f}<{floor:.2f}")

    if len(text) < min_length and not keyword_anchor:
        return RejectionReason(RejectCodes.TOO_SHORT, f"len={len(text)}")

    letters = len(_LATIN_LETTER_RE.findall(text))
    digits = len(_DIGIT_RE.findall(text))
    total = max(1, len(text.replace(" ", "")))
    symbols = total - letters - digits
    if symbols / total > max_symbol_ratio and not keyword_anchor:
        return RejectionReason(RejectCodes.SYMBOL_HEAVY,
                               f"sym={symbols}/{total}")

    if letters == 0 and digits == 0:
        return RejectionReason(RejectCodes.NO_LETTERS, "punct only")

    # Devanagari-heavy on an English-language run = hologram/Hindi label noise.
    dev_count = len(_DEVANAGARI_RE.findall(text))
    if dev_count and dev_count >= letters:
        return RejectionReason(RejectCodes.DEVANAGARI, f"dev={dev_count}")

    # Pure-letter words with no vowels and 3-8 chars are almost always
    # misread holograms ("fatrT", "EaRHIST", "3TTU" once digits stripped).
    if (letters and digits == 0 and 3 <= letters <= 8
            and not _VOWEL_RE.search(text) and not keyword_anchor):
        return RejectionReason(RejectCodes.NO_VOWELS, text)

    if _GIBBERISH_RE.match(text) and not keyword_anchor:
        return RejectionReason(RejectCodes.GIBBERISH, text)

    return None


# ---------------------------------------------------------------------------
# Spatial helpers — top-to-bottom, left-to-right with row-banding
# ---------------------------------------------------------------------------
def _bbox_center_y(bbox: Sequence[Sequence[float]]) -> float:
    return float(sum(p[1] for p in bbox) / len(bbox))


def _bbox_left_x(bbox: Sequence[Sequence[float]]) -> float:
    return float(min(p[0] for p in bbox))


def _bbox_height(bbox: Sequence[Sequence[float]]) -> float:
    ys = [p[1] for p in bbox]
    return float(max(ys) - min(ys))


def order_reading(
    items: Sequence[Tuple[str, float, Sequence[Sequence[float]]]],
) -> List[int]:
    """Return indices reordered top-to-bottom, left-to-right.

    Lines whose vertical centers fall within half the median line height of
    each other are treated as the same row, then sorted left-to-right within
    that row. This stops decorative side-bar / hologram text from getting
    interleaved with the main field block.
    """
    if not items:
        return []
    heights = sorted(_bbox_height(b) for _, _, b in items)
    median_h = heights[len(heights) // 2] or 1.0
    band = max(8.0, median_h * 0.5)

    indexed = sorted(
        range(len(items)),
        key=lambda i: (_bbox_center_y(items[i][2]), _bbox_left_x(items[i][2])),
    )

    rows: List[List[int]] = []
    current_y: Optional[float] = None
    for idx in indexed:
        y = _bbox_center_y(items[idx][2])
        if current_y is None or abs(y - current_y) > band:
            rows.append([idx])
            current_y = y
        else:
            rows[-1].append(idx)
            current_y = (current_y + y) / 2

    out: List[int] = []
    for row in rows:
        row.sort(key=lambda i: _bbox_left_x(items[i][2]))
        out.extend(row)
    return out


# ---------------------------------------------------------------------------
# Bulk filter
# ---------------------------------------------------------------------------
@dataclass
class FilterStats:
    kept: int = 0
    rejected: int = 0
    reasons: List[Tuple[str, str, float]] = None  # (text, code, conf)

    def __post_init__(self) -> None:
        if self.reasons is None:
            self.reasons = []


def confidence_based_filter(
    items: Sequence[Tuple[str, float, Sequence[Sequence[float]]]],
    *,
    min_confidence: float,
    min_length: int = 3,
    max_symbol_ratio: float = 0.40,
    drop_duplicates: bool = True,
) -> Tuple[List[int], FilterStats]:
    """Bulk-validate a parsed OCR page.

    Returns the surviving indices (in spatial reading order) and a
    ``FilterStats`` with per-line rejection reasons for audit logs.
    """
    stats = FilterStats()
    surviving: List[int] = []
    seen: set = set()
    for idx in order_reading(items):
        text_raw, conf, _ = items[idx]
        text = clean_ocr_text(text_raw)
        if not text:
            stats.rejected += 1
            stats.reasons.append((text_raw, RejectCodes.TOO_SHORT, conf))
            continue
        reason = validate_ocr_line(
            text,
            conf,
            min_confidence=min_confidence,
            min_length=min_length,
            max_symbol_ratio=max_symbol_ratio,
        )
        if reason is not None:
            stats.rejected += 1
            stats.reasons.append((text, reason.code, conf))
            continue
        if drop_duplicates:
            key = (text.upper(), round(_bbox_center_y(items[idx][2]) / 10))
            if key in seen:
                stats.rejected += 1
                stats.reasons.append((text, RejectCodes.DUPLICATE, conf))
                continue
            seen.add(key)
        surviving.append(idx)
        stats.kept += 1
    return surviving, stats


# ---------------------------------------------------------------------------
# Field-specific helpers
# ---------------------------------------------------------------------------
_NAME_BLOCKLIST = {
    "INCOME", "TAX", "DEPARTMENT", "GOVT", "GOVERNMENT", "INDIA",
    "PERMANENT", "ACCOUNT", "NUMBER", "CARD", "AADHAAR", "AADHAR",
    "UNIQUE", "IDENTIFICATION", "AUTHORITY", "REPUBLIC", "DOB",
    "BIRTH", "DATE", "FATHER", "GENDER", "MALE", "FEMALE", "SIGNATURE",
    "VID", "ENROLMENT", "ENROLLMENT", "ADDRESS",
}


def is_plausible_person_name(text: str) -> bool:
    """A 2–4 word, all-letters, no-keyword line that looks like a name."""
    parts = text.strip().split()
    if not 2 <= len(parts) <= 4:
        return False
    if any(p.upper() in _NAME_BLOCKLIST for p in parts):
        return False
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z\.'-]{1,}", p) for p in parts):
        return False
    if any(_DIGIT_RE.search(p) for p in parts):
        return False
    return True


def select_name_candidate(lines: Iterable[str]) -> Optional[str]:
    """Heuristic: the first plausible-name line that appears AFTER a
    'Name'/'नाम' label (or, failing that, the first plausible name)."""
    seen_label = False
    fallback: Optional[str] = None
    for line in lines:
        up = line.upper()
        if "NAME" in up and len(up) < 32:
            seen_label = True
            continue
        if is_plausible_person_name(line):
            if seen_label:
                return _titlecase(line)
            if fallback is None:
                fallback = _titlecase(line)
    return fallback


def _titlecase(s: str) -> str:
    return " ".join(w.capitalize() for w in s.split())
