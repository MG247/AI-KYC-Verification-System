"""Aadhaar (UIDAI) validator.

UIDAI Aadhaar numbers are 12 digits with a Verhoeff checksum on the
last digit. We only validate the FULL 12-digit number when present —
if only the last 4 are available (the masked form the OCR pipeline
deliberately retains for privacy), we report ``checksum_unavailable``
rather than failing.

Reference: UIDAI's Aadhaar verification spec uses Verhoeff's dihedral
group D5 algorithm (Dihedral / multiplication / permutation tables).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# --- Verhoeff tables ------------------------------------------------------
_D = (
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

_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def _verhoeff_ok(digits: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


@dataclass(frozen=True)
class AadhaarValidation:
    valid: bool
    normalized: Optional[str]      # 12-digit, no spaces, or just last4
    last4: Optional[str]
    checksum_ok: Optional[bool]    # None when only last4 present
    reason: Optional[str]


def validate_aadhaar(value: str) -> AadhaarValidation:
    if not value:
        return AadhaarValidation(False, None, None, None, "empty Aadhaar")
    digits = re.sub(r"\D", "", value)
    if len(digits) == 12:
        if digits.startswith("0") or digits.startswith("1"):
            return AadhaarValidation(False, digits, digits[-4:], False,
                                     "first digit must be 2-9 per UIDAI spec")
        ok = _verhoeff_ok(digits)
        return AadhaarValidation(
            valid=ok, normalized=digits, last4=digits[-4:],
            checksum_ok=ok,
            reason=None if ok else "Verhoeff checksum failed",
        )
    if len(digits) == 4:
        return AadhaarValidation(True, digits, digits, None, "only last4 available; checksum skipped")
    return AadhaarValidation(False, digits or None, None, None,
                             f"expected 12 digits (or last 4), got {len(digits)}")
