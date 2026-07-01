"""PAN format validator.

Indian PAN is a 10-character code: 5 letters + 4 digits + 1 letter.
The 4th character encodes the holder type (P=person, C=company,
H=HUF, F=firm, A=AOP, T=trust, B=BOI, L=local, J=artificial juridical,
G=government). We accept all of them but expose the parsed type so
callers can apply additional policy (e.g. retail KYC only allows P).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_PAN_RE = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
_HOLDER_TYPES = {
    "P": "individual", "C": "company", "H": "huf", "F": "firm",
    "A": "aop", "T": "trust", "B": "boi", "L": "local",
    "J": "artificial-juridical", "G": "government",
}


@dataclass(frozen=True)
class PANValidation:
    valid: bool
    normalized: Optional[str]
    holder_type: Optional[str]
    reason: Optional[str]


def validate_pan(pan: str) -> PANValidation:
    if not pan:
        return PANValidation(False, None, None, "empty PAN")
    norm = re.sub(r"\s+", "", pan).upper()
    if not _PAN_RE.match(norm):
        return PANValidation(False, norm, None, "format mismatch (expect AAAAA9999A)")
    holder = _HOLDER_TYPES.get(norm[3])
    if holder is None:
        return PANValidation(False, norm, None, f"unknown holder-type char '{norm[3]}'")
    return PANValidation(True, norm, holder, None)
