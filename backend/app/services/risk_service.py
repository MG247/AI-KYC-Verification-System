"""Risk-scoring service.

Single source of truth for how individual signals combine into a final
risk number. Kept tiny and explicit so policy changes are auditable -
no learned model in here, just a documented weighted sum.

Formula
-------
    risk = w1 * (1 - field_match_score)        # how poorly fields match
         + w2 * fraud_score                    # AI-synthesis suspicion
         + w3 * (1 - ocr_confidence)           # how unreliable the OCR was
         + w4 * hallucination_score            # how much LLM drifted from raw OCR
         + w5 * metadata_suspicion             # EXIF / JPEG-history suspicion

Hard overrides
--------------
* ``fraud_verdict == LIKELY_AI``      -> risk clamped to 0.95 (force REJECT)
* ``validation_overall == FAIL``      -> risk clamped to >= 0.75
* ``validation_overall == MISSING``   -> risk clamped to >= 0.45
* ``hallucination_score >= 0.5``      -> risk clamped to >= 0.65

Bands
-----
* LOW       < 0.30
* MEDIUM   0.30 .. 0.60
* HIGH     0.60 .. 0.85
* CRITICAL >= 0.85
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class RiskWeights:
    field_mismatch: float = 0.35
    fraud: float = 0.30
    ocr_uncertainty: float = 0.10
    hallucination: float = 0.15
    metadata: float = 0.10


class RiskService:
    LOW_BAND = 0.30
    HIGH_BAND = 0.60
    CRITICAL_BAND = 0.85

    def __init__(self, weights: Optional[RiskWeights] = None) -> None:
        self._w = weights or RiskWeights()

    def score(
        self,
        *,
        field_match_score: float,
        fraud_score: float,
        ocr_confidence: float,
        validation_overall: str,
        fraud_verdict: str,
        hallucination_score: float = 0.0,
        metadata_suspicion: float = 0.0,
    ) -> Tuple[float, str, List[str]]:
        reasons: List[str] = []
        if validation_overall == "NOT_APPLICABLE":
            fm = 0.0
        else:
            fm = 1.0 - _clip01(field_match_score)
        fr = _clip01(fraud_score)
        oc = 1.0 - _clip01(ocr_confidence)
        hl = _clip01(hallucination_score)
        md = _clip01(metadata_suspicion)

        risk = (
            self._w.field_mismatch * fm
            + self._w.fraud * fr
            + self._w.ocr_uncertainty * oc
            + self._w.hallucination * hl
            + self._w.metadata * md
        )
        reasons.append(
            f"weighted risk = {risk:.2f} "
            f"(field_mismatch={fm:.2f}, fraud={fr:.2f}, "
            f"ocr_uncertainty={oc:.2f}, hallucination={hl:.2f}, metadata={md:.2f})"
        )
        if validation_overall == "NOT_APPLICABLE":
            reasons.append("field validation not applicable for this document type")

        if fraud_verdict == "LIKELY_AI":
            risk = max(risk, 0.95)
            reasons.append("override: fraud verdict LIKELY_AI -> risk clamped to 0.95")
        if validation_overall == "FAIL":
            risk = max(risk, 0.75)
            reasons.append("override: validation FAIL -> risk clamped to >=0.75")
        elif validation_overall == "MISSING":
            risk = max(risk, 0.45)
            reasons.append("override: validation MISSING -> risk clamped to >=0.45")
        if hl >= 0.5:
            risk = max(risk, 0.65)
            reasons.append(f"override: hallucination={hl:.2f} -> risk clamped to >=0.65")

        risk = _clip01(risk)
        band = self._band(risk)
        return risk, band, reasons

    @classmethod
    def _band(cls, risk: float) -> str:
        if risk >= cls.CRITICAL_BAND:
            return "CRITICAL"
        if risk >= cls.HIGH_BAND:
            return "HIGH"
        if risk >= cls.LOW_BAND:
            return "MEDIUM"
        return "LOW"


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


_lock = threading.Lock()
_svc: Optional[RiskService] = None


def get_risk_service() -> RiskService:
    global _svc
    if _svc is None:
        with _lock:
            if _svc is None:
                _svc = RiskService()
    return _svc
