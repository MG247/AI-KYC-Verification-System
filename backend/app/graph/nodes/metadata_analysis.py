"""Node - metadata analysis (parallel branch).

Runs in parallel with ``validate_fields`` and ``detect_fraud`` after OCR
completes. Re-uses the forensic detector to surface the *metadata-flavoured*
signals (EXIF / JPEG history / filename) separately from the pixel-level
fraud signals. This makes the audit trail clearer: a reviewer can tell at
a glance whether the suspicion came from the document's history (uploaded
file looks edited) or its content (the image itself looks generated).

Why a separate node
-------------------
* Splits one heavy fraud check into two independent state writes so the
  decision layer can reason about metadata vs. pixel-content risk
  separately - useful when a real document has stripped EXIF (perfectly
  innocent) but the pixels are clean.
* Runs in parallel with validation and pixel-fraud, so the wall-clock
  cost of the metadata pass is hidden under the OCR's tail latency.

Idempotent and side-effect-free: it never modifies the file, just reads
EXIF and JPEG markers.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from app.graph.state import KYCState
from app.services.ai_generator_detector import SignalCode, get_ai_detector

logger = logging.getLogger(__name__)

# Signals owned by this node. The pixel-content signals (FFT, ELA, noise,
# entropy, saturation, edge density) belong to ``detect_fraud``.
_METADATA_SIGNALS = {SignalCode.METADATA, SignalCode.JPEG_HISTORY}


def metadata_analysis(state: KYCState) -> Dict[str, Any]:
    started = time.perf_counter()
    path = state.get("document_path") or ""
    try:
        report = get_ai_detector().detect(path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("metadata analysis failed")
        return {
            "metadata_results": {"available": False, "reason": str(exc), "signals": []},
            "errors": [f"metadata: {exc}"],
            "audit_logs": [{"node": "metadata_analysis", "error": str(exc)}],
            "timings": {"metadata": time.perf_counter() - started},
        }

    md_signals = [s for s in report.signals if s.code in _METADATA_SIGNALS]
    triggered = [s for s in md_signals if s.triggered]
    summary = {
        "available": True,
        "file_name": report.file_name,
        "signals": [s.model_dump() for s in md_signals],
        "triggered_codes": [s.code.value for s in triggered],
        "suspicion": round(sum(s.score * s.weight for s in md_signals), 3),
    }

    audit = {
        "node": "metadata_analysis",
        "triggered": summary["triggered_codes"],
        "suspicion": summary["suspicion"],
    }
    return {
        "metadata_results": summary,
        "audit_logs": [audit],
        "timings": {"metadata": time.perf_counter() - started},
    }
