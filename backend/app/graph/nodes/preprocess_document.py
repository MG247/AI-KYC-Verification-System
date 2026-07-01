"""Node 0 - document quality gate.

Cheap pre-OCR image triage. Reads the image once, measures focus and
exposure characteristics, and writes a quality report to the state. The
OCR engine still does its own normalization (CLAHE, deskew, resize)
inside ``ocr_service`` so this node deliberately does NOT rewrite the
image - it only decides whether the document is usable and surfaces
why if not. Failing quality is not a hard reject; it adds an audit
breadcrumb and lowers the OCR-confidence prior the risk engine sees.

Why a separate node
-------------------
* Pulls quality signals out of the OCR engine so reviewers can see
  *why* an OCR result was weak (blurry input vs. font mangle).
* Gives downstream nodes a cheap short-circuit: e.g. metadata_analysis
  can skip its EXIF dive when the file is unreadable.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from app.graph.state import KYCState

logger = logging.getLogger(__name__)

# Tuned for KYC card scans on consumer phones. A pristine PAN scan lands
# around variance ~600; a phone-snap of a card on a wood table ~120; a
# motion-blurred capture ~30.
_BLUR_THRESHOLD = 80.0
# Fraction of pixels at full saturation. ID-card holograms produce a
# small bright patch (<3%); a flash glare blowout is >8%.
_GLARE_THRESHOLD = 0.08
# Mean luminance below this is "scanner cap left on" black; above 250 is
# blank white sheet.
_DARK_MEAN = 15.0
_BRIGHT_MEAN = 245.0


def _read_grayscale(path: str) -> np.ndarray | None:
    if not path or not Path(path).exists():
        return None
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    except Exception:  # noqa: BLE001
        return None
    return img


def _focus_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _glare_fraction(gray: np.ndarray) -> float:
    return float((gray >= 250).sum()) / float(gray.size or 1)


def preprocess_document(state: KYCState) -> Dict[str, Any]:
    started = time.perf_counter()
    path = state.get("document_path") or ""
    report: Dict[str, Any] = {
        "checked": False, "blurry": False, "glare": False,
        "blank": False, "too_small": False,
        "focus_variance": 0.0, "glare_fraction": 0.0,
        "mean_luminance": 0.0, "width": 0, "height": 0,
    }
    errors: list[str] = []

    gray = _read_grayscale(path)
    if gray is None:
        # PDFs and unreadable inputs fall through — OCR will handle the
        # PDF case via pdf utils; we just record that we couldn't peek.
        report["reason"] = "not a readable image (likely PDF or missing file)"
        return {
            "preprocessed_path": path,
            "preprocess_report": report,
            "timings": {"preprocess": time.perf_counter() - started},
        }

    h, w = gray.shape[:2]
    focus = _focus_variance(gray)
    glare = _glare_fraction(gray)
    mean_lum = float(gray.mean())

    report.update({
        "checked": True,
        "focus_variance": round(focus, 2),
        "glare_fraction": round(glare, 4),
        "mean_luminance": round(mean_lum, 2),
        "width": int(w),
        "height": int(h),
        "blurry": focus < _BLUR_THRESHOLD,
        "glare": glare > _GLARE_THRESHOLD,
        "blank": (mean_lum < _DARK_MEAN) or (mean_lum > _BRIGHT_MEAN),
        "too_small": min(h, w) < 300,
    })

    if report["blank"]:
        errors.append("preprocess: image appears blank")
    if report["too_small"]:
        errors.append(f"preprocess: image too small ({w}x{h})")
    if report["blurry"]:
        logger.info("preprocess: low focus variance %.1f (threshold %.1f)", focus, _BLUR_THRESHOLD)
    if report["glare"]:
        logger.info("preprocess: high glare fraction %.3f", glare)

    return {
        "preprocessed_path": path,
        "preprocess_report": report,
        "errors": errors,
        "audit_logs": [{"node": "preprocess_document", **report}],
        "timings": {"preprocess": time.perf_counter() - started},
    }
