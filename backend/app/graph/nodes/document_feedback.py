"""Node - document feedback summarizer.

Translates the raw pipeline signals into user-facing structured findings:
"is this AI-generated?", "does the filename match what we detected?",
"does the metadata look tampered?", "is the image too poor to read?",
"do the fields match the user's claim?". Each finding has a category, a
severity (LOW/MEDIUM/HIGH/CRITICAL), and a one-line message a compliance
reviewer (or end-user-facing UI) can show without further interpretation.

Why a separate node
-------------------
* Keeps ``decision_node`` lean and policy-focused (APPROVE/REVIEW/REJECT).
* Feedback is derived purely from already-computed state - no new I/O,
  no LLM calls. Cheap and deterministic.
* Sits between ``final_review`` and ``decision_node`` so the decision
  node can read the rolled-up ``document_status``.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List

from app.graph.state import KYCState

# Re-use the same filename hint list the forensic detector uses, plus a
# couple of generic "fake" tokens we'd never want in a legit upload.
_FILENAME_AI_HINTS = (
    "ai_", "_ai", "ai-gen", "generated", "diffusion", "synth",
    "midjourney", "stable_diff", "dalle", "fake",
)


def _filename_doc_hint(name: str) -> str | None:
    """Cheap heuristic: what doc type does the filename look like?"""
    n = name.lower()
    if "aadhaar" in n or "aadhar" in n: return "AADHAAR"
    if "pan" in n:                       return "PAN"
    if "passport" in n:                  return "PASSPORT"
    if "dl" in n or "driving" in n:      return "DRIVING_LICENSE"
    if "gst" in n:                       return "GST_CERTIFICATE"
    return None


def _filename_has_ai_hint(name: str) -> List[str]:
    n = name.lower()
    return [tag for tag in _FILENAME_AI_HINTS if tag in n]


def _add(findings: List[Dict[str, Any]], category: str, severity: str, message: str, **extra: Any) -> None:
    findings.append({"category": category, "severity": severity, "message": message, **extra})


def document_feedback(state: KYCState) -> Dict[str, Any]:
    """Build human-readable findings + a rolled-up ``document_status``."""
    started = time.perf_counter()
    findings: List[Dict[str, Any]] = []

    fraud_verdict   = str(state.get("fraud_verdict", "INSUFFICIENT_DATA"))
    fraud_score     = float(state.get("fraud_score") or 0.0)
    validation_overall = str(state.get("validation_overall", "MISSING"))
    risk_band       = str(state.get("risk_band", "HIGH"))
    hallu           = float(state.get("hallucination_score") or 0.0)
    metadata        = state.get("metadata_results") or {}
    preprocess      = state.get("preprocess_report") or {}
    extracted       = state.get("extracted_data") or {}
    dynamic         = state.get("dynamic_extracted_data") or {}
    generic         = state.get("generic_document_analysis") or {}
    detected_type   = str(state.get("document_type", "UNKNOWN"))
    detected_label  = str(state.get("document_type_label") or detected_type)
    hinted_type     = state.get("document_type_hint")
    document_path   = str(state.get("document_path") or "")
    fname           = Path(document_path).name if document_path else ""

    # ---- AI-generated / fake document --------------------------------------
    if fraud_verdict == "LIKELY_AI":
        _add(findings, "AI_GENERATED", "CRITICAL",
             f"Forensic signals indicate this document was AI-generated "
             f"(score={fraud_score:.2f}).",
             fraud_score=fraud_score,
             triggered=state.get("fraud_reasons") or [])
    elif fraud_verdict == "SUSPICIOUS":
        _add(findings, "POSSIBLE_FORGERY", "HIGH",
             f"Forensic signals suspicious but not conclusive "
             f"(score={fraud_score:.2f}). Manual review recommended.",
             fraud_score=fraud_score)

    # ---- Metadata tampering / editor history -------------------------------
    triggered_md = metadata.get("triggered_codes") or []
    if "METADATA" in triggered_md:
        _add(findings, "METADATA_ERROR", "HIGH",
             "EXIF metadata names a known image generator or aggressive editor.",
             signals=triggered_md)
    if "JPEG_HISTORY" in triggered_md:
        _add(findings, "METADATA_ERROR", "MEDIUM",
             "JPEG quantization history is inconsistent with a captured photo "
             "(e.g. PNG / re-saved file masquerading as a phone snap).",
             signals=triggered_md)

    # ---- Filename red flags ------------------------------------------------
    if fname:
        ai_hints = _filename_has_ai_hint(fname)
        if ai_hints:
            _add(findings, "FILENAME_ERROR", "HIGH",
                 f"Filename contains AI/forgery tokens: {ai_hints}",
                 filename=fname, tokens=ai_hints)
        fname_type = _filename_doc_hint(fname)
        if fname_type and detected_type != "UNKNOWN" and fname_type != detected_type:
            _add(findings, "FILENAME_ERROR", "MEDIUM",
                 f"Filename suggests {fname_type} but OCR detected {detected_type}.",
                 filename=fname, filename_suggests=fname_type, detected=detected_type)

    # ---- Hint vs detected mismatch -----------------------------------------
    if hinted_type and detected_type != "UNKNOWN" and str(hinted_type).upper() != detected_type:
        _add(findings, "DOCUMENT_TYPE_MISMATCH", "MEDIUM",
             f"Caller hinted {hinted_type} but the document classifies as {detected_type}.",
             hinted=str(hinted_type), detected=detected_type)

    if validation_overall == "NOT_APPLICABLE" or dynamic:
        _add(findings, "DYNAMIC_DOCUMENT", "MEDIUM",
             f"Document appears to be {detected_label}; dynamic JSON extraction was used for review.",
             detected=detected_type,
             label=detected_label,
             dynamic_field_count=len(dynamic))
    for concern in generic.get("visual_concerns") or []:
        _add(findings, "VISUAL_REVIEW", "MEDIUM", str(concern)[:240])

    # ---- Image quality -----------------------------------------------------
    if preprocess.get("blank"):
        _add(findings, "LOW_QUALITY", "CRITICAL", "Image appears blank or unreadable.")
    if preprocess.get("too_small"):
        _add(findings, "LOW_QUALITY", "HIGH",
             f"Image resolution too low ({preprocess.get('width')}x{preprocess.get('height')}).")
    if preprocess.get("blurry"):
        _add(findings, "LOW_QUALITY", "MEDIUM",
             f"Image is blurry (focus variance {preprocess.get('focus_variance')}). "
             "OCR confidence may be unreliable.")
    if preprocess.get("glare"):
        _add(findings, "LOW_QUALITY", "LOW",
             f"Glare/reflection covers {preprocess.get('glare_fraction', 0):.1%} of the image.")

    # ---- Field-level outcomes ----------------------------------------------
    if validation_overall == "FAIL":
        mismatched = []
        for c in (state.get("validation_results") or {}).get("checks", []) or []:
            if c.get("required") and c.get("status") == "FAIL":
                mismatched.append(c.get("field"))
        _add(findings, "FIELDS_MISMATCH", "HIGH",
             f"Required fields do not match user data: {mismatched or 'see validation_results'}.",
             mismatched=mismatched)
    elif validation_overall == "MISSING":
        missing = []
        for c in (state.get("validation_results") or {}).get("checks", []) or []:
            if c.get("required") and c.get("status") == "MISSING":
                missing.append(c.get("field"))
        _add(findings, "FIELDS_MISSING", "MEDIUM",
             f"Could not extract required fields: {missing or 'see validation_results'}.",
             missing=missing)

    # ---- Hallucination / LLM drift ----------------------------------------
    if hallu >= 0.5:
        _add(findings, "OCR_HALLUCINATION", "HIGH",
             f"LLM rectifications drift far from raw OCR (score={hallu:.2f}). "
             "Likely fabricated values rather than re-extraction.",
             hallucination_score=hallu)
    elif hallu >= 0.3:
        _add(findings, "OCR_HALLUCINATION", "MEDIUM",
             f"Some LLM-rectified values are not substring-grounded in raw OCR "
             f"(score={hallu:.2f}).",
             hallucination_score=hallu)

    # ---- Roll up to a single status ---------------------------------------
    status = _rollup_status(findings, validation_overall, risk_band, fraud_verdict)

    return {
        "document_feedback": findings,
        "document_status": status,
        "audit_logs": [{
            "node": "document_feedback",
            "status": status,
            "categories": sorted({f["category"] for f in findings}),
        }],
        "timings": {"feedback": time.perf_counter() - started},
    }


def _rollup_status(
    findings: List[Dict[str, Any]],
    validation_overall: str,
    risk_band: str,
    fraud_verdict: str,
) -> str:
    """Single-word verdict for UIs and dashboards.

    Priority: FAKE_AI > TAMPERED > FILENAME_SUSPECT > LOW_QUALITY >
    DATA_MISMATCH > NEEDS_REVIEW > AUTHENTIC."""
    cats = {f["category"] for f in findings}
    sevs = {f["category"]: f["severity"] for f in findings}

    if "AI_GENERATED" in cats:
        return "FAKE_AI"
    if "POSSIBLE_FORGERY" in cats or sevs.get("METADATA_ERROR") == "HIGH":
        return "TAMPERED"
    if any(f["category"] == "FILENAME_ERROR" and f["severity"] in {"HIGH", "CRITICAL"} for f in findings):
        return "FILENAME_SUSPECT"
    if any(f["category"] == "LOW_QUALITY" and f["severity"] in {"HIGH", "CRITICAL"} for f in findings):
        return "LOW_QUALITY"
    if "FIELDS_MISMATCH" in cats or "OCR_HALLUCINATION" in cats:
        return "DATA_MISMATCH"
    if validation_overall == "PASS" and risk_band == "LOW" and fraud_verdict == "LIKELY_AUTHENTIC":
        return "AUTHENTIC"
    return "NEEDS_REVIEW"
