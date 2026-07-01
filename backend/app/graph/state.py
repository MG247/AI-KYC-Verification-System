"""LangGraph shared state for the KYC pipeline.

State design notes
------------------
* ``total=False`` because no single node sets every key; each node returns
  a partial dict that LangGraph merges into the master state.
* The pipeline fans out at one point — ``validate_fields``, ``detect_fraud``
  and ``metadata_analysis`` all run on the post-OCR state in parallel. Two
  parallel nodes that both return ``{"errors": [...]}`` would normally race
  and the last write would clobber the other. To keep parallel updates
  safe we declare ``Annotated[..., reducer]`` for every list/dict field
  more than one node may touch (errors, timings, audit_logs,
  decision_reasons, fraud_reasons, llm_concerns, llm_rectifications).
* Risk bands now include ``CRITICAL`` (>= 0.85) so the decision layer can
  separate "auto-reject" from "high but maybe rectifiable".
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


def _merge_dict(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow-merge two dicts. Right wins on key collision.

    Used as the reducer for ``timings`` so parallel nodes can each contribute
    their own per-node timing without overwriting siblings."""
    if not left:
        return dict(right or {})
    if not right:
        return dict(left)
    merged = dict(left)
    merged.update(right)
    return merged


class KYCState(TypedDict, total=False):
    # ---- inputs ----
    request_id: str
    user_data: Dict[str, Any]
    document_path: str
    document_type_hint: Optional[str]
    ocr_engine: str

    # ---- preprocessing ----
    preprocessed_path: Optional[str]      # may equal document_path if no rewrite needed
    preprocess_report: Dict[str, Any]     # blur / glare / dims / rotated / etc.

    # ---- ocr ----
    ocr_success: bool
    raw_ocr_text: str
    extracted_data: Dict[str, Any]
    typed_extracted_data: Dict[str, Any]   # per-doc-type fields ONLY (strict)
    corrected_fields: Dict[str, Any]      # post-rectification view of extracted_data
    ocr_confidence: float
    document_type: str
    document_type_label: str
    classification_confidence: float
    dynamic_extracted_data: Dict[str, Any]
    generic_document_analysis: Dict[str, Any]

    # ---- validation ----
    validation_results: Dict[str, Any]
    field_match_score: float
    validation_overall: str

    # ---- fraud / metadata (split into two parallel branches) ----
    fraud_score: float
    fraud_reasons: Annotated[List[str], operator.add]
    fraud_verdict: str
    metadata_results: Dict[str, Any]      # EXIF / JPEG history / filename signatures

    # ---- llm (Azure OpenAI second-opinion) ----
    llm_enabled: bool
    llm_skipped_reason: Optional[str]
    llm_rectifications: Annotated[List[Dict[str, Any]], operator.add]
    llm_recommendation: Optional[str]     # APPROVE | REVIEW | REJECT
    llm_confidence: Optional[float]
    llm_reasoning: Optional[str]
    llm_concerns: Annotated[List[str], operator.add]
    hallucination_score: float            # [0,1] — how much LLM-rectified output drifts from raw OCR

    # ---- risk ----
    risk_score: float
    risk_band: str                        # LOW | MEDIUM | HIGH | CRITICAL

    # ---- final ----
    final_decision: str                   # APPROVE | REVIEW | REJECT
    decision_reasons: Annotated[List[str], operator.add]
    document_feedback: Annotated[List[Dict[str, Any]], operator.add]
    document_status: str                  # AUTHENTIC | NEEDS_REVIEW | DATA_MISMATCH | LOW_QUALITY | FILENAME_SUSPECT | TAMPERED | FAKE_AI

    # ---- diagnostics / audit ----
    errors: Annotated[List[str], operator.add]
    timings: Annotated[Dict[str, float], _merge_dict]
    audit_logs: Annotated[List[Dict[str, Any]], operator.add]
    retry_count: int
    processing_time: float                # total wall-clock for the request


# Back-compat alias — older imports kept working through the refactor.
KycState = KYCState
