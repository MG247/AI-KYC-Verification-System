"""API response schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class KYCVerifyResponse(BaseModel):
    """Result of the full KYC pipeline. Mirrors the LangGraph state."""

    request_id: str
    document_type: str
    document_type_label: Optional[str] = None
    ocr_engine: str = "paddle"
    classification_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    final_decision: str
    risk_band: str
    risk_score: float = Field(ge=0.0, le=1.0)
    validation_overall: str
    field_match_score: float = Field(ge=0.0, le=1.0)
    fraud_verdict: str
    fraud_score: float = Field(ge=0.0, le=1.0)
    ocr_confidence: float = Field(ge=0.0, le=1.0)

    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    typed_extracted_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Strict per-doc-type extraction: only the fields valid for the classified document type.",
    )
    corrected_fields: Dict[str, Any] = Field(default_factory=dict)
    dynamic_extracted_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dynamic key-value JSON for documents outside the fixed KYC schemas.",
    )
    generic_document_analysis: Dict[str, Any] = Field(default_factory=dict)
    document_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Final view of what's on the document: typed_extracted_data with corrected_fields applied on top.",
    )
    validation_results: Dict[str, Any] = Field(default_factory=dict)
    fraud_reasons: List[str] = Field(default_factory=list)
    metadata_results: Dict[str, Any] = Field(default_factory=dict)
    preprocess_report: Dict[str, Any] = Field(default_factory=dict)
    decision_reasons: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    timings: Dict[str, float] = Field(default_factory=dict)
    processing_time: float = 0.0
    audit_logs: List[Dict[str, Any]] = Field(default_factory=list)
    hallucination_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # ---- document-level feedback ----
    document_status: str = "NEEDS_REVIEW"  # AUTHENTIC | NEEDS_REVIEW | DATA_MISMATCH | LOW_QUALITY | FILENAME_SUSPECT | TAMPERED | FAKE_AI
    document_feedback: List[Dict[str, Any]] = Field(default_factory=list)

    # ---- LLM second-opinion (only populated when Azure OpenAI is configured) ----
    llm_enabled: bool = False
    llm_recommendation: Optional[str] = None
    llm_confidence: Optional[float] = None
    llm_reasoning: Optional[str] = None
    llm_concerns: List[str] = Field(default_factory=list)
    llm_rectifications: List[Dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    ocr: Dict[str, Any]
    llm: Dict[str, Any] = Field(default_factory=dict)
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
