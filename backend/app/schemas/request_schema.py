"""API request schemas."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.ocr_schema import OCREngine


class KYCVerifyRequest(BaseModel):
    """Body for ``POST /v1/kyc/verify`` when sent as JSON.

    ``document_path`` is a server-readable path. The multipart variant
    of the endpoint takes the file directly and constructs the path
    internally; this schema is what the orchestrator consumes.
    """

    model_config = ConfigDict(extra="forbid")

    document_path: str = Field(..., description="Server-side path to the KYC document")
    user_data: Dict[str, Any] = Field(default_factory=dict, description="Ground-truth user profile")
    document_type_hint: Optional[str] = Field(
        default=None,
        description="Optional document type hint: PAN | AADHAAR | PASSPORT | DRIVING_LICENSE | GST_CERTIFICATE",
    )
    ocr_engine: OCREngine = Field(
        default=OCREngine.PADDLE,
        description="OCR engine: paddle | gpt_vision",
    )
    request_id: Optional[str] = None


class AIDetectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_path: str
