"""Generic document analysis for documents outside fixed KYC schemas.

Known KYC documents use deterministic extractors. Everything else can still
be useful to an operator: identify the apparent document type, extract a
dynamic key-value JSON object, and capture any visual concerns from GPT
Vision when available.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import mimetypes
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import AzureOpenAISettings, get_azure_settings
from app.schemas.ocr_schema import GenericDocumentAnalysis
from app.services.llm_reasoner import _build_http_client
from app.utils.pdf_utils import encode_image_to_bytes, render_pdf

logger = logging.getLogger(__name__)


_SYSTEM_GENERIC = """You analyze business/KYC-adjacent documents for a fintech review console.
Return ONLY a JSON object. Do not invent values. Preserve exact text values when readable.

Output schema:
{
  "document_type_label": "short human document type, e.g. Electricity Bill, Bank Statement, Invoice",
  "document_category": "one of KYC_ID, FINANCIAL, ADDRESS_PROOF, BUSINESS_PROOF, TAX, CONTRACT, OTHER",
  "confidence": 0.0,
  "raw_text": "readable text from the document, line separated",
  "dynamic_fields": {"field_name": "value", "nested_or_repeating_data": []},
  "summary": "one sentence summary of what this document appears to show",
  "visual_concerns": ["visible concern about tampering/editing/AI generation, if any"],
  "extraction_notes": ["short notes about uncertainty, unreadable areas, or omitted fields"]
}

Rules:
- Build dynamic_fields from the document itself; choose field names that match the document.
- Include dates, IDs, names, addresses, amounts, account/customer numbers, issuer names, and totals when visible.
- If this is a known KYC ID, still return dynamic_fields, but do not fabricate missing values.
- Visual concerns are advisory only; do not claim a final forensic verdict.
"""


_GENERIC_PATTERNS: Tuple[Tuple[str, str, float, Tuple[str, ...]], ...] = (
    ("Bank Statement", "FINANCIAL", 0.78, ("statement of account", "account statement", "ifsc", "transaction", "closing balance")),
    ("Salary Slip", "FINANCIAL", 0.78, ("salary slip", "pay slip", "earnings", "deductions", "net pay")),
    ("Invoice", "BUSINESS_PROOF", 0.72, ("tax invoice", "invoice no", "invoice number", "bill to", "total amount")),
    ("Utility Bill", "ADDRESS_PROOF", 0.72, ("electricity bill", "water bill", "gas bill", "consumer number", "bill amount")),
    ("Rent Agreement", "ADDRESS_PROOF", 0.72, ("rent agreement", "lease agreement", "lessor", "lessee", "tenant")),
    ("Udyam Registration", "BUSINESS_PROOF", 0.80, ("udyam registration", "udyam", "enterprise type", "msme")),
    ("Shop Establishment Certificate", "BUSINESS_PROOF", 0.74, ("shop", "establishment", "certificate of registration")),
    ("Cheque", "FINANCIAL", 0.70, ("pay", "rupees", "ifsc", "micr", "account payee")),
)

_KV_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9\s/&().,#-]{1,52})\s*[:\-]\s*(.{2,180})\s*$")
_WS_RE = re.compile(r"\s+")


class GenericDocumentAnalyzer:
    def __init__(self, settings: Optional[AzureOpenAISettings] = None) -> None:
        self._settings = settings or get_azure_settings()
        self._client = None
        self._client_lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._settings.is_configured

    def analyze_text(self, raw_text: str, *, hint: Optional[str] = None, timeout_s: float = 12.0) -> GenericDocumentAnalysis:
        fallback = _heuristic_analysis(raw_text, hint=hint)
        if not self.enabled or not raw_text.strip():
            return fallback

        payload = {
            "hint": hint,
            "raw_ocr_text": raw_text[:9000],
        }
        data = self._chat_json(
            messages=[
                {"role": "system", "content": _SYSTEM_GENERIC},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            label="generic_text_analysis",
            timeout_s=timeout_s,
        )
        return _coerce_analysis(data, fallback=fallback, source="gpt_text") if data else fallback

    def analyze_with_vision(
        self,
        file_path: str | Path,
        *,
        hint: Optional[str] = None,
        require_vision: bool = False,
        timeout_s: float = 25.0,
    ) -> GenericDocumentAnalysis:
        path = Path(file_path)
        fallback = _heuristic_analysis("", hint=hint)
        if not self.enabled:
            if require_vision:
                raise RuntimeError("GPT Vision OCR requires configured Azure OpenAI credentials")
            return fallback

        images = _image_payloads(path)
        if not images:
            if require_vision:
                raise RuntimeError("GPT Vision OCR could not prepare an image payload")
            return fallback

        content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": json.dumps({"hint": hint, "file_name": path.name}, ensure_ascii=False),
            }
        ]
        for mime, b64 in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
            })

        data = self._chat_json(
            messages=[
                {"role": "system", "content": _SYSTEM_GENERIC},
                {"role": "user", "content": content},
            ],
            label="generic_vision_analysis",
            timeout_s=timeout_s,
        )
        if not data:
            if require_vision:
                raise RuntimeError("GPT Vision OCR returned no structured result")
            return fallback
        return _coerce_analysis(data, fallback=fallback, source="gpt_vision")

    def _client_or_none(self):
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                from openai import AzureOpenAI
            except ImportError as exc:
                logger.error("openai SDK missing: %s", exc)
                return None
            try:
                self._client = AzureOpenAI(
                    api_key=self._settings.api_key,
                    azure_endpoint=self._settings.endpoint,
                    api_version=self._settings.api_version,
                    http_client=_build_http_client(),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("AzureOpenAI client init failed: %s", exc)
                return None
            return self._client

    def _chat_json(self, *, messages: List[Dict[str, Any]], label: str, timeout_s: float) -> Optional[Dict[str, Any]]:
        client = self._client_or_none()
        if client is None or not self._settings.deployment:
            return None
        started = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=self._settings.deployment,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
                timeout=timeout_s,
            )
            content = (resp.choices[0].message.content or "").strip()
            logger.info("%s ok latency=%.2fs", label, time.perf_counter() - started)
            return json.loads(content) if content else None
        except json.JSONDecodeError as exc:
            logger.warning("%s returned non-JSON: %s", label, exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s failed: %s", label, exc)
            return None


def _heuristic_analysis(raw_text: str, *, hint: Optional[str] = None) -> GenericDocumentAnalysis:
    text = raw_text or ""
    lower = text.lower()
    label = _title_hint(hint) if hint else "Unknown Document"
    category = "OTHER"
    confidence = 0.20 if text.strip() else 0.0
    for candidate_label, candidate_category, candidate_conf, tokens in _GENERIC_PATTERNS:
        if any(token in lower for token in tokens):
            label = candidate_label
            category = candidate_category
            confidence = candidate_conf
            break

    fields = _extract_key_values(text)
    if not fields and text.strip():
        fields["text_preview"] = [line for line in _clean_lines(text)[:12]]

    summary = f"Document appears to be {label}." if label != "Unknown Document" else "Document type could not be confidently identified."
    return GenericDocumentAnalysis(
        document_type_label=label,
        document_category=category,
        confidence=confidence,
        dynamic_fields=fields,
        raw_text=text,
        summary=summary,
        extraction_notes=["Heuristic extraction used; verify fields manually."],
        source="heuristic",
    )


def _extract_key_values(raw_text: str) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    for line in _clean_lines(raw_text)[:100]:
        match = _KV_RE.match(line)
        if not match:
            continue
        key = _safe_key(match.group(1))
        value = match.group(2).strip(" :-\t")
        if not key or not value or len(value) > 240:
            continue
        if key in fields:
            existing = fields[key]
            if isinstance(existing, list):
                existing.append(value)
            else:
                fields[key] = [existing, value]
        else:
            fields[key] = value
    return fields


def _clean_lines(raw_text: str) -> List[str]:
    return [_WS_RE.sub(" ", line).strip() for line in raw_text.splitlines() if line.strip()]


def _safe_key(value: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    key = key.strip("_")[:64]
    if key in {"no", "date", "name", "address"}:
        return key
    return key if len(key) >= 3 else ""


def _coerce_analysis(data: Dict[str, Any], *, fallback: GenericDocumentAnalysis, source: str) -> GenericDocumentAnalysis:
    dynamic = data.get("dynamic_fields") if isinstance(data.get("dynamic_fields"), dict) else fallback.dynamic_fields
    raw_text = str(data.get("raw_text") or fallback.raw_text or "")
    try:
        confidence = float(data.get("confidence", fallback.confidence))
    except (TypeError, ValueError):
        confidence = fallback.confidence
    return GenericDocumentAnalysis(
        document_type_label=str(data.get("document_type_label") or fallback.document_type_label or "Unknown Document")[:120],
        document_category=str(data.get("document_category") or fallback.document_category or "OTHER")[:80],
        confidence=max(0.0, min(1.0, confidence)),
        dynamic_fields=dynamic,
        raw_text=raw_text,
        summary=str(data.get("summary") or fallback.summary or "")[:800],
        visual_concerns=[str(item)[:240] for item in data.get("visual_concerns", []) if item],
        extraction_notes=[str(item)[:240] for item in data.get("extraction_notes", []) if item],
        source=source,
    )


def _image_payloads(path: Path) -> List[Tuple[str, str]]:
    if path.suffix.lower() == ".pdf":
        rendered = render_pdf(path, max_pages=2)
        payloads = []
        for page in rendered.pages[:2]:
            payloads.append(("image/png", base64.b64encode(encode_image_to_bytes(page, fmt="PNG")).decode("ascii")))
        return payloads

    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((1800, 1800))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=90, optimize=True)
    except (UnidentifiedImageError, OSError) as exc:
        raise RuntimeError(f"cannot prepare image for GPT Vision: {exc}") from exc
    _ = mime
    return [("image/jpeg", base64.b64encode(buffer.getvalue()).decode("ascii"))]


def _title_hint(hint: Optional[str]) -> str:
    if not hint:
        return "Unknown Document"
    return str(hint).replace("_", " ").title()


_lock = threading.Lock()
_analyzer: Optional[GenericDocumentAnalyzer] = None


def get_generic_document_analyzer() -> GenericDocumentAnalyzer:
    global _analyzer
    if _analyzer is None:
        with _lock:
            if _analyzer is None:
                _analyzer = GenericDocumentAnalyzer()
    return _analyzer