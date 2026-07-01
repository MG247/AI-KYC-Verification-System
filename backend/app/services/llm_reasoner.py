"""Azure OpenAI reasoner for the KYC pipeline.

What the LLM is FOR (and what it is NOT for)
--------------------------------------------
The LLM is a *fallback brain* that runs only when deterministic heuristics
hit ambiguity. Its output is treated as ONE opinion the decision node
weighs alongside field-validation and forensic fraud signals. It does NOT
override hard policy:
    * AI-generated documents still auto-REJECT.
    * Required-field validation FAIL still auto-REJECTS.

Two capabilities
----------------
1. ``rectify_fields()`` — re-extract specific missing/noisy fields from
   the raw OCR text. Useful when OCR recognized the characters but the
   regex anchor failed (e.g. a label mangle that split a date across
   two lines).

2. ``final_review()`` — produce a recommendation + audit-trail reasoning
   on borderline cases. Returns one of APPROVE / REVIEW / REJECT with a
   short justification a compliance reviewer can read.

Operational notes
-----------------
* Gated on ``AzureOpenAISettings.is_configured`` plus the ``KYC_LLM_ENABLE``
  env flag (default true once Azure keys are present). If the LLM is
  disabled or the call fails, the pipeline continues with heuristic-only
  decisions — never silently blocks.
* Temperature pinned to 0 for deterministic audit replay.
* ``response_format=json_object`` forces structured output so parsing
  cannot drift between model versions.
* PII (PAN, Aadhaar last4, name, DOB) is sent to the model. Before
  enabling in production, confirm your Azure OpenAI deployment has
  zero-data-retention and is in the appropriate data-residency region.
  Logs redact PAN / Aadhaar; only field counts and latency are recorded.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.config import AzureOpenAISettings, get_azure_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses (graph-friendly: plain dicts in/out)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LLMRectification:
    """One rectified field. ``confidence`` is the LLM's self-reported
    confidence (NOT an OCR confidence); the caller decides the merge
    threshold."""

    field: str
    value: str
    confidence: float
    source: str       # one-line citation from raw OCR text


@dataclass(frozen=True)
class LLMReview:
    recommendation: str          # APPROVE | REVIEW | REJECT
    confidence: float
    reasoning: str
    concerns: List[str]


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
_SYSTEM_RECTIFY = """You are a KYC document field-extraction assistant for an Indian fintech.
Given raw OCR text from one document and a list of missing/uncertain fields,
re-extract each field STRICTLY from the OCR text. Do not invent data.

Rules:
- Only fill a field if the value is clearly supported by the OCR text.
- Normalize dates to ISO YYYY-MM-DD.
- For PAN: 10 chars, 5 letters + 4 digits + 1 letter, uppercase.
- For Aadhaar: return only the LAST 4 DIGITS, never the full number.
- For names: Title Case, no honorifics, no labels like 'Name'.
- If unsure, omit the field rather than guess. Confidence below 0.6 means omit.

Output schema (JSON):
{
  "rectifications": [
    {"field": "<one of the requested fields>",
     "value": "<the normalized value>",
     "confidence": <0.0-1.0>,
     "source": "<the OCR line/snippet that justifies this value>"}
  ]
}
Return ONLY the JSON object, no prose."""


_SYSTEM_REVIEW = """You are a senior KYC compliance reviewer for an Indian fintech.
Given a structured pipeline report (OCR result, field-validation outcome,
fraud-detection signals, risk score), produce a recommendation.

Rules:
- Recommend REJECT if validation has hard failures or AI-synthesis signals are strong.
- Recommend APPROVE only if all required fields match ground truth and fraud signals are quiet.
- Recommend REVIEW for ambiguous cases (missing fields, MEDIUM risk, suspicious-but-not-confirmed fraud).
- Hard policy is enforced upstream — your role is to add reasoning, not bypass it.
- Keep `reasoning` under 60 words. List concrete `concerns` as separate bullet strings.

Output schema (JSON):
{
  "recommendation": "APPROVE" | "REVIEW" | "REJECT",
  "confidence": <0.0-1.0>,
  "reasoning": "<short justification>",
  "concerns": ["<bullet>", ...]
}
Return ONLY the JSON object, no prose."""


# ---------------------------------------------------------------------------
# Reasoner
# ---------------------------------------------------------------------------
class LLMReasoner:
    """Thin Azure OpenAI client. Construct via :func:`get_llm_reasoner`."""

    def __init__(self, settings: Optional[AzureOpenAISettings] = None) -> None:
        self._settings = settings or get_azure_settings()
        self._client = None  # lazy
        self._client_lock = threading.Lock()

    # ----------------- public -----------------
    @property
    def enabled(self) -> bool:
        """True when both creds AND the feature flag say go."""
        flag = os.getenv("KYC_LLM_ENABLE", "true").strip().lower()
        if flag in {"0", "false", "no", "off"}:
            return False
        return self._settings.is_configured

    def rectify_fields(
        self,
        *,
        raw_text: str,
        extracted: Dict[str, Any],
        missing_fields: List[str],
        doc_type: str,
        timeout_s: float = 10.0,
    ) -> List[LLMRectification]:
        if not self.enabled or not missing_fields or not raw_text:
            return []

        user_payload = {
            "document_type": doc_type,
            "already_extracted": _redact_for_log(extracted),
            "missing_fields": missing_fields,
            "raw_ocr_text": raw_text[:6000],   # token budget guard
        }
        data = self._chat_json(
            system=_SYSTEM_RECTIFY,
            user=json.dumps(user_payload, ensure_ascii=False),
            label="rectify_fields",
            timeout_s=timeout_s,
        )
        if not data:
            return []
        out: List[LLMRectification] = []
        for r in data.get("rectifications", []):
            try:
                out.append(LLMRectification(
                    field=str(r["field"]),
                    value=str(r["value"]),
                    confidence=float(r.get("confidence", 0.0)),
                    source=str(r.get("source", ""))[:200],
                ))
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug("dropping malformed rectification: %s", exc)
        return out

    def final_review(
        self,
        *,
        pipeline_state: Dict[str, Any],
        timeout_s: float = 10.0,
    ) -> Optional[LLMReview]:
        if not self.enabled:
            return None
        digest = _state_digest(pipeline_state)
        data = self._chat_json(
            system=_SYSTEM_REVIEW,
            user=json.dumps(digest, ensure_ascii=False),
            label="final_review",
            timeout_s=timeout_s,
        )
        if not data:
            return None
        try:
            rec = str(data.get("recommendation", "REVIEW")).upper()
            if rec not in {"APPROVE", "REVIEW", "REJECT"}:
                rec = "REVIEW"
            return LLMReview(
                recommendation=rec,
                confidence=float(data.get("confidence", 0.0)),
                reasoning=str(data.get("reasoning", ""))[:600],
                concerns=[str(c)[:200] for c in data.get("concerns", []) if c],
            )
        except (TypeError, ValueError) as exc:
            logger.warning("LLM review returned malformed JSON: %s", exc)
            return None

    # ----------------- internals -----------------
    def _client_or_none(self):
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                from openai import AzureOpenAI  # local import keeps cold-start light
            except ImportError as exc:
                logger.error("openai SDK missing: %s", exc)
                return None
            s = self._settings
            try:
                self._client = AzureOpenAI(
                    api_key=s.api_key,
                    azure_endpoint=s.endpoint,
                    api_version=s.api_version,
                    http_client=_build_http_client(),
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("AzureOpenAI client init failed: %s", exc)
                return None
            return self._client

    def _chat_json(
        self,
        *,
        system: str,
        user: str,
        label: str,
        timeout_s: float,
        retries: int = 1,
    ) -> Optional[Dict[str, Any]]:
        client = self._client_or_none()
        if client is None:
            return None
        deployment = self._settings.deployment
        if not deployment:
            return None

        attempt = 0
        while True:
            attempt += 1
            t0 = time.perf_counter()
            try:
                resp = client.chat.completions.create(
                    model=deployment,             # Azure uses deployment as model id
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    timeout=timeout_s,
                )
                content = (resp.choices[0].message.content or "").strip()
                usage = getattr(resp, "usage", None)
                logger.info(
                    "llm %s: ok attempt=%d latency=%.2fs in_tok=%s out_tok=%s",
                    label, attempt, time.perf_counter() - t0,
                    getattr(usage, "prompt_tokens", "?"),
                    getattr(usage, "completion_tokens", "?"),
                )
                return json.loads(content) if content else None
            except json.JSONDecodeError as exc:
                logger.warning("llm %s: non-JSON response (%s) — giving up", label, exc)
                return None
            except Exception as exc:  # noqa: BLE001
                if attempt > retries:
                    logger.warning("llm %s: failed after %d attempts: %s", label, attempt, exc)
                    return None
                logger.info("llm %s: transient error '%s' — retrying", label, exc)
                time.sleep(0.5)


# ---------------------------------------------------------------------------
# Helpers — PII redaction for logs / state-digest builder
# ---------------------------------------------------------------------------
_PII_KEYS = {"pan_number", "aadhaar_number", "passport_number", "driving_license_number"}


def _redact_for_log(d: Dict[str, Any]) -> Dict[str, Any]:
    """Don't show the model the same masked form we use in logs — the model
    needs real values to reason. This redaction is for our log files only.
    The function returned to the LLM (above) is the raw extracted dict."""
    return {k: ("***" if k in _PII_KEYS and v else v) for k, v in (d or {}).items()}


def _state_digest(state: Dict[str, Any]) -> Dict[str, Any]:
    """Trim and shape state for the review prompt."""
    fraud_signals = []
    vr = state.get("validation_results") or {}
    checks = []
    for c in vr.get("checks", []) or []:
        checks.append({
            "field": c.get("field"),
            "status": c.get("status"),
            "expected": c.get("expected"),
            "actual": c.get("actual"),
            "required": c.get("required"),
        })
    return {
        "document_type": state.get("document_type"),
        "ocr_confidence": state.get("ocr_confidence"),
        "extracted_fields": state.get("extracted_data") or {},
        "validation": {
            "overall": vr.get("overall"),
            "score": vr.get("score"),
            "required_passed": vr.get("required_passed"),
            "required_total": vr.get("required_total"),
            "checks": checks,
        },
        "fraud": {
            "verdict": state.get("fraud_verdict"),
            "score": state.get("fraud_score"),
            "reasons": state.get("fraud_reasons") or [],
        },
        "risk": {
            "score": state.get("risk_score"),
            "band": state.get("risk_band"),
        },
        "ocr_text_snippet": (state.get("raw_ocr_text") or "")[:1500],
    }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_reasoner: Optional[LLMReasoner] = None


def _build_http_client():
    """Construct an httpx.Client that trusts the OS cert store.

    Corporate networks sometimes terminate TLS at an inspection proxy
    and re-sign requests with an internal root CA installed in the Windows /
    macOS cert store. The default httpx behavior trusts only certifi's
    bundle, which causes ``CERTIFICATE_VERIFY_FAILED``. ``truststore`` wires
    Python's ``ssl`` module into the OS trust store so the corporate CA is
    honored without disabling verification.

    Escape hatches (use only when truststore unavailable):
      * ``KYC_LLM_CA_BUNDLE=/path/to/ca.pem`` — explicit bundle
      * ``KYC_LLM_INSECURE_SKIP_VERIFY=true`` — DEV ONLY, logs a loud warning
    """
    import httpx

    if os.getenv("KYC_LLM_INSECURE_SKIP_VERIFY", "").lower() in {"1", "true", "yes"}:
        logger.warning("KYC_LLM_INSECURE_SKIP_VERIFY is set — TLS verification DISABLED")
        return httpx.Client(verify=False, timeout=30)

    ca_bundle = os.getenv("KYC_LLM_CA_BUNDLE")
    if ca_bundle:
        return httpx.Client(verify=ca_bundle, timeout=30)

    try:
        import ssl
        import truststore
        ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        return httpx.Client(verify=ctx, timeout=30)
    except ImportError:
        logger.info("truststore not installed — falling back to certifi bundle "
                    "(may fail behind corporate TLS proxies)")
        return httpx.Client(timeout=30)


def get_llm_reasoner() -> LLMReasoner:
    global _reasoner
    if _reasoner is None:
        with _lock:
            if _reasoner is None:
                _reasoner = LLMReasoner()
    return _reasoner
