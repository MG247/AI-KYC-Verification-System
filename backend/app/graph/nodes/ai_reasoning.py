"""Node - LLM reasoning (rectification proposals).

First of three LLM-adjacent nodes (``ai_reasoning`` -> ``rectify_fields``
-> ``final_review``). This node decides whether the LLM should engage at
all, and if so, asks it for rectification *proposals* for required
fields the deterministic extractor failed to find. It does NOT merge
those proposals into the state - merging is ``rectify_fields``'s job, so
the proposals are auditable in isolation before they touch the
extracted data.

Skip conditions (saves cost + latency)
--------------------------------------
* LLM not configured / disabled.
* Hard-reject cases already settled (LIKELY_AI fraud verdict or
  validation FAIL) - the decision node will REJECT regardless.
* Clean APPROVE cases (LIKELY_AUTHENTIC + PASS + LOW risk) - no
  ambiguity, no point spending an OpenAI call.

Hallucination guard
-------------------
For each proposed rectification we check whether the value appears as a
substring in the raw OCR text. Values that don't are *not* dropped here
(they may have been normalized, e.g. DOB regex -> ISO format), but the
fraction that lacks substring grounding is recorded as
``hallucination_score`` and fed into the risk engine downstream.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List

from app.graph.state import KYCState
from app.services.llm_reasoner import LLMRectification, get_llm_reasoner

logger = logging.getLogger(__name__)


def _missing_required_fields(state: KYCState) -> List[str]:
    vr = state.get("validation_results") or {}
    return [
        str(c.get("field"))
        for c in (vr.get("checks") or [])
        if c.get("required") and c.get("status") == "MISSING"
    ]


def _should_skip(state: KYCState) -> str | None:
    """Return a human-readable skip reason or None to proceed."""
    if state.get("fraud_verdict") == "LIKELY_AI":
        return "hard reject path (fraud LIKELY_AI)"
    if state.get("validation_overall") == "FAIL":
        return "hard reject path (validation FAIL)"
    if (
        state.get("validation_overall") == "PASS"
        and state.get("fraud_verdict") == "LIKELY_AUTHENTIC"
        and state.get("ocr_confidence", 0.0) >= 0.85
    ):
        return "clean approve path (no ambiguity to resolve)"
    return None


def _hallucination_signal(rectifications: List[LLMRectification], raw_text: str) -> float:
    """0 = every value substring-grounded in OCR text; 1 = none are.

    Scoring is per-TOKEN, not per-whole-value, because the LLM legitimately
    normalizes shape (e.g. DOB ``10/08/2006`` -> ``2006-08-10``) and a naive
    whole-string match would flag every normalization as hallucination. A
    value is considered supported when every "meaningful" token (length>=2
    alphanumeric chunk) it contains is also present in the raw OCR text.
    """
    if not rectifications:
        return 0.0
    norm_text = re.sub(r"\s+", " ", (raw_text or "").lower())
    text_token = re.sub(r"[^a-z0-9]", "", norm_text)
    weighted_unsupported = 0.0
    total_weight = 0.0
    for r in rectifications:
        val = (r.value or "").strip().lower()
        if not val:
            continue
        weight = max(r.confidence, 0.1)
        total_weight += weight

        # Cheap path: whole-string or alphanum-collapsed match still wins.
        token = re.sub(r"[^a-z0-9]", "", val)
        if val in norm_text or (token and token in text_token):
            continue

        # Token path: split into alphanumeric chunks and check each. A
        # date like "2006-08-10" yields ["2006","08","10"] — all three are
        # expected to appear somewhere in the raw OCR even if reordered.
        parts = [p for p in re.split(r"[^a-z0-9]+", val) if len(p) >= 2]
        if not parts:
            weighted_unsupported += weight
            continue
        missing = sum(1 for p in parts if p not in text_token)
        # Fraction of parts that are missing from raw text, weighted by
        # LLM confidence. A single missing token in a 3-token value
        # contributes only 1/3 of `weight`, not the whole weight.
        weighted_unsupported += weight * (missing / len(parts))
    return round(weighted_unsupported / total_weight, 3) if total_weight else 0.0


def ai_reasoning(state: KYCState) -> Dict[str, Any]:
    started = time.perf_counter()
    reasoner = get_llm_reasoner()
    timings = {"ai_reasoning": 0.0}

    if not reasoner.enabled:
        return {
            "llm_enabled": False,
            "llm_skipped_reason": "LLM disabled or not configured",
            "llm_rectifications": [],
            "hallucination_score": 0.0,
            "audit_logs": [{"node": "ai_reasoning", "skipped": "disabled"}],
            "timings": {"ai_reasoning": time.perf_counter() - started},
        }

    skip = _should_skip(state)
    if skip:
        return {
            "llm_enabled": True,
            "llm_skipped_reason": skip,
            "llm_rectifications": [],
            "hallucination_score": 0.0,
            "audit_logs": [{"node": "ai_reasoning", "skipped": skip}],
            "timings": {"ai_reasoning": time.perf_counter() - started},
        }

    missing = _missing_required_fields(state)
    if not missing:
        return {
            "llm_enabled": True,
            "llm_skipped_reason": "no missing required fields",
            "llm_rectifications": [],
            "hallucination_score": 0.0,
            "audit_logs": [{"node": "ai_reasoning", "skipped": "nothing to rectify"}],
            "timings": {"ai_reasoning": time.perf_counter() - started},
        }

    proposals = reasoner.rectify_fields(
        raw_text=state.get("raw_ocr_text", ""),
        extracted=state.get("extracted_data") or {},
        missing_fields=missing,
        doc_type=str(state.get("document_type", "UNKNOWN")),
    )
    hallu = _hallucination_signal(proposals, state.get("raw_ocr_text", ""))

    audit = {
        "node": "ai_reasoning",
        "missing_fields": missing,
        "proposed": [
            {"field": p.field, "value": p.value, "confidence": p.confidence}
            for p in proposals
        ],
        "hallucination_score": hallu,
    }
    logger.info(
        "ai_reasoning: %d proposal(s), hallucination_score=%.2f",
        len(proposals), hallu,
    )
    return {
        "llm_enabled": True,
        "llm_rectifications": [
            {"field": p.field, "value": p.value,
             "confidence": p.confidence, "source": p.source}
            for p in proposals
        ],
        "hallucination_score": hallu,
        "audit_logs": [audit],
        "timings": {"ai_reasoning": time.perf_counter() - started},
    }
