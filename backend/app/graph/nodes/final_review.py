"""Node - LLM final review.

Last LLM-touching node. Sends a digest of the (post-rectification) state
to the model and asks for an APPROVE / REVIEW / REJECT recommendation
with reasoning and concerns. The output is *advisory only* - the
``decision_node`` weighs it against deterministic policy and never
allows the LLM to override a hard reject.

Skips when the LLM was disabled or short-circuited upstream (see
``ai_reasoning._should_skip``).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from app.graph.state import KYCState
from app.services.llm_reasoner import get_llm_reasoner

logger = logging.getLogger(__name__)


def final_review(state: KYCState) -> Dict[str, Any]:
    started = time.perf_counter()
    reasoner = get_llm_reasoner()

    if not reasoner.enabled or state.get("llm_skipped_reason"):
        return {
            "audit_logs": [{
                "node": "final_review",
                "skipped": state.get("llm_skipped_reason") or "LLM disabled",
            }],
            "timings": {"final_review": time.perf_counter() - started},
        }

    review = reasoner.final_review(pipeline_state=dict(state))
    if review is None:
        return {
            "audit_logs": [{"node": "final_review", "error": "no response"}],
            "timings": {"final_review": time.perf_counter() - started},
        }

    audit = {
        "node": "final_review",
        "recommendation": review.recommendation,
        "confidence": review.confidence,
    }
    logger.info(
        "final_review: %s (conf=%.2f)", review.recommendation, review.confidence,
    )
    return {
        "llm_recommendation": review.recommendation,
        "llm_confidence": review.confidence,
        "llm_reasoning": review.reasoning,
        "llm_concerns": review.concerns,
        "audit_logs": [audit],
        "timings": {"final_review": time.perf_counter() - started},
    }
