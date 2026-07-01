"""Node - final decision.

Order of precedence:
  1. Hard policy: LIKELY_AI fraud -> REJECT. Validation FAIL -> REJECT.
     Risk band CRITICAL -> REJECT.
  2. LLM recommendation (if available, and not contradicting hard policy):
     * LLM REJECT with high confidence -> REJECT
     * LLM APPROVE with high confidence AND validation PASS AND
       hallucination_score < 0.3 -> APPROVE
     * Otherwise the LLM downgrades a marginal case to REVIEW.
  3. Heuristic fallback: LOW band + PASS -> APPROVE; HIGH band -> REJECT;
     anything else -> REVIEW.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from app.graph.state import KYCState

_LLM_TRUST_FLOOR = 0.70
_HALLUCINATION_BLOCK = 0.30


def decision_node(state: KYCState) -> Dict[str, Any]:
    started = time.perf_counter()
    reasons: List[str] = []
    band = str(state.get("risk_band", "HIGH"))
    fraud_verdict = str(state.get("fraud_verdict", "INSUFFICIENT_DATA"))
    validation_overall = str(state.get("validation_overall", "MISSING"))
    llm_rec = state.get("llm_recommendation")
    llm_conf = float(state.get("llm_confidence") or 0.0)
    llm_reasoning = state.get("llm_reasoning")
    llm_concerns = state.get("llm_concerns") or []
    hallu = float(state.get("hallucination_score") or 0.0)

    if fraud_verdict == "LIKELY_AI":
        decision = "REJECT"
        reasons.append("HARD: document flagged as AI-generated")
    elif validation_overall == "FAIL":
        decision = "REJECT"
        reasons.append("HARD: required field comparison failed")
    elif band == "CRITICAL":
        decision = "REJECT"
        reasons.append("HARD: risk band CRITICAL")
    elif validation_overall == "NOT_APPLICABLE":
        decision = "REVIEW"
        reasons.append("document uses dynamic extraction; no fixed KYC validation rules apply")
    elif llm_rec == "REJECT" and llm_conf >= _LLM_TRUST_FLOOR:
        decision = "REJECT"
        reasons.append(f"LLM recommends REJECT (conf={llm_conf:.2f})")
    elif (
        llm_rec == "APPROVE"
        and llm_conf >= _LLM_TRUST_FLOOR
        and validation_overall == "PASS"
        and fraud_verdict != "SUSPICIOUS"
        and hallu < _HALLUCINATION_BLOCK
    ):
        decision = "APPROVE"
        reasons.append(f"LLM recommends APPROVE (conf={llm_conf:.2f})")
    elif band == "LOW" and validation_overall == "PASS":
        decision = "APPROVE"
    elif band == "HIGH":
        decision = "REJECT"
    else:
        decision = "REVIEW"
        if validation_overall == "MISSING":
            reasons.append("one or more required fields could not be extracted")
        if fraud_verdict == "SUSPICIOUS":
            reasons.append("fraud signals SUSPICIOUS - manual review")
        if hallu >= _HALLUCINATION_BLOCK:
            reasons.append(f"hallucination signal {hallu:.2f} above APPROVE floor")
        if llm_rec == "REVIEW":
            reasons.append(f"LLM also recommends REVIEW (conf={llm_conf:.2f})")

    if llm_reasoning:
        reasons.append(f"LLM: {llm_reasoning}")
    for c in llm_concerns:
        reasons.append(f"LLM concern: {c}")

    elapsed = time.perf_counter() - started
    return {
        "final_decision": decision,
        "decision_reasons": reasons,
        "audit_logs": [{
            "node": "decision_node",
            "decision": decision, "band": band,
            "hallucination_score": hallu,
        }],
        "timings": {"decision": elapsed},
    }
