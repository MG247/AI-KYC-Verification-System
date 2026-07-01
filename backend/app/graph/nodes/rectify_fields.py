"""Node - apply LLM rectifications.

Takes the rectification *proposals* gathered in ``ai_reasoning``,
filters them by confidence, and merges the survivors into
``extracted_data`` / ``corrected_fields``. If anything changed,
re-runs validation so the risk engine sees the updated picture.

Why a separate node from ``ai_reasoning``
-----------------------------------------
* Auditability: proposals are recorded before the merge, so a reviewer
  can see what the model suggested vs. what was actually accepted.
* Pure: no LLM call here. Deterministic merge + re-validation, easy to
  unit test in isolation.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from app.graph.nodes.validate_fields import _rebuild_ocr_response
from app.graph.state import KYCState
from app.services.validation_service import get_validation_service

logger = logging.getLogger(__name__)

# Only accept a rectification when the model is reasonably sure. Below this
# we keep the field MISSING and let the decision node route the case to
# REVIEW instead of risking a wrong APPROVE on a low-confidence guess.
_MERGE_CONF_FLOOR = 0.65


def rectify_fields(state: KYCState) -> Dict[str, Any]:
    started = time.perf_counter()
    proposals: List[Dict[str, Any]] = state.get("llm_rectifications") or []
    extracted: Dict[str, Any] = dict(state.get("extracted_data") or {})

    if not proposals:
        return {
            "corrected_fields": extracted,
            "audit_logs": [{"node": "rectify_fields", "applied": 0, "skipped": "no proposals"}],
            "timings": {"rectify": time.perf_counter() - started},
        }

    applied: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for p in proposals:
        field = p.get("field")
        value = p.get("value")
        conf = float(p.get("confidence", 0.0))
        if not field or value in (None, ""):
            rejected.append({**p, "reason": "empty"})
            continue
        if conf < _MERGE_CONF_FLOOR:
            rejected.append({**p, "reason": f"conf {conf:.2f} below floor"})
            continue
        if extracted.get(field):
            rejected.append({**p, "reason": "already-extracted; refusing overwrite"})
            continue
        extracted[field] = value
        applied.append(p)

    update: Dict[str, Any] = {
        "corrected_fields": extracted,
        "audit_logs": [{
            "node": "rectify_fields",
            "applied": [a["field"] for a in applied],
            "rejected": rejected,
        }],
    }

    if applied:
        rebuilt = _rebuild_ocr_response({**state, "extracted_data": extracted})
        report = get_validation_service().validate(rebuilt, state.get("user_data") or {})
        update["extracted_data"] = extracted
        update["validation_results"] = report.model_dump(mode="json")
        update["field_match_score"] = float(report.score)
        update["validation_overall"] = report.overall.value
        logger.info(
            "rectify_fields: applied %d -> validation %s score=%.2f",
            len(applied), report.overall.value, report.score,
        )

    update["timings"] = {"rectify": time.perf_counter() - started}
    return update
