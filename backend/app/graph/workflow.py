"""LangGraph workflow for the KYC pipeline.

Graph topology
--------------
    START
      |
      v
    preprocess_document        # quality gate: blur / glare / blank
      |
      v
    extract_text               # PaddleOCR -> structured fields
      |
      +--> validate_fields ----+
      +--> detect_fraud -------+   # all three run in parallel
      +--> metadata_analysis --+
                               v
                          ai_reasoning      # LLM proposes rectifications (skippable)
                               |
                               v
                          rectify_fields    # apply proposals, re-validate
                               |
                               v
                          calculate_risk    # weighted score + band
                               |
                               v
                          final_review      # LLM advisory recommendation (skippable)
                               |
                               v
                          decision_node     # APPROVE / REVIEW / REJECT
                               |
                               v
                              END

Why this shape
--------------
* The three post-OCR branches (validate / fraud / metadata) are
  independent reads of the same artifact, so we fan them out. State
  reducers (see ``state.py``) make their parallel updates safe.
* LLM nodes sit AFTER the deterministic signals so the model has a full
  picture and can be cheaply short-circuited when the case is already
  settled (clean APPROVE or hard REJECT).
* ``rectify_fields`` is a separate node from ``ai_reasoning`` so the raw
  LLM proposals are auditable before they touch ``extracted_data``.

Operational
-----------
* RetryPolicy is attached to the I/O-bound nodes (OCR, fraud, metadata,
  LLM). Pure-Python nodes (validate, rectify, calc_risk, decision) are
  deterministic and don't benefit from retry.
* MemorySaver is wired in so requests can be replayed/inspected per
  ``thread_id`` in dev. Swap for a persistent checkpointer in prod.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from app.graph.nodes.ai_reasoning import ai_reasoning
from app.graph.nodes.calculate_risk import calculate_risk
from app.graph.nodes.decision_node import decision_node
from app.graph.nodes.detect_fraud import detect_fraud
from app.graph.nodes.document_feedback import document_feedback
from app.graph.nodes.extract_text import extract_text
from app.graph.nodes.final_review import final_review
from app.graph.nodes.metadata_analysis import metadata_analysis
from app.graph.nodes.preprocess_document import preprocess_document
from app.graph.nodes.rectify_fields import rectify_fields
from app.graph.nodes.validate_fields import validate_fields
from app.graph.state import KYCState

# Retry I/O-bound nodes only. Pure-CPU nodes are deterministic.
_IO_RETRY = RetryPolicy(max_attempts=2, initial_interval=0.5, backoff_factor=2.0)


def _build() -> StateGraph:
    g = StateGraph(KYCState)

    g.add_node("preprocess_document", preprocess_document)
    g.add_node("extract_text",        extract_text,       retry=_IO_RETRY)
    g.add_node("validate_fields",     validate_fields)
    g.add_node("detect_fraud",        detect_fraud,       retry=_IO_RETRY)
    g.add_node("metadata_analysis",   metadata_analysis,  retry=_IO_RETRY)
    g.add_node("ai_reasoning",        ai_reasoning,       retry=_IO_RETRY)
    g.add_node("rectify_fields",      rectify_fields)
    g.add_node("calculate_risk",      calculate_risk)
    g.add_node("final_review",        final_review,       retry=_IO_RETRY)
    g.add_node("document_feedback",   document_feedback)
    g.add_node("decision_node",       decision_node)

    g.add_edge(START,                "preprocess_document")
    g.add_edge("preprocess_document", "extract_text")

    # Fan out: three branches run in parallel from extract_text.
    g.add_edge("extract_text", "validate_fields")
    g.add_edge("extract_text", "detect_fraud")
    g.add_edge("extract_text", "metadata_analysis")

    # Fan in: ai_reasoning waits for all three parallel branches to finish.
    g.add_edge("validate_fields",    "ai_reasoning")
    g.add_edge("detect_fraud",       "ai_reasoning")
    g.add_edge("metadata_analysis",  "ai_reasoning")

    g.add_edge("ai_reasoning",   "rectify_fields")
    g.add_edge("rectify_fields", "calculate_risk")
    g.add_edge("calculate_risk", "final_review")
    g.add_edge("final_review",   "document_feedback")
    g.add_edge("document_feedback", "decision_node")
    g.add_edge("decision_node",  END)
    return g


# Single checkpointer instance per process; safe to share across threads.
checkpointer = MemorySaver()

graph = _build().compile(checkpointer=checkpointer)
