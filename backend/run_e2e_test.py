"""End-to-end smoke test for the KYC LangGraph pipeline.

Walks every file in ./input through the compiled graph
(OCR -> validate -> fraud -> risk -> decision) and prints a
per-document decision plus a summary table. No FastAPI server needed.

Run:
    python run_e2e_test.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.graph.workflow import graph as kyc_graph  # noqa: E402

INPUT_DIR = ROOT / "input"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("run_e2e_test")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def load_user_truth() -> Dict[str, Any]:
    p = INPUT_DIR / "sample_user.json"
    if not p.exists():
        log.warning("sample_user.json missing - validation will report NOT_APPLICABLE")
        return {}
    with p.open(encoding="utf-8") as fh:
        return json.load(fh)


def list_images() -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".pdf"}
    return sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in exts)


def infer_doc_hint(filename: str) -> str | None:
    n = filename.lower()
    if "aadhaar" in n or "aadhar" in n: return "AADHAAR"
    if "pan" in n:                       return "PAN"
    if "passport" in n:                  return "PASSPORT"
    if "dl" in n or "driving" in n:      return "DRIVING_LICENSE"
    if "gst" in n:                       return "GST_CERTIFICATE"
    return None


# ---------------------------------------------------------------------------
# Per-document run
# ---------------------------------------------------------------------------
def run_one(path: Path, truth: Dict[str, Any]) -> Tuple[str, Dict[str, Any], float]:
    banner(f"[KYC e2e] {path.name}")
    initial: Dict[str, Any] = {
        "request_id": str(uuid.uuid4()),
        "user_data": truth,
        "document_path": str(path),
        "document_type_hint": infer_doc_hint(path.name),
        "errors": [],
        "timings": {},
        "decision_reasons": [],
    }
    t0 = time.perf_counter()
    final = kyc_graph.invoke(initial, config={"configurable": {"thread_id": initial["request_id"]}})
    elapsed = time.perf_counter() - t0

    print(f"  document_type     : {final.get('document_type')}")
    if final.get("classification_confidence") is not None:
        print(f"  classify_conf     : {final.get('classification_confidence', 0.0):.2f}")
    print(f"  ocr_confidence    : {final.get('ocr_confidence', 0.0):.3f}")
    print(f"  validation_overall: {final.get('validation_overall')}")
    print(f"  field_match_score : {final.get('field_match_score', 0.0):.2f}")
    print(f"  fraud_verdict     : {final.get('fraud_verdict')}")
    print(f"  fraud_score       : {final.get('fraud_score', 0.0):.2f}")
    print(f"  risk_score        : {final.get('risk_score', 0.0):.2f}  ({final.get('risk_band')})")
    if final.get("llm_enabled"):
        if final.get("llm_skipped_reason"):
            print(f"  LLM               : skipped ({final['llm_skipped_reason']})")
        else:
            rec = final.get("llm_recommendation") or "?"
            conf = final.get("llm_confidence") or 0.0
            print(f"  LLM recommendation: {rec} (conf={conf:.2f})")
            if final.get("llm_reasoning"):
                print(f"  LLM reasoning     : {final['llm_reasoning']}")
            for r in (final.get("llm_rectifications") or []):
                print(f"  LLM rectified     : {r['field']}='{r['value']}' (conf={r['confidence']:.2f})")
            for c in (final.get("llm_concerns") or []):
                print(f"  LLM concern       : {c}")
    print(f"  FINAL DECISION    : {final.get('final_decision')}")
    print(f"  DOCUMENT STATUS   : {final.get('document_status')}")

    # ---- what's actually printed/written on the KYC document --------------
    typed = final.get("typed_extracted_data") or {}
    extracted = final.get("extracted_data") or {}
    corrected = final.get("corrected_fields") or {}
    base = typed if typed else extracted
    document_data = {**base, **{k: v for k, v in corrected.items() if k in base}}
    if document_data:
        print("  DOCUMENT DATA     :")
        for k in sorted(document_data.keys()):
            v = document_data[k]
            if v in (None, "", [], {}, "UNKNOWN") or k == "schema_type":
                continue
            tag = " (rectified)" if k in corrected and corrected[k] != base.get(k) else ""
            print(f"    - {k:18}: {v}{tag}")
    if final.get("document_feedback"):
        print("  feedback:")
        for f in final["document_feedback"]:
            print(f"    [{f['severity']:8}] {f['category']:24} {f['message']}")
    if final.get("decision_reasons"):
        print("  reasons:")
        for r in final["decision_reasons"]:
            print(f"    - {r}")
    if final.get("errors"):
        print("  errors:")
        for e in final["errors"]:
            print(f"    ! {e}")
    print(f"  total time        : {elapsed:.2f}s")
    return final.get("final_decision", "REVIEW"), final, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    if not INPUT_DIR.exists():
        log.error("input directory not found: %s", INPUT_DIR)
        return 2
    truth = load_user_truth()
    images = list_images()
    if not images:
        log.error("no input files in %s", INPUT_DIR)
        return 2

    print(f"\nGround truth keys: {list(truth.keys())}")
    print(f"Found {len(images)} file(s): {[p.name for p in images]}")

    results = []
    for img in images:
        try:
            decision, final, elapsed = run_one(img, truth)
            results.append((img.name, decision, final.get("risk_band", "?"),
                            final.get("risk_score", 0.0), elapsed))
        except Exception:  # noqa: BLE001
            log.exception("pipeline crashed on %s", img.name)
            results.append((img.name, "ERROR", "?", 0.0, 0.0))

    banner("SUMMARY")
    width = max(len(n) for n, *_ in results)
    print(f"  {'file'.ljust(width)}   decision   band     risk   time")
    for name, decision, band, risk, t in results:
        print(f"  {name.ljust(width)}   {decision:<8}   {band:<6}  {risk:.2f}   {t:.2f}s")

    rejected = sum(1 for _, d, *_ in results if d == "REJECT")
    approved = sum(1 for _, d, *_ in results if d == "APPROVE")
    review   = sum(1 for _, d, *_ in results if d == "REVIEW")
    print(f"\nAPPROVE={approved}  REVIEW={review}  REJECT={rejected}  (of {len(results)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
