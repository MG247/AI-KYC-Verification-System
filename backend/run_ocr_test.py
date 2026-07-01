"""Smoke test for app.utils.image_utils and app.services.ocr_service.

Reads images and sample_user.json from ./input, runs each image through the
preprocessing utilities and the OCR service, and prints a verification report
that compares extracted fields against the ground-truth user JSON.

Run:
    python run_ocr_test.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2

# Make `app` importable when running from the repo root.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.schemas.ocr_schema import DocumentType, OCRRequest, ValidationStatus
from app.services.ocr_service import get_ocr_service
from app.services.validation_service import get_validation_service
from app.utils.image_utils import (
    assess_quality,
    blur_variance,
    is_blank,
    load_image,
    preprocess_for_ocr,
)

INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("run_ocr_test")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def load_user_truth() -> Dict[str, str]:
    p = INPUT_DIR / "sample_user.json"
    if not p.exists():
        log.warning("sample_user.json missing — field comparison will be skipped")
        return {}
    with p.open(encoding="utf-8") as fh:
        data = json.load(fh)
    print(f"\nGround-truth user data ({p.name}):")
    for k, v in data.items():
        print(f"  {k:<14} = {v}")
    return data


def list_images() -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".pdf"}
    files = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in exts)
    return files


# ---------------------------------------------------------------------------
# Stage 1 — image_utils
# ---------------------------------------------------------------------------
def test_image_utils(image_path: Path) -> bool:
    banner(f"[image_utils] {image_path.name}")
    if image_path.suffix.lower() == ".pdf":
        print("  (PDF — skipping image_utils stage)")
        return True
    try:
        loaded = load_image(image_path)
    except Exception as exc:  # noqa: BLE001
        log.error("load_image failed: %s", exc)
        return False

    print(f"  dimensions      : {loaded.width} x {loaded.height}")
    print(f"  blur variance   : {blur_variance(loaded.array):.2f}")
    print(f"  is_blank        : {is_blank(loaded.array)}")

    quality = assess_quality(loaded.array)
    print(f"  quality score   : {quality.quality_score:.3f}  ({quality.verdict.value})")
    print(f"  brightness/contrast : {quality.brightness:.1f} / {quality.contrast:.1f}")

    t0 = time.perf_counter()
    prepared, stages = preprocess_for_ocr(loaded.array)
    dt = (time.perf_counter() - t0) * 1000
    print(f"  preprocessing   : {stages}  ({dt:.1f} ms)")
    print(f"  output shape    : {prepared.shape}")

    out_path = OUTPUT_DIR / f"prep_{image_path.stem}.png"
    cv2.imwrite(str(out_path), prepared)
    print(f"  saved preview   : {out_path.relative_to(ROOT)}")
    return True


# ---------------------------------------------------------------------------
# Stage 2 — ocr_service
# ---------------------------------------------------------------------------
def _norm(value: str | None) -> str:
    return (value or "").replace(" ", "").upper()


def _infer_doc_type(filename: str) -> DocumentType | None:
    """Map an input filename to its likely DocumentType.

    Without this, we'd hint every file as the truth-file's `document_type`,
    which would force the validator to check PAN fields against an Aadhaar
    card. The file-name heuristic keeps each image tested under its own
    rule set.
    """
    n = filename.lower()
    if "aadhaar" in n or "aadhar" in n: return DocumentType.AADHAAR
    if "pan" in n:                       return DocumentType.PAN
    if "passport" in n:                  return DocumentType.PASSPORT
    if "dl" in n or "driving" in n:      return DocumentType.DRIVING_LICENSE
    if "gst" in n:                       return DocumentType.GST_CERTIFICATE
    return None


def test_ocr_service(image_path: Path, truth: Dict[str, str]) -> bool:
    banner(f"[ocr_service] {image_path.name}")
    svc = get_ocr_service()
    validator = get_validation_service()

    doc_type = _infer_doc_type(image_path.name)
    request = OCRRequest(file_path=str(image_path), document_type=doc_type, extract_qr=True)
    response = svc.recognize(request)

    print(f"  success         : {response.success}")
    print(f"  detected type   : {response.document_type.value}")
    print(f"  processing time : {response.processing_time:.3f} s")
    print(f"  confidence avg  : {response.confidence_score:.3f}")
    if response.confidence_report:
        cr = response.confidence_report
        print(f"  conf min/median : {cr.min:.3f} / {cr.median:.3f}")
        print(f"  low-conf lines  : {cr.low_confidence_count}  suspicious: {cr.suspicious_count}")

    if not response.success:
        err = response.error
        log.error("OCR failed: %s - %s", err.code.value if err else "?", err.message if err else "")
        return False

    f = response.fields
    print("\n  extracted fields:")
    for k, v in f.model_dump().items():
        if v:
            print(f"    {k:<14} = {v}")

    snippet = response.raw_text.strip().splitlines()[:6]
    print("\n  raw text (first 6 lines):")
    for line in snippet:
        print(f"    | {line}")

    # Document-aware validation: only fields relevant to THIS doc type are
    # checked. Unrelated truth keys (e.g. Aadhaar on a PAN card) are skipped.
    report = validator.validate(response, truth)
    print(f"\n  validation: {report.overall.value}  "
          f"score={report.score:.2f}  "
          f"required={report.required_passed}/{report.required_total}  "
          f"optional={report.optional_passed}/{report.optional_total}")
    for chk in report.checks:
        tag = chk.status.value
        req = "req" if chk.required else "opt"
        exp = chk.expected if chk.expected is not None else "-"
        act = chk.actual if chk.actual is not None else "-"
        suffix = f"  ({chk.reason})" if chk.reason else ""
        print(f"    [{tag:<14}] {req}  {chk.field:<22} expected={exp!r} got={act!r}{suffix}")

    return report.overall == ValidationStatus.PASS


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
        log.error("no input images found in %s", INPUT_DIR)
        return 2

    print(f"\nFound {len(images)} input file(s): {[p.name for p in images]}")

    results = {}
    for img in images:
        try:
            ok_prep = test_image_utils(img)
        except Exception as exc:  # noqa: BLE001
            log.exception("image_utils crashed on %s", img.name)
            ok_prep = False
        try:
            ok_ocr = test_ocr_service(img, truth)
        except Exception as exc:  # noqa: BLE001
            log.exception("ocr_service crashed on %s", img.name)
            ok_ocr = False
        results[img.name] = (ok_prep, ok_ocr)

    banner("SUMMARY")
    width = max(len(n) for n in results)
    print(f"  {'file'.ljust(width)}   image_utils   ocr_service")
    for name, (a, b) in results.items():
        print(f"  {name.ljust(width)}   {'PASS' if a else 'FAIL':<11}   {'PASS' if b else 'FAIL'}")

    failed = sum(1 for a, b in results.values() if not (a and b))
    print(f"\n{len(results) - failed}/{len(results)} files fully passed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
