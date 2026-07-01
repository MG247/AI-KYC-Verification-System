"""Production OCR service for KYC document processing.

Engine policy
-------------
* **PaddleOCR is the primary engine.** It is initialized lazily inside a
  thread-safe singleton, downloads model weights once into the user cache,
  and is reused for the life of the process.
* **EasyOCR is an OPTIONAL fallback**, gated behind
  ``KYC_OCR_ENABLE_EASYOCR_FALLBACK=true``. It is never imported unless
  Paddle initialization fails AND the flag is set, keeping the cold-start
  footprint minimal.

Threading & async
-----------------
PaddleOCR releases the GIL inside its C++ kernels; we expose
``recognize_async`` that off-loads to a thread pool so FastAPI and
LangGraph nodes remain non-blocking. The singleton is guarded by an RLock
so worker prefork (gunicorn/uvicorn ``--workers``) is safe.

Idempotency & caching
---------------------
Results are keyed by ``sha256(file_bytes) + document_type + flags``, so
retries from upstream short-circuit. The cache is in-process LRU — swap
for Redis in multi-replica deployments without touching this module.

Troubleshooting
---------------
* "paddlepaddle is not installed" — ``pip install paddlepaddle==3.0.0``.
* "No module named 'langchain_text_splitters'" — paddleocr 3.x pulls
  this transitively; ``pip install langchain-text-splitters``.
* "PDX has already been initialized" — only one PaddleOCR pipeline may
  exist per process. The singleton handles this; do not construct
  ``PaddleOCR`` directly elsewhere.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import mimetypes
import re
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import OCRSettings, get_settings
from app.schemas.ocr_schema import (
    BBox,
    BaseFields,
    ConfidenceReport,
    DocumentType,
    ExtractedFields,
    GenericDocumentAnalysis,
    OCRError,
    OCRErrorCode,
    OCREngine,
    OCRLine,
    OCRMetadata,
    OCRRequest,
    OCRResponse,
    OCRWord,
    QualityVerdict,
)
from app.utils.image_utils import (
    LoadedImage,
    assess_quality,
    extract_qr_codes,
    is_blank,
    load_image,
    preprocess_for_ocr,
)
from app.utils.pdf_utils import render_pdf
from app.utils.text_cleaner import (
    clean_ocr_text,
    confidence_based_filter,
    select_name_candidate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class OCRException(Exception):
    code: OCRErrorCode = OCRErrorCode.UNKNOWN

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class FileValidationError(OCRException):
    code = OCRErrorCode.UNSUPPORTED_FORMAT


class FileTooLargeError(OCRException):
    code = OCRErrorCode.FILE_TOO_LARGE


class CorruptFileError(OCRException):
    code = OCRErrorCode.CORRUPT_FILE


class BlankImageError(OCRException):
    code = OCRErrorCode.BLANK_IMAGE


class LowQualityError(OCRException):
    code = OCRErrorCode.LOW_QUALITY


class OCRFailedError(OCRException):
    code = OCRErrorCode.OCR_FAILED


# ---------------------------------------------------------------------------
# Engine singleton (PaddleOCR primary, EasyOCR optional fallback)
# ---------------------------------------------------------------------------
@dataclass
class EngineHandle:
    backend: str            # "paddle" | "easyocr"
    instance: Any
    version: str
    device: str


class OCREngineRegistry:
    """Thread-safe lazy holder for the active OCR backend.

    Exposed as a class (not a free singleton instance) so tests can call
    ``OCREngineRegistry.reset()`` between cases.
    """

    _lock = threading.RLock()
    _handle: Optional[EngineHandle] = None
    _init_error: Optional[str] = None

    # ------------- public API -------------
    @classmethod
    def get(cls, settings: Optional[OCRSettings] = None) -> EngineHandle:
        if cls._handle is not None:
            return cls._handle
        with cls._lock:
            if cls._handle is not None:
                return cls._handle
            s = settings or get_settings()
            handle = cls._init_paddle(s)
            if handle is None and s.enable_easyocr_fallback:
                logger.warning("PaddleOCR init failed; trying EasyOCR fallback (flag enabled)")
                handle = cls._init_easyocr(s)
            if handle is None:
                raise OCRFailedError(
                    "No OCR backend available",
                    details={
                        "primary": "paddleocr",
                        "init_error": cls._init_error,
                        "hint": "pip install paddleocr paddlepaddle==3.0.0",
                    },
                )
            cls._handle = handle
            logger.info(
                "OCR backend ready: %s v%s device=%s",
                handle.backend, handle.version, handle.device,
            )
            return handle

    @classmethod
    def _silence_paddle(cls) -> None:
        """Quiet down PaddlePaddle / PaddleX init chatter.

        PaddleX prints colored 'Creating model' / 'Model files already exist'
        banners via its own logger, paddle shells out to ``where ccache``
        (Windows) which echoes 'INFO: Could not find files...' to stdout,
        and emits a ccache UserWarning. None of these are actionable for
        our service; we silence each at the right layer.
        """
        import os, sys, warnings, logging as _logging
        warnings.filterwarnings("ignore", message=r".*ccache.*")
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        os.environ.setdefault("GLOG_minloglevel", "3")
        os.environ.setdefault("FLAGS_call_stack_level", "0")
        # Mute the "INFO: Could not find files" Windows `where` shellout:
        # paddle runs it through cmd.exe which writes to stdout. Patch
        # subprocess.run/Popen to swallow ccache-probe output. Cheap & local.
        try:
            import subprocess
            _orig_run = subprocess.run

            def _quiet_run(*a, **kw):  # type: ignore[no-untyped-def]
                cmd = a[0] if a else kw.get("args", "")
                if isinstance(cmd, (list, tuple)):
                    cmd_str = " ".join(str(c) for c in cmd)
                else:
                    cmd_str = str(cmd)
                if "ccache" in cmd_str or "where ccache" in cmd_str:
                    kw.setdefault("stdout", subprocess.DEVNULL)
                    kw.setdefault("stderr", subprocess.DEVNULL)
                return _orig_run(*a, **kw)

            subprocess.run = _quiet_run  # type: ignore[assignment]
        except Exception:  # noqa: BLE001
            pass
        for name in ("paddlex", "paddle", "ppocr"):
            _logging.getLogger(name).setLevel(_logging.WARNING)
        _ = sys

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._handle = None
            cls._init_error = None

    @classmethod
    def health(cls) -> Dict[str, Any]:
        """Health-check payload for /healthz endpoints. Forces init if cold."""
        try:
            h = cls.get()
            payload = {
                "status": "ok",
                "backend": h.backend,
                "version": h.version,
                "device": h.device,
            }
            try:
                from app.services.generic_document_analyzer import get_generic_document_analyzer
                payload["gpt_vision"] = {"available": get_generic_document_analyzer().enabled}
            except Exception:  # noqa: BLE001
                payload["gpt_vision"] = {"available": False}
            return payload
        except OCRException as exc:
            return {"status": "degraded", "error": exc.message, "details": exc.details}

    # ------------- internals -------------
    @classmethod
    def _init_paddle(cls, s: OCRSettings) -> Optional[EngineHandle]:
        cls._silence_paddle()
        try:
            import paddleocr  # type: ignore
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as exc:
            cls._init_error = f"paddleocr import failed: {exc}"
            logger.error("PaddleOCR not importable: %s", exc)
            return None

        # PaddleX resets its logger to INFO during its own import; force it
        # back to WARNING here so the per-model 'Creating model' banner is
        # suppressed during construction below.
        import logging as _logging
        for name in ("paddlex", "paddleocr", "ppocr", "paddle"):
            _logging.getLogger(name).setLevel(_logging.WARNING)
        try:
            import paddlex.utils.logging as _plog  # type: ignore
            _plog._logger.setLevel(_logging.WARNING)
            for h in _plog._logger.handlers:
                h.setLevel(_logging.WARNING)
        except Exception:  # noqa: BLE001
            pass

        # Resolve and pin the device. paddleocr v3 does not take use_gpu in
        # its constructor; device selection happens via paddle.set_device.
        try:
            import paddle  # type: ignore
            device = s.effective_device
            try:
                paddle.set_device(device)
            except Exception as dev_exc:  # noqa: BLE001
                logger.warning("paddle.set_device(%s) failed (%s); using cpu", device, dev_exc)
                device = "cpu"
                paddle.set_device("cpu")
            # Thread budget for CPU inference.
            try:
                paddle.set_num_threads(int(s.paddle_cpu_threads))
            except Exception:  # noqa: BLE001
                pass
        except ImportError as exc:
            cls._init_error = f"paddlepaddle not installed: {exc}"
            logger.error(
                "paddlepaddle missing — install: pip install paddlepaddle==3.0.0  (%s)", exc,
            )
            return None

        version = getattr(paddleocr, "__version__", "unknown")
        logger.info(
            "Initializing PaddleOCR v%s lang=%s textline_ori=%s doc_ori=%s unwarp=%s device=%s threads=%d",
            version, s.paddle_lang, s.paddle_use_textline_orientation,
            s.paddle_use_doc_orientation_classify, s.paddle_use_doc_unwarping,
            device, s.paddle_cpu_threads,
        )

        kwargs: Dict[str, Any] = {
            "lang": s.paddle_lang,
            "use_textline_orientation": s.paddle_use_textline_orientation,
            "use_doc_orientation_classify": s.paddle_use_doc_orientation_classify,
            "use_doc_unwarping": s.paddle_use_doc_unwarping,
            "text_det_box_thresh": s.paddle_text_det_box_thresh,
            "text_rec_score_thresh": s.paddle_text_rec_score_thresh,
        }
        if s.paddle_ocr_version:
            kwargs["ocr_version"] = s.paddle_ocr_version

        try:
            instance = PaddleOCR(**kwargs)
        except TypeError:
            # Older PaddleOCR (2.x) — fall back to the legacy signature.
            logger.info("PaddleOCR legacy (v2.x) signature detected")
            instance = PaddleOCR(
                use_angle_cls=s.paddle_use_textline_orientation,
                lang=s.paddle_lang,
                use_gpu=s.paddle_use_gpu,
                show_log=False,
            )
        except Exception as exc:  # noqa: BLE001
            cls._init_error = f"PaddleOCR construction failed: {exc}"
            logger.exception("PaddleOCR construction failed")
            return None

        return EngineHandle(backend="paddle", instance=instance, version=version, device=device)

    @classmethod
    def _init_easyocr(cls, s: OCRSettings) -> Optional[EngineHandle]:
        try:
            import easyocr  # type: ignore
        except ImportError as exc:
            logger.error("EasyOCR fallback unavailable: %s", exc)
            return None
        try:
            reader = easyocr.Reader([s.paddle_lang or "en"], gpu=s.paddle_use_gpu, verbose=False)
            return EngineHandle(
                backend="easyocr",
                instance=reader,
                version=getattr(easyocr, "__version__", "unknown"),
                device="gpu" if s.paddle_use_gpu else "cpu",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("EasyOCR init failed: %s", exc)
            return None


def startup_banner(settings: Optional[OCRSettings] = None) -> Dict[str, Any]:
    """Eagerly initialize PaddleOCR and print a one-line status banner.

    Call this from your FastAPI ``startup`` event or LangGraph entry point
    to avoid cold-start latency on the first user request.
    """
    s = settings or get_settings()
    health = OCREngineRegistry.health()
    logger.info(
        "KYC-OCR startup | engine=%s version=%s device=%s status=%s",
        health.get("backend"),
        health.get("version"),
        health.get("device"),
        health.get("status"),
    )
    if health["status"] != "ok":
        logger.error("OCR engine NOT ready: %s", health)
    _ = s  # reserved for future banner extensions
    return health


# ---------------------------------------------------------------------------
# Tiny LRU result cache
# ---------------------------------------------------------------------------
class _LRU:
    def __init__(self, capacity: int) -> None:
        self._cap = capacity
        self._store: "OrderedDict[str, OCRResponse]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[OCRResponse]:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return self._store[key]
            return None

    def put(self, key: str, value: OCRResponse) -> None:
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self._cap:
                self._store.popitem(last=False)


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------
def _to_legacy_fields(typed: "BaseFields") -> ExtractedFields:
    """Project a per-doc-type typed payload onto the legacy union schema.

    Kept so consumers of `OCRResponse.fields` (validation, audit, the
    legacy `extracted_data` dict on `KYCState`) keep working unchanged.
    The authoritative typed payload is what the graph node reads and
    emits — this is purely a back-compat surface.
    """
    out = ExtractedFields()
    d = typed.model_dump(exclude_none=True)
    # Aadhaar uses `masked_aadhaar`; legacy schema calls it `aadhaar_number`.
    if "masked_aadhaar" in d:
        out.aadhaar_number = d["masked_aadhaar"]
    if "dl_number" in d:
        out.driving_license_number = d["dl_number"]
    # Passport: legacy `name` is the concatenation of surname + given_name.
    if "surname" in d or "given_name" in d:
        parts = [d.get("given_name", ""), d.get("surname", "")]
        joined = " ".join(p for p in parts if p).strip()
        if joined:
            out.name = joined
    # GST: legacy `name` is the legal name.
    if "legal_name" in d and not out.name:
        out.name = d["legal_name"]
    # Direct mappings.
    for k in ("pan_number", "aadhaar_last4", "passport_number", "gstin",
              "name", "father_name", "dob", "gender", "address"):
        if k in d and getattr(out, k) is None:
            setattr(out, k, d[k])
    out.qr_codes = list(typed.qr_codes)
    out.tables = list(typed.tables)
    return out


class FieldExtractor:
    """Regex + heuristic parser for canonical KYC fields."""

    PAN_RE = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")
    AADHAAR_RE = re.compile(r"\b(\d{4})\s?(\d{4})\s?(\d{4})\b")
    PASSPORT_RE = re.compile(r"\b([A-PR-WYa-pr-wy][0-9]{7})\b")
    DL_RE = re.compile(r"\b([A-Z]{2}[-\s]?\d{2}[-\s]?(?:19|20)\d{2}[-\s]?\d{7})\b")
    GSTIN_RE = re.compile(r"\b(\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d])\b")
    # DOB: numeric or month-name format. No leading digit-boundary because
    # OCR routinely runs the label into the date ("D0810/08/2006"); the
    # trailing (?!\d) still rejects runaway year-like sequences.
    DOB_RE = re.compile(
        r"(\d{1,2}[\/\-\.\s](?:0?[1-9]|1[0-2]|[A-Za-z]{3,9})[\/\-\.\s]\d{2,4})(?!\d)"
    )
    # Labelled DOB. Tolerates 'DOB', 'D.O.B', the OCR mangle 'D0B'/'DOB',
    # and Hindi 'जन्म'. Optional separator between label and date.
    DOB_LABEL_RE = re.compile(
        r"(?:D[\.\s]?[O0Q][\.\s]?B|DATE\s*OF\s*BIRTH|BIRTH\s*DATE|जन्म)"
        r"[^\d]{0,4}"
        r"(\d{1,2}[\/\-\.\s](?:0?[1-9]|1[0-2]|[A-Za-z]{3,9})[\/\-\.\s]\d{2,4})",
        re.IGNORECASE,
    )
    # Gender: FEMALE first (longest-match), require standalone word, refuse
    # lone "M"/"F" that often come from "F/Father's" or middle initials.
    GENDER_RE = re.compile(r"\b(FEMALE|MALE|TRANSGENDER)\b", re.IGNORECASE)
    GENDER_LABEL_RE = re.compile(
        r"(?:GENDER|SEX|लिंग)\s*[:\-]?\s*(FEMALE|MALE|TRANSGENDER|F|M)\b",
        re.IGNORECASE,
    )
    NAME_HINT_RE = re.compile(r"(?:^|\n)\s*Name[:\s]+([A-Z][A-Z\s\.]{2,})", re.IGNORECASE)
    FATHER_HINT_RE = re.compile(
        r"(?:Father(?:'s)? Name|S/O|D/O|W/O)[:\s]+([A-Z][A-Z\s\.]{2,40})", re.IGNORECASE
    )

    DOC_HINTS: Dict[DocumentType, Tuple[re.Pattern, ...]] = {
        DocumentType.PAN: (re.compile(r"INCOME\s*TAX", re.I), re.compile(r"Permanent Account", re.I)),
        DocumentType.AADHAAR: (re.compile(r"Unique Identification", re.I), re.compile(r"Aadhaar", re.I)),
        DocumentType.PASSPORT: (re.compile(r"REPUBLIC OF INDIA", re.I), re.compile(r"Passport", re.I)),
        DocumentType.DRIVING_LICENSE: (re.compile(r"Driving Licen[cs]e", re.I),),
        DocumentType.GST_CERTIFICATE: (re.compile(r"Goods and Services Tax", re.I),),
        DocumentType.BANK_STATEMENT: (re.compile(r"Statement of Account", re.I), re.compile(r"IFSC", re.I)),
        DocumentType.SALARY_SLIP: (re.compile(r"Salary Slip|Pay Slip|Earnings", re.I),),
    }

    @classmethod
    def detect_type(cls, text: str) -> DocumentType:
        for doc_type, patterns in cls.DOC_HINTS.items():
            if any(p.search(text) for p in patterns):
                return doc_type
        return DocumentType.UNKNOWN

    @classmethod
    def extract(cls, text: str, doc_type: DocumentType) -> ExtractedFields:
        fields = ExtractedFields()
        upper = text.upper()

        if m := cls.PAN_RE.search(upper):
            fields.pan_number = m.group(1)

        if m := cls.AADHAAR_RE.search(text):
            last4 = m.group(3)
            fields.aadhaar_last4 = last4
            fields.aadhaar_number = f"XXXX XXXX {last4}"  # UIDAI masking

        if m := cls.PASSPORT_RE.search(text):
            fields.passport_number = m.group(1).upper()
        if m := cls.DL_RE.search(upper):
            fields.driving_license_number = re.sub(r"[-\s]", "", m.group(1))
        if m := cls.GSTIN_RE.search(upper):
            fields.gstin = m.group(1)

        # DOB: prefer a labelled match ("DOB: 10/08/2006"); fall back to a
        # bare date pattern. The labelled form tolerates the common OCR
        # mangle 'D0B' (zero instead of O).
        dob_m = cls.DOB_LABEL_RE.search(text) or cls.DOB_RE.search(text)
        if dob_m:
            fields.dob = cls._normalize_dob(dob_m.group(1))

        # Gender: only trust a labelled match (avoids the 'F/Father's' bug).
        # If unlabelled, require FEMALE/MALE as a standalone word, never
        # single-letter, and FEMALE is tried first to win longest-match.
        gender_raw: Optional[str] = None
        if m := cls.GENDER_LABEL_RE.search(text):
            gender_raw = m.group(1).upper()
        elif m := cls.GENDER_RE.search(text):
            gender_raw = m.group(1).upper()
        if gender_raw:
            fields.gender = {"M": "MALE", "F": "FEMALE"}.get(gender_raw, gender_raw)

        # Smarter name extraction: prefer "Name: <X>" label match, otherwise
        # walk the (already reading-ordered) lines and pick the first
        # plausible person-name that isn't a header keyword.
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if m := cls.NAME_HINT_RE.search(text):
            fields.name = cls._clean_name(m.group(1))
        else:
            fields.name = select_name_candidate(lines)
        if m := cls.FATHER_HINT_RE.search(text):
            fields.father_name = cls._clean_name(m.group(1))

        _ = doc_type
        return fields

    @staticmethod
    def _clean_name(raw: str) -> str:
        return re.sub(r"\s+", " ", raw).strip().title()

    @staticmethod
    def _normalize_dob(raw: str) -> str:
        """Coerce a noisy date string into ISO ``YYYY-MM-DD``.

        Accepts ``DD/MM/YYYY``, ``DD-MM-YY``, ``DD MMM YYYY`` and the
        space-separated variants PaddleOCR sometimes emits. Returns the
        raw string unchanged if no parser matches (so the audit trail
        still records what was on the card)."""
        from datetime import datetime as _dt
        s = re.sub(r"\s+", " ", raw).strip().replace(".", "/").replace("-", "/")
        # Insert a separator if missing between space-only tokens.
        s = re.sub(r"\s", "/", s) if " " in s and "/" not in s else s
        for fmt in (
            "%d/%m/%Y", "%d/%m/%y",
            "%d/%b/%Y", "%d/%b/%y",
            "%d/%B/%Y", "%d/%B/%y",
        ):
            try:
                return _dt.strptime(s, fmt).date().isoformat()
            except ValueError:
                continue
        return raw


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
@dataclass
class _OCRPage:
    words: List[OCRWord]
    lines: List[OCRLine]
    text: str


class OCRService:
    """Process-wide facade. Hold one instance per process."""

    def __init__(self, settings: Optional[OCRSettings] = None) -> None:
        self._settings = settings or get_settings()
        self._cache = _LRU(self._settings.result_cache_size) if self._settings.enable_result_cache else None
        self._executor = ThreadPoolExecutor(
            max_workers=self._settings.ocr_executor_workers,
            thread_name_prefix="ocr",
        )

    # ---------------- Public ----------------
    def recognize(self, request: OCRRequest | Dict[str, Any]) -> OCRResponse:
        if isinstance(request, dict):
            request = OCRRequest(**request)
        req_id = request.request_id or str(uuid.uuid4())
        log = logging.LoggerAdapter(logger, {"request_id": req_id})
        started = time.perf_counter()
        try:
            return self._recognize(request, req_id, log, started)
        except OCRException as exc:
            log.warning("OCR rejected: %s (%s)", exc.message, exc.code.value)
            return self._error_response(req_id, request, exc, started)
        except Exception as exc:  # noqa: BLE001
            log.exception("Unhandled OCR error")
            return self._error_response(
                req_id, request, OCRFailedError(str(exc) or "internal error"), started,
            )

    async def recognize_async(self, request: OCRRequest | Dict[str, Any]) -> OCRResponse:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.recognize, request)

    def health(self) -> Dict[str, Any]:
        return OCREngineRegistry.health()

    def benchmark(self, file_path: str, runs: int = 5) -> Dict[str, float]:
        timings: List[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            self.recognize(OCRRequest(file_path=file_path))
            timings.append(time.perf_counter() - t0)
        return {
            "runs": float(runs),
            "min_s": min(timings),
            "max_s": max(timings),
            "mean_s": sum(timings) / len(timings),
            "p50_s": float(median(timings)),
        }

    # ---------------- Pipeline ----------------
    def _recognize(
        self,
        request: OCRRequest,
        req_id: str,
        log: logging.LoggerAdapter,
        started: float,
    ) -> OCRResponse:
        path = Path(request.file_path)
        size, mime = self._validate_file(path)

        cache_key = self._cache_key(path, request)
        if self._cache is not None and (cached := self._cache.get(cache_key)) is not None:
            log.info("cache hit")
            cached = cached.model_copy(deep=True)
            cached.metadata.cache_hit = True
            cached.metadata.request_id = req_id
            return cached

        if request.ocr_engine == OCREngine.GPT_VISION:
            response = self._recognize_with_gpt_vision(request, req_id, path, size, mime, started, log)
            if self._cache is not None:
                self._cache.put(cache_key, response)
            return response

        is_pdf = path.suffix.lower() == ".pdf"
        embedded_text: List[str] = []
        tables: List[List[List[str]]] = []
        if is_pdf:
            try:
                pdf = render_pdf(path, extract_tables=request.extract_tables)
            except ValueError as exc:
                raise CorruptFileError(str(exc)) from exc
            pages = pdf.pages
            embedded_text = pdf.embedded_text
            tables = pdf.tables
            page_count = pdf.page_count
        else:
            try:
                loaded: LoadedImage = load_image(path)
            except ValueError as exc:
                raise CorruptFileError(str(exc)) from exc
            pages = [loaded.array]
            page_count = 1

        quality = assess_quality(pages[0])
        if is_blank(pages[0]):
            raise BlankImageError("Image appears blank", details={"std": quality.contrast})
        if quality.verdict == QualityVerdict.REJECTED:
            raise LowQualityError(
                "Image quality below acceptable threshold",
                details={"quality_score": quality.quality_score, "blur": quality.blur_score},
            )

        engine = OCREngineRegistry.get(self._settings)
        per_page: List[_OCRPage] = []
        preprocessing_applied: List[str] = []
        for idx, page_img in enumerate(pages):
            prepared, applied = preprocess_for_ocr(page_img)
            if idx == 0:
                preprocessing_applied = applied
            per_page.append(self._ocr_with_retry(engine, prepared, page_index=idx, log=log))

        all_words = [w for p in per_page for w in p.words]
        all_lines = [l for p in per_page for l in p.lines]
        ocr_text = "\n\n".join(p.text for p in per_page if p.text)
        raw_text = ("\n\n".join(embedded_text) + ("\n\n" + ocr_text if ocr_text else "")
                    if embedded_text else ocr_text)

        if not raw_text.strip():
            raise OCRFailedError("OCR produced no text")

        qrs: List[str] = extract_qr_codes(pages[0]) if (request.extract_qr and pages) else []
        # Classification runs on OCR text + the request hint. We deliberately
        # let the classifier override a wrong hint (e.g. caller said PAN but
        # the doc is clearly Aadhaar) — downstream document_feedback will
        # surface the mismatch.
        from app.services.document_classifier import classify as _classify
        from app.services.extractors import EXTRACTORS as _EXTRACTORS, ExtractorContext as _Ctx, extract as _extract_fields
        from app.services.generic_document_analyzer import get_generic_document_analyzer

        classification = _classify(raw_text, hint=request.document_type)
        doc_type = classification.doc_type
        classification_confidence = classification.confidence
        generic_analysis: Optional[GenericDocumentAnalysis] = None
        if doc_type not in _EXTRACTORS or (classification.confidence < 1.0 and request.document_type is None):
            generic_analysis = get_generic_document_analyzer().analyze_text(
                raw_text,
                hint=doc_type.value if doc_type != DocumentType.UNKNOWN else None,
            )
            if (
                doc_type in _EXTRACTORS
                and classification.confidence < 1.0
                and _is_non_kyc_label(generic_analysis.document_type_label)
                and not _label_supports_doc_type(generic_analysis.document_type_label, doc_type)
            ):
                doc_type = DocumentType.UNKNOWN
                classification_confidence = max(classification.confidence, generic_analysis.confidence)
        ctx = _Ctx(lines=all_lines, words=all_words, raw_text=raw_text, doc_type=doc_type)
        typed_fields = _extract_fields(ctx)
        # Side-channel containers ride along on every fields payload.
        typed_fields.qr_codes = qrs
        typed_fields.tables = tables
        # Adapter: legacy callers consume `OCRResponse.fields` as the
        # union `ExtractedFields`. Project the typed payload onto it.
        fields = _to_legacy_fields(typed_fields)

        confidence_report = self._build_confidence_report(all_lines)
        elapsed = time.perf_counter() - started

        response = OCRResponse(
            success=True,
            ocr_engine=OCREngine.PADDLE,
            document_type=doc_type,
            document_type_label=(
                generic_analysis.document_type_label if generic_analysis else doc_type.value
            ),
            classification_confidence=classification_confidence,
            raw_text=raw_text,
            lines=all_lines,
            words=all_words,
            fields=fields,
            typed_fields=typed_fields,
            dynamic_fields=generic_analysis.dynamic_fields if generic_analysis else {},
            generic_analysis=generic_analysis,
            confidence_score=confidence_report.average,
            confidence_report=confidence_report,
            processing_time=elapsed,
            metadata=OCRMetadata(
                request_id=req_id,
                file_name=path.name,
                file_size_bytes=size,
                mime_type=mime,
                page_count=page_count,
                engine=engine.backend,
                engine_version=f"{engine.backend}:{engine.version}",
                preprocessing_applied=preprocessing_applied,
                quality=quality,
                finished_at=datetime.utcnow(),
            ),
        )
        log.info(
            "ocr ok engine=%s pages=%d words=%d conf=%.3f time=%.3fs",
            engine.backend, page_count, len(all_words), confidence_report.average, elapsed,
        )
        if self._cache is not None:
            self._cache.put(cache_key, response)
        return response

    def _recognize_with_gpt_vision(
        self,
        request: OCRRequest,
        req_id: str,
        path: Path,
        size: int,
        mime: Optional[str],
        started: float,
        log: logging.LoggerAdapter,
    ) -> OCRResponse:
        from app.services.document_classifier import classify as _classify
        from app.services.extractors import ExtractorContext as _Ctx, extract as _extract_fields
        from app.services.generic_document_analyzer import get_generic_document_analyzer

        analysis = get_generic_document_analyzer().analyze_with_vision(
            path,
            hint=request.document_type.value if request.document_type else None,
            require_vision=True,
        )
        raw_text = analysis.raw_text or _fields_to_text(analysis.dynamic_fields)
        if not raw_text.strip():
            raise OCRFailedError("GPT Vision OCR produced no readable text")

        lines, words = _pseudo_ocr_items(raw_text, confidence=max(analysis.confidence, 0.75))
        classification = _classify(raw_text, hint=request.document_type)
        doc_type = classification.doc_type
        classification_confidence = max(classification.confidence, analysis.confidence if doc_type == DocumentType.UNKNOWN else 0.0)
        if (
            doc_type != DocumentType.UNKNOWN
            and classification.confidence < 1.0
            and _is_non_kyc_label(analysis.document_type_label)
            and not _label_supports_doc_type(analysis.document_type_label, doc_type)
        ):
            doc_type = DocumentType.UNKNOWN
            classification_confidence = max(classification.confidence, analysis.confidence)
        ctx = _Ctx(lines=lines, words=words, raw_text=raw_text, doc_type=doc_type)
        typed_fields = _extract_fields(ctx)
        typed_fields.tables = []
        typed_fields.qr_codes = []
        fields = _to_legacy_fields(typed_fields)
        confidence_report = self._build_confidence_report(lines)
        elapsed = time.perf_counter() - started

        response = OCRResponse(
            success=True,
            ocr_engine=OCREngine.GPT_VISION,
            document_type=doc_type,
            document_type_label=analysis.document_type_label or doc_type.value,
            classification_confidence=classification_confidence,
            raw_text=raw_text,
            lines=lines,
            words=words,
            fields=fields,
            typed_fields=typed_fields,
            dynamic_fields=analysis.dynamic_fields,
            generic_analysis=analysis,
            confidence_score=max(confidence_report.average, analysis.confidence),
            confidence_report=confidence_report,
            processing_time=elapsed,
            metadata=OCRMetadata(
                request_id=req_id,
                file_name=path.name,
                file_size_bytes=size,
                mime_type=mime,
                page_count=1,
                engine=OCREngine.GPT_VISION.value,
                engine_version="azure-openai-vision",
                preprocessing_applied=[],
                finished_at=datetime.utcnow(),
            ),
        )
        log.info(
            "ocr ok engine=%s lines=%d conf=%.3f time=%.3fs",
            OCREngine.GPT_VISION.value, len(lines), response.confidence_score, elapsed,
        )
        return response

    # ---------------- Backend dispatch ----------------
    def _ocr_with_retry(
        self,
        engine: EngineHandle,
        image: np.ndarray,
        *,
        page_index: int,
        log: logging.LoggerAdapter,
    ) -> _OCRPage:
        s = self._settings
        last_exc: Optional[Exception] = None
        for attempt in range(1, s.retry_max_attempts + 1):
            try:
                if engine.backend == "paddle":
                    return self._run_paddle(engine.instance, image, page_index)
                return self._run_easyocr(engine.instance, image, page_index)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                delay = s.retry_base_delay_s * (2 ** (attempt - 1))
                log.warning("OCR attempt %d/%d failed: %s — retrying in %.2fs",
                            attempt, s.retry_max_attempts, exc, delay)
                time.sleep(delay)
        raise OCRFailedError(
            f"OCR engine failed after {s.retry_max_attempts} attempts: {last_exc}"
        )

    # ---------------- PaddleOCR (primary) ----------------
    def _run_paddle(self, paddle_inst: Any, image: np.ndarray, page_index: int) -> _OCRPage:
        """Execute PaddleOCR and normalize its output across v2.x and v3.x."""
        result = None
        if hasattr(paddle_inst, "predict"):
            try:
                result = paddle_inst.predict(image)
            except Exception:
                result = None
        if result is None:
            result = paddle_inst.ocr(image)
            page = self._parse_paddle_v2(result, page_index)
        else:
            page = self._parse_paddle_v3(result, page_index)
        return self._postfilter(page, page_index)

    def _postfilter(self, page: _OCRPage, page_index: int) -> _OCRPage:
        """Apply the enterprise cleaning layer: dedupe, drop noise,
        reorder top-to-bottom + left-to-right. Logs every rejection so
        compliance can audit why a line was discarded."""
        if not page.lines:
            return page
        s = self._settings
        items = [(l.text, l.confidence, l.bbox) for l in page.lines]
        kept_idx, stats = confidence_based_filter(
            items,
            min_confidence=s.min_keep_confidence,
            min_length=s.ocr_min_text_length,
            max_symbol_ratio=s.ocr_max_symbol_ratio,
            drop_duplicates=s.ocr_drop_duplicates,
        )
        if s.ocr_log_rejections and stats.rejected:
            preview = ", ".join(f"{code}:'{text[:18]}'" for text, code, _ in stats.reasons[:8])
            logger.info(
                "page %d post-filter: kept=%d rejected=%d  %s",
                page_index, stats.kept, stats.rejected, preview,
            )

        kept_lines: List[OCRLine] = []
        kept_words: List[OCRWord] = []
        for i in kept_idx:
            orig = page.lines[i]
            cleaned = clean_ocr_text(orig.text)
            new_line = orig.model_copy(update={"text": cleaned})
            kept_lines.append(new_line)
            kept_words.append(OCRWord(
                text=cleaned,
                confidence=orig.confidence,
                bbox=orig.bbox,
                page=page_index,
            ))
        return _OCRPage(
            words=kept_words,
            lines=kept_lines,
            text="\n".join(l.text for l in kept_lines),
        )

    @staticmethod
    def _parse_paddle_v3(result: Any, page_index: int) -> _OCRPage:
        words: List[OCRWord] = []
        lines: List[OCRLine] = []
        for page_res in result or []:
            data = page_res if isinstance(page_res, dict) else getattr(page_res, "json", lambda: {})()
            if isinstance(data, dict) and "res" in data:
                data = data["res"]
            texts = data.get("rec_texts", []) if isinstance(data, dict) else []
            scores = data.get("rec_scores", []) if isinstance(data, dict) else []
            polys = OCRService._first_nonempty(data, "rec_polys", "dt_polys", "rec_boxes") if isinstance(data, dict) else []
            for text, score, poly in zip(texts, scores, polys):
                try:
                    conf = float(score)
                    bbox = OCRService._poly_to_bbox(poly)
                    line = OCRLine(text=str(text), confidence=conf, bbox=bbox, page=page_index)
                    lines.append(line)
                    words.append(OCRWord(text=str(text), confidence=conf, bbox=bbox, page=page_index))
                except (TypeError, ValueError) as exc:
                    logger.debug("skipping malformed paddle-v3 entry: %s", exc)
        return _OCRPage(words=words, lines=lines, text="\n".join(l.text for l in lines))

    @staticmethod
    def _parse_paddle_v2(raw: Any, page_index: int) -> _OCRPage:
        words: List[OCRWord] = []
        lines: List[OCRLine] = []
        if not raw:
            return _OCRPage(words=words, lines=lines, text="")
        page = raw[0] if (isinstance(raw, list) and raw and isinstance(raw[0], list)
                          and raw[0] and isinstance(raw[0][0], list)) else raw
        for entry in page or []:
            try:
                bbox_raw, payload = entry[0], entry[1]
                text, conf = ((payload[0], float(payload[1]))
                              if isinstance(payload, (list, tuple)) else (str(payload), 1.0))
                bbox = OCRService._poly_to_bbox(bbox_raw)
                line = OCRLine(text=text, confidence=conf, bbox=bbox, page=page_index)
                lines.append(line)
                words.append(OCRWord(text=text, confidence=conf, bbox=bbox, page=page_index))
            except (IndexError, TypeError, ValueError) as exc:
                logger.debug("skipping malformed paddle-v2 entry: %s", exc)
        return _OCRPage(words=words, lines=lines, text="\n".join(l.text for l in lines))

    @staticmethod
    def _first_nonempty(data: Dict[str, Any], *keys: str) -> Any:
        for key in keys:
            value = data.get(key)
            if value is None:
                continue
            try:
                if len(value) == 0:
                    continue
            except TypeError:
                pass
            return value
        return []

    # ---------------- EasyOCR (optional fallback) ----------------
    def _run_easyocr(self, reader: Any, image: np.ndarray, page_index: int) -> _OCRPage:
        rgb = image[:, :, ::-1] if image.ndim == 3 else image
        raw = reader.readtext(rgb, detail=1, paragraph=False)
        words: List[OCRWord] = []
        lines: List[OCRLine] = []
        for entry in raw or []:
            try:
                bbox_raw, text, conf = entry[0], str(entry[1]), float(entry[2])
                bbox = OCRService._poly_to_bbox(bbox_raw)
                line = OCRLine(text=text, confidence=conf, bbox=bbox, page=page_index)
                lines.append(line)
                words.append(OCRWord(text=text, confidence=conf, bbox=bbox, page=page_index))
            except (IndexError, TypeError, ValueError) as exc:
                logger.debug("skipping malformed easyocr entry: %s", exc)
        page = _OCRPage(words=words, lines=lines, text="\n".join(l.text for l in lines))
        return self._postfilter(page, page_index)

    # ---------------- Helpers ----------------
    @staticmethod
    def _poly_to_bbox(poly: Any) -> BBox:
        """Normalize any 4-point polygon (numpy / list / tuple) to our BBox shape."""
        pts = np.asarray(poly, dtype=float).reshape(-1, 2)
        if pts.shape[0] == 2:  # axis-aligned [x1,y1,x2,y2]
            (x1, y1), (x2, y2) = pts
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        if pts.shape[0] != 4:
            # Take the min-area rect corners as a safe approximation.
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            pts = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
        return tuple(tuple(map(float, p)) for p in pts)  # type: ignore[return-value]

    def _build_confidence_report(self, lines: List[OCRLine]) -> ConfidenceReport:
        if not lines:
            return ConfidenceReport(average=0.0, median=0.0, min=0.0)
        confs = [l.confidence for l in lines]
        s = self._settings
        low_lines = [l.text for l in lines if l.confidence < s.low_confidence_threshold]
        suspicious = sum(1 for c in confs if c < s.suspicious_confidence_threshold)
        return ConfidenceReport(
            average=float(sum(confs) / len(confs)),
            median=float(median(confs)),
            min=float(min(confs)),
            low_confidence_count=len(low_lines),
            suspicious_count=suspicious,
            low_confidence_lines=low_lines[:25],
        )

    def _validate_file(self, path: Path) -> Tuple[int, Optional[str]]:
        if not path.exists() or not path.is_file():
            raise FileValidationError(f"File not found: {path}")
        ext = path.suffix.lower()
        if ext not in self._settings.allowed_extensions:
            raise FileValidationError(
                f"Extension '{ext}' is not allowed",
                details={"allowed": sorted(self._settings.allowed_extensions)},
            )
        size = path.stat().st_size
        limit = self._settings.max_file_size_mb * 1024 * 1024
        if size > limit:
            raise FileTooLargeError(
                f"File size {size} exceeds {limit} bytes",
                details={"size": size, "limit": limit},
            )
        if size == 0:
            raise CorruptFileError("File is empty")
        mime, _ = mimetypes.guess_type(str(path))
        if mime and not any(mime.startswith(p) for p in self._settings.allowed_mime_prefixes):
            raise FileValidationError(
                f"MIME type '{mime}' is not allowed", details={"mime": mime},
            )
        try:
            with path.open("rb") as fh:
                header = fh.read(8)
        except OSError as exc:
            raise CorruptFileError(f"Cannot read file: {exc}") from exc
        if not _looks_like_supported(header, ext):
            raise FileValidationError("File header does not match its extension")
        return size, mime

    @staticmethod
    def _cache_key(path: Path, request: OCRRequest) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        h.update(b"|engine=" + request.ocr_engine.value.encode())
        h.update((request.document_type.value if request.document_type else "AUTO").encode())
        h.update(b"|tbl=" + (b"1" if request.extract_tables else b"0"))
        return h.hexdigest()

    def _error_response(
        self,
        req_id: str,
        request: OCRRequest,
        exc: OCRException,
        started: float,
    ) -> OCRResponse:
        path = Path(request.file_path)
        try:
            size = path.stat().st_size if path.exists() else 0
        except OSError:
            size = 0
        return OCRResponse(
            success=False,
            ocr_engine=request.ocr_engine,
            document_type=request.document_type or DocumentType.UNKNOWN,
            processing_time=time.perf_counter() - started,
            metadata=OCRMetadata(
                request_id=req_id,
                file_name=path.name,
                file_size_bytes=size,
                finished_at=datetime.utcnow(),
            ),
            error=OCRError(code=exc.code, message=exc.message, details=exc.details or None),
        )


# ---------------------------------------------------------------------------
# Magic-byte sniff
# ---------------------------------------------------------------------------
_MAGIC = {
    b"\x89PNG\r\n\x1a\n": {".png"},
    b"\xff\xd8\xff": {".jpg", ".jpeg"},
    b"GIF87a": {".gif"},
    b"GIF89a": {".gif"},
    b"BM": {".bmp"},
    b"II*\x00": {".tif", ".tiff"},
    b"MM\x00*": {".tif", ".tiff"},
    b"RIFF": {".webp"},
    b"%PDF": {".pdf"},
}


def _looks_like_supported(header: bytes, ext: str) -> bool:
    for magic, exts in _MAGIC.items():
        if header.startswith(magic) and ext in exts:
            return True
    return ext in {".tif", ".tiff", ".webp"}


def _label_supports_doc_type(label: str, doc_type: DocumentType) -> bool:
    text = (label or "").lower()
    tokens = {
        DocumentType.PAN: ("pan", "permanent account"),
        DocumentType.AADHAAR: ("aadhaar", "aadhar", "uidai", "unique identification"),
        DocumentType.PASSPORT: ("passport",),
        DocumentType.DRIVING_LICENSE: ("driving", "licence", "license", "dl"),
        DocumentType.GST_CERTIFICATE: ("gst", "goods and services"),
    }.get(doc_type, ())
    return any(token in text for token in tokens)


def _is_non_kyc_label(label: str) -> bool:
    text = (label or "").strip().lower()
    if not text or text in {"unknown", "unknown document", "document"}:
        return False
    return not any(
        _label_supports_doc_type(label, doc_type)
        for doc_type in (
            DocumentType.PAN,
            DocumentType.AADHAAR,
            DocumentType.PASSPORT,
            DocumentType.DRIVING_LICENSE,
            DocumentType.GST_CERTIFICATE,
        )
    )


def _fields_to_text(fields: Dict[str, Any]) -> str:
    lines = []
    for key, value in fields.items():
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=False)
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    return "\n".join(lines)


def _pseudo_ocr_items(raw_text: str, *, confidence: float) -> Tuple[List[OCRLine], List[OCRWord]]:
    lines: List[OCRLine] = []
    words: List[OCRWord] = []
    conf = float(max(0.0, min(1.0, confidence)))
    for idx, text in enumerate(line.strip() for line in raw_text.splitlines() if line.strip()):
        bbox: BBox = ((0.0, float(idx)), (1.0, float(idx)), (1.0, float(idx + 1)), (0.0, float(idx + 1)))
        word = OCRWord(text=text, confidence=conf, bbox=bbox, page=0)
        line = OCRLine(text=text, confidence=conf, bbox=bbox, page=0, words=[word])
        lines.append(line)
        words.append(word)
    return lines, words


# ---------------------------------------------------------------------------
# Module-level service singleton
# ---------------------------------------------------------------------------
_service_lock = threading.Lock()
_service: Optional[OCRService] = None


def get_ocr_service() -> OCRService:
    """Process-wide service singleton (FastAPI/LangGraph DI friendly)."""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = OCRService()
    return _service
