"""PDF helpers: page rendering, embedded-text extraction, table extraction.

We prefer ``pdfplumber`` because:
  * it gives us native (vector) text without OCR when the PDF was generated
    digitally (most bank statements, GST certs, salary slips);
  * it exposes table structure for tabular KYC artifacts;
  * it ships with rendering via the built-in ``to_image`` API, avoiding
    extra system dependencies that break installs on Windows CI runners.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pdfplumber
from PIL import Image

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class RenderedPDF:
    pages: List[np.ndarray]                       # BGR uint8
    embedded_text: List[str] = field(default_factory=list)
    tables: List[List[List[str]]] = field(default_factory=list)
    page_count: int = 0
    used_embedded_text: bool = False


def _pil_to_bgr(im: Image.Image) -> np.ndarray:
    if im.mode != "RGB":
        im = im.convert("RGB")
    arr = np.array(im)
    return arr[:, :, ::-1].copy()


def render_pdf(
    path: str | Path,
    *,
    dpi: Optional[int] = None,
    max_pages: Optional[int] = None,
    extract_tables: bool = False,
) -> RenderedPDF:
    """Render a PDF to per-page BGR arrays and harvest any embedded text.

    Raises:
        ValueError: if the PDF is corrupt, password-protected, or empty.
    """
    s = get_settings()
    dpi = dpi or s.pdf_render_dpi
    max_pages = max_pages or s.max_pdf_pages

    p = Path(path)
    if not p.exists():
        raise ValueError(f"PDF not found: {p}")

    result = RenderedPDF(pages=[])
    try:
        with pdfplumber.open(str(p)) as pdf:
            if not pdf.pages:
                raise ValueError("PDF contains zero pages")

            for idx, page in enumerate(pdf.pages[:max_pages]):
                # 1) Embedded text first — fast, lossless, no OCR error.
                try:
                    text = page.extract_text() or ""
                except Exception as exc:  # noqa: BLE001
                    logger.debug("page %d text extraction failed: %s", idx, exc)
                    text = ""
                if text.strip():
                    result.embedded_text.append(text)
                    result.used_embedded_text = True

                # 2) Tables (optional — costlier).
                if extract_tables:
                    try:
                        tables = page.extract_tables() or []
                        for t in tables:
                            result.tables.append(
                                [[(c or "").strip() for c in row] for row in t]
                            )
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("page %d table extraction failed: %s", idx, exc)

                # 3) Render image for OCR fallback / scanned PDFs.
                try:
                    pim = page.to_image(resolution=dpi).original
                    result.pages.append(_pil_to_bgr(pim))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("page %d render failed: %s", idx, exc)

            result.page_count = len(pdf.pages)
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to open PDF: {exc}") from exc

    if not result.pages and not result.embedded_text:
        raise ValueError("PDF produced no usable pages or text")

    return result


def encode_image_to_bytes(img: np.ndarray, fmt: str = "PNG") -> bytes:
    """Useful when handing a rendered page off to a remote OCR worker."""
    pil = Image.fromarray(img[:, :, ::-1] if img.ndim == 3 else img)
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return buf.getvalue()
