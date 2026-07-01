"""Central configuration for the KYC OCR engine.

Settings are loaded from environment variables (and the ``.env`` file via
python-dotenv) with strong typing through pydantic-settings. PaddleOCR is
the primary / required engine; EasyOCR can be enabled as an optional
secondary fallback for hostile environments where PaddleOCR cannot
initialize (e.g. unsupported Python version, missing system libs).

Troubleshooting
---------------
* **paddlepaddle missing** — install with ``pip install paddlepaddle==3.0.0``
  (CPU) or the CUDA variant for GPU.
* **Python 3.14** — paddlepaddle currently ships wheels for Python
  3.8–3.12. Use a 3.11 venv (which this project does at ``.venv/``).
* **Windows** — install the VC++ 2019 redistributable; PaddleOCR ships
  native DLLs that depend on it.
* **First run downloads models** to ``~/.paddlex/official_models``
  (~400 MB). Pre-warm the cache in your Docker image to avoid cold-start
  latency in production.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import FrozenSet, Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env once at import time so subprocesses/threads see the same env.
load_dotenv(override=False)


class OCRSettings(BaseSettings):
    """Strongly-typed settings for the OCR subsystem."""

    model_config = SettingsConfigDict(
        env_prefix="KYC_OCR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- PaddleOCR (primary engine, v3.x API) ---------------------------
    paddle_lang: str = Field(default="en", description="Primary OCR language code (en, ch, fr, ...).")
    paddle_use_textline_orientation: bool = Field(
        default=True,
        description="Rotate cropped lines via PP-LCNet textline-orientation classifier (v3.x replacement for use_angle_cls).",
    )
    paddle_use_doc_orientation_classify: bool = Field(
        default=False,
        description="Whole-document orientation. Off by default — we deskew via OpenCV which is faster.",
    )
    paddle_use_doc_unwarping: bool = Field(
        default=False,
        description="UVDoc page un-warping. Heavy; enable only for warped/folded scans.",
    )
    paddle_use_gpu: bool = Field(default=False)
    paddle_device: Optional[str] = Field(
        default=None,
        description="Explicit Paddle device string (e.g. 'gpu:0', 'cpu'). Overrides paddle_use_gpu when set.",
    )
    paddle_cpu_threads: int = Field(default=max(1, (os.cpu_count() or 4) // 2))
    paddle_text_det_box_thresh: float = Field(default=0.60, ge=0.0, le=1.0,
        description="Tuned for KYC cards — higher rejects hologram speckles.")
    paddle_text_rec_score_thresh: float = Field(default=0.60, ge=0.0, le=1.0,
        description="PaddleOCR-side recognition floor. Service applies a further filter at low_confidence_threshold.")
    paddle_ocr_version: Optional[str] = Field(
        default="PP-OCRv4",
        description="Lightweight mobile family is fast and accurate enough for ID cards; "
                    "override to PP-OCRv5 for hand-written or low-res scans.",
    )

    # ---- OCR post-filter (KYC-tuned) -----------------------------------
    ocr_min_text_length: int = Field(default=3, ge=1, le=10)
    ocr_max_symbol_ratio: float = Field(default=0.40, ge=0.0, le=1.0)
    ocr_drop_duplicates: bool = Field(default=True)
    ocr_log_rejections: bool = Field(default=True)

    # ---- Optional fallback ----------------------------------------------
    enable_easyocr_fallback: bool = Field(
        default=False,
        description="If True and PaddleOCR fails to initialize, fall back to EasyOCR. Off by default.",
    )

    # ---- I/O limits & security ------------------------------------------
    max_file_size_mb: int = Field(default=15, ge=1, le=100)
    max_pdf_pages: int = Field(default=20, ge=1, le=200)
    pdf_render_dpi: int = Field(default=300, ge=72, le=600)
    allowed_extensions: FrozenSet[str] = Field(
        default=frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".pdf"})
    )
    allowed_mime_prefixes: FrozenSet[str] = Field(
        default=frozenset({"image/", "application/pdf"})
    )

    # ---- Preprocessing --------------------------------------------------
    target_long_edge_px: int = Field(default=1600, ge=600, le=4000)
    blur_variance_threshold: float = Field(default=80.0)
    min_image_dim_px: int = Field(default=200)
    enable_denoise: bool = Field(default=True)
    enable_clahe: bool = Field(default=True)

    # ---- Caching & concurrency ------------------------------------------
    enable_result_cache: bool = Field(default=True)
    result_cache_size: int = Field(default=512)
    ocr_executor_workers: int = Field(default=max(2, (os.cpu_count() or 4)))

    # ---- Retry policy ---------------------------------------------------
    retry_max_attempts: int = Field(default=3, ge=1, le=10)
    retry_base_delay_s: float = Field(default=0.25, ge=0.0)

    # ---- Confidence -----------------------------------------------------
    low_confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    suspicious_confidence_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    # Hard floor for a line to enter the output at all (post-filter cut).
    min_keep_confidence: float = Field(default=0.55, ge=0.0, le=1.0)

    # ---- Paths ----------------------------------------------------------
    temp_dir: Path = Field(default=Path(os.getenv("TMPDIR", "/tmp")) / "kyc_ocr")

    @field_validator("temp_dir")
    @classmethod
    def _ensure_temp_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def effective_device(self) -> str:
        """Resolve the Paddle device string actually used at init time."""
        if self.paddle_device:
            return self.paddle_device
        return "gpu:0" if self.paddle_use_gpu else "cpu"


class AzureOpenAISettings(BaseSettings):
    """Azure OpenAI credentials (loaded from .env, NEVER committed).

    These are consumed by downstream LangGraph / fraud-detection nodes;
    the OCR pipeline itself does not call OpenAI. Centralizing the config
    here keeps secret-handling in one auditable place.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    api_key: Optional[str] = Field(default=None, alias="AZURE_OPENAI_API_KEY")
    endpoint: Optional[str] = Field(default=None, alias="AZURE_OPENAI_ENDPOINT")
    deployment: Optional[str] = Field(default=None, alias="AZURE_OPENAI_DEPLOYMENT")
    api_version: str = Field(default="2024-10-21", alias="AZURE_OPENAI_API_VERSION")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.endpoint and self.deployment)


@lru_cache(maxsize=1)
def get_settings() -> OCRSettings:
    """Process-wide cached OCR settings instance."""
    return OCRSettings()


@lru_cache(maxsize=1)
def get_azure_settings() -> AzureOpenAISettings:
    """Process-wide cached Azure OpenAI settings instance."""
    return AzureOpenAISettings()
