"""AI-generated document detector for KYC forensics.

Why this exists
---------------
2024+ fraud trend: synthetic PAN/Aadhaar/passport images generated via
Stable Diffusion, Midjourney, "AI ID generator" web tools, and Photoshop
Generative Fill are being submitted to onboarding flows to open accounts
under fabricated identities. Catching them before OCR validation saves
downstream review time and audit cost.

Approach
--------
Weighted ensemble of *interpretable* forensic signals. Each signal is a
pure function over the decoded image (and file metadata), returns a
score in [0, 1] where higher means "more AI-like", plus a human-readable
detail string so a compliance reviewer can audit WHY the system flagged
the document.

Signals
-------
1. METADATA       — EXIF Software/Make tags reveal generators or editors
2. FFT_SPECTRUM   — natural photos follow ~1/f^α; many diffusion outputs flatten it
3. ELA            — error-level analysis: uniform error → no real JPEG history
4. NOISE_RESIDUAL — too-clean high-frequency residual ≈ no sensor noise
5. COLOR_ENTROPY  — generated images often cluster too smoothly in color space
6. SATURATION     — diffusion outputs are characteristically oversaturated
7. JPEG_HISTORY   — PNG / missing quantization tables on a "captured" document
8. EDGE_DENSITY   — over-smoothed or pathologically sharp edge distributions

Limitations (read before deploying)
-----------------------------------
* This is HEURISTIC forensics, not a CNN. It catches casual fakes (consumer
  AI tools at default settings, screenshots of generated images, naive
  Photoshop edits). A sophisticated adversary who post-processes a
  diffusion output (re-photographs the screen, adds sensor noise, strips
  EXIF, re-saves as low-quality JPEG) can defeat individual signals.
* Treat the score as ONE input into a layered decision: also run face
  match against the document photo, device fingerprinting, sanctions
  screening, and a deep-learning detector trained on labeled synthetic
  IDs when you can afford to build one.
* Default thresholds were chosen for the Indian-KYC document mix (PAN /
  Aadhaar / passport on white-ish backgrounds). Re-calibrate against your
  own labeled dataset before tightening the AI threshold in production.
"""

from __future__ import annotations

import io
import logging
import re
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import ExifTags, Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Known generator / editor signatures
# ---------------------------------------------------------------------------
# Substrings (case-insensitive) we look for in EXIF Software / XMP fields,
# and in the filename itself. Editor tags (Photoshop) are weaker evidence —
# legitimate scans are sometimes post-processed — but still worth surfacing.
_AI_GENERATOR_TAGS: Tuple[str, ...] = (
    "stable diffusion", "stable-diffusion", "sdxl",
    "midjourney", "mj_",
    "dall-e", "dalle", "dall_e",
    "firefly", "adobe firefly",
    "runway", "runwayml",
    "leonardo.ai", "leonardo ai",
    "flux.1", "flux1",
    "imagen",
    "ideogram",
    "playground ai",
    "novelai",
    "automatic1111", "a1111",
    "comfyui", "comfy_ui",
    "invokeai",
    "generative fill", "generative_fill",
    "ai generated",
)

_EDITOR_TAGS: Tuple[str, ...] = (
    "photoshop", "gimp", "pixlr", "canva", "figma",
    "lightroom", "snapseed",
)

# Filename hints — weak but cheap to check.
_FILENAME_AI_HINTS: Tuple[str, ...] = (
    "ai_", "_ai", "ai-gen", "generated", "diffusion", "synth", "fake_",
    "midjourney", "stable_diff", "dalle",
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class AIVerdict(str, Enum):
    LIKELY_AUTHENTIC = "LIKELY_AUTHENTIC"
    SUSPICIOUS = "SUSPICIOUS"
    LIKELY_AI = "LIKELY_AI"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class SignalCode(str, Enum):
    METADATA = "METADATA"
    FFT_SPECTRUM = "FFT_SPECTRUM"
    ELA = "ELA"
    NOISE_RESIDUAL = "NOISE_RESIDUAL"
    COLOR_ENTROPY = "COLOR_ENTROPY"
    SATURATION = "SATURATION"
    JPEG_HISTORY = "JPEG_HISTORY"
    EDGE_DENSITY = "EDGE_DENSITY"
    LAYOUT_ANCHOR = "LAYOUT_ANCHOR"   # doc-template structural checks (QR / emblem / aspect)


class SignalScore(BaseModel):
    """One forensic signal's contribution to the final verdict."""

    model_config = ConfigDict(frozen=True)

    code: SignalCode
    score: float = Field(ge=0.0, le=1.0)   # higher = more AI-like
    triggered: bool                         # crossed the per-signal threshold
    weight: float = Field(ge=0.0, le=1.0)
    detail: str                             # human-readable measurement


class AIDetectionResult(BaseModel):
    file_name: str
    is_ai_generated: bool
    verdict: AIVerdict
    confidence: float = Field(ge=0.0, le=1.0)  # weighted aggregate score
    signals: List[SignalScore] = Field(default_factory=list)
    explanation: str
    processing_time_s: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _load_bgr(path: Path) -> Optional[np.ndarray]:
    """Decode an image to BGR ndarray. Prefers Pillow (EXIF-aware) and
    falls back to OpenCV. Returns ``None`` if the file is undecodable."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            arr = np.array(im)[:, :, ::-1].copy()  # RGB → BGR
            return arr
    except (UnidentifiedImageError, OSError):
        pass
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return arr


# ---------------------------------------------------------------------------
# Signal 1 — Metadata / EXIF
# ---------------------------------------------------------------------------
def _signal_metadata(path: Path) -> Tuple[float, str]:
    """Inspect EXIF + filename for AI-generator signatures.

    Strongest single signal we have: if a file literally says
    ``Software: Stable Diffusion`` we are done. Editor tags (Photoshop)
    get a moderate score — legitimate scans are sometimes cleaned up in
    Photoshop, so we surface but don't auto-fail.
    """
    detail_parts: List[str] = []
    score = 0.0

    name_l = path.name.lower()
    if any(h in name_l for h in _FILENAME_AI_HINTS):
        score = max(score, 0.55)
        detail_parts.append(f"filename hint: '{path.stem}'")

    software = ""
    make = ""
    model = ""
    try:
        with Image.open(path) as im:
            exif = im.getexif() or {}
            inv = {v: k for k, v in ExifTags.TAGS.items()}
            software = str(exif.get(inv.get("Software", -1), "") or "").strip()
            make = str(exif.get(inv.get("Make", -1), "") or "").strip()
            model = str(exif.get(inv.get("Model", -1), "") or "").strip()
            # XMP (where Firefly / Adobe Generative Fill leave fingerprints)
            xmp = ""
            try:
                xmp_bytes = im.info.get("XML:com.adobe.xmp") or im.info.get("xmp") or b""
                xmp = xmp_bytes.decode("utf-8", errors="ignore") if isinstance(xmp_bytes, bytes) else str(xmp_bytes)
            except Exception:  # noqa: BLE001
                xmp = ""
    except (UnidentifiedImageError, OSError):
        return 0.30, "EXIF unreadable"

    haystack = " ".join([software, make, model]).lower()
    xmp_l = xmp.lower()

    for tag in _AI_GENERATOR_TAGS:
        if tag in haystack or tag in xmp_l:
            return 0.98, f"generator signature '{tag}' in metadata"

    for tag in _EDITOR_TAGS:
        if tag in haystack:
            score = max(score, 0.50)
            detail_parts.append(f"editor tag '{tag}' in Software")
            break

    if not make and not model and path.suffix.lower() in {".jpg", ".jpeg"}:
        # A "captured" document JPEG with NO camera Make/Model is mildly
        # suspicious — phones and scanners almost always stamp these.
        score = max(score, 0.35)
        detail_parts.append("JPEG missing camera Make/Model")

    if not detail_parts:
        return score, f"clean (software='{software or '-'}', make='{make or '-'}')"
    return score, "; ".join(detail_parts)


# ---------------------------------------------------------------------------
# Signal 2 — FFT spectral slope
# ---------------------------------------------------------------------------
def _signal_fft_spectrum(gray: np.ndarray) -> Tuple[float, str]:
    """Natural images follow a ~1/f^α power spectrum (α ≈ 1.8–2.5).
    Many diffusion / GAN outputs flatten the high-frequency tail because
    the upsampler smooths out fine detail. We fit the log-log slope of
    the radial power spectrum and score flatter slopes as more AI-like.
    """
    h, w = gray.shape[:2]
    # Down-sample large images to keep the FFT cheap and the slope stable.
    if max(h, w) > 1024:
        scale = 1024.0 / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = gray.shape[:2]

    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    mag = np.abs(f) + 1e-8

    cy, cx = h // 2, w // 2
    y, x = np.indices(mag.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)

    radial_sum = np.bincount(r.ravel(), mag.ravel())
    radial_cnt = np.bincount(r.ravel())
    radial = radial_sum / np.maximum(radial_cnt, 1)
    # Fit slope on mid-band [4 .. min(h,w)/4] — skip the DC spike and the
    # noisy tail near the Nyquist limit.
    lo, hi = 4, max(8, min(h, w) // 4)
    radial = radial[lo:hi]
    if radial.size < 8 or np.any(radial <= 0):
        return 0.0, "spectrum unmeasurable"

    log_r = np.log(np.arange(lo, lo + len(radial)))
    log_m = np.log(radial)
    slope, _ = np.polyfit(log_r, log_m, 1)
    # Natural images: slope around -1.8 to -2.4.
    # Flatter slope (e.g. -1.0) ⇒ smoothed high-freq ⇒ AI-like.
    # Map -2.2 → 0.0, -1.0 → 1.0.
    score = _clip01((slope + 2.2) / 1.2)
    return score, f"spectral_slope={slope:.2f} (natural ~ -2.0)"


# ---------------------------------------------------------------------------
# Signal 3 — Error-level analysis (ELA)
# ---------------------------------------------------------------------------
def _signal_ela(bgr: np.ndarray) -> Tuple[float, str]:
    """Re-encode as JPEG at a known quality and compute the per-pixel
    absolute difference. Genuine photos show *high variance* in the diff
    (edges and texture compress differently); AI-generated images often
    produce uniformly low diff because they have no real compression
    history and lack natural micro-texture.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=90)
    buf.seek(0)
    re_rgb = np.array(Image.open(buf).convert("RGB"))
    re_bgr = cv2.cvtColor(re_rgb, cv2.COLOR_RGB2BGR)
    if re_bgr.shape != bgr.shape:
        re_bgr = cv2.resize(re_bgr, (bgr.shape[1], bgr.shape[0]))

    diff = cv2.absdiff(bgr, re_bgr).astype(np.float32)
    mean_d = float(diff.mean())
    std_d = float(diff.std())
    # ratio of variance to mean: high in natural images (edge structure),
    # low in synthetic images (smooth, featureless diff).
    ratio = std_d / max(mean_d, 1e-3)
    # Natural ratio observed ~1.0–1.8 on KYC scans.
    # ratio < 0.7 → suspicious; map 1.2 → 0.0, 0.4 → 1.0
    score = _clip01((1.2 - ratio) / 0.8)
    return score, f"ela_mean={mean_d:.2f} std={std_d:.2f} ratio={ratio:.2f}"


# ---------------------------------------------------------------------------
# Signal 4 — Sensor-noise residual
# ---------------------------------------------------------------------------
def _signal_noise_residual(gray: np.ndarray) -> Tuple[float, str]:
    """Camera sensors leave a high-frequency noise residual with a
    characteristic variance band. We isolate it with a median high-pass
    and flag values that are too low (no sensor → likely synthetic) or
    pathologically high (noise artificially injected to evade detection).
    """
    blurred = cv2.medianBlur(gray, 3)
    residual = gray.astype(np.float32) - blurred.astype(np.float32)
    noise_std = float(residual.std())

    if noise_std < 1.5:
        score = _clip01((1.5 - noise_std) / 1.5)
        verdict = "too clean (no sensor noise)"
    elif noise_std > 14:
        score = _clip01((noise_std - 14) / 10)
        verdict = "abnormally high noise"
    else:
        score = 0.0
        verdict = "within natural range"
    return score, f"noise_std={noise_std:.2f} - {verdict}"


# ---------------------------------------------------------------------------
# Signal 5 — Color histogram entropy
# ---------------------------------------------------------------------------
def _signal_color_entropy(bgr: np.ndarray) -> Tuple[float, str]:
    """Joint-color histogram entropy. Real photographs typically span a
    richer color manifold than diffusion outputs (which tend to cluster
    around a few dominant chroma modes). We compute Shannon entropy on
    a 16×16×16 joint histogram and score lower entropy as more AI-like.
    """
    pixels = bgr.reshape(-1, 3)
    if pixels.shape[0] > 200_000:
        idx = np.random.default_rng(0).choice(pixels.shape[0], 200_000, replace=False)
        pixels = pixels[idx]
    hist, _ = np.histogramdd(pixels, bins=(16, 16, 16), range=((0, 255), (0, 255), (0, 255)))
    p = hist[hist > 0]
    p = p / p.sum()
    entropy = float(-(p * np.log2(p)).sum())
    # Natural KYC scans: ~7.5–10.5. Diffusion: often ≤ 7.0.
    score = _clip01((8.0 - entropy) / 3.0)
    return score, f"color_entropy={entropy:.2f} bits (natural>=8)"


# ---------------------------------------------------------------------------
# Signal 6 — Saturation outlier
# ---------------------------------------------------------------------------
def _signal_saturation(bgr: np.ndarray) -> Tuple[float, str]:
    """Diffusion / SDXL outputs are characteristically oversaturated —
    a side-effect of CLIP-guided training. Phone-captured KYC docs sit
    in the 60–130 saturation range; values above ~160 are anomalous."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat_mean = float(hsv[:, :, 1].mean())
    if sat_mean > 160:
        score = _clip01((sat_mean - 160) / 60)
    elif sat_mean < 15:
        # Pure greyscale outputs (some toy AI generators) are also unusual.
        score = _clip01((15 - sat_mean) / 15) * 0.6
    else:
        score = 0.0
    return score, f"saturation_mean={sat_mean:.1f}"


# ---------------------------------------------------------------------------
# Signal 7 — JPEG history
# ---------------------------------------------------------------------------
def _signal_jpeg_history(path: Path) -> Tuple[float, str]:
    """A genuine "photo of an ID card" almost always arrives as JPEG with
    populated quantization tables. PNG / WebP for a captured document is
    unusual — many AI tools export PNG by default. Missing or trivial
    quant tables on a JPEG are another tell."""
    ext = path.suffix.lower()
    if ext == ".png":
        return 0.55, "format=PNG (unusual for a captured document)"
    if ext == ".webp":
        return 0.45, "format=WebP (unusual for a captured document)"
    if ext in {".jpg", ".jpeg"}:
        try:
            with Image.open(path) as im:
                qt = getattr(im, "quantization", None)
                if not qt:
                    return 0.45, "JPEG missing quantization tables"
                # Some AI tools emit a single default quant table; real
                # cameras emit two (luma + chroma).
                if len(qt) == 1:
                    return 0.30, "JPEG has only one quantization table"
        except (UnidentifiedImageError, OSError):
            return 0.0, "JPEG unreadable"
    return 0.0, f"format={ext or 'unknown'} ok"


# ---------------------------------------------------------------------------
# Signal 8 — Edge density
# ---------------------------------------------------------------------------
def _signal_edge_density(gray: np.ndarray) -> Tuple[float, str]:
    """Canny edge density. Real ID scans land in roughly [0.04, 0.18].
    Very low density means the image is suspiciously smooth (a hallmark
    of low-step diffusion output); very high density can indicate
    upscaler ringing or aggressive sharpening to fake detail."""
    edges = cv2.Canny(gray, 60, 160)
    density = float(edges.mean() / 255.0)
    if density < 0.02:
        score = _clip01((0.02 - density) / 0.02)
        verdict = "over-smoothed"
    elif density > 0.28:
        score = _clip01((density - 0.28) / 0.15)
        verdict = "over-sharp / ringing"
    else:
        score = 0.0
        verdict = "natural"
    return score, f"edge_density={density:.3f} - {verdict}"


# ---------------------------------------------------------------------------
# Signal 9 — Layout anchor (Aadhaar-specific structural template check)
# ---------------------------------------------------------------------------
def _signal_layout_anchor_aadhaar(bgr: np.ndarray) -> Tuple[float, str]:
    """Aadhaar template assertions a real card MUST satisfy.

    Diffusion models are great at *looking* like an Aadhaar but bad at
    placing structural anchors. We check three:

    1. **QR present in the lower-right quadrant.** UIDAI prints a Secure
       QR there on every Aadhaar issued since 2018. No QR → strong AI signal.
    2. **Aspect ratio ~ 1.59:1** (standard CR-80 ID card). Generated
       images often default to 1:1 or 4:3.
    3. **High-contrast emblem region in the top band.** The Ashoka emblem
       + "GOVT OF INDIA" header gives a Canny-edge density >> 0.10 in the
       top 25% of a real Aadhaar; smooth diffusion outputs collapse here.

    Combined by `max` — any single anchor failure is enough to flag.
    """
    h, w = bgr.shape[:2]
    if h < 100 or w < 100:
        return 0.5, "image too small for layout check"

    # 1. QR detection.
    qr_score = 1.0
    qr_detail = "QR not detected"
    try:
        detector = cv2.QRCodeDetector()
        retval, points = detector.detect(bgr)
        if retval and points is not None and len(points) > 0:
            pts = points.reshape(-1, 2)
            cx_qr = float(pts[:, 0].mean())
            cy_qr = float(pts[:, 1].mean())
            # Expected: right 40%, bottom 60%.
            in_x = cx_qr >= 0.55 * w
            in_y = cy_qr >= 0.35 * h
            if in_x and in_y:
                qr_score = 0.0
                qr_detail = f"QR at ({cx_qr/w:.2f}, {cy_qr/h:.2f}) - expected region"
            else:
                qr_score = 0.6
                qr_detail = f"QR off-position ({cx_qr/w:.2f}, {cy_qr/h:.2f})"
    except cv2.error:
        qr_score = 0.5
        qr_detail = "QR detector errored"

    # 2. Aspect ratio.
    aspect = w / float(h)
    aspect_target = 1.59
    aspect_dev = abs(aspect - aspect_target) / aspect_target
    if aspect_dev > 0.20:
        aspect_score = _clip01((aspect_dev - 0.20) / 0.30 + 0.5)
        aspect_detail = f"aspect={aspect:.2f} (target 1.59, dev {aspect_dev:.0%})"
    else:
        aspect_score = 0.0
        aspect_detail = f"aspect={aspect:.2f} ok"

    # 3. Top-band edge density — header / emblem region.
    top = bgr[: max(1, h // 4), :]
    gray_top = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY) if top.ndim == 3 else top
    edges_top = cv2.Canny(gray_top, 60, 160)
    top_density = float(edges_top.mean() / 255.0)
    # Real Aadhaars: top-band density observed ~0.10-0.25. Diffusion
    # outputs collapse to <0.04.
    if top_density < 0.04:
        emblem_score = 1.0
        emblem_detail = f"top-band edge density {top_density:.3f} - missing emblem/header"
    elif top_density < 0.08:
        emblem_score = 0.6
        emblem_detail = f"top-band edge density {top_density:.3f} - weak emblem"
    else:
        emblem_score = 0.0
        emblem_detail = f"top-band edge density {top_density:.3f} ok"

    score = max(qr_score, aspect_score, emblem_score)
    detail = f"qr: {qr_detail}; {aspect_detail}; emblem: {emblem_detail}"
    return score, detail


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DetectorConfig:
    """Weights and thresholds. Per-doc-type profiles override the default."""

    weights: Dict[SignalCode, float]
    suspicious_threshold: float = 0.40   # below → LIKELY_AUTHENTIC
    ai_threshold: float = 0.62           # above → LIKELY_AI
    # When set, the detector uses these profiles instead of the global
    # weights/thresholds when a `doc_type` is provided at call time.
    per_doc_profiles: Dict[str, "DetectorConfig"] = None  # type: ignore[assignment]


def _aadhaar_profile() -> DetectorConfig:
    """Aadhaar-specific tuning. Tighter thresholds + LAYOUT_ANCHOR weight.

    Justification for tightening to (0.35, 0.55): with the new
    LAYOUT_ANCHOR signal, an AI-generated Aadhaar without a real QR or
    with a wrong aspect ratio jumps ~0.25 in raw score — a healthy margin
    above the new threshold. False positives on real Aadhaars remain
    bounded because real cards score 0.0 on LAYOUT_ANCHOR.
    """
    return DetectorConfig(
        weights={
            SignalCode.LAYOUT_ANCHOR:  0.25,
            SignalCode.METADATA:       0.18,
            SignalCode.FFT_SPECTRUM:   0.14,
            SignalCode.ELA:            0.12,
            SignalCode.NOISE_RESIDUAL: 0.12,
            SignalCode.EDGE_DENSITY:   0.09,
            SignalCode.JPEG_HISTORY:   0.05,
            SignalCode.COLOR_ENTROPY:  0.03,
            SignalCode.SATURATION:     0.02,
        },
        suspicious_threshold=0.35,
        ai_threshold=0.55,
    )


def _default_config() -> DetectorConfig:
    cfg = DetectorConfig(
        weights={
            SignalCode.METADATA:       0.28,
            SignalCode.FFT_SPECTRUM:   0.16,
            SignalCode.ELA:            0.14,
            SignalCode.NOISE_RESIDUAL: 0.14,
            SignalCode.COLOR_ENTROPY:  0.08,
            SignalCode.SATURATION:     0.06,
            SignalCode.JPEG_HISTORY:   0.10,
            SignalCode.EDGE_DENSITY:   0.04,
        },
    )
    object.__setattr__(cfg, "per_doc_profiles", {"AADHAAR": _aadhaar_profile()})
    return cfg


class AIGeneratedDocumentDetector:
    """Forensic detector for AI-synthesized KYC documents.

    Stateless after construction — safe to share across threads. Pass a
    custom :class:`DetectorConfig` to override weights or thresholds
    after calibrating on your own labeled dataset.
    """

    PER_SIGNAL_TRIGGER = 0.50  # signal flagged in the audit trail above this

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self._cfg = config or _default_config()

    # ------------------------ public ------------------------
    def detect(self, file_path: str | Path, doc_type: Optional[str] = None) -> AIDetectionResult:
        """Run forensic detection. Pass `doc_type` (e.g. "AADHAAR") to
        engage the per-doc-type weights and structural anchor checks.
        """
        started = time.perf_counter()
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            return self._insufficient(path, started, "file not found")

        bgr = _load_bgr(path)
        if bgr is None or bgr.size == 0:
            return self._insufficient(path, started, "image undecodable")

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr

        cfg = self._resolve_config(doc_type)

        signals: List[SignalScore] = []
        signals.append(self._wrap(cfg, SignalCode.METADATA,       *_signal_metadata(path)))
        signals.append(self._wrap(cfg, SignalCode.FFT_SPECTRUM,   *_signal_fft_spectrum(gray)))
        signals.append(self._wrap(cfg, SignalCode.ELA,            *_signal_ela(bgr)))
        signals.append(self._wrap(cfg, SignalCode.NOISE_RESIDUAL, *_signal_noise_residual(gray)))
        signals.append(self._wrap(cfg, SignalCode.COLOR_ENTROPY,  *_signal_color_entropy(bgr)))
        signals.append(self._wrap(cfg, SignalCode.SATURATION,     *_signal_saturation(bgr)))
        signals.append(self._wrap(cfg, SignalCode.JPEG_HISTORY,   *_signal_jpeg_history(path)))
        signals.append(self._wrap(cfg, SignalCode.EDGE_DENSITY,   *_signal_edge_density(gray)))

        if (doc_type or "").upper() == "AADHAAR":
            signals.append(self._wrap(cfg, SignalCode.LAYOUT_ANCHOR,
                                       *_signal_layout_anchor_aadhaar(bgr)))

        total_w = sum(s.weight for s in signals) or 1.0
        confidence = float(sum(s.score * s.weight for s in signals) / total_w)

        if confidence >= cfg.ai_threshold:
            verdict = AIVerdict.LIKELY_AI
        elif confidence >= cfg.suspicious_threshold:
            verdict = AIVerdict.SUSPICIOUS
        else:
            verdict = AIVerdict.LIKELY_AUTHENTIC

        triggered = [s for s in signals if s.triggered]
        explanation = self._explain(verdict, confidence, triggered)
        elapsed = time.perf_counter() - started

        result = AIDetectionResult(
            file_name=path.name,
            is_ai_generated=(verdict == AIVerdict.LIKELY_AI),
            verdict=verdict,
            confidence=confidence,
            signals=signals,
            explanation=explanation,
            processing_time_s=elapsed,
        )
        logger.info(
            "ai-detector %s (doc=%s): %s conf=%.3f triggered=%d time=%.3fs",
            path.name, doc_type or "?", verdict.value, confidence, len(triggered), elapsed,
        )
        return result

    # ------------------------ internals ------------------------
    def _resolve_config(self, doc_type: Optional[str]) -> DetectorConfig:
        if not doc_type:
            return self._cfg
        profiles = getattr(self._cfg, "per_doc_profiles", None) or {}
        return profiles.get(str(doc_type).upper(), self._cfg)

    def _wrap(self, cfg: DetectorConfig, code: SignalCode, score: float, detail: str) -> SignalScore:
        return SignalScore(
            code=code,
            score=_clip01(score),
            triggered=score >= self.PER_SIGNAL_TRIGGER,
            weight=cfg.weights.get(code, 0.0),
            detail=detail,
        )

    @staticmethod
    def _explain(verdict: AIVerdict, conf: float, triggered: List[SignalScore]) -> str:
        if not triggered:
            return f"{verdict.value} (confidence={conf:.2f}); no individual signal crossed its threshold."
        names = ", ".join(f"{s.code.value}({s.score:.2f})" for s in triggered)
        return f"{verdict.value} (confidence={conf:.2f}); triggered: {names}."

    def _insufficient(self, path: Path, started: float, reason: str) -> AIDetectionResult:
        return AIDetectionResult(
            file_name=path.name,
            is_ai_generated=False,
            verdict=AIVerdict.INSUFFICIENT_DATA,
            confidence=0.0,
            signals=[],
            explanation=f"INSUFFICIENT_DATA: {reason}",
            processing_time_s=time.perf_counter() - started,
        )


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_detector: Optional[AIGeneratedDocumentDetector] = None


def get_ai_detector() -> AIGeneratedDocumentDetector:
    global _detector
    if _detector is None:
        with _lock:
            if _detector is None:
                _detector = AIGeneratedDocumentDetector()
    return _detector


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    if len(sys.argv) < 2:
        print("usage: python -m app.services.ai_generator_detector <image-path> [<image-path> ...]")
        raise SystemExit(2)
    det = get_ai_detector()
    for arg in sys.argv[1:]:
        r = det.detect(arg)
        print(f"\n=== {r.file_name} ===")
        print(f"  verdict   : {r.verdict.value}")
        print(f"  confidence: {r.confidence:.3f}")
        print(f"  time      : {r.processing_time_s*1000:.1f} ms")
        print(f"  explain   : {r.explanation}")
        print("  signals:")
        for s in r.signals:
            flag = "X" if s.triggered else " "
            print(f"    [{flag}] {s.code.value:<15} score={s.score:.2f} w={s.weight:.2f}  {s.detail}")
