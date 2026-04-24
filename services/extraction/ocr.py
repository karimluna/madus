"""Text extraction: fitz native layer first, PaddleOCR fallback for scanned pages.
Native extraction is ~100x faster and free. OCR only runs when a page
has no embedded text (scanned PDFs, image-only pages).

Uses PP-OCRv5_mobile_det for lighter CPU footprint.
PaddleOCR 3.0+ uses .predict() instead of .ocr() and returns
a dict with 'rec_texts' instead of nested lists.

ref: https://arxiv.org/abs/1911.08947
ref: https://github.com/PaddlePaddle/Paddle/issues/77340 (MKL-DNN crash)
"""

import threading
import logging

import fitz
import numpy as np
from paddleocr import PaddleOCR

logging.getLogger("ppocr").setLevel(logging.WARNING)

_ocr = None
_lock = threading.Lock()

# Minimum characters on a page to trust the native text layer.
# Pages below this threshold are treated as image-only and sent to OCR.
_MIN_NATIVE_CHARS = 50


def _get_ocr() -> PaddleOCR:
    """Lazy singleton with double-checked locking.
    OCR model is only loaded if at least one page needs it.
    """
    global _ocr
    if _ocr is None:
        with _lock:
            if _ocr is None:
                _ocr = PaddleOCR(
                    use_textline_orientation=True,
                    lang="en",
                    enable_mkldnn=False,  # avoids crash on some CPU configs
                    text_detection_model_name="PP-OCRv5_mobile_det",
                )
    return _ocr


def _ocr_page(page: fitz.Page) -> str:
    """Rasterize a single page and run PaddleOCR on it.
    150 DPI is the minimum for reliable CRNN character recognition.
    """
    pix = page.get_pixmap(dpi=150)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

    result = _get_ocr().predict(img)
    if not result:
        return ""

    page_result = result[0]

    if isinstance(page_result, dict):
        # PaddleOCR 3.0+
        texts = page_result.get("rec_texts", [])
    else:
        # Legacy fallback for PaddleOCR < 3.0
        texts = [line[1][0] for line in page_result if line and len(line) > 1]

    return " ".join(texts)


def extract_text_chunks(pdf_path: str) -> list[str]:
    """Extract one text chunk per page.

    Fast path: fitz native text layer (no ML, microseconds per page).
    Slow path: PaddleOCR, only triggered for scanned/image-only pages.
    A mixed PDF (some native, some scanned) is handled page by page.
    """
    doc = fitz.open(pdf_path)
    chunks = []

    for page in doc:
        native = page.get_text().strip()

        if len(native) >= _MIN_NATIVE_CHARS:
            # Native text layer is present and usable
            chunks.append(native)
        else:
            # Scanned or image-only page: fall back to OCR
            ocr_text = _ocr_page(page)
            if ocr_text:
                chunks.append(ocr_text)

    doc.close()
    return chunks
