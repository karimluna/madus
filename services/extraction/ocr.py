"""PaddleOCR text extraction from PDF pages.
Uses DB detection + CRNN recognition pipeline at 150 DPI,
the sweet spot between accuracy and memory.

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


def _get_ocr() -> PaddleOCR:
    """Lazy singleton with double-checked locking."""
    global _ocr
    if _ocr is None:
        with _lock:
            if _ocr is None:
                _ocr = PaddleOCR(
                    use_textline_orientation=True,
                    lang="en",
                    enable_mkldnn=False,  # avoids crash on some CPU configs
                )
    return _ocr


def extract_text_chunks(pdf_path: str) -> list[str]:
    """Extract text from each page using PaddleOCR 3.0+.
    A 20-page PDF should complete in under 5 seconds on CPU.
    """
    ocr = _get_ocr()
    doc = fitz.open(pdf_path)
    chunks = []

    for page in doc:
        pix = page.get_pixmap(dpi=150)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.h, pix.w, pix.n
        )
        result = ocr.predict(img)

        if not result:
            continue

        page_result = result[0]

        # PaddleOCR 3.0+ returns dict with 'rec_texts'
        if isinstance(page_result, dict):
            texts = page_result.get("rec_texts", [])
        else:
            # Legacy fallback for PaddleOCR < 3.0
            texts = [line[1][0] for line in page_result if line and len(line) > 1]

        if texts:
            chunks.append(" ".join(texts))

    doc.close()
    return chunks