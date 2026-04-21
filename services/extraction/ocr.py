"""PaddleOCR text extraction from PDF pages.
Uses DB detection + CRNN recognition pipeline at 150 DPI,
the sweet spot between accuracy and memory

ref: https://arxiv.org/abs/1911.08947"""


import threading 
import fitz 
import numpy as np
from paddleocr import PaddleOCR
import logging
import os

logging.getLogger("ppocr").setLevel(logging.WARNING)
_ocr = None 
_lock = threading.Lock()


def _get_ocr() -> PaddleOCR:
    global _ocr
    if _ocr is None:
        with _lock:
            if _ocr is None:
                _ocr = PaddleOCR(use_textline_orientation=True, lang="en", enable_mkldnn=False) # https://github.com/PaddlePaddle/Paddle/issues/77340
    return _ocr  
    

def extract_text_chunks(pdf_path: str) -> list[str]:
    """Extract text from each page using PaddleOCR.
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

        page = result[0]

        # handle new + old formats safely
        if isinstance(page, dict):  
            texts = page.get("rec_texts", [])
        else:  # legacy fallback
            texts = [line[1][0] for line in page if line and len(line) > 1]

        if texts:
            chunks.append(" ".join(texts))
    
    doc.close()
    return chunks