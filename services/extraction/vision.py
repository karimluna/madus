"""Layout detection for figure regions using OpenCV.
Classical image processing: grayscale -> Otsu threshold -> morphological
closing -> contour detection -> area + aspect-ratio filtering.

No deep learning needed because figures have dense pixel regions that
connected-component analysis finds reliably.

ref: Otsu's method: https://en.wikipedia.org/wiki/Otsu%27s_method
ref: OpenCV morphological ops: https://docs.opencv.org/5.0.0-alpha/d9/d61/tutorial_py_morphological_ops.html
"""

import base64

import cv2
import fitz
import numpy as np

from core.models import ImageChunk


def _detect_figures(page_img) -> list[tuple[int, int, int, int]]:
    """Run the OpenCV pipeline on a single page image.
    Returns bounding boxes (x, y, w, h) for detected figure regions.
    """
    gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological closing with wide horizontal kernel connects
    # dense figure pixels without expanding region boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_area = page_img.shape[0] * page_img.shape[1]
    min_area = 0.02 * page_area  # 2% of page area

    bboxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        # Aspect ratio filter: text columns are tall/narrow,
        # figures tend toward landscape or square
        if 0.5 < w / h < 6.0:
            bboxes.append((x, y, w, h))
    return bboxes


def extract_images(pdf_path: str) -> list[ImageChunk]:
    """Extract figure regions from each page as base64-encoded PNGs."""
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=150)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

        # PyMuPDF gives RGB(A), OpenCV expects BGR
        if pix.n == 4:
            page_img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            page_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        bboxes = _detect_figures(page_img)

        for x, y, w, h in bboxes:
            crop = page_img[y : y + h, x : x + w]
            _, buf = cv2.imencode(".png", crop)
            b64 = base64.b64encode(buf).decode("utf-8")
            chunks.append(ImageChunk(page=page_num, image_b64=b64, bbox=(x, y, w, h)))
    doc.close()
    return chunks
