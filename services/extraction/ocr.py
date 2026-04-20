from paddleocr import PaddleOCR
import fitz, numpy as np

_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

def extract_text_chunks(pdf_path: str) -> list[str]:
    doc = fitz.open(pdf_path)
    chunks = []
    for page in doc: # recognition
        pix = page.get_pixmap(dpi=150)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        result = _ocr.ocr(img, cls=True) 
        if result and result[0]:
            text = " ".join(line[1][0] for line in result[0])
            chunks.append(text)
    return chunks