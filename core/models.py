"""
State schemas for langgraph.
"""
from pydantic import BaseModel, Field
from typing import Optional
import uuid


class ImageChunk(BaseModel):
    page: int 
    image_b64: str
    bbox: tuple[int, int, int, int] # (x, y, w, h) in PDF points


class TableChunk(BaseModel):
    page: int
    markdown: str
    raw_cells: list[list[str]] = []


class DocumentState(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pdf_path: str
    question: str

    text_chunks: list[str] = []
    images: list[ImageChunk] = []
    tables: list[TableChunk] = []

    text_answer: Optional[str] = None
    image_answer: Optional[str] = None
    table_answer: Optional[str] = None

    critique: Optional[str] = None
    needs_retry: bool = False
    retry_count: int = 0        # fir reflexion section

    final_answer: Optional[str] = None
    confidence: Optional[float] = None