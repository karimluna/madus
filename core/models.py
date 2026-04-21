"""
State schemas for langgraph.
https://langchain-ai.github.io/langgraph/concepts/low_level/#state
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


# here we let use default values so that langgraph can construct
# partial states without errors: 
class DocumentState(BaseModel):
    '''`pdf_path` and `question` muste be set before invoking the graph'''
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pdf_path: str = ""
    question: str = ""

    # populated by extraction node 
    text_chunks: list[str] = []
    images: list[ImageChunk] = []
    tables: list[TableChunk] = []

    # populated by orchestator
    active_agents: list[str] = []

    # populated by specialist agents
    text_answer: Optional[str] = None
    image_answer: Optional[str] = None
    table_answer: Optional[str] = None

    # populated by critics
    critique: Optional[str] = None
    needs_retry: bool = False
    retry_count: int = 0        # fir reflexion section

    # populated by summarizer
    final_answer: Optional[str] = None
    confidence: Optional[float] = None