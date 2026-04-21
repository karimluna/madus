"""POST /api/analyze. The main document analysis endpoint.
Receives a PDF file and a question, returns the analyzed answer

Flow:
1. Save uploaded PDF to temp file
2. Check Redis cache (SHA-256 content address)
3. If cache hit, return cached result (~15ms)
4. If cache miss, run the full LangGraph pipeline
5. Cache the result, optionally write to Databricks
6. Return the final answer
"""

import os
import tempfile
import logging

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from core.models import DocumentState
from core.cache import get_cached, set_cached
from core.databricks_sink import write_to_kb
from services.api.app import get_graph

logger = logging.getLogger(__name__)
router = APIRouter()


class AnalyzeResponse(BaseModel):
    doc_id: str
    final_answer: str
    confidence: float | None = None
    cached: bool = False


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    question: str = Form(...),
):
    """Analyze a PDF document and answer a question about it."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Save uploaded file to temp location
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        tmp.close()

        # Check cache
        cached_state = get_cached(tmp.name)
        if cached_state and cached_state.final_answer:
            logger.info("Cache hit for doc_id=%s", cached_state.doc_id)
            return AnalyzeResponse(
                doc_id=cached_state.doc_id,
                final_answer=cached_state.final_answer,
                confidence=cached_state.confidence,
                cached=True,
            )

        # Run the graph
        graph = get_graph()
        initial = DocumentState(
            pdf_path=tmp.name,
            question=question,
        )
        result = await graph.ainvoke(initial.model_dump())

        # Convert result back to DocumentState
        if isinstance(result, dict):
            final_state = DocumentState(**result)
        else:
            final_state = result

        # Cache the result
        set_cached(tmp.name, final_state)

        # Optional Databricks sink
        try:
            write_to_kb(final_state)
        except Exception as e:
            logger.warning("Databricks sink failed: %s", e)

        return AnalyzeResponse(
            doc_id=final_state.doc_id,
            final_answer=final_state.final_answer or "Unable to determine an answer.",
            confidence=final_state.confidence,
            cached=False,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp.name)
        except OSError:
            pass