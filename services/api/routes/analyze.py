"""POST /api/analyze : the main document analysis endpoint.
Receives a PDF file and a question, returns the analyzed answer.

Flow:
1. Save uploaded PDF to temp file
2. Check Redis cache (SHA-256 content + question address)
3. If cache hit, return cached result (~15ms)
4. If cache miss, run the full LangGraph pipeline
5. Cache the result, optionally write to Databricks
6. Return the final answer

POST /api/analyze/stream : SSE variant that emits progress events
during long-running cold runs so the demo doesn't show a blank screen.
"""

import os
import json
import tempfile
import logging
import asyncio

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.models import DocumentState
from core.cache import get_cached, set_cached
from core.databricks_sink import write_to_kb

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

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        tmp.close()

        # Cache key now includes the question, same PDF, different
        # question produces an independent entry
        cached_state = get_cached(tmp.name, question)
        if cached_state and cached_state.final_answer:
            logger.info("Cache hit for doc_id=%s", cached_state.doc_id)
            return AnalyzeResponse(
                doc_id=cached_state.doc_id,
                final_answer=cached_state.final_answer,
                confidence=cached_state.confidence,
                cached=True,
            )

        from services.api.app import get_graph
        graph = get_graph()
        initial = DocumentState(pdf_path=tmp.name, question=question)
        result = await graph.ainvoke(initial.model_dump())

        final_state = DocumentState(**result) if isinstance(result, dict) else result

        set_cached(tmp.name, question, final_state)

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
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@router.post("/analyze/stream")
async def analyze_stream(
    file: UploadFile = File(...),
    question: str = Form(...),
):
    """SSE variant of /analyze. Emits progress events so the client
    can show a live status indicator during the cold-run pipeline.

    Events are newline-delimited JSON strings prefixed with 'data: ',
    following the Server-Sent Events spec. The final event carries
    the complete result payload.

    Example client (JavaScript):
        const es = new EventSource('/api/analyze/stream')
        es.onmessage = (e) => console.log(JSON.parse(e.data))
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    content = await file.read()

    async def event_stream():
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        try:
            tmp.write(content)
            tmp.flush()
            tmp.close()

            def emit(stage: str, detail: str = "") -> str:
                return f"data: {json.dumps({'stage': stage, 'detail': detail})}\n\n"

            # Cache check
            cached_state = get_cached(tmp.name, question)
            if cached_state and cached_state.final_answer:
                yield emit("cache_hit")
                yield f"data: {json.dumps({'stage': 'done', 'doc_id': cached_state.doc_id, 'final_answer': cached_state.final_answer, 'confidence': cached_state.confidence, 'cached': True})}\n\n"
                return

            yield emit("extracting", "Running OCR, layout detection, and table parsing")

            # Run the grap LangGraph is async so we await it directly.
            # We cannot yield mid-graph without restructuring the graph itself,
            # so we emit coarse-grained stage markers around the full call...
            from services.api.app import get_graph
            graph = get_graph()
            initial = DocumentState(pdf_path=tmp.name, question=question)

            # Emit agent stage before blocking on the graph
            yield emit("running_agents", "Text, image, and table agents running in parallel")

            result = await graph.ainvoke(initial.model_dump())
            final_state = DocumentState(**result) if isinstance(result, dict) else result

            yield emit("critic", "Critic evaluating agent outputs")

            set_cached(tmp.name, question, final_state)

            try:
                write_to_kb(final_state)
            except Exception as e:
                logger.warning("Databricks sink failed: %s", e)

            yield f"data: {json.dumps({'stage': 'done', 'doc_id': final_state.doc_id, 'final_answer': final_state.final_answer or 'Unable to determine an answer.', 'confidence': final_state.confidence, 'cached': False})}\n\n"

        except Exception as e:
            logger.exception("Streaming analysis failed")
            yield f"data: {json.dumps({'stage': 'error', 'detail': str(e)})}\n\n"
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    return StreamingResponse(event_stream(), media_type="text/event-stream")



