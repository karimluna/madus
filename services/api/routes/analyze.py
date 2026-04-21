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
        cached_state = await get_cached(tmp.name, question)
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

        # 90 second overall timeout — prevents infinite hangs
        result = await asyncio.wait_for(
            graph.ainvoke(initial.model_dump()),
            timeout=90.0,
        )

        final_state = DocumentState(**result) if isinstance(result, dict) else result

        await set_cached(tmp.name, question, final_state)

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

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Analysis timed out")
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
    """SSE variant. Each yield is flushed immediately by uvicorn
    because we use an async generator with small payloads.

    The extraction runs in a ProcessPoolExecutor (via runners.py)
    so the event loop is free to yield SSE events between stages.
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

            # Cache check, now with socket timeouts so it won't hang
            cached_state = get_cached(tmp.name, question)

            if cached_state and cached_state.final_answer:
                yield emit("cache_hit")
                yield emit("done", json.dumps({
                    "doc_id": cached_state.doc_id,
                    "final_answer": cached_state.final_answer,
                    "confidence": cached_state.confidence,
                    "cached": True,
                }))
                return

            yield emit("extracting", "Running OCR, layout detection, and table parsing in background process")

            from services.api.app import get_graph
            graph = get_graph()
            initial = DocumentState(pdf_path=tmp.name, question=question)

            yield emit("running_agents", "Agents processing the document")

            # Run with timeout: 90 seconds max for CPU bounds
            try:
                result = await asyncio.wait_for(
                    graph.ainvoke(initial.model_dump()),
                    timeout=90.0,
                )
            except asyncio.TimeoutError:
                yield emit("error", "Analysis timed out after 90 seconds")
                return

            final_state = DocumentState(**result) if isinstance(result, dict) else result

            set_cached(tmp.name, question, final_state)

            try:
                write_to_kb(final_state)
            except Exception as e:
                logger.warning("Databricks sink failed: %s", e)

            yield emit("done", json.dumps({
                "doc_id": final_state.doc_id,
                "final_answer": final_state.final_answer or "Unable to determine an answer.",
                "confidence": final_state.confidence,
                "cached": False,
            }))

        except Exception as e:
            logger.exception("Streaming analysis failed")
            yield emit("error", str(e))
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            # Prevent proxy buffering, each event must be sent immediately
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no", 
        },
    )