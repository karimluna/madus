"""Integration tests, real LLM calls.
These tests require OPENAI_API_KEY or a running Ollama instance.
"""

import os
import pytest

# Skip entire module in CI unless explicitly enabled
pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_INTEGRATION_TESTS"),
    reason="Set RUN_INTEGRATION_TESTS=1 to run integration tests",
)


@pytest.fixture
def digital_pdf():
    path = os.path.join(os.path.dirname(__file__), "..", "fixtures", "digital.pdf")
    if not os.path.exists(path):
        pytest.skip("digital.pdf fixture not found")
    return path


@pytest.fixture
def image_heavy_pdf():
    path = os.path.join(os.path.dirname(__file__), "..", "fixtures", "image_heavy.pdf")
    if not os.path.exists(path):
        pytest.skip("image_heavy.pdf fixture not found")
    return path


@pytest.fixture
def scanned_pdf():
    path = os.path.join(os.path.dirname(__file__), "..", "fixtures", "scanned.pdf")
    if not os.path.exists(path):
        pytest.skip("scanned.pdf fixture not found")
    return path


class TestExtractionIntegration:
    def test_ocr_extracts_text(self, digital_pdf):
        from services.extraction.ocr import extract_text_chunks

        chunks = extract_text_chunks(digital_pdf)
        assert len(chunks) > 0
        assert any(len(c) > 50 for c in chunks)

    def test_vision_extracts_images(self, image_heavy_pdf):
        from services.extraction.vision import extract_images

        images = extract_images(image_heavy_pdf)
        assert len(images) > 0
        assert all(img.image_b64 for img in images)

    def test_table_extraction(self, digital_pdf):
        from services.extraction.chunker import extract_tables

        tables = extract_tables(digital_pdf)
        # May or may not have tables depending on fixture
        for t in tables:
            assert t.markdown
            assert t.raw_cells


class TestGraphIntegration:
    @pytest.mark.asyncio
    async def test_text_only_pipeline(self, digital_pdf):
        from core.models import DocumentState
        from services.reasoning.graph.builder import build_graph

        graph = build_graph()
        result = await graph.ainvoke(
            DocumentState(
                pdf_path=digital_pdf,
                question="What is the main topic of this document?",
            ).model_dump()
        )

        if isinstance(result, dict):
            state = DocumentState(**result)
        else:
            state = result

        assert state.final_answer is not None
        assert len(state.final_answer) > 10
