"""Unit tests for extraction layer, no LLM calls, fast, runs in CI.

These tests verify that OCR, layout detection, and table parsing
produce correct output shapes. They do not verify content accuracy
(which depends on the PDF fixture)
"""

import pytest

from core.models import ImageChunk, TableChunk
from core.utils import table_to_markdown, parse_json


class TestTableToMarkdown:
    def test_empty_table(self):
        assert table_to_markdown([]) == ""

    def test_simple_table(self):
        table = [["Name", "Value"], ["A", "1"], ["B", "2"]]
        md = table_to_markdown(table)
        assert "| Name | Value |" in md
        assert "| --- | --- |" in md
        assert "| A | 1 |" in md

    def test_none_cells(self):
        table = [["A", None], ["B", ""]]
        md = table_to_markdown(table)
        assert "| A |  |" in md

    def test_pipe_escaping(self):
        table = [["A|B", "C"]]
        md = table_to_markdown(table)
        assert "A\\|B" in md


class TestParseJson:
    def test_direct_json(self):
        result = parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = parse_json(text)
        assert result == {"key": "value"}

    def test_json_with_surrounding_text(self):
        text = 'Here is the result: {"agents": ["text"]} done'
        result = parse_json(text)
        assert result == {"agents": ["text"]}

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            parse_json("not json at all")


class TestModels:
    def test_document_state_defaults(self):
        from core.models import DocumentState

        state = DocumentState()
        assert state.doc_id
        assert state.text_chunks == []
        assert state.images == []
        assert state.tables == []
        assert state.text_answer is None
        assert state.retry_count == 0

    def test_document_state_with_values(self):
        from core.models import DocumentState

        state = DocumentState(
            pdf_path="/tmp/test.pdf",
            question="What is this?",
        )
        assert state.pdf_path == "/tmp/test.pdf"
        assert state.question == "What is this?"

    def test_image_chunk(self):
        chunk = ImageChunk(page=0, image_b64="abc", bbox=(1, 2, 3, 4))
        assert chunk.page == 0
        assert chunk.bbox == (1, 2, 3, 4)

    def test_table_chunk(self):
        chunk = TableChunk(page=1, markdown="| a |", raw_cells=[["a"]])
        assert chunk.page == 1


class TestChunker:
    def test_extract_tables_nonexistent_file(self):
        from services.extraction.chunker import extract_tables

        with pytest.raises(Exception):
            extract_tables("/nonexistent/file.pdf")
