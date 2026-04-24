"""Shared test fixtures."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir():
    return FIXTURES_DIR


@pytest.fixture
def sample_pdf_path():
    """Path to a sample PDF for testing. Skips if not available"""
    path = FIXTURES_DIR / "sample.pdf"
    if not path.exists():
        pytest.skip("Sample PDF not found in tests/fixtures/")
    return str(path)
