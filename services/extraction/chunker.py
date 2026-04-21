"""Table extraction via pdfplumber's lattice algorithm.
Detects table structure from visible PDF ruling lines.
For scanned PDFs without ruling lines, the image agent handles
table interpretation visually.

ref: pdfplumber: https://github.com/jsvine/pdfplumber
"""

import pdfplumber

from core.models import TableChunk
from core.utils import table_to_markdown


def extract_tables(pdf_path: str) -> list[TableChunk]:
    """Extract tables from each page. Returns empty list for
    pages without detectable table structures.
    """
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            for table in page.extract_tables():
                clean_cells = [
                    [str(cell or "") for cell in row] for row in table
                ]
                chunks.append(
                    TableChunk(
                        page=i,
                        markdown=table_to_markdown(table),
                        raw_cells=clean_cells,
                    )
                )
    return chunks