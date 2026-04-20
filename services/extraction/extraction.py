import pdfplumber

def extract_tables(pdf_path: str) -> list[TableChunk]:
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            for table in page.extract_tables():
                chunks.append(TableChunk(page=i, markdown=_to_md(table), raw_cells=table))
    return chunks