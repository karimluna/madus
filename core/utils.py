"""Shared utilities used across reasonign nodes"""

import json
import re


def parse_json(text: str) -> dict:
    """Extract JSON from LLM output. Handles markdown code blocks
    and trailing prose that LLMd sometimes add.

    regex-based extraction is fragile but pragmatic, but force small
    models to output strict JSON files is unreliable"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # find first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM output: {text[:200]}")


def table_to_markdown(table: list[list[str | None]]) -> str:
    """Convert pdfplumber table cells to markdown format"""
    if not table:
        return ""
    rows = []
    for i, row in enumerate(table):
        cells = [str(cell or "").replace("|", "\\|") for cell in row]
        rows.append("| " + " | ".join(cells) + " |")
        if i == 0:
            rows.append("| " + " | ".join(["---"] * len(cells)) + " |")

    return "\n".join(rows)
