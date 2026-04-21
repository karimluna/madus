"""Summarizer: synthesizes agent answers into a final coherent response.
If the critique couldn't be resolved, includes a note about limitations.
"""

from core.models import DocumentState
from core.config import get_chat_llm, load_prompt
from core.utils import parse_json

_template = load_prompt("summarizer.txt")


async def summarizer_node(state: DocumentState) -> dict:
    """Combine all agent answers into a final response with confidence."""
    critique_note = ""
    if state.critique and state.needs_retry:
        critique_note = (
            f"Note: The system was unable to fully resolve the question. "
            f"Unresolved feedback: {state.critique}"
        )

    prompt = (
        _template
        .replace("{{question}}", state.question)
        .replace("{{text_answer}}", state.text_answer or "N/A")
        .replace("{{image_answer}}", state.image_answer or "N/A")
        .replace("{{table_answer}}", state.table_answer or "N/A")
        .replace("{{critique_note}}", critique_note)
    )

    llm = get_chat_llm()
    response = await llm.ainvoke(prompt)

    try:
        data = parse_json(response.content)
        return {
            "final_answer": data.get("final_answer", response.content),
            "confidence": data.get("confidence"),
        }
    except ValueError:
        # LLM didn't return JSON: use raw content as answer
        return {"final_answer": response.content, "confidence": None}