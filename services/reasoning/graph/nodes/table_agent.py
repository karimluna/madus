"""Table agent: LLM reasoning over structured table data.
Tables are extracted by pdfplumber's lattice algorithm and
converted to markdown for the LLM.
"""

from core.models import DocumentState
from core.config import get_chat_llm, load_prompt

_template = load_prompt("table_agent.txt")


async def table_agent_node(state: DocumentState) -> dict:
    """Interpret table data to answer the question.
    Returns early if table modality is not active or no tables found.
    """
    if "table" not in state.active_agents or not state.tables:
        return {"table_answer": state.table_answer or "No table content available."}

    critique_section = ""
    if state.critique:
        critique_section = f"Previous attempt feedback: {state.critique}"

    tables_md = "\n\n".join(
        f"Table {i+1} (page {t.page}):\n{t.markdown}"
        for i, t in enumerate(state.tables)
    )

    prompt = (
        _template
        .replace("{{question}}", state.question)
        .replace("{{tables}}", tables_md)
        .replace("{{critique_section}}", critique_section)
    )

    llm = get_chat_llm()
    response = await llm.ainvoke(prompt)
    return {"table_answer": response.content}