"""General Orchestrator: ReAct loop.
Maps context c_t = (q, history, critique) to a decision about
which agents to dispatch. On retry, reads the critic's verbal
feedback to selectively re-dispatch only the weak agent.

ref: ReAct paper: https://arxiv.org/abs/2210.03629
ref: nibzard/awesome-agentic-patterns - Hierarchical Orchestration
"""

from core.models import DocumentState
from core.config import get_chat_llm, load_prompt
from core.utils import parse_json

_template = load_prompt("orchestrator.txt")


async def orchestrator_node(state: DocumentState) -> dict:
    """Decide which agents are necessary based on available content
    and critic feedback. Returns active_agents list.
    """
    prompt = (
        _template.replace("{{question}}", state.question)
        .replace("{{text_count}}", str(len(state.text_chunks)))
        .replace("{{image_count}}", str(len(state.images)))
        .replace("{{table_count}}", str(len(state.tables)))
        .replace("{{critique}}", state.critique or "N/A")
        .replace("{{retry_count}}", str(state.retry_count))
    )

    llm = get_chat_llm()
    response = await llm.ainvoke(prompt)
    data = parse_json(response.content)

    agents = data.get("agents", [])
    # Validate: only include agents that have data
    valid = []
    if "text" in agents and state.text_chunks:
        valid.append("text")
    if "image" in agents and state.images:
        valid.append("image")
    if "table" in agents and state.tables:
        valid.append("table")
    # Default: include all modalities with data
    if not valid:
        if state.text_chunks:
            valid.append("text")
        if state.images:
            valid.append("image")
        if state.tables:
            valid.append("table")

    return {"active_agents": valid}
