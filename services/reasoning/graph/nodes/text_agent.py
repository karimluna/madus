"""Text agent: BM25/hybrid retrieval + LLM reasoning over text chunks.
On retry, the critique from the Reflexion loop is threaded into the
prompt as verbal feedback, no weight updates needed.

ref: Reflexion paper: https://arxiv.org/abs/2303.11366
"""

from core.models import DocumentState
from core.config import get_chat_llm, load_prompt
from services.reasoning.tools.retrieval import retrieve_hybrid
import asyncio


_template = load_prompt("text_agent.txt")


async def text_agent_node(state: DocumentState) -> dict:
    """Retrieve relevant chunks and answer the question.
    Returns early if text modality is not active or no chunks available.
    """
    if "text" not in state.active_agents or not state.text_chunks:
        return {"text_answer": state.text_answer or "No text content available."}

    relevant = await asyncio.to_thread(
        retrieve_hybrid, state.text_chunks, state.doc_id, state.question, k=5
    )

    critique_section = ""
    if state.critique:
        critique_section = f"Previous attempt feedback: {state.critique}"

    prompt = (
        _template.replace("{{question}}", state.question)
        .replace("{{context}}", "\n\n".join(relevant))
        .replace("{{critique_section}}", critique_section)
    )

    llm = get_chat_llm()
    response = await llm.ainvoke(prompt)
    return {"text_answer": response.content}
