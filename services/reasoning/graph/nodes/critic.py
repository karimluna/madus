"""Reflexion critic: evaluates agent outputs and produces verbal
feedback v_k that conditions the next attempt. The verbal reflection
is a surrogate gradient, it encodes why the output failed and what
to change, without touching any parameters.

Retry is bounded at 2 retries (3 total attempts). Without this bound
you have an unbounded loop.

ref: Reflexion paper: https://arxiv.org/abs/2303.11366
"""

from core.models import DocumentState
from core.config import get_chat_llm, load_prompt
from core.utils import parse_json

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

_template = load_prompt("critic.txt")

MAX_RETRIES = 2


async def critic_node(state: DocumentState) -> dict:
    """Evaluate all agent answers and decide if retry is needed."""
    prompt = (
        _template.replace("{{question}}", state.question)
        .replace("{{text_answer}}", state.text_answer or "N/A")
        .replace("{{image_answer}}", state.image_answer or "N/A")
        .replace("{{table_answer}}", state.table_answer or "N/A")
    )

    llm = get_chat_llm()
    response = await llm.ainvoke(prompt)
    data = parse_json(response.content)

    logger.info("=== CRITIC ===")
    logger.info("Sufficient: %s", data.get("sufficient"))
    logger.info("Critique: %s", data.get("critique"))
    logger.info("Retry count: %d", state.retry_count)
    logger.info("Text answer preview: %s", (state.text_answer or "")[:200])

    sufficient = data.get("sufficient", True)
    needs_retry = not sufficient and state.retry_count < MAX_RETRIES
    # if the answer is not good after 3 attempts the document likely
    # does not contain the answer
    return {
        "critique": data.get("critique", ""),
        "needs_retry": needs_retry,
        "retry_count": state.retry_count + 1,
    }


def route_after_critic(state: DocumentState) -> str:
    """Conditional edge: retry via orchestrator or pass to summarizer."""
    if state.needs_retry and state.retry_count <= MAX_RETRIES:
        return "orchestrator"

    # we force the summarizer so the user understands why the answer is partial
    return "summarizer"
