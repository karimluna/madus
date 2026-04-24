"""Image agent: vision LLM analysis of detected figure regions.
Supports GPT-4o (API) or moondream:latest (local GPU via Ollama).
Caps at 4 images per request to control token costs.

ref: GPT-4o image token pricing: https://openai.com/api/pricing/
"""

from langchain_core.messages import HumanMessage

from core.models import DocumentState
from core.config import get_chat_llm, get_settings, load_prompt

_template = load_prompt("image_agent.txt")


async def _openai_vision(state: DocumentState) -> dict:
    """GPT-4o path: send base64 images via the multimodal API."""
    llm = get_chat_llm(vision=True)

    critique_section = ""
    if state.critique:
        critique_section = f"Previous attempt feedback: {state.critique}"

    content = [
        {
            "type": "text",
            "text": _template.replace("{{question}}", state.question).replace(
                "{{critique_section}}", critique_section
            ),
        }
    ]
    # Cap at 4 images -> GPT-4o charges per image tile :)
    for img in state.images[:4]:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img.image_b64}",
                    "detail": "high",
                },  # detail: "high" tiles each image into 512x512 cropts
            }  # at 170 tokens per tile, costing more but capturing
        )  # fine chart detail. Use "low" (fixed 85 tokens/image)
        # if layout matters more than pixel content.

    response = await llm.ainvoke([HumanMessage(content=content)])
    return {"image_answer": response.content}


async def _ollama_vision(state: DocumentState) -> dict:
    """Qwen2-VL path via Ollama's OpenAI-compatible API.
    Uses the same ChatOpenAI client pointed at Ollama.
    """
    llm = get_chat_llm(vision=True)

    critique_section = ""
    if state.critique:
        critique_section = f"Previous attempt feedback: {state.critique}"

    content = [
        {
            "type": "text",
            "text": _template.replace("{{question}}", state.question).replace(
                "{{critique_section}}", critique_section
            ),
        }
    ]
    for img in state.images[:4]:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img.image_b64}",
                },
            }
        )

    response = await llm.ainvoke([HumanMessage(content=content)])
    return {"image_answer": response.content}


async def image_agent_node(state: DocumentState) -> dict:
    """Analyze detected figure regions with a vision LLM.
    Returns early if image modality is not active or no images found.
    """
    if "image" not in state.active_agents or not state.images:
        return {"image_answer": state.image_answer or "No visual content detected."}

    s = get_settings()
    if s.vision_backend == "local":
        return await _ollama_vision(state)
    return await _openai_vision(state)
