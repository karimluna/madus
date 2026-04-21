"""Environment based configuration and LLM factory: one env var switches 
between OpenAI and Ollama backends.

ref: nibzard/awesome-agentic-patterns: the pattern of a factory
behind an env var is standard in production agent systems.
"""

import os 
from pathlib import Path
from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings
from langchain_openai import ChatOpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PROJECT_ROOT / "configs" / "prompts"


# `pydantic-settings` picks environment variables in real time with lowercases
class Settings(BaseSettings):
    llm_backend: Literal["local", "openai"] = "local"
    vision_backend: Literal["local", "colflow", "openai"] = "local" # colflow is optional as is more computationally demanding but still local
    embedding_backend: Literal["local", "openai"] = "local"
    openai_api_key: str = ""
    redis_host: str = "localhost"
    redis_port: int = 6379
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    databricks_host: str = ""
    databricks_http_path: str = ""
    databricks_token: str = ""

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    """Returns instatiation of the Settings class."""
    return Settings()


def get_chat_llm(vision: bool = False) -> ChatOpenAI:
    """Return an LLM instance. Ollama exposes the same /v1 API surface so 
    LangChain doesn't know the difference. Ollama has higher latency but 
    zero marginal cost"""
    s = get_settings()
    backend = s.vision_backend if vision else s.llm_backend
    if backend == "local":
        return ChatOpenAI(
            model="qwen2-vl" if vision else "qwen2.5:1.5b",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            temperature=0,
        )
    return ChatOpenAI(
        model="gpt-4o" if vision else "gpt-4o-mini",
        temperature=0,
    )


def load_prompt(name: str) -> str:
    """Load a prompt template from configs/prompts/.
 
    Raises a clear error if the prompts directory or file
    is missing.
    """
    if not PROMPTS_DIR.exists():
        raise FileNotFoundError(
            f"Prompts directory not found: {PROMPTS_DIR}\n"
            f"Create it with: mkdir -p {PROMPTS_DIR}\n"
            f"Then add the prompt templates listed in configs/prompts/."
        )
    path = PROMPTS_DIR / name
    if not path.exists():
        available = [p.name for p in PROMPTS_DIR.glob("*.txt")]
        raise FileNotFoundError(
            f"Prompt template not found: {path}\n"
            f"Available prompts: {available if available else 'none — directory is empty'}\n"
            f"Expected files: orchestrator.txt, text_agent.txt, image_agent.txt, "
            f"table_agent.txt, critic.txt, summarizer.txt"
        )
    return path.read_text()