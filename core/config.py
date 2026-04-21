"""Environment based configuration and LLM factory: one env var switches 
between OpenAI and Ollama backends.

ref: nibzard/awesome-agentic-patterns: the pattern of a factory
behind an env var is standard in production agent systems.
"""

import os 
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings
from langchain_openai import ChatOpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PROJECT_ROOT / "configs" / "prompts"


class Settings(BaseSettings):
    llm_backend: str = "local"
    vision_backend: str = "local"
    embedding_backend: str = "local"
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
    return Settings


def get_chat_llm(vision: bool = False) -> ChatOpenAI:
    """Return an LLM instance. Ollama exposes the same /v1 API surface so 
    LangChain doesn't know the difference. Ollama has higher latency but 
    zero marginal cost"""
    s = get_settings()
    if s.llm_backend == "local":
        return ChatOpenAI(
            model="moondream" if vision else "qwen2.5:1.5b",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            temperature=0,
        )
    return ChatOpenAI(
        model="gpt-4o" if vision else "gpt-4o-mini",
        temperature=0,
    )


def load_prompt(name: str) -> str:
    """Load a prompt template form configs/prompts/."""
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text()


