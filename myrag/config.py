"""全局配置：集中读取环境变量。"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

# 加载 .env（只第一次生效）
load_dotenv()

class Settings:
    """简单配置对象（用类而不是 pydantic BaseSettings，避免引入额外依赖）。"""

    def __init__(self) -> None:
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.openai_base_url: str | None = os.getenv("OPENAI_BASE_URL") or None
        self.llm_model: str = os.getenv("LLM_MODEL", "qwen-plus")
        self.llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
        self.llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))

    def validate(self) -> None:
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY 未设置。请复制 .env.example 为 .env 并填入千问 DashScope key。"
            )

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """全局单例。"""
    return Settings()