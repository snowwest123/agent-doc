"""LLM 抽象层。

设计目标：
1. 把 LangChain ChatModel 包一层，方便将来换供应商、加缓存、加监控。
2. 暴露 invoke / ainvoke / stream / count_tokens / with_structured_output 这几个高频接口。
3. 通过 from_env() 一行构造，符合"约定优于配置"。
"""
from __future__ import annotations

from typing import Any, Type, TypeVar

import tiktoken
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from myrag.config import get_settings

T = TypeVar("T", bound=BaseModel)

class LLMEndpoint:
    """对 LangChain ChatModel 的薄封装。"""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._temperature = temperature
        self._max_tokens = max_tokens

        # 真正干活的是 LangChain 的 ChatOpenAI（OpenAI 兼容模式也能复用）
        self._llm: BaseChatModel = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )

    # ===== 核心调用接口 =====

    def invoke(self, messages: list[BaseMessage]) -> str:
        """同步调用，返回纯文本。"""
        response = self._llm.invoke(messages)
        return self._extract_content(response)

    async def ainvoke(self, messages: list[BaseMessage]) -> str:
        """异步调用，返回纯文本。"""
        response = await self._llm.ainvoke(messages)
        return self._extract_content(response)

    def stream(self, messages: list[BaseMessage]):
        """同步流式，返回迭代器（每次一个 AIMessageChunk）。"""
        return self._llm.stream(messages)

    async def astream(self, messages: list[BaseMessage]):
        """异步流式。"""
        async for chunk in self._llm.astream(messages):
            yield chunk

    # ===== 高级能力 =====

    def with_structured_output(self, schema: Type[T]) -> Any:
        """让 LLM 返回结构化对象（Pydantic 模型）。

        关键：这一步把 Pydantic schema 转成 JSON Schema，通过 prompt 约束 LLM 输出。
        Agent 工程化必备（Quivr 的 SplittedInput / FinalAnswer 都用它）。
        """
        return self._llm.with_structured_output(schema)

    # ===== Token 计数 =====

    def count_tokens(self, text: str | list[BaseMessage]) -> int:
        """估算 token 数。text 可以是字符串或消息列表（后者按对话格式累加）。"""
        enc = tiktoken.get_encoding("cl100k_base")
        if isinstance(text, str):
            return len(enc.encode(text))
        # 对话格式：每条消息 4 个 token 包裹
        total = 3 # 每轮对话开头 <|im_start|>assistant<|im_sep|>
        for msg in text:
            total += 4  # <|im_start|>{role}<|im_sep|>
            total += len(enc.encode(str(msg.content)))
        return total

    # ===== 工厂方法 =====

    @classmethod
    def from_env(cls) -> "LLMEndpoint":
        """从环境变量构造（最常用入口）。"""
        settings = get_settings()
        settings.validate()
        return cls(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    # ===== 信息 =====

    def info(self) -> dict[str, Any]:
        """返回当前配置（用于调试 / Brain.print_info）。"""
        return {
            "model": self._model,
            "base_url": self._base_url or "openai-default",
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

    def __repr__(self) -> str:
        return (
            f"LLMEndpoint(model={self._model!r}, "
            f"base_url={self._base_url!r}, "
            f"temperature={self._temperature})"
        )

    # ===== 内部 =====

    @staticmethod
    def _extract_content(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # list 里可能是 str 或 dict（如 {"type": "text", "text": "..."}）
            parts = []
            for p in content:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict) and "text" in p:
                    parts.append(p["text"])
            return "".join(parts)
        return str(content)