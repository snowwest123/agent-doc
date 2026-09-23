"""工具抽象层。

什么是"工具"？
- LLM 不能访问外部世界（数据库、API）
- 我们把"能干的活"包装成工具，让它按 JSON 输出指令来调用
- LangChain 的 BaseTool 已经做了一半，我们做我们项目的封装
"""
from __future__ import annotations

import abc
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel

class ToolInput(BaseModel):
    """工具的输入 schema（Pydantic 模型）。"""
    pass

class ToolWrapper:
    """工具的封装：包一层让用户更容易用。

    为什么不直接用 LangChain BaseTool？
    - 我们的格式更统一（format_input / format_output）
    - 方便在 Brain 里统一管理（注册、激活、停用）
    """

    def __init__(
        self,
        tool: BaseTool,
        name: str,
        description: str,
    ) -> None:
        self.tool = tool
        self.name = name
        self.description = description

    def format_input(self, raw_input: str) -> dict[str, Any]:
        """把 LLM 给的自然语言转换成工具需要的参数。"""
        # 默认实现：直接当成单一参数
        return {"input": raw_input}

    def format_output(self, raw_output: Any) -> str:
        """把工具输出格式化成字符串，方便塞回 prompt。"""
        return str(raw_output)

class ToolRegistry:
    """工具注册中心：管理所有可用工具。

    使用方式：
        registry.register("weather", weather_tool_wrapper)
        tool = registry.get("weather")
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolWrapper] = {}

    def register(self, name: str, wrapper: ToolWrapper) -> None:
        if name in self._tools:
            raise ValueError(f"工具 {name} 已存在")
        self._tools[name] = wrapper

    def get(self, name: str) -> ToolWrapper | None:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_descriptions(self) -> list[dict[str, str]]:
        """给 LLM 看的工具描述列表（用于 prompt）。"""
        return [
            {"name": w.name, "description": w.description}
            for w in self._tools.values()
        ]

# 全局默认注册表（也可每个 Brain 自己持有一个）
default_registry = ToolRegistry()