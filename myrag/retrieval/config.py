"""Retrieval 配置：把 yaml 转成强类型配置。

为什么用 pydantic？
- yaml 解析出来是 dict，没类型检查
- pydantic 自动转类型 + 校验 + 报错友好
- IDE 补全友好
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

class LLMConfig(BaseModel):
    """LLM 相关配置。"""
    model: str = "qwen-plus"
    temperature: float = 0.3
    max_tokens: int = 4096
    base_url: str | None = None

class RetrievalConfig(BaseModel):
    """检索流程的所有可调项。"""

    # === LLM 配置 ===
    llm: LLMConfig = Field(default_factory=LLMConfig)

    # === 检索策略 ===
    use_vector: bool = True
    use_bm25: bool = True
    use_rerank: bool = True

    # 召回数量
    k_vector: int = 50
    k_bm25: int = 50
    k_final: int = 5             # 最终给 LLM 的 chunk 数

    # 分数阈值
    score_threshold: float = 0.0  # 低于此分数丢弃

    # === Prompt 配置 ===
    system_prompt: str = (
        "你是「{brain_name}」——一个基于文档回答问题的助手。"
        "请只根据下面的文档内容回答，不要编造。"
        "如果文档里没有答案，请直接说「我不知道」。"
        "\n\n文档内容：\n{context}"
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RetrievalConfig":
        """从 yaml 文件加载。"""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """导出到 yaml。"""
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(), f, allow_unicode=True, sort_keys=False)
