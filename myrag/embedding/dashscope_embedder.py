"""千问 DashScope Embedding 实现。

为什么用原生 SDK 而不是 OpenAI 兼容？
- 你之前调试过：DashScope OpenAI 兼容模式的 embeddings 有 bug
- 原生 SDK 走 dashscope.TextEmbedding.call() 接口，更稳定
"""
from __future__ import annotations

import asyncio
import os
from typing import List

import dashscope
from dashscope import TextEmbedding

from myrag.embedding.embedder import EmbedderBase


class DashScopeEmbedder(EmbedderBase):
    """千问 Embedding（支持 text-embedding-v1/v2/v3）。"""

    def __init__(
        self,
        model: str = "text-embedding-v2",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("需要 OPENAI_API_KEY 或 DASHSCOPE_API_KEY")
        # 设置 dashscope 全局 api_key
        dashscope.api_key = self.api_key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """同步批量 embed。DashScope 每次最多 25 条，超过要分批。"""
        all_embeddings: List[List[float]] = []
        batch_size = 25

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = TextEmbedding.call(model=self.model, input=batch)
            # 改进：加上 request_id方便排查
            if resp.status_code != 200:
                raise RuntimeError(
                    f"DashScope embed failed: {resp.message} "
                    f"(code={resp.code}, request_id={resp.request_id})"
                )

            # 提取向量：响应里每个 item 有 embedding 字段
            batch_embeddings = [item["embedding"] for item in resp.output["embeddings"]]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """单个查询的 embed（多数场景下和 embed_documents 用同一接口）。"""
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步版本：用 to_thread 把同步调用包成异步。"""
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        return (await self.aembed_documents([text]))[0]

    @classmethod
    def from_env(cls) -> "DashScopeEmbedder":
        """从环境变量构造（最常用入口）。"""
        model = os.getenv("EMBED_MODEL", "text-embedding-v2")
        return cls(model=model)