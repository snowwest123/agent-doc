"""Reranker 抽象：对初筛的 chunks 按 query 相关性重排序。

为什么需要 reranker？
- 向量检索是"近似"匹配，可能把次相关的排在前面
- Reranker 用更精细的 cross-encoder 模型重排，准确率更高
- 代价：reranker 比向量检索慢 10x，所以只对 top-K (K=50) 重排
"""
from __future__ import annotations

import abc
import asyncio
import os
from typing import List

from langchain_core.documents import Document

class RerankerBase(abc.ABC):
    """所有 reranker 的基类。"""

    @abc.abstractmethod
    def rerank(
        self, query: str, documents: List[Document], top_n: int = 5
    ) -> List[Document]:
        """重排序，返回 top_n 个最相关的文档。"""

class DashScopeReranker(RerankerBase):
    """千问通用 Rerank（gte-rerank 系列）。

    用 dashscope.TextReRank.call() 调用。
    """

    def __init__(
        self,
        model: str = "gte-rerank",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("需要 OPENAI_API_KEY 或 DASHSCOPE_API_KEY")
        import dashscope
        dashscope.api_key = self.api_key

    def rerank(
        self, query: str, documents: List[Document], top_n: int = 5
    ) -> List[Document]:
        """调用 DashScope Rerank API。"""
        from dashscope import TextReRank

        if not documents:
            return []

        # DashScope rerank 接受 (query, [doc_text]) 格式
        doc_texts = [doc.page_content for doc in documents]
        resp = TextReRank.call(
            model=self.model,
            query=query,
            documents=doc_texts,
            top_n=min(top_n, len(doc_texts)),
            return_documents=False,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"DashScope rerank failed: {resp.message}")

        # resp.output.results 是 [{index, relevance_score}, ...]，按 score 排序
        results = resp.output["results"]
        reranked = []
        for r in results:
            original_doc = documents[r["index"]]
            # 把分数写到 metadata 里，方便后续阈值过滤
            original_doc.metadata["rerank_score"] = r["relevance_score"]
            reranked.append(original_doc)
        return reranked

    async def arerank(
        self, query: str, documents: List[Document], top_n: int = 5
    ) -> List[Document]:
        """异步版本。"""
        return await asyncio.to_thread(self.rerank, query, documents, top_n)

    @classmethod
    def from_env(cls) -> "DashScopeReranker":
        return cls()