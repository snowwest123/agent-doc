"""Embedding 抽象层：把文本转成向量。

为什么需要这个抽象？
- 不同供应商的 embedding API 不一样（OpenAI、Cohere、DashScope、本地模型）
- Brain 应该能切换 embedding 不改业务代码
- 接口要兼容 langchain.embeddings.Embeddings（向量库用它）
"""
from __future__ import annotations

import abc
from typing import List

from langchain_core.embeddings import Embeddings

class EmbedderBase(abc.ABC):
    """所有 embedder 的基类。"""

    @abc.abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量 embed 多个文档。"""

    @abc.abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """embed 单个查询。"""

def to_langchain_embedder(embedder: EmbedderBase) -> Embeddings:
    """把我们自己的 Embedder 包成 LangChain 的 Embeddings。

    为什么？因为 langchain 的 FAISS/Milvus 等都用 Embeddings 接口。
    """
    class _Adapter(Embeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return embedder.embed_documents(texts)

        def embed_query(self, text: str) -> List[float]:
            return embedder.embed_query(text)

    return _Adapter()