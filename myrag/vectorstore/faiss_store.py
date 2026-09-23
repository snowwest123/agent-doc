"""FAISS 向量库实现。

FAISS = Facebook AI Similarity Search，Meta 开源的相似度搜索库。
- 优点：本地、CPU 跑、零运维
- 缺点：单机、单进程
- 适用：开发期、小到中等规模（< 100万 向量）
"""
from __future__ import annotations

import asyncio
import pickle
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from myrag.embedding.embedder import to_langchain_embedder
from myrag.embedding.embedder import EmbedderBase
from myrag.vectorstore.base import VectorStoreBase

class FAISSStore(VectorStoreBase):
    """基于 langchain-community FAISS 的实现。"""

    def __init__(self, embedder: EmbedderBase) -> None:
        self.embedder = embedder
        # 转成 LangChain 的 Embeddings 接口（langchain FAISS 需要）
        self._lc_embeddings = to_langchain_embedder(embedder)
        # FAISS 实例：初始为空
        self._faiss: FAISS | None = None


    #  加一个方法：
    async def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[tuple[Document, float]]:
        """带分数的相似度搜索。"""
        if self._faiss is None:
            return []
        return await asyncio.to_thread(
            self._faiss.similarity_search_with_score, query, k=k
        )

    async def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
        texts = [doc.page_content for doc in documents]
        embeddings = await self.embedder.aembed_documents(texts)
        
        if self._faiss is None:
            # 第一次：先 embed 再用 from_embeddings（更明确）
            self._faiss = await asyncio.to_thread(
                FAISS.from_embeddings,
                text_embeddings=list(zip(texts, embeddings)),
                embedding=self._lc_embeddings,
                metadatas=[doc.metadata for doc in documents],
            )
        else:
            await asyncio.to_thread(self._faiss.add_documents, documents)
    async def similarity_search(
        self, query: str, k: int = 4
    ) -> List[Document]:
        """相似度搜索。"""
        if self._faiss is None:
            return []
        return await asyncio.to_thread(
            self._faiss.similarity_search, query, k=k
        )

    def save_local(self, path: str | Path) -> None:
        """保存到本地目录。"""
        if self._faiss is None:
            raise ValueError("FAISS 是空的，没东西可保存")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._faiss.save_local(str(path))

    def load_local(self, path: str | Path) -> None:
        """从本地加载（必须用相同的 embedder）。"""
        path = Path(path)
        # FAISS.load_local 需要 allow_dangerous_deserialization=True
        self._faiss = FAISS.load_local(
            str(path),
            self._lc_embeddings,
            allow_dangerous_deserialization=True,
        )

    @property
    def nb_docs(self) -> int:
        """文档数量（粗略估算）。"""
        if self._faiss is None:
            return 0
        return self._faiss.index.ntotal