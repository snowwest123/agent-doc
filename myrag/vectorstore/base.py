"""向量库抽象：存向量，支持相似度搜索。

设计参考 LangChain 的 VectorStore 接口，但我们做简化版。
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import List

from langchain_core.documents import Document

class VectorStoreBase(abc.ABC):
    """所有向量库的基类。"""

    @abc.abstractmethod
    async def add_documents(self, documents: List[Document]) -> None:
        """往库里加文档（会自动 embed）。"""

    @abc.abstractmethod
    async def similarity_search(
        self, query: str, k: int = 4
    ) -> List[Document]:
        """相似度搜索：返回与 query 最相关的 k 个文档。"""

    @abc.abstractmethod
    def save_local(self, path: str | Path) -> None:
        """持久化到本地。"""

    @abc.abstractmethod
    def load_local(self, path: str | Path) -> None:
        """从本地加载。"""