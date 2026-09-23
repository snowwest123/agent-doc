from __future__ import annotations

import os

from myrag.embedding.embedder import EmbedderBase
from myrag.vectorstore.base import VectorStoreBase


def create_vector_store(embedder: EmbedderBase, session_id: str) -> VectorStoreBase:
    backend = os.getenv("VECTOR_STORE", "faiss").strip().lower()
    if backend == "hologres":
        from myrag.vectorstore.hologres_store import HologresVectorStore

        return HologresVectorStore(embedder=embedder, tenant_id=session_id)
    if backend == "faiss":
        from myrag.vectorstore.faiss_store import FAISSStore

        return FAISSStore(embedder=embedder)
    raise ValueError(f"不支持的 VECTOR_STORE：{backend}")


def is_hologres_vector_store() -> bool:
    return os.getenv("VECTOR_STORE", "faiss").strip().lower() == "hologres"
