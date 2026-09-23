"""混合检索器：向量 + BM25 + Reranker 三路融合。

流程：
1. 向量检索：召回 top-50（语义相似）
2. BM25 检索：召回 top-50（关键词精确）
3. 去重合并：相同的 chunk 只保留一份
4. Reranker：对合并后的 top-50 重排，挑出 top-5
5. 分数阈值过滤：丢弃分数太低的（防止幻觉）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document

from myrag.retrieval.bm25_retriever import BM25Retriever
from myrag.retrieval.reranker import RerankerBase

@dataclass
class RetrievalResult:
    """检索结果：携带分数信息。"""
    doc: Document
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    source: str = ""  # "vector" / "bm25" / "both"

class HybridRetriever:
    """三路混合检索器。"""

    def __init__(
        self,
        vector_retriever,           # FAISSStore 实例
        documents: List[Document],  # 用于 BM25
        reranker: RerankerBase | None = None,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.documents = documents
        self.reranker = reranker
        # 初始化 BM25
        self.bm25 = BM25Retriever(documents) if documents else None

    async def retrieve(
        self,
        query: str,
        k_vector: int = 50,
        k_bm25: int = 50,
        k_final: int = 5,
        score_threshold: float = 0.0,
    ) -> List[RetrievalResult]:
        """完整检索流程。"""
        # 1. 向量召回（拿 (doc, score)）
        vector_results = await self.vector_retriever.similarity_search_with_score(
            query, k=k_vector
        ) if hasattr(self.vector_retriever, "similarity_search_with_score") else []

        # 2. BM25 召回
        bm25_results = self.bm25.search(query, k=k_bm25) if self.bm25 else []

        # 3. 合并去重（按 page_content 去重）
        merged: dict[str, RetrievalResult] = {}
        for doc, score in vector_results:
            key = doc.page_content[:100]  # 用前 100 字符做 key
            if key not in merged:
                merged[key] = RetrievalResult(doc=doc, vector_score=score, source="vector")
            else:
                merged[key].vector_score = score
                merged[key].source = "both" if merged[key].source == "bm25" else "vector"

        for doc, score in bm25_results:
            key = doc.page_content[:100]
            if key not in merged:
                merged[key] = RetrievalResult(doc=doc, bm25_score=score, source="bm25")
            else:
                merged[key].bm25_score = score
                merged[key].source = "both" if merged[key].source == "vector" else "bm25"

        candidates = list(merged.values())
        if not candidates:
            return []

        # 4. Rerank（如果有）
        if self.reranker:
            docs_only = [r.doc for r in candidates]
            try:
                reranked_docs = await self.reranker.arerank(query, docs_only, top_n=k_final)
            except Exception as e:
                # Rerank 失败（限流、未开通服务等）→ 优雅降级到向量排序
                import logging
                logging.warning(f"Rerank failed, falling back to vector_score: {e}")
                reranked_docs = None

            if reranked_docs:
                reranked_results = []
                for doc in reranked_docs:
                    key = doc.page_content[:100]
                    if key in merged:
                        merged[key].rerank_score = doc.metadata.get("rerank_score", 0.0)
                        reranked_results.append(merged[key])
                return reranked_results[:k_final]

        # 没有 reranker 或 reranker 失败 → 按 vector_score 排序
        candidates.sort(key=lambda r: r.vector_score, reverse=True)

        # 5. 阈值过滤
        return [r for r in candidates[:k_final] if r.vector_score >= score_threshold]