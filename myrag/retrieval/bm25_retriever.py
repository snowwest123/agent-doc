"""BM25 关键词检索。

BM25 = Best Matching 25，经典的信息检索算法。
- 基于词频和文档长度归一化
- 优点：精确匹配专有名词、代码、缩写
- 缺点：不懂语义（同义词匹配不到）

和向量检索互补，所以生产系统都两个一起用。
"""
from __future__ import annotations

import math
import re
from typing import List

from langchain_core.documents import Document


class BM25Retriever:
    """一个最简的 BM25 实现（中文按字符切分）。"""

    def __init__(self, documents: List[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b

        # 1. 切分所有文档
        self.tokenized = [self._tokenize(d.page_content) for d in documents]
        # 2. 文档长度
        self.doc_lens = [len(t) for t in self.tokenized]
        self.avg_doc_len = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        # 3. 倒排索引 + DF
        self.df: dict[str, int] = {}
        self.tf: list[dict[str, int]] = []
        for tokens in self.tokenized:
            tf_doc: dict[str, int] = {}
            for t in tokens:
                tf_doc[t] = tf_doc.get(t, 0) + 1
            self.tf.append(tf_doc)
            for t in set(tokens):
                self.df[t] = self.df.get(t, 0) + 1

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简单切分：中文按字，英文按词。"""
        tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", text)
        return [t.lower() for t in tokens]

    def _score(self, query_tokens: list[str], doc_idx: int) -> float:
        """BM25 评分公式。"""
        score = 0.0
        doc_len = self.doc_lens[doc_idx]
        tf_doc = self.tf[doc_idx]
        n_docs = len(self.documents)
        for q in query_tokens:
            if q not in tf_doc:
                continue
            f = tf_doc[q]
            df = self.df.get(q, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += idf * numerator / denominator
        return score

    def search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        """返回 top-k (文档, 分数) 列表。"""
        query_tokens = self._tokenize(query)
        scored = [(i, self._score(query_tokens, i)) for i in range(len(self.documents))]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:k]
        return [(self.documents[i], s) for i, s in top if s > 0]