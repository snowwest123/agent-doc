from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from pathlib import Path
from typing import List

import psycopg
from langchain_core.documents import Document
from psycopg import sql

from myrag.embedding.embedder import EmbedderBase
from myrag.vectorstore.base import VectorStoreBase


class HologresVectorStore(VectorStoreBase):
    def __init__(
        self,
        embedder: EmbedderBase,
        tenant_id: str,
        table_name: str | None = None,
        dimension: int | None = None,
    ) -> None:
        self.embedder = embedder
        self.tenant_id = tenant_id
        self.table_name = table_name or os.getenv("HOLOGRES_VECTOR_TABLE", "rag_chunks")
        self.dimension = dimension or int(os.getenv("HOLOGRES_VECTOR_DIM", "1536"))
        self._validate_table_name(self.table_name)

    async def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
        texts = [doc.page_content for doc in documents]
        embeddings = await self.embedder.aembed_documents(texts)
        rows = []
        for doc, embedding in zip(documents, embeddings):
            self._validate_embedding(embedding)
            metadata = dict(doc.metadata or {})
            rows.append((
                str(uuid.uuid4()),
                self.tenant_id,
                doc.page_content,
                json.dumps(metadata, ensure_ascii=False),
                embedding,
            ))
        await asyncio.to_thread(self._insert_rows, rows)

    async def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        results = await self.similarity_search_with_score(query, k=k)
        return [doc for doc, _ in results]

    async def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple[Document, float]]:
        embedding = await self.embedder.aembed_query(query)
        self._validate_embedding(embedding)
        return await asyncio.to_thread(self._search_rows, embedding, k)

    async def delete_tenant(self) -> None:
        await asyncio.to_thread(self._delete_tenant)

    def save_local(self, path: str | Path) -> None:
        return None

    def load_local(self, path: str | Path) -> None:
        return None

    @property
    def nb_docs(self) -> int:
        return self._count_rows()

    def ensure_table(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {table} (
                            id TEXT NOT NULL,
                            tenant_id TEXT NOT NULL,
                            content TEXT NOT NULL,
                            metadata JSONB,
                            embedding REAL[] NOT NULL CHECK (
                                array_ndims(embedding) = 1
                                AND array_length(embedding, 1) = {dimension}
                            ),
                            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (tenant_id, id)
                        )
                        WITH (
                            orientation = 'column',
                            distribution_key = 'tenant_id',
                            vectors = {vectors}
                        )
                        """
                    ).format(
                        table=self._table_identifier(),
                        dimension=sql.Literal(self.dimension),
                        vectors=sql.Literal(json.dumps({
                            "embedding": {
                                "algorithm": "HGraph",
                                "distance_method": "Cosine",
                                "builder_params": {
                                    "base_quantization_type": "fp32",
                                    "use_reorder": True,
                                    "precise_quantization_type": "fp32",
                                    "precise_io_type": "reader_io",
                                },
                            }
                        })),
                    )
                )
            conn.commit()

    def _insert_rows(self, rows: list[tuple[str, str, str, str, list[float]]]) -> None:
        self.ensure_table()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    sql.SQL(
                        """
                        INSERT INTO {table} (id, tenant_id, content, metadata, embedding)
                        VALUES (%s, %s, %s, %s::jsonb, %s::real[])
                        """
                    ).format(table=self._table_identifier()),
                    rows,
                )
            conn.commit()

    def _search_rows(self, embedding: list[float], k: int) -> list[tuple[Document, float]]:
        self.ensure_table()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        SELECT content, metadata, cosine_distance(embedding, %s::real[]) AS score
                        FROM {table}
                        WHERE tenant_id = %s
                        ORDER BY score DESC
                        LIMIT %s
                        """
                    ).format(table=self._table_identifier()),
                    (embedding, self.tenant_id, k),
                )
                rows = cur.fetchall()
        results = []
        for content, metadata, score in rows:
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            results.append((Document(page_content=content, metadata=metadata or {}), float(score)))
        return results

    def _delete_tenant(self) -> None:
        self.ensure_table()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("DELETE FROM {table} WHERE tenant_id = %s").format(
                        table=self._table_identifier()
                    ),
                    (self.tenant_id,),
                )
            conn.commit()

    def _count_rows(self) -> int:
        try:
            self.ensure_table()
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("SELECT COUNT(*) FROM {table} WHERE tenant_id = %s").format(
                            table=self._table_identifier()
                        ),
                        (self.tenant_id,),
                    )
                    return int(cur.fetchone()[0])
        except Exception:
            return 0

    def _connect(self):
        host = os.getenv("HOLOGRES_HOST")
        if not host:
            raise ValueError("VECTOR_STORE=hologres 时需要设置 HOLOGRES_HOST")
        return psycopg.connect(
            host=host,
            port=int(os.getenv("HOLOGRES_PORT", "80")),
            dbname=os.getenv("HOLOGRES_DB", ""),
            user=os.getenv("HOLOGRES_USER", ""),
            password=os.getenv("HOLOGRES_PASSWORD", ""),
            connect_timeout=int(os.getenv("HOLOGRES_CONNECT_TIMEOUT", "10")),
        )

    def _table_identifier(self):
        parts = self.table_name.split(".")
        if len(parts) == 1:
            return sql.Identifier(parts[0])
        return sql.Identifier(parts[0], parts[1])

    def _validate_embedding(self, embedding: list[float]) -> None:
        if len(embedding) != self.dimension:
            raise ValueError(f"Embedding 维度不匹配：期望 {self.dimension}，实际 {len(embedding)}")

    @staticmethod
    def _validate_table_name(table_name: str) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", table_name):
            raise ValueError("HOLOGRES_VECTOR_TABLE 只能是 table 或 schema.table 格式")
