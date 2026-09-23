"""Step 5 验证：三路混合检索。"""
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rich.console import Console
from rich.panel import Panel

from myrag.brain import Brain
from myrag.embedding.dashscope_embedder import DashScopeEmbedder
from myrag.retrieval.hybrid import HybridRetriever
from myrag.retrieval.reranker import DashScopeReranker

console = Console(legacy_windows=False)

async def main() -> None:
    console.print(Panel.fit("Step 5: 三路混合检索", style="bold magenta"))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # 故意放容易混淆专有名词和同义词的内容
        (tmp_path / "python.txt").write_text(
            """Python 是一种广泛使用的高级编程语言。
                由 Guido van Rossum 于 1991 年首次发布。
                主要特点：语法简洁、跨平台、生态丰富。
                常见框架：Django、Flask、FastAPI。
            """,
            encoding="utf-8",
        )
        (tmp_path / "rag.txt").write_text(
            """RAG（Retrieval-Augmented Generation，检索增强生成）
                是一种让 LLM
                结合外部知识库回答问题的技术。
                核心步骤：
                1. 把文档切片2. 用 Embedding 模型向量化
                3. 检索最相关的 chunks
                4. 拼接 prompt 让 LLM 生成答案""",
            encoding="utf-8",
        )
        (tmp_path / "python_web.txt").write_text(
            """FastAPI 是 Python 的现代 Web 框架，基于 Starlette。
                特点：异步、自动 OpenAPI 文档、类型注解。
                适合做 AI应用的 API 服务。""",
            encoding="utf-8",
        )

        brain = await Brain.from_files(
            name="技术知识库",
            file_paths=[
                tmp_path / "python.txt",
                tmp_path / "rag.txt",
                tmp_path / "python_web.txt",
            ],
            embedder=DashScopeEmbedder.from_env(),
        )

        # 手动演示三路召回
        retriever = HybridRetriever(
            vector_retriever=brain.vector_store,
            documents=brain.knowledge,
            reranker=DashScopeReranker.from_env(),
        )

        # 三类问题对比
        questions = [
            "Python 是谁发明的？",            # 关键词明确 → BM25 优势
            "什么是 RAG 的核心流程？",        # 抽象概念 → 向量优势
            "FastAPI 适合做什么？",           # 多义词 → 三路融合
        ]

        for q in TYuestions:
            console.print(f"\n[bold yellow]问：[/bold yellow] {q}")
            results = await retriever.retrieve(q, k_final=3)
            console.print(f"  检索到 {len(results)} 条相关文档：")
            for r in results:
                name = r.doc.metadata.get("original_file_name", "?")
                src = r.source
                rs = r.rerank_score if r.rerank_score else r.vector_score
                preview = r.doc.page_content[:60].replace("\n", " ")
                console.print(f"    [{src}|{rs:.3f}] {name}: {preview}...")

            answer = await brain.ask(q)
            console.print(f"  [bold green]答：[/bold green] {answer}")

    console.print(Panel("Step 5 完成", style="bold green"))

if __name__ == "__main__":
    asyncio.run(main())