"""Brain：把 LLM + 文件 + 存储粘合起来的"第二大脑"。

为什么需要 Brain 这个抽象？
- LLMEndpoint 只懂调模型，不知道有哪些文档
- Storage 只懂存文件，不知道怎么用 LLM
- Brain 把它们组织起来，提供一个简单的 ask() 接口

设计要点（参考 quivr-core/brain/brain.py）：
- 用 @classmethod factory 模式构造（Brain.from_files）
- 异步主链路（async def ask）
- 内部状态（storage、llm、knowledge）通过属性暴露
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

from myrag.llm import LLMEndpoint
from myrag.processor.registry import ProcessorRegistry
from myrag.storage import LocalStorage, QuivrFile, StorageBase
import asyncio
from contextlib import asynccontextmanager
from myrag.retrieval.config import RetrievalConfig
from myrag.retrieval.hybrid import HybridRetriever
from myrag.retrieval.reranker import DashScopeReranker
from myrag.tools.base import ToolRegistry, ToolWrapper, default_registry
import json
import re

class Brain:
    """一个"第二大脑"。

    当前最简实现：
    - knowledge 用 list[Document] 存（还没接向量库）
    - ask() 把全部 knowledge 拼到 prompt 里问 LLM
    """

    def __init__(
        self,
        name: str,
        llm: LLMEndpoint,
        storage: StorageBase | None = None,
        knowledge: list | None = None,
        vector_store: "VectorStoreBase | None" = None,
        retrieval_config: "RetrievalConfig | None" = None,
        brain_id: uuid.UUID | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.name = name
        self.id = brain_id or uuid.uuid4()
        self.llm = llm
        self.storage = storage
        # knowledge 是 LangChain Document 列表
        self.knowledge: list = knowledge or []
        self.vector_store = vector_store
        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.tool_registry = tool_registry or default_registry
    @asynccontextmanager
    async def temp_storage():
        """自动清理的临时 storage。"""
        import tempfile, shutil
        path = Path(tempfile.mkdtemp(prefix="myrag_"))
        try:
            yield LocalStorage(dir_path=path)
        finally:
            shutil.rmtree(path, ignore_errors=True)
    @classmethod
    async def from_files(
        cls,
        name: str,
        file_paths: list[str | Path],
        llm: LLMEndpoint | None = None,
        embedder: "EmbedderBase | None" = None,
        storage_dir: str | Path | None = None,
        use_vector_store: bool = True,
        config_path: str | Path | None = None,
    ) -> "Brain":
        """工厂方法：从一组文件路径构造 Brain。

        步骤：
        1. 准备 storage（如果不传就用内存临时目录）
        2. 依次上传文件到 storage
        3. 按文件后缀选 processor 处理
        4. 把所有处理后的 Document 收集成 knowledge
        5. 构造 Brain 返回
        """
        import asyncio
        import tempfile

        # 1. Storage（默认用临时目录）
        if storage_dir is None:
            storage_dir = tempfile.mkdtemp(prefix="myrag_")
        storage = LocalStorage(dir_path=storage_dir)

        # 2. LLM（默认从环境变量）
        if llm is None:
            llm = LLMEndpoint.from_env()

        # 3. 上传文件到 storage
        quivr_files = await asyncio.gather(
            *[storage.upload_file(fp) for fp in file_paths]
        )

        # 每个文件处理互相独立
        async def process_one(qf):
            proc = ProcessorRegistry.get(qf.file_extension)
            if proc is None:
                raise ValueError(...)
            return await proc.process_file(qf)

        all_docs_nested = await asyncio.gather(*[process_one(qf) for qf in quivr_files])
        all_docs = [d for docs in all_docs_nested for d in docs]

        # 5. 接向量库

        vector_store = None
        if use_vector_store:
            if embedder is None:
                from myrag.embedding.dashscope_embedder import DashScopeEmbedder
                embedder = DashScopeEmbedder.from_env()
            from myrag.vectorstore.factory import create_vector_store
            vector_store = create_vector_store(embedder, str(uuid.uuid4()))
            await vector_store.add_documents(all_docs)

        retrieval_config = (
            RetrievalConfig.from_yaml(config_path) if config_path else RetrievalConfig()
        )

        brain = cls(
            name=name,
            llm=llm,
            storage=storage,
            knowledge=all_docs,
            vector_store=vector_store,
            retrieval_config=retrieval_config,
        )
        return brain

    async def ask_with_tool(self, question: str) -> str:
        """带工具调用的问答：LLM 可以决定是否调用工具。

        流程：
        1. 把工具描述塞到 system prompt
        2. LLM 看到问题是否需要工具
        3. 如果需要，输出 JSON 调用指令
        4. 我们解析 JSON，执行工具
        5. 把工具结果喂回 LLM，让它生成最终答案
        """

        # 1. 拼工具描述
        tools_desc = "\n".join(
            f"- {t['name']}: {t['description']}"
            for t in self.tool_registry.get_descriptions()
        )

        system_prompt = (
            "你是一个能调用工具的助手。\n\n"
            f"可用工具：\n{tools_desc}\n\n"
            "如果需要调用工具，请输出 JSON 格式：\n"
            '{"tool": "工具名", "args": {"参数名": "值"}}\n\n'
            "如果不需要，直接回答问题。"
        )

        # 2. 第一次 LLM 调用，决定要不要调
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]
        response = await self.llm.ainvoke(messages)
        response_text = response.strip()

        # 3. 解析是否调用工具
        # 找 JSON 块
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not json_match:
            # 没调用工具，直接返回
            return response_text

        try:
            tool_call = json.loads(json_match.group())
            tool_name = tool_call["tool"]
            tool_args = tool_call["args"]
        except (json.JSONDecodeError, KeyError):
            return response_text

        # 4. 执行工具
        wrapper = self.tool_registry.get(tool_name)
        if not wrapper:
            return f"错误：未知工具 {tool_name}"

        tool_input = wrapper.format_input(question)
        tool_input.update(tool_args)
        try:
            tool_result = await wrapper.tool.ainvoke(tool_input)
            tool_output = wrapper.format_output(tool_result)
        except Exception as e:
            tool_output = f"工具执行错误：{e}"

        # 5. 把工具结果喂回 LLM，让它生成最终答案
        final_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
            SystemMessage(content=f"工具 {tool_name} 的返回结果：\n{tool_output}"),
            HumanMessage(content="基于工具结果，给出最终答案。"),
        ]
        final_response = await self.llm.ainvoke(final_messages)
        return final_response
    async def _retrieve_docs(self, question: str) -> list:
        """共享的检索逻辑。"""
        cfg = self.retrieval_config
        if self.vector_store is None or self.vector_store.nb_docs == 0:
            return self.knowledge
        
        reranker = DashScopeReranker.from_env() if cfg.use_rerank else None
        retriever = HybridRetriever(
            vector_retriever=self.vector_store,
            documents=self.knowledge if cfg.use_bm25 else [],
            reranker=reranker,
        )
        results = await retriever.retrieve(
            question,
            k_vector=cfg.k_vector,
            k_bm25=cfg.k_bm25,
            k_final=cfg.k_final,
        )
        return [r.doc for r in results]

    def _build_context(self, docs: list) -> str:
        """拼 context。"""
        parts = []
        for i, doc in enumerate(docs):
            name = doc.metadata.get("original_file_name", f"doc_{i}")
            parts.append(f"--- {name} ---\n{doc.page_content}")
        return "\n\n".join(parts) if parts else "（无文档）"

    async def ask(self, question: str, use_graph: bool = True, chat_history: list | None = None) -> str:
        """问 Brain 提问。

        use_graph=True: 走 LangGraph（推荐，含 rewrite/retrieve/generate 节点）
        use_graph=False: 直接走混合检索 + LLM（老逻辑）

        chat_history: 多轮对话历史（仅 use_graph=True 时生效）
        """
        if use_graph and self.vector_store is not None and self.vector_store.nb_docs > 0:
            return await self._ask_via_graph(question, chat_history or [])

        # 回退：直接走混合检索 + LLM（向后兼容 Step 5/6 行为）
        return await self._ask_simple(question)

    async def _ask_via_graph(self, question: str, chat_history: list) -> str:
        """走 LangGraph 工作流。"""
        from myrag.workflow.graph import build_rag_graph, run_rag

        graph = build_rag_graph(self.llm, self.vector_store)
        result = await run_rag(graph, question, chat_history=chat_history)
        return result["answer"]

    async def _ask_simple(self, question: str) -> str:
        """简单 RAG：混合检索 + LLM（Step 5/6 行为）。"""
        docs = await self._retrieve_docs(question)
        context = self._build_context(docs)
        messages = [
            SystemMessage(content=self.retrieval_config.system_prompt.format(
                brain_name=self.name, context=context
            )),
            HumanMessage(content=question),
        ]
     
        return await self.llm.ainvoke(messages)

    async def suggest_questions(self, count: int = 5) -> list[str]:
        """基于已上传文档生成可提问问题。"""
        docs = await self._retrieve_docs("概括这些文档中最值得追问的主题")
        context = self._build_context(docs)
        messages = [
            SystemMessage(content=(
                "你是文档问答助手，需要基于给定文档生成用户可以直接点击提问的问题。"
                "只返回 JSON 字符串数组，不要返回 Markdown，不要返回解释。"
                f"数组长度最多 {count} 个，问题必须具体、自然，并且能从文档中找到依据。\n\n"
                f"文档内容：\n{context}"
            )),
            HumanMessage(content="请生成可提问问题。"),
        ]
        text = await self.llm.ainvoke(messages)
        return self._parse_suggested_questions(text, count)

    def _parse_suggested_questions(self, text: str, count: int) -> list[str]:
        try:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            raw = json.loads(match.group() if match else text)
            questions = [str(item).strip() for item in raw if str(item).strip()]
        except Exception:
            questions = [
                line.strip(" -0123456789.、\t")
                for line in text.splitlines()
                if line.strip(" -0123456789.、\t")
            ]
        seen: set[str] = set()
        result: list[str] = []
        for question in questions:
            if question in seen:
                continue
            seen.add(question)
            result.append(question)
            if len(result) >= count:
                break
        return result


    async def ask_streaming(self, question: str):
        """流式问答：逐 chunk 返回答案。

        Yields:
            str: 每次返回新增的文本片段
        """
        docs = await self._retrieve_docs(question)
        context = self._build_context(docs)

        # 3. 拼 prompt
        messages = [
            SystemMessage(content=self.retrieval_config.system_prompt.format(
                brain_name=self.name, context=context
            )),
            HumanMessage(content=question),
        ]

        # 4. 流式调 LLM，逐 chunk yield 文本
        async for chunk in self.llm.astream(messages):
            # chunk 是 AIMessageChunk，content 可能是 str 或 list
            content = chunk.content if isinstance(chunk.content, str) else "".join(chunk.content)
            if content:
                yield content
    
    def print_info(self) -> None:
        """漂亮地打印 Brain 信息。"""
        tree = Tree(f"[bold cyan]{self.name}[/bold cyan] (id={str(self.id)[:8]}...)")
        tree.add(f"LLM: {self.llm.__repr__()}")
        tree.add(f"Storage: {type(self.storage).__name__ if self.storage else 'None'}")
        tree.add(f"Knowledge: {len(self.knowledge)} chunks")

        for i, doc in enumerate(self.knowledge):
            doc_tree = tree.add(
                f"  [{i}] {doc.metadata.get('original_file_name', '?')}"
            )
            doc_tree.add(f"    preview: {doc.page_content[:60]}...")

        console = Console()
        console.print(Panel(tree, title="Brain Info", border_style="bold"))

    def __repr__(self) -> str:
        return (
            f"Brain(name={self.name!r}, "
            f"docs={len(self.knowledge)}, "
            f"llm={self.llm.__class__.__name__})"
        )
