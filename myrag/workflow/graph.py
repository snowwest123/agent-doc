"""LangGraph 工作流：把所有节点串起来。"""
from __future__ import annotations

import logging
from functools import partial

from langgraph.graph import END, START, StateGraph

from myrag.llm import LLMEndpoint
from myrag.workflow.nodes import (
    filter_history_node,
    generate_rag_node,
    retrieve_node,
    rewrite_node,
)
from myrag.workflow.state import AgentState

logger = logging.getLogger("myrag.workflow")

def build_rag_graph(llm: LLMEndpoint, retriever):
    """构造 LangGraph：filter → rewrite → retrieve → generate。

    使用 partial 把 llm 和 retriever 绑到节点函数上（闭包），
    这样节点函数签名 (state) → LLMEndpoint 不需要出现在 state 里。
    """
    # 用 partial 把 llm 注入到节点函数
    filter_fn = partial(filter_history_node, llm=llm)
    rewrite_fn = partial(rewrite_node, llm=llm)
    retrieve_fn = partial(retrieve_node, retriever=retriever)
    generate_fn = partial(generate_rag_node, llm=llm)

    # 1. 创建图
    workflow = StateGraph(AgentState)

    # 2. 加节点
    workflow.add_node("filter_history", filter_fn)
    workflow.add_node("rewrite", rewrite_fn)
    workflow.add_node("retrieve", retrieve_fn)
    workflow.add_node("generate_rag", generate_fn)

    # 3. 加边（执行顺序）
    workflow.add_edge(START, "filter_history")
    workflow.add_edge("filter_history", "rewrite")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "generate_rag")
    workflow.add_edge("generate_rag", END)

    # 4. 编译
    return workflow.compile()

async def run_rag(graph, question: str, chat_history: list = None) -> dict:
    """便捷函数：调用 graph 执行一次。"""
    initial_state = {
        "question": question,
        "chat_history": chat_history or [],
        "retrieved_docs": [],
        "answer": "",
        "rewritten_question": "",
        "messages": [],
    }
    return await graph.ainvoke(initial_state)