"""LangGraph 节点实现。

每个节点是一个普通 Python 函数：
- 输入：当前的 AgentState
- 输出：要更新的字段（dict）

LangGraph 会自动合并到 state 里。
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from myrag.llm import LLMEndpoint
from myrag.workflow.state import AgentState

logger = logging.getLogger("myrag.workflow")

async def filter_history_node(state: AgentState, llm: LLMEndpoint) -> dict[str, Any]:
    """节点 1: 过滤 chat_history，只保留最近的几轮。
    
    为什么需要？LLM 的 context window 有限，
    全量历史会爆 token。
 """
    history = state.get("chat_history", [])
    max_pairs = 3  # 只保留最近 3 轮对话
    if len(history) <= max_pairs * 2:
        return {"chat_history": history}

    # 保留最近 max_pairs 对（一对 = 一条 Human + 一条 AI）
    filtered = history[-(max_pairs * 2):]
    logger.info(f"Filtered chat_history: {len(history)} -> {len(filtered)}")
    return {"chat_history": filtered}

async def rewrite_node(state: AgentState, llm: LLMEndpoint) -> dict[str, Any]:
    """节点 2: 把模糊的问题改写成清晰版本。
    
    例子：
        "她是谁？" → "Chloé 是谁？"（利用 chat_history）
        "RAG 怎么做？" → "RAG 包含哪些核心步骤？"（更具体）
    """
    question = state["question"]
    history = state.get("chat_history", [])
    history_text = "\n".join(
        f"{m.type}: {m.content}" for m in history[-6:]
    )

    if not history_text:
        # 没有历史，直接返回原问题
        return {"rewritten_question": question}

    prompt = (
        "根据以下对话历史，把用户最后的问题改写得更清楚、更独立。"
        "只返回改写后的问题，不要加任何解释。\n\n"
        f"对话历史：\n{history_text}\n\n"
        f"原问题：{question}\n\n"
        "改写后的问题："
    )
    rewritten = await llm.ainvoke([HumanMessage(content=prompt)])
    rewritten = rewritten.strip()
    logger.info(f"Rewritten: '{question}' -> '{rewritten}'")
    return {"rewritten_question": rewritten}

async def retrieve_node(state: AgentState, retriever: Any) -> dict[str, Any]:
    """节点 3: 用 retriever 召回相关文档。
    
    retriever 是 HybridRetriever 或 FAISSStore 实例。
    """
    query = state.get("rewritten_question") or state["question"]
    logger.info(f"Retrieving for: '{query}'")
    docs = await retriever.similarity_search(query, k=5) if hasattr(retriever, "similarity_search") else []
    logger.info(f"Retrieved {len(docs)} docs")
    return {"retrieved_docs": docs}

async def generate_rag_node(state: AgentState, llm: LLMEndpoint) -> dict[str, Any]:
    """节点 4: 拼 prompt + 调 LLM 生成答案。

    prompt 包含三部分：
    1. system 指令（基于文档回答）
    2. 对话上文（chat_history），让 LLM 保持连贯
    3. 检索到的文档（context）
    4. 当前问题
    """
    from langchain_core.messages import AIMessage

    question = state["question"]
    docs = state.get("retrieved_docs", [])
    history = state.get("chat_history", [])

    # 1. 把 history 拼成对话上文
    history_text = ""
    for msg in history[-6:]:  # 最近 3 轮
        if isinstance(msg, HumanMessage):
            history_text += f"用户：{msg.content}\n"
        elif isinstance(msg, (SystemMessage, AIMessage)):
            history_text += f"助手：{msg.content}\n"

    # 2. 拼 context
    context_parts = []
    for i, doc in enumerate(docs):
        name = doc.metadata.get("original_file_name", f"doc_{i}")
        context_parts.append(f"--- {name} ---\n{doc.page_content}")
    context = "\n\n".join(context_parts) if context_parts else "（无文档）"

    # 3. 完整 prompt：把上文 + context + 问题 一次性给 LLM
    system_instruction = (
        "你是一个基于文档回答问题的助手。"
        "请只根据下面的文档内容回答，不要编造。"
        "如果文档里没有答案，请直接说「我不知道」。"
        "如果用户的问题依赖上文，请结合对话历史理解。"
    )
    user_prompt = (
        f"对话历史：\n{history_text if history_text else '（无）'}\n\n"
        f"文档内容：\n{context}\n\n"
        f"当前问题：{question}"
    )

    messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=user_prompt),
    ]
    answer = await llm.ainvoke(messages)

    # 4. 把本轮问答写回 history（add_messages 会自动合并到 state["messages"]）
    return {
        "answer": answer,
        "messages": [
            HumanMessage(content=question),
            SystemMessage(content=answer),
        ],
    }