"""LangGraph 工作流的共享 state。

LangGraph 的核心概念：
- StateGraph = 状态机
- 每个节点接收 state，返回更新（部分字段）
- LangGraph 自动合并更新到 state
- 节点通过 state 共享数据
"""
from __future__ import annotations

from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """整个工作流共享的 state。
字段说明：
        messages: 消息列表，add_messages 让 LangGraph 自动追加（不覆盖）
        question: 用户当前问题
        chat_history: 多轮对话历史
        retrieved_docs: 检索到的相关文档        answer: 最终答案
        rewritten_question: query rewrite 后的版本
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    chat_history: list[BaseMessage]
    retrieved_docs: list[Any]  # List[Document]
    answer: str
    rewritten_question: str