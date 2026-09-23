"""计算器工具：LLM 调它做数学计算。

这是最简单的工具示例，演示工具调用的完整链路。
"""
from __future__ import annotations

import re

from langchain_core.tools import tool

from myrag.tools.base import ToolWrapper

# LangChain 的 @tool 装饰器自动把函数包装成 BaseTool
@tool
def calculator(expression: str) -> str:
    """计算数学表达式。输入形如 '2 + 3 * 4'，返回结果。"""
    try:
        # 简单安全检查：只允许数字、运算符、括号、空格
        if not re.match(r"^[\d\s+\-*/().]+$", expression):
            return "错误：只支持数字和 + - * / ( ) 运算符"
        result = eval(expression)  # 注意：生产环境应该用 ast.parse 更安全
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"

def create_calculator_tool() -> ToolWrapper:
    """创建包装好的 Calculator 工具。"""
    return ToolWrapper(
        tool=calculator,
        name="calculator",
        description="计算数学表达式，例如 '2 + 3 * 4'",
    )