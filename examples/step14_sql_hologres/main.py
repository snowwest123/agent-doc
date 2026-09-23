"""Step 14: Text-to-SQL Agent 接阿里云 Hologres。

成功连上后的完整演示。
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rich.console import Console
from rich.panel import Panel

from myrag.agents.sql_agent import TextToSQLAgent
from myrag.llm import LLMEndpoint
from myrag.tools.base import ToolRegistry
from myrag.tools.sql_tool import create_hologres_tools

console = Console(legacy_windows=False)


# 自定义 Agent：system prompt 适配 Hologres
class HologresAgent(TextToSQLAgent):
    def _build_system_prompt(self, db_path: str) -> str:
        return (
            "你是一个 SQL 数据分析助手，可以调用工具查询阿里云 Hologres 数据库。\n\n"
            "可用工具：\n"
            "- get_hologres_schema: 查看数据库的所有表结构\n"
            "- hologres_query: 执行 SELECT 查询\n\n"
            "工作流程：\n"
            "1. 先用 get_hologres_schema 看表结构\n"
            "2. 用 hologres_query 查数据\n"
            "3. SQL 报错时根据错误信息重写 SQL\n"
            "4. 用自然语言回答用户\n\n"
            "输出 JSON：\n"
            '{"action": "tool_call", "tool": "工具名", "args": {...}, "reasoning": "..."}\n'
            "或：\n"
            '{"action": "final_answer", "answer": "...", "reasoning": "..."}\n'
        )


async def main() -> None:
    console.print(Panel.fit("Step 14: Text-to-SQL + 阿里云 Hologres", style="bold magenta"))

    registry = ToolRegistry()
    schema_tool, query_tool = create_hologres_tools()
    registry.register("get_hologres_schema", schema_tool)
    registry.register("hologres_query", query_tool)

    agent = HologresAgent(
        llm=LLMEndpoint.from_env(),
        tool_registry=registry,
        max_iterations=5,
    )

    questions = [
        "数据库里有哪些表？",
        "北京客户一共消费了多少钱？",
        "销售额最高的前3 个产品是什么？",
    ]

    for q in questions:
        console.print(f"\n[bold yellow]问：[/bold yellow] {q}")
        result = await agent.run(q, db_path="hologres")

        for step in result.steps:
            if step.action == "tool_call":
                console.print(
                    f"  [cyan]Step {step.step}: {step.tool_name}({step.tool_args})[/cyan]"
                )
                preview = (step.tool_result or "")[:100].replace("\n", " ")
                console.print(f"    结果: {preview}...")

        console.print(f"  [bold green]答：[/bold green] {result.answer}")

    console.print(Panel("Step 14 完成", style="bold green"))


if __name__ == "__main__":
    asyncio.run(main())