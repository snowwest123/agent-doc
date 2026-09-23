"""Step 12 压轴：完整的 Text-to-SQL Agent。

演示：
1. Schema 感知
2. 多轮工具调用
3. SQL 自修正
4. 自然语言总结
"""
import asyncio
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rich.console import Console
from rich.panel import Panel

from myrag.agents.sql_agent import TextToSQLAgent
from myrag.llm import LLMEndpoint
from myrag.tools.base import ToolRegistry
from myrag.tools.sql_tool import create_sql_tools

console = Console(legacy_windows=False)

async def main() -> None:
    console.print(Panel.fit("Step 12: Text-to-SQL Agent 完整版", style="bold magenta"))

    # 1. 准备测试数据库（电商场景）
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "ecommerce.db"
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY, name TEXT, city TEXT, signup_date TEXT
            );
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY, customer_id INTEGER,
                product TEXT, amount REAL, order_date TEXT
            );
            INSERT INTO customers VALUES (1, '张三', '北京', '2024-01-15');
            INSERT INTO customers VALUES (2, '李四', '上海', '2024-02-20');
            INSERT INTO customers VALUES (3, '王五', '广州', '2024-03-10');
            INSERT INTO customers VALUES (4, '赵六', '北京', '2024-04-05');
            INSERT INTO orders VALUES (1, 1, '拿铁', 32, '2024-05-01');
            INSERT INTO orders VALUES (2, 1, '芝士蛋糕', 35, '2024-05-02');
            INSERT INTO orders VALUES (3, 2, '美式', 28, '2024-05-03');
            INSERT INTO orders VALUES (4, 2, '拿铁', 32, '2024-05-04');
            INSERT INTO orders VALUES (5, 3, '司康饼', 28, '2024-05-05');
            INSERT INTO orders VALUES (6, 4, '红茶', 25, '2024-05-06');
            INSERT INTO orders VALUES (7, 1, '美式', 28, '2024-05-07');
        """)
        conn.commit()
        conn.close()
        console.print(f"[bold cyan]数据库：[/bold cyan] {db_path}\n")

        # 2. 设置 Agent
        registry = ToolRegistry()
        schema_tool, query_tool = create_sql_tools()
        registry.register("get_db_schema", schema_tool)
        registry.register("sql_query", query_tool)

        agent = TextToSQLAgent(
            llm=LLMEndpoint.from_env(),
            tool_registry=registry,
            max_iterations=5,
        )

        # 3. 各种类型的查询
        questions = [
            "数据库里有哪些表？",
            "北京客户一共消费了多少钱？",
            "销售额最高的前3 个产品是什么？各卖了多少？",
            "5 月份每天的订单总金额是多少？按日期排序",
        ]

        for q in questions:
            console.print(f"\n[bold yellow]问：[/bold yellow] {q}")

            result = await agent.run(q, str(db_path))

            # 打印 Agent 思维链
            console.print(f"  [dim]Agent 走了 {len(result.steps)} 步：[/dim]")
            for step in result.steps:
                if step.action == "tool_call":
                    console.print(
                        f"    [cyan]Step {step.step}[/cyan] →调 {step.tool_name}({step.tool_args})"
                    )
                    preview = (step.tool_result or "")[:100].replace("\n", " ")
                    console.print(f"      结果: {preview}...")
                else:
                    console.print(f"    [cyan]Step {step.step}[/cyan] → final_answer")

            console.print(f"  [bold green]答：[/bold green] {result.answer}")

    console.print(Panel("Step 12 完成（压轴）", style="bold green"))

if __name__ == "__main__":
    asyncio.run(main())