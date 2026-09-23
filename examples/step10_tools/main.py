"""Step 10 验证：Tool 调用。"""
import asyncio
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rich.console import Console
from rich.panel import Panel

from myrag.llm import LLMEndpoint
from myrag.brain import Brain
from myrag.tools.base import default_registry
from myrag.tools.calculator import create_calculator_tool
from myrag.tools.sql_tool import create_sql_tool

console = Console(legacy_windows=False)

async def main() -> None:
    console.print(Panel.fit("Step 10: Tool 系统", style="bold magenta"))

    # 1. 注册工具
    default_registry.register("calculator", create_calculator_tool())
    default_registry.register("sql_query", create_sql_tool())
    console.print(f"[bold cyan]已注册工具：[/bold cyan] {default_registry.list_tools()}")

    # 2. 准备测试 SQLite 数据库
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT,
                price REAL,
                category TEXT
            )
        """)
        conn.execute("INSERT INTO products VALUES (1, '拿铁', 32, '咖啡')")
        conn.execute("INSERT INTO products VALUES (2, '美式', 28, '咖啡')")
        conn.execute("INSERT INTO products VALUES (3, '芝士蛋糕', 35, '甜点')")
        conn.execute("INSERT INTO products VALUES (4, '司康饼', 28, '甜点')")
        conn.execute("INSERT INTO products VALUES (5, '红茶', 25, '茶')")
        conn.commit()
        conn.close()
        console.print(f"[bold cyan]数据库：[/bold cyan] {db_path}")

        # 3. 创建 Brain（带工具）
        brain = Brain(
            name="数据查询助手",
            llm=LLMEndpoint.from_env(),
        )

        # 4. 测试：纯计算问题
        console.print("\n[bold yellow]=== 测试 1：计算器 ===[/bold yellow]")
        q = "一打鸡蛋12 个，3 打一共多少个？再算一下 3 打鸡蛋每个 5 元，总价多少？"
        answer = await brain.ask_with_tool(q)
        console.print(f"[bold green]答：[/bold green] {answer}")

        # 5. 测试：SQL 查询
        console.print("\n[bold yellow]=== 测试 2：SQL 查询 ===[/bold yellow]")
        q = f"数据库在 {db_path}，帮我查所有甜点类产品的价格，按价格降序排列"
        answer = await brain.ask_with_tool(q)
        console.print(f"[bold green]答：[/bold green] {answer}")

        # 6. 测试：组合
        console.print("\n[bold yellow]=== 测试 3：SQL + 计算 ===[/bold yellow]")
        q = f"数据库在 {db_path}，帮我查所有咖啡类产品一共多少钱（每类数量算1）"
        answer = await brain.ask_with_tool(q)
        console.print(f"[bold green]答：[/bold green] {answer}")

    console.print(Panel("Step 10 完成", style="bold green"))

if __name__ == "__main__":
    asyncio.run(main())