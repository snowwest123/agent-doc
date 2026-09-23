"""Step 3 验证：上传文件 → 创建 Brain → 提问 → 打印答案。"""
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rich.console import Console
from rich.panel import Panel

from myrag.brain import Brain

console = Console(legacy_windows=False)

async def main() -> None:
    console.print(Panel.fit("Step 3: Brain 简易版", style="bold magenta"))

    # 1. 准备测试文件
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "menu.txt").write_text(
            """幻海咖啡馆菜单：
            - 拿铁 ¥32
            - 美式 ¥28
            - 芝士蛋糕 ¥35
            - 司康饼 ¥28
            营业时间：周一至周五 08:00 - 22:00
            """,
            encoding="utf-8",
        )
        (tmp_path / "policy.txt").write_text(
            """退换政策：
            本店商品均为现做饮品，恕不退换。
            如发现品质问题，请当场反馈，我们将免费重做。
            """,
            encoding="utf-8",
        )

        # 2. 创建 Brain
        brain = await Brain.from_files(
            name="幻海咖啡馆助手",
            file_paths=[tmp_path / "menu.txt", tmp_path / "policy.txt"],
        )

        # 3. 打印 Brain 信息
        console.print()
        brain.print_info()

        # 4. 提问测试
        console.print()
        questions = [
            "拿铁多少钱？",
            "营业到几点？",
            "能退换吗？",
            "店里有没有 wifi？",  # 文档里没答案，应该回答"我不知道"
        ]
        for q in questions:
            console.print(f"\n[bold yellow]问：[/bold yellow] {q}")
            answer = await brain.ask(q)
            console.print(f"[bold green]答：[/bold green] {answer}")

    console.print(Panel("Step 3 完成", style="bold green"))

if __name__ == "__main__":
    asyncio.run(main())