"""Step 9 验证：流式输出。"""
import asyncio
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from myrag.brain import Brain
from myrag.embedding.dashscope_embedder import DashScopeEmbedder

console = Console(legacy_windows=False)

async def main() -> None:
    console.print(Panel.fit("Step 9: 流式输出", style="bold magenta"))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "menu.txt").write_text(
            """幻海咖啡馆菜单：
            - 拿铁 ¥32
            - 美式 ¥28
            - 芝士蛋糕 ¥35
            - 司康饼 ¥28
            营业时间：周一至周五 08:00 - 22:00
            招牌饮品：幻海拿铁、海盐焦糖玛奇朵、星光美式""",
            encoding="utf-8",
        )
        (tmp_path / "policy.txt").write_text(
            """退换政策：现做饮品恕不退换。
            会员制度：消费满 200 元自动升级 9 折""",
            encoding="utf-8",
        )

        brain = await Brain.from_files(
            name="幻海知识库",
            file_paths=[tmp_path / "menu.txt", tmp_path / "policy.txt"],
            embedder=DashScopeEmbedder.from_env(),
        )

        # === 对比 1：非流式（一次性返回） ===
        console.print("\n[bold yellow]=== 对比 1：非流式 ===[/bold yellow]")
        start = time.time()
        answer = await brain.ask("店里有什么咖啡？各多少钱？")
        elapsed = time.time() - start
        console.print(f"[bold green]答（用了 {elapsed:.2f}s）：[/bold green] {answer}")

        # === 对比 2：流式（逐字返回） ===
        console.print("\n[bold yellow]=== 对比 2：流式 ===[/bold yellow]")

        # 用 Rich Live 实现实时滚动显示
        accumulated = ""
        start = time.time()
        first_token_time = None

        with Live("", refresh_per_second=10, console=console) as live:
            async for chunk in brain.ask_streaming("店里有什么咖啡？各多少钱？"):
                if first_token_time is None:
                    first_token_time = time.time() - start
                    ttft = first_token_time
                    console.print(f"  [dim]首字延迟: {ttft:.2f}s[/dim]")
                accumulated += chunk
                live.update(Panel(accumulated, title="实时回答", border_style="green"))

        total = time.time() - start
        console.print(f"\n[bold green]完整答案（总 {total:.2f}s）：[/bold green] {accumulated}")

        # === 对比 3：流式 + chainlit 风格 ===
        console.print("\n[bold yellow]=== 对比 3：模拟前端逐字显示 ===[/bold yellow]")
        console.print("[bold cyan]答：[/bold cyan]", end=" ")
        async for chunk in brain.ask_streaming("营业时间是什么时候？"):
            console.print(chunk, end="", style="green")
        console.print()  # 换行

    console.print(Panel("Step 9 完成", style="bold green"))

if __name__ == "__main__":
    asyncio.run(main())