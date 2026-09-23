"""Step 6 验证：YAML 驱动检索参数。"""
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rich.console import Console
from rich.panel import Panel

from myrag.brain import Brain
from myrag.embedding.dashscope_embedder import DashScopeEmbedder
from myrag.retrieval.config import RetrievalConfig

console = Console(legacy_windows=False)

CONFIG_PATH = Path(__file__).parent / "workflow.yaml"

async def main() -> None:
    console.print(Panel.fit("Step 6: YAML 配置驱动", style="bold magenta"))

    # 1. 先把 yaml 打印出来（让你看到生效的配置）
    cfg = RetrievalConfig.from_yaml(CONFIG_PATH)
    console.print(f"\n[bold cyan]当前 YAML 配置：[/bold cyan]")
    console.print(cfg.model_dump_json(indent=2))
    console.print()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "menu.txt").write_text(
            """幻海咖啡馆菜单：
                - 拿铁 ¥32
                - 美式 ¥28
                营业时间：周一至周五 08:00 - 22:00
                招牌饮品：幻海拿铁、海盐焦糖""",
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
            config_path=CONFIG_PATH,
        )

        # 提问验证        
        questions = [
            "拿铁多少钱？",
            "金卡会员几折？",  # 文档里没金卡，只有 9 折普通会员
            "营业到几点？",
        ]

        for q in questions:
            console.print(f"\n[bold yellow]问：[/bold yellow] {q}")
            answer = await brain.ask(q)
            console.print(f"[bold green]答：[/bold green] {answer}")

    console.print(Panel("Step 6 完成", style="bold green"))

if __name__ == "__main__":
    asyncio.run(main())