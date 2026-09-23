"""Step 4 验证：上传多个文件 → 用 RAG 检索 → 回答。"""
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rich.console import Console
from rich.panel import Panel

from myrag.brain import Brain
from myrag.embedding.dashscope_embedder import DashScopeEmbedder

console = Console(legacy_windows=False)

async def main() -> None:
    console.print(Panel.fit("Step 4: 真正的 RAG（向量检索）", style="bold magenta"))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # 故意多放几个文件，验证"只检索相关"是否生效
        (tmp_path / "menu.txt").write_text(
            """幻海咖啡馆菜单：
                - 拿铁 ¥32
                - 美式 ¥28
                - 芝士蛋糕 ¥35
                - 司康饼 ¥28
                营业时间：周一至周五 08:00 - 22:00，周末 09:00 - 23:00
                招牌饮品：幻海拿铁、海盐焦糖玛奇朵、星光美式""",
            encoding="utf-8",
        )
        (tmp_path / "policy.txt").write_text(
            """退换政策：
                本店商品均为现做饮品，恕不退换。
                如发现品质问题，请当场反馈，我们将免费重做。
                会员制度：消费满 200 元自动升级 9 折
            """,
            encoding="utf-8",
        )
        (tmp_path / "company.txt").write_text(
            """幻海科技有限公司成立于 2018 年，
            是一家专注于 AI 产品研发的科技公司。
            总部位于虚构市，员工 120 人，研发人员占 60%。
            公司主营：对话式数据分析平台。""",
            encoding="utf-8",
        )

        # 构造 Brain（带向量库）
        brain = await Brain.from_files(
            name="幻海知识库",
            file_paths=[tmp_path / "menu.txt", tmp_path / "policy.txt", tmp_path / "company.txt"],
            embedder=DashScopeEmbedder.from_env(),
        )

        console.print(
            f"\n[bold cyan]向量库文档数：[/bold cyan] "
            f"{brain.vector_store.nb_docs if brain.vector_store else 0}\n"
        )

        # 提问：分别针对不同文档
        questions = [
            "拿铁多少钱？",            # menu.txt
            "能退换吗？",              # policy.txt
            "幻海科技有多少员工？",    # company.txt
            "公司主营什么？",          # company.txt
        ]

        for q in questions:
            console.print(f"\n[bold yellow]问：[/bold yellow] {q}")
            answer = await brain.ask(q)
            console.print(f"[bold green]答：[/bold green] {answer}")

    console.print(Panel("Step 4 完成", style="bold green"))

if __name__ == "__main__":
    asyncio.run(main())