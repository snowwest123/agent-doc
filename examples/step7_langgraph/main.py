"""Step 7 验证：LangGraph 工作流。"""
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.panel import Panel


from myrag.brain import Brain
from myrag.embedding.dashscope_embedder import DashScopeEmbedder
from myrag.workflow.graph import build_rag_graph, run_rag

console = Console(legacy_windows=False)
async def main() -> None:
    console.print(Panel.fit("Step 7: LangGraph 工作流", style="bold magenta"))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "menu.txt").write_text(
            """幻海咖啡馆菜单：
                - 拿铁 ¥32
                - 美式 ¥28
                营业时间：周一至周五 08:00 - 22:00
                招牌饮品：幻海拿铁、海盐焦糖
            """,
            encoding="utf-8",
        )
        (tmp_path / "policy.txt").write_text(
            """退换政策：现做饮品恕不退换。
               会员制度：消费满 200 元自动升级 9 折
            """,
            encoding="utf-8",
        )
        brain = await Brain.from_files(
            name="幻海知识库",
            file_paths=[tmp_path / "menu.txt", tmp_path / "policy.txt"],
            embedder=DashScopeEmbedder.from_env(),
        )
        # 直接调 LangGraph（不经过 Brain）
        graph = build_rag_graph(brain.llm, brain.vector_store)

        questions = [
            "拿铁多少钱？",
            "营业到几点？",
            "有金卡吗？",  # 测试 query rewrite（如果加 history 会更明显）
        ]

        # 真实场景：把前几轮的问答拼到 history，演示多轮对话
        running_history: list = []

        for q in questions:
            console.print(f"\n[bold yellow]问：[/bold yellow] {q}")

            # 跑 LangGraph（传入当前累积的 history）
            result = await run_rag(graph, q, chat_history=running_history)

            # 打印每个节点的输出
            console.print(f"  rewrite 后的问题: {result.get('rewritten_question', '?')}")
            console.print(f"  检索到 {len(result.get('retrieved_docs', []))} 条")
            console.print(f"  [bold green]答：[/bold green] {result['answer']}")

            # 把本轮问答加进 running_history，下一轮会用到
            running_history.append(HumanMessage(content=q))
            running_history.append(SystemMessage(content=result['answer']))

    console.print(Panel("Step 7 完成", style="bold green"))

if __name__ == "__main__":
    asyncio.run(main())