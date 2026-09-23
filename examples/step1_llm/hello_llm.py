"""Step 1 验证脚本：调通千问。"""
import sys
from pathlib import Path

# 让脚本能找到 myrag 包（开发期常见做法）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.panel import Panel

from myrag.llm import LLMEndpoint

console = Console()

def main() -> None:
    console.print(Panel.fit("🚀 Step 1: LLMEndpoint 验证", style="bold magenta"))

    # 1. 从环境变量构造
    llm = LLMEndpoint.from_env()
    console.print(f"[bold cyan]LLM 配置：[/bold cyan] {llm.info()}\n")

    # 2. 同步调用
    messages = [
        SystemMessage(content="你是一个简洁的助手，回答控制在 50 字以内。"),
        HumanMessage(content="用一句话介绍 RAG 是什么。"),
    ]
    console.print("[bold yellow]→ 同步调用 invoke()[/bold yellow]")
    answer = llm.invoke(messages)
    console.print(f"[bold green]回答：[/bold green] {answer}\n")

    # 3. 流式调用
    console.print("[bold yellow]→ 流式调用 stream()[/bold yellow]")
    full = ""
    for chunk in llm.stream(messages):
        # chunk 是 AIMessageChunk，content 可能是 str 或 list
        piece = chunk.content if isinstance(chunk.content, str) else "".join(chunk.content)
        full += piece
        console.print(piece, end="", style="green")
    console.print("\n")

    # 4. token 计数
    tokens = llm.count_tokens(answer)
    console.print(f"[bold cyan]回答的 token 数：[/bold cyan] {tokens}\n")

    console.print(Panel("✅ Step 1 完成", style="bold green"))

if __name__ == "__main__":
    main()