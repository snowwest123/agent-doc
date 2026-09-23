"""Step 2 验证脚本：上传 txt 文件 → 用 SimpleTxtProcessor 读取。"""
import asyncio
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rich.console import Console
from rich.panel import Panel

from myrag.processor.implementations import SimpleTxtProcessor
from myrag.storage import LocalStorage

console = Console(force_terminal=False, legacy_windows=False)


async def main() -> None:
    console.print(Panel.fit("Step 2: 文件读取", style="bold magenta"))

    # 1. 准备测试文件
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "a.txt").write_text(
            "Python 是一门广泛使用的编程语言。\n由 Guido van Rossum 于 1991 年发布。",
            encoding="utf-8",
        )
        (tmp_path / "b.txt").write_text(
            "RAG 是检索增强生成，包含 retrieve 和 generate 两个步骤。",
            encoding="utf-8",
        )
        # 故意带 ¥ 符号，验证编码处理
        (tmp_path / "c.txt").write_text(
            "幻海咖啡馆的招牌饮品价格：拿铁 ¥32、芝士蛋糕 ¥35。",
            encoding="utf-8",
        )

        # 2. 用 LocalStorage 加载
        storage = LocalStorage(dir_path=tmp_path)
        files = await storage.get_files()
        console.print(f"[bold cyan]已加载文件数：[/bold cyan] {storage.nb_files()}\n")
        for f in files:
            console.print(f"  - {f.path.name}  (id={str(f.file_id)[:8]}...)")

        # 3. 用 SimpleTxtProcessor 读
        processor = SimpleTxtProcessor()
        console.print()
        for f in files:
            docs = await processor.process_file(f)
            for doc in docs:
                preview = doc.page_content[:80].replace("\n", " ")
                console.print(f"  [green]{f.path.name}[/green]: {preview}...")
                console.print(f"    metadata: {doc.metadata}\n", style="dim")

    console.print(Panel("Step 2 完成", style="bold green"))


if __name__ == "__main__":
    asyncio.run(main())
