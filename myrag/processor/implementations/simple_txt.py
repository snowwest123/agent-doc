"""最简单的 txt 文件处理器。"""
from __future__ import annotations

import asyncio

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from myrag.processor.processor_base import ProcessorBase
from myrag.storage.file import QuivrFile


class SimpleTxtProcessor(ProcessorBase):
    """纯文本文件处理器。

    关键：encoding="utf-8"，避免 Windows 默认 GBK 报错（你之前在 quivr 里遇到过）。
    """

    supported_extensions = [".txt"]

    async def process_file_inner(self, file: QuivrFile) -> list[Document]:
        # TextLoader 是同步的；用 asyncio.to_thread 转异步，避免阻塞事件循环
        loader = TextLoader(str(file.path), encoding="utf-8", autodetect_encoding=True)
        docs = await asyncio.to_thread(loader.load)
        return docs
