"""文件处理抽象：把任意格式文件转成 LangChain Document 列表。

为什么有两层（process_file + process_file_inner）？
- process_file（外层）：通用处理——校验、统计、包裹
- process_file_inner（内层）：子类只需实现"怎么读这个文件"
这样新增一个文件类型只需写 process_file_inner，通用逻辑白嫖。
"""
from __future__ import annotations

import abc
import logging
from typing import Any

from langchain_core.documents import Document

from myrag.storage.file import QuivrFile

logger = logging.getLogger("myrag")


class ProcessorBase(abc.ABC):
    """所有文件处理器的基类。"""

    supported_extensions: list[str] = []

    def check_supported(self, file: QuivrFile) -> None:
        """检查文件后缀是否被本 processor 支持。"""
        if file.file_extension not in self.supported_extensions:
            raise ValueError(
                f"{self.__class__.__name__} 不支持 {file.file_extension}，"
                f"支持的后缀：{self.supported_extensions}"
            )

    async def process_file(self, file: QuivrFile) -> list[Document]:
        """外层入口：校验 + 调子类实现 + 后处理。"""
        logger.info(f"Processing {file.path.name}")
        self.check_supported(file)
        docs = await self.process_file_inner(file)

        # 给每个 Document 附加文件元数据
        for doc in docs:
            doc.metadata = {
                **doc.metadata,
                **file.metadata,
                "file_id": str(file.file_id),
                "file_extension": file.file_extension,
                "original_file_name": file.path.name,
            }
        return docs

    @abc.abstractmethod
    async def process_file_inner(self, file: QuivrFile) -> list[Document]:
        """子类实现：真正读取文件内容，返回 Document 列表。"""
