"""文件抽象：把磁盘上的一个文件包装成带元数据的对象。

为什么需要这个抽象？
- 同一个文件可能被多种 processor 处理（元数据要共享）
- 文件来源可以是本地、S3、URL（不只是磁盘路径）
- 元数据会一路传递到向量库的 Document.metadata，影响检索过滤
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, AsyncIterator

from pydantic import BaseModel, Field


class QuivrFile(BaseModel):
    """一个被 Brain 管理的文件。

    字段：
        path: 文件的实际路径（可能是本地或远程，但当前实现只支持本地）
        file_id: 全局唯一 ID，由 Brain 在上传时分配
        file_extension: 后缀（小写，含点号，如 ".txt"）
        metadata: 任意业务元数据，会附加到每个 Document 上
    """

    path: Path
    file_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    file_extension: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """pydantic 钩子：构造后自动从 path 推断 file_extension。"""
        if not self.file_extension:
            self.file_extension = self.path.suffix.lower()

    async def open(self) -> AsyncIterator[bytes]:
        """异步读取文件二进制内容（生成器形式，节省内存）。"""
        try:
            import aiofiles  # noqa: F401

            async with aiofiles.open(self.path, "rb") as f:
                while chunk := await f.read(8192):
                    yield chunk
        except ImportError:
            # 没装 aiofiles 时退化为一次性读取（小文件够用）
            with open(self.path, "rb") as f:
                yield f.read()
