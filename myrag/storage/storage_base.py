"""存储抽象：所有存储后端（本地、S3、内存）都继承这个接口。

设计要点：
- 所有方法都是 async（将来对接 S3 / OSS 必然是异步 IO）
- get_files() 总是返回新 list（避免外部修改内部状态）
"""
from __future__ import annotations

import abc
import uuid
from pathlib import Path

from myrag.storage.file import QuivrFile


class StorageBase(abc.ABC):
    """抽象存储基类。"""

    @abc.abstractmethod
    async def upload_file(
        self,
        file_path: str | Path,
        exists_ok: bool = False,
        metadata: dict | None = None,
    ) -> QuivrFile:
        """上传文件到存储后端，返回包装后的 QuivrFile。"""

    @abc.abstractmethod
    async def get_files(self) -> list[QuivrFile]:
        """列出所有文件。"""

    @abc.abstractmethod
    async def remove_file(self, file_id: uuid.UUID) -> None:
        """按 ID 删除文件。"""

    @abc.abstractmethod
    def nb_files(self) -> int:
        """文件数量。"""
