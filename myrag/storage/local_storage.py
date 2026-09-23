"""本地文件系统存储。"""
from __future__ import annotations

import asyncio
import shutil
import uuid
from pathlib import Path

from myrag.storage.file import QuivrFile
from myrag.storage.storage_base import StorageBase


class LocalStorage(StorageBase):
    """把文件存在本地磁盘的某个目录下。"""

    def __init__(self, dir_path: str | Path) -> None:
        self.dir_path = Path(dir_path)
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self._file_index: dict[uuid.UUID, Path] = {}
        self._scan_existing_files()

    def _scan_existing_files(self) -> None:
        """启动时扫描目录，把已有文件加入索引。"""
        for p in self.dir_path.iterdir():
            if p.is_file():
                file_id = uuid.uuid4()
                self._file_index[file_id] = p

    async def upload_file(
        self,
        file_path: str | Path,
        exists_ok: bool = False,
        metadata: dict | None = None,
    ) -> QuivrFile:
        src = Path(file_path)
        if not src.exists():
            raise FileNotFoundError(f"{src} not found")
        if not src.is_file():
            raise ValueError(f"{src} is not a file")

        dst = self.dir_path / src.name
        if dst.exists() and not exists_ok:
            raise FileExistsError(f"{dst} already exists")

        if src.resolve() != dst.resolve():
            await asyncio.to_thread(shutil.copy2, src, dst)

        file_id = uuid.uuid4()
        self._file_index[file_id] = dst

        return QuivrFile(
            path=dst,
            file_id=file_id,
            metadata=metadata or {},
        )

    async def get_files(self) -> list[QuivrFile]:
        return [
            QuivrFile(path=path, file_id=fid, metadata={})
            for fid, path in self._file_index.items()
        ]

    async def remove_file(self, file_id: uuid.UUID) -> None:
        path = self._file_index.pop(file_id, None)
        if path is None:
            raise KeyError(f"file_id {file_id} not in storage")
        if path.exists():
            path.unlink()

    def nb_files(self) -> int:
        return len(self._file_index)
