"""处理器注册表：根据文件后缀找对应的 processor。

为什么需要这个？
- 每种文件类型一个 processor（txt / pdf / docx...）
- Brain 拿到一个文件时，不知道用哪个 processor 处理
- 注册表 = 后缀 → processor 的映射
"""
from __future__ import annotations

from myrag.processor.implementations import SimpleTxtProcessor
from myrag.processor.processor_base import ProcessorBase

class ProcessorRegistry:
    """维护 "后缀 → processor 实例" 的映射。

    使用方式：
        ProcessorRegistry.register(".pdf", PDFProcessor())
        proc = ProcessorRegistry.get(".pdf")
    """

    _registry: dict[str, ProcessorBase] = {}

    @classmethod
    def register(cls, extension: str, processor: ProcessorBase) -> None:
        """注册一个后缀对应的 processor。"""
        ext = extension.lower()
        if not ext.startswith("."):
            ext = "." + ext
        cls._registry[ext] = processor

    @classmethod
    def get(cls, extension: str) -> ProcessorBase | None:
        """根据后缀取 processor，没有就返回 None。"""
        ext = extension.lower()
        return cls._registry.get(ext)

    @classmethod
    def supported_extensions(cls) -> list[str]:
        """列出所有支持的后缀。"""
        return list(cls._registry.keys())

# 默认注册 .txt
ProcessorRegistry.register(".txt", SimpleTxtProcessor())