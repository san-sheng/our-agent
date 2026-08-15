"""内置工具包：M1 的 3 个核心工具。

这里提供一个「开箱即用」的 registry——把 3 个内置工具注册好，
供 agent/loop.py 直接使用。以后加新工具（M4 的编辑、搜索等），
在这里多注册一个就行。

为什么不用自动发现（扫描目录里的所有类）：
M1 保持显式——注册了哪些工具、顺序如何，一眼看穿。
自动发现是 M4 再考虑的优化。
"""

from __future__ import annotations

from ..base import Tool
from ..registry import ToolRegistry
from .read_file import ReadFile
from .run_command import RunCommand
from .write_file import WriteFile


def default_registry() -> ToolRegistry:
    """创建并注册全部内置工具的 registry。"""
    registry = ToolRegistry()
    registry.register(ReadFile())
    registry.register(WriteFile())
    registry.register(RunCommand())
    return registry
