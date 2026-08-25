"""内置工具包：M1 的 3 个核心工具 + M2 第 4 步的 load_skill。

这里提供一个「开箱即用」的 registry——把内置工具注册好，
供 agent/loop.py 直接使用。以后加新工具（M4 的编辑、搜索等），
在这里多注册一个就行。

为什么不用自动发现（扫描目录里的所有类）：
保持显式——注册了哪些工具、顺序如何，一眼看穿。
自动发现是 M4 再考虑的优化。

M2 第 4 步：LoadSkill 的 skills_dir 可注入（测试用临时目录），
与 agent/loop.py 的 Agent(skills_dir=...) 保持一致——元数据扫描
和技能加载必须指向同一个目录，否则模型会看到目录里没有的技能。
"""

from __future__ import annotations

from pathlib import Path

from ..base import Tool
from ..registry import ToolRegistry
from .load_skill import LoadSkill
from .read_file import ReadFile
from .run_command import RunCommand
from .write_file import WriteFile


def default_registry(skills_dir: Path | None = None) -> ToolRegistry:
    """创建并注册全部内置工具的 registry。"""
    registry = ToolRegistry()
    registry.register(ReadFile())
    registry.register(WriteFile())
    registry.register(RunCommand())
    registry.register(LoadSkill(skills_dir=skills_dir))
    return registry