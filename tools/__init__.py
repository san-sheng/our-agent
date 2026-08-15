"""工具系统包：Agent 的「手脚」。

对外暴露：
- Tool / ToolRegistry：工具抽象与注册表
- default_registry()：开箱即用的内置工具注册表

注意：这里不做 `from .builtin import *` 之类的重导出——
保持导入路径清晰，agent/loop.py 里显式 import 需要的符号。
"""

from .base import Tool
from .builtin import default_registry
from .registry import ToolRegistry

__all__ = ["Tool", "ToolRegistry", "default_registry"]
