"""工具注册表 —— 模型看到的「工具名 → Python 函数」的映射。

对应书 Ch4 的工具选择流程：模型根据工具描述决定「要调哪个工具」，
框架则根据工具名找到对应的函数并执行。registry 就是这个名字解析表。

职责三个：
- register(): 把工具收进来（名字查重，防止两个工具同名）
- schemas():  生成全部工具定义，交给 LLM 客户端随请求发给模型
- execute():  模型决定调用后，按名字执行并返回结果

为什么错误在 registry 层处理（对应书 Ch4 错误恢复·工具层）：
工具层的错误是「模型的输入」，不是「系统的故障」。
- 幻觉调用（模型编造不存在的工具名）→ 返回错误信息给模型
- 参数畸形（缺参数/类型错）→ 同样作为错误信息回填
模型需要看到错误才能自我纠正，所以错误不终止会话、不抛异常——
抛异常会打断整个 ReAct 循环，回填错误信息则让循环继续。
"""

from __future__ import annotations

import json
from typing import Any

from .base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """收编一个工具。名字重复直接报错——重名会让模型无法区分。"""
        if not tool.name:
            raise ValueError("工具必须声明 name")
        if tool.name in self._tools:
            raise ValueError(f"工具重名: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """按名字取工具，没有就返回 None（execute 里再决定怎么处理）。"""
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        """全部工具定义——这就是发给模型的 tools 参数。

        顺序即模型看到的顺序，先注册的先出现。
        """
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """执行一个工具调用，返回字符串结果。

        无论成功失败都返回字符串（不抛异常）——因为返回值是
        tool 消息的内容，要回填给模型继续思考。
        """
        tool = self._tools.get(name)
        if tool is None:
            # 模型幻觉编造了不存在的工具名：把它作为错误告诉模型，
            # 模型看到「未知工具」后会重新选择，而不是卡死
            return json.dumps(
                {"error": f"未知工具: {name}"}, ensure_ascii=False
            )
        try:
            result = tool.run(**arguments)
            # run() 返回字符串就直接用；返回其他类型（dict/list）序列化，
            # 保证回填给模型的永远是字符串（tool 消息的 content 是字符串）
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False, default=str)
        except TypeError as exc:
            # 参数畸形：模型传的参数和 schema 对不上
            # （缺参数 / 多了未知参数 / 类型不匹配）
            return json.dumps(
                {"error": f"参数错误: {exc}"}, ensure_ascii=False
            )
        except Exception as exc:
            # 工具内部执行失败：也是模型的输入，回填让它调整
            return json.dumps(
                {"error": f"工具执行失败: {exc}"}, ensure_ascii=False
            )
