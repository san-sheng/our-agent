"""工具抽象 —— Agent 的「手脚」接口。

这个模块定义工具的最小接口，对应书 Ch4 的核心思想：
每个工具 = 名字 + 描述 + 参数 schema + 执行函数。

为什么这样设计（对应书里的三个原则）：

1. 描述的艺术（Ch4「工具描述的艺术」）
   模型靠 description 决定「什么时候用这个工具」，
   所以描述要写「什么时候用」+ 边界条件（反例），而不是「能做什么」。
   写不好的后果是模型频繁选错工具——此时应优先修描述，而不是换模型。

2. 参数保真性（Ch4「参数传递的保真性」）
   工具必须原样接收模型传来的参数，不在模型不知情的情况下修改。
   书里 Cursor 的教训：工具悄悄把弯引号转成直引号，模型反复失败还找不出原因——
   模型感知到的世界与工具操作的世界之间，不能存在系统性偏差。

3. schema 显式声明，不自动生成
   每个工具手写 parameters（JSON Schema），让每一步都看得见摸得着。
   从函数签名自动推断 schema 是真实框架的做法，但那是 M4 的优化方向——
   学习阶段先看清「工具定义长什么样」。
"""

from __future__ import annotations

from typing import Any


class Tool:
    """所有工具的基类。子类声明 4 个类属性 + 实现 run()。

    - name:        模型看到的工具名（必须全局唯一，registry 里查重）
    - description: 什么时候用 + 边界条件（反例）
    - parameters:  JSON Schema，声明参数类型/约束/示例
    - run():       实际执行，返回字符串（作为 tool 消息回填给模型）
    """

    name: str = ""
    description: str = ""
    parameters: dict = {}

    def run(self, *args: Any, **kwargs: Any) -> str:
        """执行工具。

        args/kwargs 是模型传来的参数（由 registry 按名字分发，实际调用
        走 **kwargs；*args 只是为了让类型检查器接受子类的具体签名——
        每个工具声明自己的 run(path=..., offset=..., ...)，和只有
        **kwargs 的基类签名不兼容，会触发 reportIncompatibleMethodOverride）。
        返回字符串——这个字符串会原样作为 tool 消息内容回填给模型，
        所以返回的信息要自包含：成功/失败、关键数据、错误原因。
        """
        raise NotImplementedError(f"{type(self).__name__} 没有实现 run()")

    def schema(self) -> dict:
        """生成 OpenAI function calling 格式的工具定义。

        LLM 客户端（llm/client.py）把每个工具的 schema() 结果
        收集成 tools 参数发给模型，模型就能看到这个工具的存在、
        知道什么时候用、以及怎么传参。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
