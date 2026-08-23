"""agent/loop.py —— ReAct 主循环：把 LLM + 工具串起来的「胶水层」。

对应书 Ch1 的核心机制：模型思考 → 决定调工具 → 框架执行 → 结果回填 →
再思考……直到模型不再调工具、输出最终回答。

关键设计（每个都是取舍，注释写「为什么」）：

1. 轨迹 = 局部变量，每次 run() 全新
   一次任务一条轨迹（system + user + assistant + tool 消息列表）。
   跨任务记忆是 M3 的事，M1 刻意不做——单任务语义清晰。

2. assistant 消息原样回填（含 tool_calls）
   轨迹的累积性（书 Ch1 原文）：模型下一轮必须看到自己之前想了什么、
   调了什么、得到了什么，才能理解「当前处于任务的哪个阶段」。
   这是 ReAct 为什么是「循环」而不是「一问一答」的根本原因。

3. tool 消息必须带 tool_call_id
   OpenAI 协议硬性要求：tool 消息要对应 assistant 那条 tool_calls 里的 id，
   不带会报错。tool_calls 从 SDK 对象转成纯 dict 再回填——
   保证发给模型的 messages 是标准 JSON 结构，SDK 序列化不炸。

4. 坏 JSON 参数 → 回填错误让模型自纠
   参数来自模型 = 不可信输入。json.loads 失败不抛异常，
   而是作为 tool 消息回填错误（延续 registry「错误是模型的输入」原则）。

5. max_iterations 熔断（书 Ch1 护栏·控制流层错误恢复）
   模型可能陷入死循环（反复调同一个失败工具、反复思考不回答）。
   达到上限就停止，返回错误 + 最后状态，不让 token 白烧。

6. 循环内 LLM 异常不吞，向上抛
   重试已由 llm/client.py 处理（API 层：指数退避 + 抖动）。
   重试耗尽 = 无法恢复的错误，loop 层不假装能处理，抛给 CLI 层显示。
"""

from __future__ import annotations

import json
from typing import Any

from llm.client import LLMClient
from tools.registry import ToolRegistry

# 系统提示词：Agent 的「岗位说明书」，整个对话中不变（静态前缀）。
# 写得简短但覆盖三个关键行为：
# - 什么时候调工具（需要信息/要执行操作时）
# - 看到错误怎么办（调整策略，不要重复同样的失败调用）
# - 什么时候停（任务完成或信息足够就直接回答）
SYSTEM_PROMPT = (
    "你是 our-agent，一个能调用工具完成任务的中文助手。\n"
    "需要读取文件、执行命令等操作时，调用对应的工具；\n"
    "工具返回的错误信息是给你的反馈，根据错误调整策略，不要重复同样的失败调用；\n"
    "任务完成或信息足够时，直接输出最终答案（中文），不要再调用工具。"
)


class Agent:
    """ReAct 主循环。持有 LLM 客户端 + 工具注册表，run() 执行一个任务。"""

    def __init__(
        self,
        client: LLMClient,
        registry: ToolRegistry,
        max_iterations: int = 10,
        verbose: bool = True,
    ) -> None:
        self._client = client
        self._registry = registry
        self.max_iterations = max_iterations
        self.verbose = verbose
        # M2 多轮会话：跨 run 的历史 + 缓存命中统计（度量尺子）
        self._history: list[dict[str, Any]] = []
        self.cache_stats: dict[str, int] = {"hit": 0, "miss": 0}

    def run(self, task: str) -> str:
        """执行一个任务，返回最终回答字符串。

        M2 多轮对话：轨迹分两层——
        - self._history：跨 run 的会话历史（上一次任务留下的完整轨迹）
        - 本次 messages：system + history + 新 user 任务

        任务完成后本次轨迹存回 history，下一轮 run 就能看到上一轮
        做了什么、得到了什么——这就是「多轮记忆」（会话内）。
        跨任务的长期记忆是 M3 的事，M2 只做会话内记忆。

        熔断时不保存轨迹：未完成的任务会误导下一轮（看着像做完了），
        而且熔断往往是一堆重复失败的工具调用，保存只会膨胀上下文。
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self._history,
            {"role": "user", "content": task},
        ]
        # 新 user 消息的位置（初始 messages 的最后一个元素）。
        # 任务完成后 messages[user_start:] 就是本次新增的完整轨迹：
        # 从新 user 开始，到最终 assistant 回答结束。
        user_start = len(messages) - 1
        # 工具定义也是静态前缀的一部分：每轮都原样发送（KV Cache 友好）
        schemas = self._registry.schemas()

        for iteration in range(self.max_iterations):
            resp = self._client.chat(messages, tools=schemas)
            # 记录每个响应的 token 用量 + 缓存命中（M2 度量尺子）
            self._record_usage(resp.get("usage") or {})
            content = resp.get("content")
            raw_tool_calls = resp.get("tool_calls") or []

            # SDK 对象 → 纯 dict（协议要求的回填格式）
            tool_calls: list[dict[str, Any]] = []
            for tc in raw_tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )

            # assistant 消息原样回填（含思考内容 + tool_calls）——
            # 下一轮模型必须看到自己刚才做了什么
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            if not tool_calls:
                # 模型不再调工具 → content 就是最终回答
                answer = content if content is not None else "(模型没有输出内容)"
                if self.verbose:
                    print(f"\n[Agent] 完成（第 {iteration + 1} 轮）")
                # 任务成功完成 → 本次轨迹（新 user 起的所有消息）存入会话历史
                self._history.extend(messages[user_start:])
                return answer

            if self.verbose:
                names = ", ".join(tc["function"]["name"] for tc in tool_calls)
                print(f"[Agent] 第 {iteration + 1} 轮调用工具: {names}")

            # 逐个执行工具调用，结果作为 tool 消息回填
            for tc in tool_calls:
                name = tc["function"]["name"]
                raw_args = tc["function"]["arguments"]
                try:
                    # arguments 是模型生成的 JSON 字符串，可能非法——
                    # 解析失败就回填错误，让模型自己改正（错误是模型的输入）
                    args: dict[str, Any] = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError as exc:
                    result = json.dumps(
                        {"error": f"参数不是合法 JSON: {exc}"}, ensure_ascii=False
                    )
                else:
                    result = self._registry.execute(name, args)
                messages.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": result}
                )
            # 回到循环顶部：模型看到完整轨迹（含工具结果）继续思考

        # 熔断：达到 max_iterations 还没出最终回答。
        # 注意这里不保存历史（见 run docstring 的设计取舍）。
        last = messages[-1]
        last_content = last.get("content") if isinstance(last, dict) else None
        hint = f"模型最后输出: {last_content[:200]}" if last_content else "模型没有输出内容"
        return f"任务未完成：已达到最大迭代次数 {self.max_iterations}。{hint}"

    def reset(self) -> None:
        """清空会话历史与缓存统计（REPL 的「新会话」命令用）。

        回到 M1 的单任务模式：下一轮 run 从干净的 system + user 开始。
        """
        self._history.clear()
        self.cache_stats = {"hit": 0, "miss": 0}

    def _record_usage(self, usage: dict[str, Any]) -> None:
        """累计本轮 token 用量与 Prompt Cache 命中（M2 度量尺子）。

        DeepSeek 的 usage 字段：
        - prompt_cache_hit_tokens: 命中服务商缓存的前缀 token 数
        - prompt_cache_miss_tokens: 未命中、需重新计算的 token 数
        命中率 = hit / (hit + miss)。前缀稳定时应该接近 100%——
        哪天命中率掉下来，就是哪里破坏了缓存前缀（如改了 system prompt）。

        FakeClient（测试）可能不返回 usage —— .get(..., 0) 兜底为 0，
        不影响循环逻辑（度量是附加信息，不是核心依赖）。
        """
        hit = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
        miss = int(usage.get("prompt_cache_miss_tokens", 0) or 0)
        self.cache_stats["hit"] += hit
        self.cache_stats["miss"] += miss
        if self.verbose:
            total = hit + miss
            rate = f"{hit / total:.0%}" if total else "n/a"
            print(
                f"[usage] prompt={usage.get('prompt_tokens', 0)} "
                f"cache_hit={hit} miss={miss} 命中率={rate}"
            )
