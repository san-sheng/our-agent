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

7. 状态栏（M2 第 2 步，书 Ch2 §2.6）
   每轮请求末尾注入一条 user-role meta 消息：时间戳 / 工具计数 / TODO。
   「上下文窗口是一台只有一半的检索引擎」——模型擅长检索、不擅长归纳，
   状态栏就是那个「提炼层」。三条铁律：
   - 用代码维护，绝不让大模型去读历史总结（20 行代码就够的事）
   - 写成键值对，不是散文（散文 = 让模型重新扫描一遍，效果更差）
   - 模型无条件信任状态栏 → 内容必须准确，且只来自可信观测
   状态栏只拼进发送给模型的请求，不进核心轨迹 messages——
   过期状态栏不该成为会话历史的一部分。

8. 熔断 [UNFINISHED] 摘要（M2 第 2 步，2026-08-24 拍板）
   熔断不再「什么都不留」：生成键值对摘要
   （任务目标 / 已完成步骤 / 下一步 / 失败点，复用状态栏格式）写入 history。
   原始垃圾轨迹（一堆重复失败的工具调用）不死记——摘要让下一轮
   能「接着办完」，半截子轨迹则会误导模型以为任务已完成。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable

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

# 状态栏 / 失败摘要的标题标记。测试靠它辨认「框架注入的 meta 消息」，
# 与对话主体（用户任务、模型回答）区分开。
STATUS_BAR_HEADER = "【状态栏】"
UNFINISHED_HEADER = "【状态栏 · UNFINISHED】"


class Agent:
    """ReAct 主循环。持有 LLM 客户端 + 工具注册表，run() 执行一个任务。"""

    def __init__(
        self,
        client: LLMClient,
        registry: ToolRegistry,
        max_iterations: int = 10,
        verbose: bool = True,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._client = client
        self._registry = registry
        self.max_iterations = max_iterations
        self.verbose = verbose
        # M2 第 2 步：状态栏/摘要的时间来源。默认真实时钟；
        # 测试注入固定时间让断言可复现（书 Ch2 §2.6：状态栏必须可验证）
        self._clock = clock
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

        M2 第 2 步（状态栏 + 熔断摘要）：
        - 每轮请求末尾注入状态栏（时间/工具计数/TODO），只进请求不进历史
        - 熔断时写 [UNFINISHED] 摘要进 history（2026-08-24 拍板）：
          什么也不留会让「接着办完」无从下手，留原始轨迹又会误导
          （看着像做完了）——折中是存一条紧凑的失败摘要。
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
        # 本轮工具调用计数 —— 状态栏的数据来源，用代码维护
        # （书 Ch2 §2.6：状态栏绝不让大模型去「读历史总结」）
        tool_counts: dict[str, int] = {}

        for iteration in range(self.max_iterations):
            # 状态栏：上下文末尾的 user-role meta 消息（书 Ch2 §2.6）。
            # 每轮重新生成、即时反映进展（时间/工具计数/进度/TODO）；
            # 只拼进请求副本（req），不进核心轨迹 messages——
            # 会话历史不被过期状态栏污染，也守住「历史只追加不改写」。
            req = list(messages) + [
                {"role": "user", "content": self._status_bar(iteration + 1, tool_counts)}
            ]
            resp = self._client.chat(req, tools=schemas)
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
                # 计数：模型确实发起了这次调用（无论成败）——状态栏如实反映
                tool_counts[name] = tool_counts.get(name, 0) + 1
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
        # M2 第 2 步（2026-08-24 拍板）：不留原始垃圾轨迹，
        # 写一条 [UNFINISHED] 摘要（复用状态栏格式）进 history。
        last = messages[-1]
        last_content = last.get("content") if isinstance(last, dict) else None
        hint = f"模型最后输出: {last_content[:200]}" if last_content else "模型没有输出内容"
        summary = self._unfinished_summary(task, tool_counts)
        failed_user = dict(messages[user_start])
        failed_user["content"] = f"{failed_user['content']}\n\n{summary}"
        self._history.append(failed_user)
        return f"任务未完成：已达到最大迭代次数 {self.max_iterations}。{hint}"

    def reset(self) -> None:
        """清空会话历史与缓存统计（REPL 的「新会话」命令用）。

        回到 M1 的单任务模式：下一轮 run 从干净的 system + user 开始。
        """
        self._history.clear()
        self.cache_stats = {"hit": 0, "miss": 0}

    def _status_bar(self, iteration: int, tool_counts: dict[str, int]) -> str:
        """生成状态栏文本（书 Ch2 §2.6：上下文末尾的 user-role meta 消息）。

        三格信息（M2 第 2 步定的）：时间戳 / 工具计数 / TODO 进度。
        刻意写成键值对（一行一值）而不是散文——论文实验证明：
        键值对让模型「瞥一眼」就能定位；散文等于让它再扫描一遍，效果更差。
        """
        now = self._clock()
        total = sum(tool_counts.values())
        detail = "、".join(f"{name}×{n}" for name, n in tool_counts.items()) or "无"
        return (
            f"{STATUS_BAR_HEADER}\n"
            f"时间: {now:%Y-%m-%d %H:%M:%S}\n"
            f"工具: 已调用 {total} 次（{detail}）\n"
            f"进度: 第 {iteration}/{self.max_iterations} 轮\n"
            "TODO: 输出最终回答"
        )

    def _unfinished_summary(self, task: str, tool_counts: dict[str, int]) -> str:
        """生成熔断 [UNFINISHED] 摘要（复用状态栏格式）。

        四个字段对应 2026-08-24 拍板：任务目标 / 已完成步骤 / 下一步 / 失败点。
        任务目标截断到 100 字符——摘要要紧凑（书 Ch2 §2.6：低 token 成本）。
        键值对 + 明确的 UNFINISHED 标记，让下一轮模型「瞥一眼」就知道
        上次任务卡在哪、该从哪继续。
        """
        now = self._clock()
        total = sum(tool_counts.values())
        detail = "、".join(f"{name}×{n}" for name, n in tool_counts.items()) or "无"
        goal = task if len(task) <= 100 else task[:97] + "..."
        return (
            f"{UNFINISHED_HEADER}\n"
            f"时间: {now:%Y-%m-%d %H:%M:%S}\n"
            f"任务目标: {goal}\n"
            f"已完成: 工具调用 {total} 次（{detail}）\n"
            "下一步: 输出最终回答（未完成）\n"
            f"失败点: 达到最大迭代次数 {self.max_iterations}（熔断）"
        )

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