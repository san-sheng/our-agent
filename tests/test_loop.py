"""agent/loop.py 的行为测试。

核心思路：**不真调 API**。用一个「假 LLM 客户端」预置好响应序列，
驱动 ReAct 循环，验证循环逻辑本身（书 Ch1 的轨迹累积、熔断、错误自纠）。

为什么能这样测：Agent 只依赖 client.chat(messages, tools) 的返回格式
（{"role", "content", "tool_calls"}），不关心背后是不是真模型——
把「模型行为」参数化，测的是框架（Harness），这正是 Harness 工程的思想：
模型之外的部分可以且应该被确定性测试。

M2 第 2 步：每轮请求末尾会被注入一条状态栏 meta 消息（user 角色）。
断言轨迹结构时用 _strip_status() 滤掉它——状态栏是框架的「仪表盘」，
不属于对话主体；断言失败摘要时则直接查 history 里的 [UNFINISHED] 块。
"""

from __future__ import annotations

import json
import unittest
from datetime import datetime
from types import SimpleNamespace
from typing import Callable

from agent.loop import STATUS_BAR_HEADER, UNFINISHED_HEADER, Agent
from llm.client import LLMClient
from tools.builtin import default_registry


def _tc(tool_id: str, name: str, arguments: str) -> SimpleNamespace:
    """构造一个假的 tool_call 对象（模拟 SDK 的 ChatCompletionMessageToolCall）。"""
    return SimpleNamespace(
        id=tool_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class _FakeClient:
    """假 LLM 客户端：按顺序弹出预置响应，并记录每次收到的 messages。

    requests 记录让测试能验证「轨迹累积」——第二轮请求里
    必须包含第一轮的工具结果（tool 消息）。
    """

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.requests: list[list[dict]] = []

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        self.requests.append(list(messages))
        return self._responses.pop(0)


def _make_agent(
    fake: _FakeClient, max_iterations: int = 10, clock: Callable | None = None
) -> Agent:
    # 类型上 client 是 LLMClient，但 FakeClient 实现了相同的 chat 接口——
    # 鸭子类型，测试注入用
    kwargs: dict = {}
    if clock is not None:
        kwargs["clock"] = clock
    return Agent(fake, default_registry(), max_iterations=max_iterations, verbose=False, **kwargs)  # type: ignore[arg-type]


def _strip_status(messages: list[dict]) -> list[dict]:
    """滤掉框架注入的状态栏 meta 消息，剩下核心轨迹。

    状态栏是上下文末尾的 user-role meta 消息（书 Ch2 §2.6），
    不属于对话主体；断言轨迹结构时先滤掉，断言才聚焦在「轨迹」本身。
    """
    return [
        m
        for m in messages
        if not (
            m["role"] == "user"
            and isinstance(m.get("content"), str)
            and m["content"].startswith(STATUS_BAR_HEADER)
        )
    ]


class LoopTest(unittest.TestCase):
    def test_direct_answer(self):
        """模型第一轮就回答，不调工具 → run() 直接返回回答。"""
        fake = _FakeClient(
            [{"role": "assistant", "content": "直接回答：2+2=4", "tool_calls": None}]
        )
        agent = _make_agent(fake)
        self.assertEqual(agent.run("2+2=?"), "直接回答：2+2=4")
        # 只调了一次 LLM，没有工具消息
        self.assertEqual(len(fake.requests), 1)

    def test_tool_loop_then_answer(self):
        """先调工具、拿到结果后再回答 → 轨迹累积正确。

        第一轮：模型要调 run_command("echo hello")
        第二轮：模型看到工具结果后给出最终回答
        """
        fake = _FakeClient(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_tc("call_1", "run_command", '{"command": "echo hello"}')],
                },
                {"role": "assistant", "content": "执行结果：hello", "tool_calls": None},
            ]
        )
        agent = _make_agent(fake)
        self.assertEqual(agent.run("跑个命令"), "执行结果：hello")

        # 验证轨迹累积：第二次请求里包含 system + user + assistant(tool_calls) + tool
        # （滤掉状态栏 meta 消息后，剩下的是核心轨迹）
        second = _strip_status(fake.requests[1])
        roles = [m["role"] for m in second]
        self.assertEqual(roles, ["system", "user", "assistant", "tool"])
        # tool 消息带 tool_call_id（协议要求），内容里有执行结果
        tool_msg = second[-1]
        self.assertEqual(tool_msg["tool_call_id"], "call_1")
        self.assertIn("hello", tool_msg["content"])
        # assistant 消息原样回填了 tool_calls（模型能看到自己调了什么）
        self.assertIn("tool_calls", second[2])

    def test_bad_json_arguments_self_correct(self):
        """模型给的 arguments 不是合法 JSON → 回填错误让模型自纠，循环不崩。"""
        fake = _FakeClient(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_tc("call_1", "read_file", "不是json{")],
                },
                {"role": "assistant", "content": "我修正参数", "tool_calls": None},
            ]
        )
        agent = _make_agent(fake)
        self.assertEqual(agent.run("读文件"), "我修正参数")
        # tool 消息内容包含 JSON 解析错误提示
        tool_msg = _strip_status(fake.requests[1])[-1]
        self.assertIn("不是合法 JSON", tool_msg["content"])

    def test_max_iterations_break(self):
        """模型每次都调工具、永不出最终回答 → 熔断返回错误。"""
        always_tool = {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tc("c", "run_command", '{"command": "echo x"}')],
        }
        fake = _FakeClient([always_tool, always_tool, always_tool])
        agent = _make_agent(fake, max_iterations=3)
        result = agent.run("会死循环吗")
        self.assertIn("已达到最大迭代次数 3", result)
        # 只调了 3 次 LLM，没多烧
        self.assertEqual(len(fake.requests), 3)

    # ── M2 多轮对话（会话历史）──

    def test_multi_turn_history(self):
        """同一 agent 连续 run 两次：第二次的请求包含第一次的完整轨迹（多轮记忆）。"""
        fake = _FakeClient(
            [
                {"role": "assistant", "content": "第一轮回答", "tool_calls": None},
                {"role": "assistant", "content": "第二轮回答", "tool_calls": None},
            ]
        )
        agent = _make_agent(fake)
        self.assertEqual(agent.run("第一个问题"), "第一轮回答")
        self.assertEqual(agent.run("第二个问题"), "第二轮回答")

        # 第二次 run 的第一次请求 = system + 上一轮轨迹(user + assistant) + 新 user
        # （滤掉状态栏 meta 消息）
        first_of_second = _strip_status(fake.requests[1])
        roles = [m["role"] for m in first_of_second]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])
        self.assertEqual(first_of_second[1]["content"], "第一个问题")
        self.assertEqual(first_of_second[2]["content"], "第一轮回答")
        self.assertEqual(first_of_second[3]["content"], "第二个问题")

    def test_reset_clears_history(self):
        """reset() 后下一轮 run 回到干净轨迹（无历史）。"""
        fake = _FakeClient(
            [
                {"role": "assistant", "content": "第一轮回答", "tool_calls": None},
                {"role": "assistant", "content": "重置后回答", "tool_calls": None},
            ]
        )
        agent = _make_agent(fake)
        agent.run("问题一")
        agent.reset()
        agent.run("问题二")
        roles = [m["role"] for m in _strip_status(fake.requests[1])]
        self.assertEqual(roles, ["system", "user"])

    def test_break_saves_unfinished_summary(self):
        """熔断后写 [UNFINISHED] 摘要进 history（2026-08-24 拍板）。

        不存原始垃圾轨迹（一堆重复失败的工具调用），只存一条键值对摘要：
        任务目标 / 已完成 / 下一步 / 失败点。下一轮 run 能看到失败原因，
        从而能「接着办完」该任务。
        """
        always_tool = {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tc("c", "run_command", '{"command": "echo x"}')],
        }
        fake = _FakeClient([always_tool, always_tool, always_tool])
        clock = lambda: datetime(2026, 8, 25, 1, 2, 3)
        agent = _make_agent(fake, max_iterations=3, clock=clock)
        agent.run("会死循环吗")

        # 换一个 fake，模拟用户在同一会话里继续追问
        fake2 = _FakeClient(
            [{"role": "assistant", "content": "好", "tool_calls": None}]
        )
        agent._client = fake2  # type: ignore[assignment]  # 测试注入，鸭子类型
        agent.run("新问题")
        stripped = _strip_status(fake2.requests[0])
        # history 只存了一条：任务 user 消息 + [UNFINISHED] 摘要（没有垃圾轨迹）
        roles = [m["role"] for m in stripped]
        self.assertEqual(roles, ["system", "user", "user"])
        summary = stripped[1]["content"]
        self.assertIn(UNFINISHED_HEADER, summary)
        self.assertIn("时间: 2026-08-25 01:02:03", summary)
        self.assertIn("任务目标: 会死循环吗", summary)
        self.assertIn(
            "已完成: 工具调用 3 次（run_command×3），失败 0 次（无）", summary
        )
        self.assertIn("下一步: 输出最终回答（未完成）", summary)
        self.assertIn("失败点: 达到最大迭代次数 3（熔断）", summary)

    def test_usage_stats_accumulate(self):
        """带 usage 的响应：cache_stats 跨 run 累计，命中/未命中和正确。"""
        fake = _FakeClient(
            [
                {
                    "role": "assistant",
                    "content": "答一",
                    "tool_calls": None,
                    "usage": {
                        "prompt_tokens": 100,
                        "prompt_cache_hit_tokens": 60,
                        "prompt_cache_miss_tokens": 40,
                    },
                },
                {
                    "role": "assistant",
                    "content": "答二",
                    "tool_calls": None,
                    "usage": {
                        "prompt_tokens": 120,
                        "prompt_cache_hit_tokens": 100,
                        "prompt_cache_miss_tokens": 20,
                    },
                },
            ]
        )
        agent = _make_agent(fake)
        agent.run("一")
        agent.run("二")
        self.assertEqual(agent.cache_stats["hit"], 160)
        self.assertEqual(agent.cache_stats["miss"], 60)

    # ── M2 第 2 步：Agent 状态栏（时间戳 + 工具计数 + TODO）──

    def test_status_bar_injected_each_request(self):
        """每轮请求末尾注入状态栏（user-role meta 消息，书 Ch2 §2.6）。

        格子：时间戳（注入时钟，可复现）/ 工具计数 / 失败计数 /
        进度（含剩余预算）/ TODO。剩余轮数多时不出现「请收敛」警告。
        """
        fake = _FakeClient(
            [{"role": "assistant", "content": "好", "tool_calls": None}]
        )
        agent = _make_agent(fake, clock=lambda: datetime(2026, 8, 25, 1, 2, 3))
        agent.run("简单任务")
        status = fake.requests[0][-1]
        self.assertEqual(status["role"], "user")
        self.assertTrue(status["content"].startswith(STATUS_BAR_HEADER))
        self.assertIn("时间: 2026-08-25 01:02:03", status["content"])
        self.assertIn("工具: 已调用 0 次（无）", status["content"])
        self.assertIn("失败: 无", status["content"])
        self.assertIn("进度: 第 1/10 轮（剩余 9 轮）", status["content"])
        self.assertIn("TODO: 输出最终回答", status["content"])
        # 剩余 9 轮 > 阈值 3：不出现收敛警告
        self.assertNotIn("⚠", status["content"])

    def test_status_bar_updates_tool_counts(self):
        """工具循环中状态栏实时更新：第二轮的工具计数反映第一轮的调用。"""
        fake = _FakeClient(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_tc("call_1", "run_command", '{"command": "echo hello"}')],
                },
                {"role": "assistant", "content": "好", "tool_calls": None},
            ]
        )
        agent = _make_agent(fake)
        agent.run("跑个命令")
        status2 = fake.requests[1][-1]
        self.assertIn("工具: 已调用 1 次（run_command×1）", status2["content"])
        # 成功调用不记入失败
        self.assertIn("失败: 无", status2["content"])
        self.assertIn("进度: 第 2/10 轮（剩余 8 轮）", status2["content"])

    def test_status_bar_not_persisted_to_history(self):
        """状态栏是 meta 消息，不属于对话主体 —— 不进会话历史。

        否则上一轮的过期状态（旧时间、旧计数）会污染多轮记忆。
        多轮请求里 history 部分应该是「干干净净」的任务文本。
        """
        fake = _FakeClient(
            [
                {"role": "assistant", "content": "第一轮回答", "tool_calls": None},
                {"role": "assistant", "content": "第二轮回答", "tool_calls": None},
            ]
        )
        agent = _make_agent(fake)
        agent.run("第一个问题")
        agent.run("第二个问题")
        first_of_second = fake.requests[1]
        # history 里的任务消息不带状态栏
        self.assertEqual(first_of_second[1]["content"], "第一个问题")
        self.assertNotIn(STATUS_BAR_HEADER, first_of_second[1]["content"])
        # 状态栏只出现在本次请求的末尾
        self.assertIn(STATUS_BAR_HEADER, first_of_second[-1]["content"])

    def test_status_bar_counts_failures(self):
        """工具失败计入状态栏：幻觉调用不存在的工具 → 失败统计反映。

        模型容易忽略自己反复调同一个失败工具——状态栏替它数出来，
        错误结果（registry 返回 {"error": ...}）才算失败，成功调用不算。
        """
        fake = _FakeClient(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_tc("call_1", "no_such_tool", "{}")],
                },
                {"role": "assistant", "content": "好", "tool_calls": None},
            ]
        )
        agent = _make_agent(fake)
        agent.run("调个不存在的工具")
        status2 = fake.requests[1][-1]
        # 模型确实发起了调用（无论成败）→ 工具计数 +1
        self.assertIn("工具: 已调用 1 次（no_such_tool×1）", status2["content"])
        # 失败统计单独列出
        self.assertIn("失败: no_such_tool×1", status2["content"])

    def test_status_bar_warns_near_max(self):
        """剩余轮数 ≤ WARN_THRESHOLD 时状态栏追加「请收敛」警告。

        模型不知道 max_iterations 预算——剩余 3 轮（阈值边界）就该提醒它收尾。
        """
        tool_resp = {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tc("c", "run_command", '{"command": "echo x"}')],
        }
        fake = _FakeClient(
            [
                tool_resp,
                tool_resp,
                {"role": "assistant", "content": "好", "tool_calls": None},
            ]
        )
        agent = _make_agent(fake, max_iterations=4)
        agent.run("跑命令")
        # 第 1 轮请求：剩余 3 轮 = 阈值边界 → 警告出现
        status1 = fake.requests[0][-1]
        self.assertIn("进度: 第 1/4 轮（剩余 3 轮）", status1["content"])
        self.assertIn("⚠ 剩余 3 轮：请收敛，尽快输出最终回答", status1["content"])
        # 第 2 轮请求：剩余 2 轮 → 警告持续
        status2 = fake.requests[1][-1]
        self.assertIn("⚠ 剩余 2 轮：请收敛，尽快输出最终回答", status2["content"])


if __name__ == "__main__":
    unittest.main()