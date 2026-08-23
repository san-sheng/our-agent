"""agent/loop.py 的行为测试。

核心思路：**不真调 API**。用一个「假 LLM 客户端」预置好响应序列，
驱动 ReAct 循环，验证循环逻辑本身（书 Ch1 的轨迹累积、熔断、错误自纠）。

为什么能这样测：Agent 只依赖 client.chat(messages, tools) 的返回格式
（{"role", "content", "tool_calls"}），不关心背后是不是真模型——
把「模型行为」参数化，测的是框架（Harness），这正是 Harness 工程的思想：
模型之外的部分可以且应该被确定性测试。
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from agent.loop import Agent
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


def _make_agent(fake: _FakeClient, max_iterations: int = 10) -> Agent:
    # 类型上 client 是 LLMClient，但 FakeClient 实现了相同的 chat 接口——
    # 鸭子类型，测试注入用
    return Agent(fake, default_registry(), max_iterations=max_iterations, verbose=False)  # type: ignore[arg-type]


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
        second = fake.requests[1]
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
        tool_msg = fake.requests[1][-1]
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
        first_of_second = fake.requests[1]
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
        roles = [m["role"] for m in fake.requests[1]]
        self.assertEqual(roles, ["system", "user"])

    def test_break_does_not_save_history(self):
        """熔断的任务不写入历史：下一次 run 的请求不含熔断轮的轨迹。"""
        always_tool = {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tc("c", "run_command", '{"command": "echo x"}')],
        }
        fake = _FakeClient([always_tool, always_tool, always_tool])
        agent = _make_agent(fake, max_iterations=3)
        agent.run("会死循环吗")
        # 熔断后 history 应为空：换一个 fake 再跑，请求只有 system + user
        fake2 = _FakeClient(
            [{"role": "assistant", "content": "好", "tool_calls": None}]
        )
        agent._client = fake2  # type: ignore[assignment]  # 测试注入，鸭子类型
        agent.run("新问题")
        roles = [m["role"] for m in fake2.requests[0]]
        self.assertEqual(roles, ["system", "user"])

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


if __name__ == "__main__":
    unittest.main()
