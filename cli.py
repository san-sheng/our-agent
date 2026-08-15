"""cli.py —— our-agent 的命令行入口（REPL）。

两种用法：
1. 交互模式：直接 `python cli.py`，进入 REPL，每行输入一个任务
2. 单次模式：`python cli.py "读 DESIGN.md，统计有多少行"`，执行一次退出
   （方便脚本调用和 demo 验收）

为什么是 REPL 而不是「一个任务跑完就退出」：
DESIGN.md §6.2 的 M1 范围——命令行 REPL。每轮输入 = 新任务（新轨迹），
多轮对话历史是 M2 上下文工程的事，M1 刻意不做。
"""

from __future__ import annotations

import sys

from agent.loop import Agent
from llm.client import LLMClient
from tools.builtin import default_registry


def _build_agent() -> Agent:
    """从 config.toml 读配置创建 Agent。

    配置缺失（api_key/base_url/model）时 LLMClient 构造会抛 ValueError——
    不在这里捕获，让错误在启动时立刻暴露（fail fast，DESIGN.md §6.2）。
    """
    client = LLMClient()  # 显式传参 > config.toml，缺项报错
    return Agent(client, default_registry())


def run_once(agent: Agent, task: str) -> str:
    """执行单个任务并返回回答（供单次模式和 REPL 共用）。"""
    return agent.run(task)


def repl() -> None:
    """交互模式：循环读输入，每行一个任务。"""
    agent = _build_agent()
    print("our-agent M1 · 输入任务，空行跳过，Ctrl-C / exit / 退出 结束")
    while True:
        try:
            task = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break
        if not task:
            continue
        if task in {"exit", "quit", "退出"}:
            print("再见")
            break
        print(run_once(agent, task))


def main() -> int:
    if len(sys.argv) > 1:
        # 单次模式：任务从命令行参数来，执行完退出
        task = " ".join(sys.argv[1:])
        print(run_once(_build_agent(), task))
        return 0
    repl()
    return 0


if __name__ == "__main__":
    sys.exit(main())
