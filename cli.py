"""cli.py —— our-agent 的命令行入口（多轮会话 REPL）。

两种用法：
1. 交互模式：直接 `python cli.py`，进入 REPL——多轮会话（M2），
   每条输入追加到同一会话历史，模型记得上一轮做了什么
2. 单次模式：`python cli.py "读 DESIGN.md，统计有多少行"`，执行一次退出
   （方便脚本调用和 demo 验收）

M2 多轮对话：Agent 持有会话历史（agent/loop.py 的 self._history），
REPL 复用一个 agent 实例，输入「新会话」清空记忆回到单任务模式。
system 与工具定义作为静态前缀每轮原样发送（KV Cache 友好），
历史消息只追加不改写——这是 M2 缓存机制（Prompt Cache 命中率）的地基。
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
    """交互模式：循环读输入，每条输入追加到同一会话（M2 多轮对话）。"""
    agent = _build_agent()
    print(
        "our-agent M2 · 多轮会话（输入「新会话」清空记忆，"
        "Ctrl-C / exit / 退出 结束）"
    )
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
        if task in {"reset", "新会话", "new"}:
            agent.reset()
            print("[会话已重置]")
            continue
        print(run_once(agent, task))
        # 每轮结束后显示累计缓存命中情况（M2 度量尺子：
        # 命中率接近 100% = 前缀稳定；掉下来 = 哪里破坏了缓存前缀）
        hit, miss = agent.cache_stats["hit"], agent.cache_stats["miss"]
        total = hit + miss
        rate = f"{hit / total:.0%}" if total else "n/a"
        print(f"[缓存] 累计命中 {hit} / 未命中 {miss}（命中率 {rate}）")


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
