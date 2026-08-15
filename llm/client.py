"""LLM 客户端 —— Agent 的「大脑」接口。

这个模块只做一件事：把消息列表发给模型，拿回回复。
它不关心 Agent 怎么循环、怎么组织上下文（那是 agent/loop.py 的事），
只负责和模型 API 对话，并处理好「对话可能失败」这件事。

为什么要单独一个模块：
- 以后换模型（DeepSeek → 别的）只改这一个文件
- 错误处理集中在这里，循环层不用关心网络细节

为什么用 openai SDK 而不是自己写 HTTP：
- DeepSeek 是 OpenAI 兼容协议，直接换 base_url 就能用
- 把精力留给 Agent 逻辑（上下文、循环、工具），而不是 HTTP 管道
- 书里配套代码也是这个思路
"""

from __future__ import annotations

import random
import time
import tomllib
from pathlib import Path
from typing import Any

import openai

# 配置文件：项目根目录 config.toml（已加入 .gitignore，不进 git）
# 所有可配置项（api_key / base_url / model）都在配置文件里——
# 缺哪项就在 __init__ 里明确报错，不设静默兜底（避免新人被隐藏默认值误导）
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"

# 值得重试的 HTTP 状态码：
# 429 限流、5xx 服务端临时故障 —— 重试有意义（服务端可能恢复）
# 401/400/404 等 —— 重试没意义（请求本身有问题，重试一万次也一样）
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _load_config() -> dict[str, Any]:
    """读 config.toml。文件不存在 → 返回空 dict（调用方决定怎么处理缺 key）。"""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def _llm_config() -> dict[str, Any]:
    """返回配置文件的 [llm] 段（api_key / base_url / model），没有就空 dict。"""
    return _load_config().get("llm", {})


def _is_retryable(exc: Exception) -> bool:
    """判断一个异常是否值得重试。

    书里的错误恢复分层（API 层）：限流、超时、连接中断 → 静默重试。
    注意「静默」——重试不应该打断流程、也不应该让用户看到。

    但有个工程细节必须区分：不是所有错误都值得重试。
    - 超时 / 连接断 / 限流 / 5xx：服务端或网络临时问题，等一会儿可能就好了
    - 401（key 无效）/ 400（参数错）：请求本身错了，重试只会浪费时间
    """
    if isinstance(exc, openai.APITimeoutError):
        return True
    if isinstance(exc, openai.APIConnectionError):
        return True
    if isinstance(exc, openai.APIStatusError):
        # APIStatusError 覆盖所有 4xx/5xx（401、429、500 都是它的子类）
        return exc.status_code in RETRYABLE_STATUS_CODES
    return False


class LLMClient:
    """封装模型调用的最小客户端。

    设计要点：
    1. 可配置项（key/base_url/model）全部由外部传入，代码里不做死
    2. 每次调用都是无状态的：调用方负责维护消息列表，client 只做一次往返
       —— 对应书里说的：每次 API 调用都是无状态的，
          所有模型需要的信息必须在请求的消息列表中完整提供
    3. 重试策略：指数退避 + 抖动（见 chat() 里的注释）
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
    ) -> None:
        # 优先级：显式传参 > config.toml。没有兜底——缺哪项就明确报错，
        # 让配置错误在启动时立刻暴露（fail fast），而不是带病运行
        cfg = _llm_config()
        api_key = api_key or cfg.get("api_key")
        base_url = base_url or cfg.get("base_url")
        model = model or cfg.get("model")

        missing = [
            name
            for name, val in (
                ("api_key", api_key),
                ("base_url", base_url),
                ("model", model),
            )
            if not val
        ]
        if missing:
            raise ValueError(
                f"缺少配置: {', '.join(missing)}。"
                f"请在 {CONFIG_PATH} 的 [llm] 段补齐，或通过参数显式传入。"
            )

        # 注意：openai SDK 自己也带重试（默认 2 次），
        # 我们把它关掉（max_retries=0），重试逻辑完全自己控制——
        # 学习项目，每一步都要看得见摸得着
        self._client = openai.OpenAI(
            api_key=api_key, base_url=base_url, max_retries=0
        )
        self.model = model
        self.max_retries = max_retries

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """发一次 chat completion，返回 assistant 消息（dict 形式）。

        messages: 消息列表（system / user / assistant / tool）
        tools:    工具定义列表；None 表示不启用工具调用
        返回: {"role": "assistant", "content": str | None, "tool_calls": [...] | None}

        tool_calls 保留 SDK 原始对象（ChatCompletionMessageToolCall），
        由调用方（agent/loop.py）决定怎么解析——LLM 客户端不关心循环逻辑。
        """
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
                if tools:
                    kwargs["tools"] = tools
                resp = self._client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                return {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": msg.tool_calls,
                }
            except Exception as exc:
                if not _is_retryable(exc):
                    raise  # 不值得重试的错误直接抛，不浪费时间
                last_error = exc
                if attempt == self.max_retries - 1:
                    break  # 最后一次尝试也失败 → 跳出循环，抛给上层
                # 指数退避 + 抖动：
                #   第 1 次失败等 ~1 秒，第 2 次 ~2 秒，第 3 次 ~4 秒……
                #   抖动（random.random()）让多个并发请求不会同时重试，
                #   否则退避会失效（大家同时打回去，服务端又被压垮）
                delay = (2**attempt) + random.random()
                time.sleep(delay)
        raise RuntimeError(
            f"LLM 调用失败（已重试 {self.max_retries} 次）: {last_error}"
        ) from last_error
