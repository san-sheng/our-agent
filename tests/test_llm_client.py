"""LLM 客户端的错误处理行为测试。

现在 key 还没到位，不碰真实 API——用 mock 模拟 openai client，
专门验证「重试逻辑」这个知识点：

1. 可重试错误（429 限流）→ 自动重试，最后成功返回
2. 不可重试错误（401 key 无效）→ 立即抛出，不做无谓重试
3. 可重试错误一直失败 → 重试耗尽后抛 RuntimeError
4. 重试之间有退避等待（sleep 被调用）
"""

import unittest
from unittest import mock

import openai

from llm.client import LLMClient


def _fake_response(content: str = "hi"):
    """构造一个假的 chat completion 响应。"""
    resp = mock.MagicMock()
    resp.choices[0].message.content = content
    resp.choices[0].message.tool_calls = None
    return resp


def _rate_limit_error():
    """构造一个真实的 429 限流异常（openai 的 APIStatusError 子类）。"""
    return openai.RateLimitError(
        message="rate limited",
        response=mock.MagicMock(status_code=429),
        body=None,
    )


class LLMClientTest(unittest.TestCase):
    def setUp(self):
        self.client = LLMClient(api_key="test-key", max_retries=3)

    def test_retryable_then_success(self):
        """429 一次 → 自动重试 → 成功返回。"""
        with (
            mock.patch("llm.client.time.sleep") as sleep_mock,
            mock.patch.object(
                self.client._client.chat.completions,
                "create",
                side_effect=[_rate_limit_error(), _fake_response()],
            ),
        ):
            result = self.client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(result["content"], "hi")
        sleep_mock.assert_called_once()  # 确实走了退避分支

    def test_non_retryable_raises_immediately(self):
        """401（key 无效）→ 立即抛，不重试、不 sleep。"""
        err = openai.AuthenticationError(
            message="bad key",
            response=mock.MagicMock(status_code=401),
            body=None,
        )
        with (
            mock.patch("llm.client.time.sleep") as sleep_mock,
            mock.patch.object(
                self.client._client.chat.completions, "create", side_effect=err
            ),
        ):
            with self.assertRaises(openai.AuthenticationError):
                self.client.chat([{"role": "user", "content": "hi"}])
        sleep_mock.assert_not_called()

    def test_retries_exhausted_raises_runtime_error(self):
        """429 一直失败 → 重试耗尽 → RuntimeError。"""
        with (
            mock.patch("llm.client.time.sleep") as sleep_mock,
            mock.patch.object(
                self.client._client.chat.completions,
                "create",
                side_effect=_rate_limit_error(),
            ),
        ):
            with self.assertRaises(RuntimeError):
                self.client.chat([{"role": "user", "content": "hi"}])
        # max_retries=3 → 共尝试 3 次；最后一次失败不 sleep，所以 sleep 2 次
        self.assertEqual(sleep_mock.call_count, 2)

    def test_missing_key_raises_value_error(self):
        """config.toml 没 key（[llm] 段为空）→ 构造时抛 ValueError。"""
        with mock.patch("llm.client._llm_config", return_value={}):
            with self.assertRaises(ValueError) as ctx:
                LLMClient()
        msg = str(ctx.exception)
        self.assertIn("api_key", msg)
        self.assertIn("base_url", msg)
        self.assertIn("model", msg)

    def test_partial_config_raises(self):
        """配置缺 base_url → 明确报错列出缺失项（白板报错，不静默兜底）。"""
        with mock.patch(
            "llm.client._llm_config",
            return_value={"api_key": "cfg-key", "model": "custom-model"},
        ):
            with self.assertRaises(ValueError) as ctx:
                LLMClient()
        self.assertIn("base_url", str(ctx.exception))

    def test_key_from_config(self):
        """不传 api_key，但从 config 能读到 → 构造成功，用 config 的 key。"""
        with mock.patch(
            "llm.client._llm_config",
            return_value={
                "api_key": "cfg-key",
                "base_url": "https://api.example.com",
                "model": "custom-model",
            },
        ):
            client = LLMClient()
        self.assertEqual(client._client.api_key, "cfg-key")

    def test_model_base_url_from_config(self):
        """base_url/model 也优先从 config 读（换模型不动代码）。"""
        with mock.patch(
            "llm.client._llm_config",
            return_value={
                "api_key": "cfg-key",
                "base_url": "https://api.example.com",
                "model": "custom-model",
            },
        ):
            client = LLMClient()
        self.assertEqual(client.model, "custom-model")
        self.assertEqual(
            str(client._client.base_url).rstrip("/"), "https://api.example.com"
        )


if __name__ == "__main__":
    unittest.main()
