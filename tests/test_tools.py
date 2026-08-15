"""工具系统的行为测试。

覆盖三个知识点（对应书 Ch4）：

1. registry 的「名字 → 函数」映射：注册、查重、生成 schema、执行
2. 工具层错误处理：未知工具 / 参数畸形 / 工具内部异常
   —— 都返回错误字符串回填给模型，不抛异常打断循环
3. 三个内置工具的关键行为：
   - read_file 的 offset/limit + 截断显式可见
   - write_file 的覆盖语义 + 自动建目录
   - run_command 的危险命令黑名单 + 输出截断
"""

import json
import tempfile
import unittest
from pathlib import Path

from tools.base import Tool
from tools.builtin import default_registry
from tools.registry import ToolRegistry


class _DummyTool(Tool):
    """测试用工具：把参数原样返回，验证 registry 的参数保真性。"""

    name = "dummy"
    description = "测试工具"
    parameters = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
    }

    def run(self, x: int) -> str:
        return f"got:{x}"


class _ExplodingTool(Tool):
    """测试用工具：内部抛异常，验证 registry 不崩溃、回填错误。"""

    name = "explode"
    description = "总是失败的测试工具"
    parameters = {"type": "object", "properties": {}}

    def run(self, **kwargs) -> str:
        raise RuntimeError("boom")


class RegistryTest(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register(_DummyTool())

    def test_register_and_get(self):
        """注册后能按名字取到，取不到返回 None。"""
        self.assertIsInstance(self.registry.get("dummy"), _DummyTool)
        self.assertIsNone(self.registry.get("nope"))

    def test_duplicate_name_rejected(self):
        """重名注册直接报错——重名会让模型无法区分工具。"""
        with self.assertRaises(ValueError):
            self.registry.register(_DummyTool())

    def test_schemas_generate_openai_format(self):
        """schema() 生成 OpenAI function calling 格式（llm/client.py 直接用）。"""
        schemas = self.registry.schemas()
        self.assertEqual(len(schemas), 1)
        s = schemas[0]
        self.assertEqual(s["type"], "function")
        self.assertEqual(s["function"]["name"], "dummy")
        self.assertEqual(s["function"]["parameters"]["properties"]["x"]["type"], "integer")

    def test_execute_success(self):
        """参数按 schema 传对 → 正常执行，返回字符串。"""
        self.assertEqual(self.registry.execute("dummy", {"x": 42}), "got:42")

    def test_execute_unknown_tool_returns_error(self):
        """模型幻觉编造工具名 → 返回错误信息，不抛异常。"""
        result = self.registry.execute("nonexistent", {})
        self.assertIn("未知工具", result)

    def test_execute_missing_arg_returns_error(self):
        """参数畸形（缺 required 参数）→ TypeError 被捕获，返回错误信息。"""
        result = self.registry.execute("dummy", {})
        self.assertIn("参数错误", result)

    def test_execute_tool_exception_returns_error(self):
        """工具内部抛异常 → 捕获并回填错误，不打断循环。"""
        self.registry.register(_ExplodingTool())
        result = self.registry.execute("explode", {})
        self.assertIn("工具执行失败", result)
        self.assertIn("boom", result)


class ReadFileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "sample.txt"
        # 5 行内容，方便测 offset/limit
        self.path.write_text("line1\nline2\nline3\nline4\nline5\n", encoding="utf-8")
        tool = default_registry().get("read_file")
        assert tool is not None, "read_file 应该已注册"
        self.tool = tool

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_full(self):
        """默认读全部（5 行内），返回带行号和总行数。"""
        result = self.tool.run(str(self.path))
        self.assertIn("共 5 行", result)
        self.assertIn("1|line1", result)
        self.assertIn("5|line5", result)

    def test_read_with_offset_limit(self):
        """offset/limit 分段读取：第 2-3 行。"""
        result = self.tool.run(str(self.path), offset=2, limit=2)
        self.assertIn("第 2-3 行", result)
        self.assertIn("2|line2", result)
        self.assertIn("3|line3", result)
        self.assertNotIn("line4", result)

    def test_read_not_exist(self):
        """文件不存在 → 返回错误信息，不抛异常。"""
        result = self.tool.run(str(self.path) + ".missing")
        self.assertIn("文件不存在", result)

    def test_read_directory(self):
        """传目录 → 返回错误信息。"""
        result = self.tool.run(self.tmp.name)
        self.assertIn("目录", result)

    def test_read_offset_beyond_range(self):
        """offset 超出文件范围 → 明确报错并给出总行数。"""
        result = self.tool.run(str(self.path), offset=100)
        self.assertIn("超出文件范围", result)
        self.assertIn("total_lines", result)


class WriteFileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tool = default_registry().get("write_file")
        assert tool is not None, "write_file 应该已注册"
        self.tool = tool

    def tearDown(self):
        self.tmp.cleanup()

    def test_write_creates_file(self):
        """写新文件 → 返回 ok，文件内容原样。"""
        target = Path(self.tmp.name) / "out.txt"
        result = self.tool.run(str(target), "hello\nworld")
        self.assertIn('"ok": true', result)
        self.assertEqual(target.read_text(encoding="utf-8"), "hello\nworld")

    def test_write_overwrites_existing(self):
        """覆盖已有文件 → 旧内容被完全替换（覆盖语义明确）。"""
        target = Path(self.tmp.name) / "out.txt"
        target.write_text("old content", encoding="utf-8")
        self.tool.run(str(target), "new content")
        self.assertEqual(target.read_text(encoding="utf-8"), "new content")

    def test_write_creates_parent_dirs(self):
        """目标路径父目录不存在 → 自动创建（显式行为，不是静默注入）。"""
        target = Path(self.tmp.name) / "a" / "b" / "out.txt"
        result = self.tool.run(str(target), "x")
        self.assertIn('"ok": true', result)
        self.assertTrue(target.exists())


class RunCommandTest(unittest.TestCase):
    def setUp(self):
        tool = default_registry().get("run_command")
        assert tool is not None, "run_command 应该已注册"
        self.tool = tool

    def test_run_echo(self):
        """正常命令 → 返回 exit_code 和 stdout。"""
        result = self.tool.run("echo hello")
        data = json.loads(result)
        self.assertEqual(data["exit_code"], 0)
        self.assertEqual(data["stdout"].strip(), "hello")

    def test_run_nonzero_exit(self):
        """命令失败（非零退出码）→ exit_code 原样返回，不抛异常。"""
        result = self.tool.run("exit 3")
        data = json.loads(result)
        self.assertEqual(data["exit_code"], 3)

    def test_dangerous_command_rejected(self):
        """黑名单命令（rm）→ 拒绝执行并返回错误。"""
        result = self.tool.run("rm -rf /tmp/something")
        self.assertIn("危险命令", result)

    def test_empty_command_rejected(self):
        """空命令 → 拒绝。"""
        result = self.tool.run("   ")
        self.assertIn("不能为空", result)


if __name__ == "__main__":
    unittest.main()
