"""run_command —— 执行工具：在 shell 里运行命令。

设计对应书 Ch4 执行工具的安全约束，但刻意保持 M1 的最小形态：

1. 危险命令黑名单（第一层安全：输入验证）
   书 Ch4 原文：黑名单只是最基础的防护层，不应作为唯一手段——
   攻击者可以通过变形命令绕过字符串匹配。M1 先做这一层，
   M4（Coding Agent 深化）再上语义解析和沙盒。

2. 快速失败
   发现危险命令立即拒绝，返回错误信息给模型——不执行、不尝试修正。
   模型看到错误后会调整命令，而不是执行了危险操作。

3. timeout 熔断
   命令可能挂起（如交互式程序），必须给超时上限。
   超时后杀掉进程并返回错误——否则 ReAct 循环会卡死在这里。

4. 输出截断
   命令输出可能巨大（几十 MB），全量回填会撑爆上下文。
   截断必须显式可见：注明截断了多少字符（书 Ch4 感知工具的
   「静默截断是危险的」同样适用于命令输出）。
"""

from __future__ import annotations

import json
import subprocess

from ..base import Tool

# 危险命令黑名单：匹配命令的第一个单词（不含参数）
# 覆盖删除、格式化、关机、权限、挖矿、fork 炸弹等危险操作
DANGEROUS_COMMANDS = {
    "rm", "rmdir", "dd", "mkfs", "mkfs.ext4", "fdisk", "parted",
    "shutdown", "reboot", "halt", "poweroff",
    "chmod", "chown", "chattr",
    "mv",  # mv 会覆盖目标文件，M1 保守起见也拦（M4 再细化）
    "kill", "killall", "pkill",  # 杀进程可能影响其他服务
    "curl", "wget",  # M1 禁网络下载，避免下载执行任意代码
    ":(){:|:&};:",  # fork 炸弹
}

# 输出截断上限：超过就截断并注明
MAX_OUTPUT_CHARS = 8000

# 命令超时（秒）
TIMEOUT_SECONDS = 30


class RunCommand(Tool):
    name = "run_command"
    description = (
        "在 shell 中执行命令。当需要运行程序、查看文件系统、统计信息、"
        "执行 git 操作等需要终端能力时使用。"
        "注意：只能执行非交互式命令（不会等待用户输入的命令）；"
        "禁止执行删除（rm）、格式化（mkfs）、关机（shutdown）、权限修改（chmod/chown）、"
        "网络下载（curl/wget）等危险操作；命令输出超过 8000 字符会被截断。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的 shell 命令，例如 'ls -la' 或 'python3 script.py'",
            },
        },
        "required": ["command"],
    }

    def run(self, command: str) -> str:
        # 空命令直接拒绝
        if not command or not command.strip():
            return '{"error": "命令不能为空"}'

        # 提取第一个单词作为命令名，查黑名单
        first_word = command.strip().split()[0]
        if first_word in DANGEROUS_COMMANDS:
            return (
                f'{{"error": "危险命令被拒绝: {first_word}", '
                f'"hint": "请改用安全的只读命令"}}'
            )

        try:
            # shell=True：命令以字符串传给 shell 执行（模型生成自然语言命令
            # 的形态）。这是执行工具的标准做法，代价是注入风险——
            # M1 用黑名单兜底，M4 深化。
            # capture_output=True：捕获 stdout/stderr，不打印到终端
            # text=True：以文本模式解码输出（而非 bytes）
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return (
                '{"error": "命令超时（>' + str(TIMEOUT_SECONDS) + 's 被终止）", '
                '"command": ' + json.dumps(command) + "}"
            )
        except OSError as exc:
            return json.dumps({"error": f"执行失败: {exc}"}, ensure_ascii=False)

        # 输出截断：截断时显式注明，不让模型误以为看到完整输出
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        truncated = False
        if len(stdout) > MAX_OUTPUT_CHARS:
            stdout = stdout[:MAX_OUTPUT_CHARS] + "\n...[输出已截断]..."
            truncated = True
        if len(stderr) > MAX_OUTPUT_CHARS:
            stderr = stderr[:MAX_OUTPUT_CHARS] + "\n...[输出已截断]..."
            truncated = True

        # 注意：必须用 json.dumps 序列化 stdout/stderr——它们是任意文本，
        # 可能包含换行、引号、非 ASCII 字符；手工拼 JSON 会制造非法 JSON
        return json.dumps(
            {
                "exit_code": result.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "truncated": truncated,
            },
            ensure_ascii=False,
        )
