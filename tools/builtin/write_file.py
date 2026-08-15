"""write_file —— 执行工具：创建或完全重写文件。

设计取舍（书 Ch4 执行工具：能力开放 vs 安全约束）：

1. M1 刻意最小化
   不检查「是否在项目目录内」、不做沙盒——那是 M4（Coding Agent 深化）
   的事。M1 只要它能写、能验证写成功，让 ReAct 闭环先跑起来。
   安全边界在 DESIGN.md §7 有约定：涉及删除、写系统目录的操作先问少爷，
   这个「先问」在 M1 由 CLI 层兜底（agent 只在本项目目录里活动）。

2. 覆盖语义明确
   write_file 语义 = 创建或完全重写。模型传什么就写什么——
   不静默追加、不静默转换内容（书 Ch4 参数保真性）。

3. 返回值自包含
   成功返回写入的路径和行数；失败返回错误原因——
   模型需要看到这些信息来决定下一步。
"""

from __future__ import annotations

from pathlib import Path

from ..base import Tool


class WriteFile(Tool):
    name = "write_file"
    description = (
        "创建新文件或完全重写已有文件。当需要写入代码、配置、文本内容时使用。"
        "注意：这会覆盖目标文件的全部现有内容，不是追加也不是局部修改。"
        "只能写文本内容；写文件前若不确定目标内容，先用 read_file 查看。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要写入的文件路径（绝对路径或相对路径），例如 /home/user/project/output.txt",
            },
            "content": {
                "type": "string",
                "description": "完整的文件内容，将原样写入（覆盖旧内容）",
            },
        },
        "required": ["path", "content"],
    }

    def run(self, path: str, content: str) -> str:
        p = Path(path)
        # 自动创建父目录——模型写嵌套路径时不用先手动 mkdir
        # （这是显式行为，写在注释里；不是静默参数注入）
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return f'{{"error": "创建目录失败: {exc}"}}'

        try:
            p.write_text(content, encoding="utf-8")
        except OSError as exc:
            return f'{{"error": "写入失败: {exc}"}}'

        # 返回写入结果的真实数字：让模型能验证（写了几行、在哪）
        # 空内容 splitlines() 返回 []，行数就是 0（旧写法 if content else 0
        # 会让 lines=0，随后 len(0) 抛 TypeError）
        lines = content.splitlines()
        return (
            f'{{"ok": true, "path": "{path}", '
            f'"lines_written": {len(lines)}}}'
        )
