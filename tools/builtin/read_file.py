"""read_file —— 感知工具：读取文本文件内容。

设计对应书 Ch4 感知工具的三个原则：

1. offset/limit 按需读取
   大文件不能一次性全读进上下文（浪费窗口、噪声淹没关键信息），
   支持指定行号范围分段读取。

2. 截断显式可见
   超过 limit 时返回信息里必须注明「已显示第 X-Y 行，共 N 行」。
   静默截断是危险的——模型会误以为自己看到了全部内容，
   基于不完整的信息做错误判断（书 Ch4 原文）。

3. 返回值带行号
   模型后续操作（M4 的编辑工具）需要精确行号定位，
   所以内容按「行号|内容」格式返回。
"""

from __future__ import annotations

from pathlib import Path

from ..base import Tool


class ReadFile(Tool):
    name = "read_file"
    description = (
        "读取文本文件内容。当需要查看代码、配置、日志或文档时使用。"
        "只能读取文本文件，不能读取图片等二进制文件。"
        "大文件请配合 offset/limit 分段读取，不要一次读整个大文件。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径（绝对路径或相对路径），例如 /home/user/project/main.py",
            },
            "offset": {
                "type": "integer",
                "description": "起始行号（从 1 开始），默认 1",
                "default": 1,
            },
            "limit": {
                "type": "integer",
                "description": "最多读取的行数，默认 500",
                "default": 500,
            },
        },
        "required": ["path"],
    }

    def run(self, path: str, offset: int = 1, limit: int = 500) -> str:
        # 参数先做基本校验，非法输入快速失败——不尝试「智能修正」
        # （书 Ch4 执行工具第一层安全：输入验证，关键是快速失败）
        if offset < 1:
            return '{"error": "offset 必须 >= 1"}'
        if limit < 1:
            return '{"error": "limit 必须 >= 1"}'

        p = Path(path)
        if not p.exists():
            return f'{{"error": "文件不存在: {path}"}}'
        if p.is_dir():
            return f'{{"error": "这是一个目录，不是文件: {path}"}}'

        try:
            # errors="replace"：遇到无法解码的字节用替换符代替——
            # 保证二进制文件不会让整个工具崩溃，而是返回可读的错误信号
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return f'{{"error": "读取失败: {exc}"}}'

        total = len(lines)
        # 切片用 offset-1（行号从 1 开始，索引从 0 开始）
        start = offset - 1
        end = min(start + limit, total)
        if start >= total:
            return (
                f'{{"error": "offset 超出文件范围", '
                f'"total_lines": {total}}}'
            )

        # 截断显式可见：无论是否截断，都告诉模型总行数和本次范围
        body = "\n".join(
            f"{i + 1}|{line}" for i, line in enumerate(lines[start:end], start=start)
        )
        return (
            f"=== {path} (第 {start + 1}-{end} 行 / 共 {total} 行) ===\n"
            f"{body}"
        )
