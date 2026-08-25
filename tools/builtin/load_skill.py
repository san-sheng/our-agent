"""load_skill —— 按需加载 Agent Skill（书 Ch2 §动态提示词与 Agent Skills）。

对应书「方式三（生产实现）」：元数据（name + description）常驻上下文，
模型判断任务需要某个技能时，通过本工具加载完整的 SKILL.md——
路由与执行分离，兼顾 KV Cache 与指令遵循。

为什么不用 read_file 直接读：
1. 模型不知道 skill 文件在哪——路径约定是内部细节，不暴露给模型
2. 专门的工具能返回「技能不存在」的标准错误（错误是模型的输入），
   并确认加载的是技能目录里的文件，而不是任意路径
3. 语义清晰：模型看到 load_skill 就知道这是「加载领域知识」的动作，
   加载结果也天然适用 M2 第 3 步的来源标记（<tool_result tool="load_skill">）

安全边界：只从 skills_dir 下按「目录名 / SKILL.md」找文件，
不接受任意路径——技能加载不是任意文件读取（M2 第 3 步的注入防御延伸：
skill 内容也是外部数据，模型加载它之前应视其为不可信内容）。
"""

from __future__ import annotations

import json
from pathlib import Path

from ..base import Tool

# 每个 Skill 的固定文件名（约定：一个技能 = 一个目录 + SKILL.md）
SKILL_FILE = "SKILL.md"


class LoadSkill(Tool):
    name = "load_skill"
    description = (
        "加载一个技能（Skill）的完整内容。"
        "当任务需要某个专业技能时，先看【可用技能】元数据列表，"
        "判断哪个技能适用，再用本工具加载它的完整 SKILL.md。"
        "技能名必须是【可用技能】里列出的名字。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "要加载的技能名，例如 notes-writer",
            },
        },
        "required": ["skill_name"],
    }

    def __init__(self, skills_dir: Path | None = None) -> None:
        """skills_dir 可注入（测试用临时目录）；默认为项目根目录下的 skills/。"""
        if skills_dir is None:
            skills_dir = Path(__file__).resolve().parent.parent.parent / "skills"
        self._skills_dir = Path(skills_dir)

    def _skill_path(self, name: str) -> Path | None:
        """按技能名找 SKILL.md 路径（目录名 = 技能名），找不到返回 None。"""
        p = self._skills_dir / name / SKILL_FILE
        return p if p.is_file() else None

    def run(self, skill_name: str) -> str:
        p = self._skill_path(skill_name)
        if p is None:
            # 模型幻觉编造了不存在的技能名：错误让模型重新选择
            # （错误是模型的输入，与 registry 的错误协议同构）
            return json.dumps(
                {
                    "error": f"未知技能: {skill_name}",
                    "hint": "参考【可用技能】元数据里列出的技能名",
                },
                ensure_ascii=False,
            )
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return json.dumps({"error": f"读取技能失败: {exc}"}, ensure_ascii=False)
        return f"=== 技能 {skill_name} 完整内容 ===\n{content}"