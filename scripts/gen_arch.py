"""gen_arch.py —— 扫描项目源码，自动生成架构图 ARCH.html。

为什么存在（对应少爷的需求「随着代码增加同时能更新架构图」）：
手绘架构图最大的问题是会过时——代码每加一个工具、每改一层结构，
图就落后一步。这个脚本用 AST 直接读源码，把「图上有什么」变成
「代码里有什么」的投影：跑一次脚本，图就是当前代码的真实状态。
以后每完成一个里程碑，跑 `python scripts/gen_arch.py` 即可。

设计取舍：
1. 纯静态扫描，不 import 任何项目模块
   —— 项目代码如果有运行时错误（缺依赖、坏配置），生成器仍然能工作。
   生成器只依赖标准库（ast/json/re/datetime/pathlib）。
2. 单文件自包含 HTML（无 CDN、无外部字体、无 JS 依赖）
   —— 双击就能打开，分享给别人也只看一个文件。
3. 已实现 vs 计划中（planned）分开渲染
   —— agent/loop.py、cli.py 还没写，图上画成虚线框；
   写完之后再跑脚本，虚线自动变实线。这本身就是「代码增长的可见进度」。
"""

from __future__ import annotations

import ast
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT_ROOT / "ARCH.html"
DESIGN_MD = PROJECT_ROOT / "DESIGN.md"

# 架构图的主体扫描范围：这几层是 Agent 的核心组件
SCAN_DIRS = ["agent", "llm", "tools"]
# 项目根的关键文件也要扫（cli.py 是入口，架构图里有它的节点）
ROOT_FILES = ["cli.py"]
# 排除：生成器自身、测试、笔记
EXCLUDE = {"scripts", "tests", "NOTES", "notes", ".venv"}

# ---------------------------------------------------------------- AST 提取 --

def _literal(node: ast.expr | None) -> Any:
    """把 AST 字面量节点安全转回 Python 值（支持 str/int/bool/list/dict/None）。

    为什么自己写而不是 ast.literal_eval：literal_eval 接收的是源码字符串，
    这里手里只有 AST 节点；且工具代码里的 parameters 都是纯字面量，
    这个受限的求值器足够用，不需要冒险 eval。
    """
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_literal(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        return {_literal(k): _literal(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        val = _literal(node.operand)
        return -val if isinstance(val, (int, float)) else None
    return None


def _class_attrs(cls: ast.ClassDef) -> dict[str, Any]:
    """提取类体里的类属性赋值（name / description / parameters 等）。"""
    attrs: dict[str, Any] = {}
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id not in attrs:
                    attrs[target.id] = _literal(stmt.value)
    return attrs


def _local_imports(tree: ast.Module) -> list[str]:
    """提取对本项目内部模块的引用（用于展示模块依赖关系）。

    过滤掉标准库和第三方：只看以 agent/llm/tools 开头或相对导入（..）的。
    """
    deps: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in {"agent", "llm", "tools"}:
                    deps.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # 相对导入（..base → tools.base）
                base = node.module or ""
                deps.append(f"..{base}" if base else "..")
            elif node.module and node.module.split(".")[0] in {"agent", "llm", "tools"}:
                deps.append(node.module)
    # 去重保序
    return list(dict.fromkeys(deps))


def _scan_py_file(path: Path) -> dict[str, Any] | None:
    """扫描单个 .py 文件，返回模块信息。解析失败返回 None（降级，不崩溃）。"""
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return None
    lines = len(text.splitlines())
    classes: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node) or ""
            classes.append(
                {
                    "name": node.name,
                    "line": node.lineno,
                    "doc": doc.splitlines()[0] if doc else "",
                }
            )
    doc = ast.get_docstring(tree) or ""
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "lines": lines,
        "classes": classes,
        "doc": doc.splitlines()[0] if doc else "",
        "imports": _local_imports(tree),
        "status": "done",
    }


def _collect_tools() -> list[dict[str, Any]]:
    """扫描 tools/builtin/ 下的全部 Tool 子类。

    不依赖 __init__.py 的注册列表：直接扫目录，新加的工具文件
    天然会被发现（这正是「随代码更新」的关键路径）。
    """
    tools: list[dict[str, Any]] = []
    builtin_dir = PROJECT_ROOT / "tools" / "builtin"
    for path in sorted(builtin_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            # 只认继承 Tool 的类（base.py 里 Tool 自身不算）
            if not any(
                isinstance(b, ast.Name) and b.id == "Tool" for b in node.bases
            ):
                continue
            attrs = _class_attrs(node)
            doc = ast.get_docstring(node) or ""
            tools.append(
                {
                    "name": attrs.get("name", node.name),
                    "class": node.name,
                    "file": str(path.relative_to(PROJECT_ROOT)),
                    "line": node.lineno,
                    "description": attrs.get("description", ""),
                    "parameters": attrs.get("parameters", {}),
                    "doc": doc.splitlines()[0] if doc else "",
                }
            )
    return tools


def _collect_register_order() -> list[str]:
    """从 tools/builtin/__init__.py 提取工具注册顺序（模型看到的顺序）。"""
    path = PROJECT_ROOT / "tools" / "builtin" / "__init__.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return []
    order: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == "register":
                for arg in call.args:
                    if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
                        order.append(arg.func.id)
    return order


def _collect_modules() -> list[dict[str, Any]]:
    """扫描核心目录下所有 .py，构建模块清单。"""
    modules: list[dict[str, Any]] = []
    for sub in SCAN_DIRS:
        base = PROJECT_ROOT / sub
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if any(part in EXCLUDE for part in path.parts):
                continue
            info = _scan_py_file(path)
            if info:
                modules.append(info)
    # 项目根的关键文件（cli.py 等）
    for fname in ROOT_FILES:
        path = PROJECT_ROOT / fname
        if not path.exists():
            continue
        info = _scan_py_file(path)
        if info:
            modules.append(info)
    return modules


def _collect_milestones() -> list[dict[str, Any]]:
    """从 DESIGN.md 提取里程碑进度 checklist。"""
    if not DESIGN_MD.exists():
        return []
    text = DESIGN_MD.read_text(encoding="utf-8")
    milestones: list[dict[str, Any]] = []
    for line in text.splitlines():
        m = re.match(r"^\s*-\s*\[( |x)\]\s+(.+)$", line)
        if m:
            milestones.append({"done": m.group(1) == "x", "text": m.group(2).strip()})
    return milestones


def _collect_flow() -> list[str]:
    """从 DESIGN.md 提取数据流说明（回退到内置描述）。"""
    if not DESIGN_MD.exists():
        return []
    text = DESIGN_MD.read_text(encoding="utf-8")
    m = re.search(r"### 4\.2 数据流（一次任务怎么走）(.*?)(?:\n---|\n## )", text, re.S)
    if not m:
        return []
    steps: list[str] = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if line.startswith("↓"):
            continue
        if line.startswith("[") or line.startswith("你输入") or line.startswith("回到"):
            # 去掉 markdown 链接语法 [x](y) 里的链接，保留文字
            clean = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
            steps.append(clean)
    return steps


# ---------------------------------------------------------------- SVG 渲染 --

def _node(
    x: float, y: float, w: float, h: float, title: str, sub: str, status: str
) -> str:
    """渲染一个 SVG 节点。status: done（实线）/ planned（虚线灰）。"""
    cls = "node" if status == "done" else "node planned"
    rect = (
        f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" rx="8"/>'
    )
    title_y = y + h / 2 - (8 if sub else 0)
    text = (
        f'<text class="th" x="{x + w / 2:.0f}" y="{title_y:.0f}" '
        f'text-anchor="middle" dominant-baseline="central">{title}</text>'
    )
    if sub:
        text += (
            f'<text class="ts" x="{x + w / 2:.0f}" y="{y + h / 2 + 10:.0f}" '
            f'text-anchor="middle" dominant-baseline="central">{sub}</text>'
        )
    return f'<g class="{cls}">{rect}{text}</g>'


def _edge(
    x1: float, y1: float, x2: float, y2: float, cls: str = "edge", label: str = ""
) -> str:
    """渲染一条带箭头的连线。"""
    out = (
        f'<line class="{cls}" x1="{x1:.0f}" y1="{y1:.0f}" '
        f'x2="{x2:.0f}" y2="{y2:.0f}"/>'
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        out += (
            f'<text class="lbl" x="{mx:.0f}" y="{my - 6:.0f}" '
            f'text-anchor="middle">{label}</text>'
        )
    return out


def _render_svg(modules: list[dict[str, Any]]) -> str:
    """架构总览：分层节点 + 数据流箭头。

    节点状态 = 文件是否真实存在：
    - agent/loop.py、cli.py 还没写 → planned（虚线）
    - llm/client.py、tools/* 已写 → done（实线）
    代码增长时，planned 会逐个变 done，SVG 自动反映。
    """
    done_files = {m["path"] for m in modules}

    W = 960
    nodes = [
        _node(
            390, 60, 180, 40, "cli.py", "命令行入口",
            "planned" if "cli.py" not in done_files else "done",
        ),
        _node(
            390, 150, 180, 56, "agent/loop.py", "ReAct 主循环",
            "planned" if "agent/loop.py" not in done_files else "done",
        ),
        _node(
            120, 280, 220, 56, "llm/client.py", "LLMClient · 大脑",
            "planned" if "llm/client.py" not in done_files else "done",
        ),
        _node(
            620, 280, 220, 56, "tools/registry.py", "ToolRegistry · 手脚",
            "planned" if "tools/registry.py" not in done_files else "done",
        ),
    ]
    # 底部工具行：从注册顺序生成（没有注册信息时按字母序兜底）
    tools = _collect_tools()
    if tools:
        n = len(tools)
        gap = 30
        total_w = n * 180 + (n - 1) * gap
        x0 = (W - total_w) / 2
        for i, t in enumerate(tools):
            cx = x0 + i * (180 + gap)
            nodes.append(_node(cx, 400, 180, 56, t["name"], t["file"], "done"))
    else:
        nodes.append(_node(390, 400, 180, 56, "(还没有工具)", "", "planned"))

    edges = [
        _edge(480, 45, 480, 60),          # 用户 → cli
        _edge(480, 100, 480, 150),        # cli → loop
        _edge(430, 206, 230, 280),        # loop 左下 → llm 顶
        _edge(530, 206, 730, 280),        # loop 右下 → registry 顶
    ]
    # registry 底边中点 → 各工具顶边中点
    if tools:
        n = len(tools)
        gap = 30
        total_w = n * 180 + (n - 1) * gap
        x0 = (W - total_w) / 2
        for i in range(n):
            cx = x0 + i * (180 + gap)
            edges.append(_edge(730, 336, cx + 90, 400))
    # 结果回填（工具行上方绕回 loop 右侧）
    edges.append(
        '<path class="edge back" d="M 760 360 C 900 360, 900 178, 570 178" />'
        '<text class="lbl" x="880" y="260" text-anchor="middle">结果回填</text>'
    )

    svg = (
        f'<svg viewBox="0 0 {W} 500" role="img" aria-label="our-agent 架构总览">'
        "<defs>"
        '<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M2 1 L8 5 L2 9" fill="none" stroke="context-stroke" '
        'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
        "</marker></defs>"
        '<text class="lbl" x="480" y="24" text-anchor="middle">你输入任务</text>'
        + "".join(nodes)
        + "".join(edges)
        + "</svg>"
    )
    return svg


# ---------------------------------------------------------------- HTML 渲染 --

CSS = """
:root {
  --ivory:    #FAF9F5; --white: #FFFFFF; --slate: #141413;
  --clay:     #D97757; --olive: #788C5D; --rust: #B04A3F; --oat: #E3DACC;
  --gray-150: #F0EEE6; --gray-300: #D1CFC5; --gray-500: #87867F; --gray-700: #3D3D3A;
  --border: 1.5px solid var(--gray-300);
  --radius-panel: 12px; --radius-row: 8px; --radius-pill: 999px;
  --serif: ui-serif, Georgia, "Times New Roman", serif;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: var(--ivory); color: var(--gray-700); font-family: var(--sans);
       line-height: 1.6; -webkit-font-smoothing: antialiased; padding: 56px 24px 120px; }
.page { max-width: 1040px; margin: 0 auto; }
.eyebrow { font-family: var(--mono); font-size: 11px; letter-spacing: 0.08em;
           text-transform: uppercase; color: var(--gray-500); }
h1 { font-family: var(--serif); font-weight: 500; letter-spacing: -0.01em;
     font-size: 34px; color: var(--slate); margin: 6px 0 4px; }
h2 { font-family: var(--serif); font-weight: 500; letter-spacing: -0.01em;
     font-size: 22px; color: var(--slate); margin: 52px 0 14px; }
h3 { font-family: var(--mono); font-weight: 500; font-size: 13px; color: var(--slate); }
p.sub { color: var(--gray-500); font-size: 14px; margin-bottom: 8px; }
.card { background: var(--white); border: var(--border); border-radius: var(--radius-panel);
        padding: 20px; }
.grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
@media (max-width: 860px) { .grid3 { grid-template-columns: 1fr; } }
.badge { border-radius: 6px; padding: 1px 7px; font-family: var(--mono); font-size: 11px; }
.badge.done { background: rgba(120,140,93,0.18); color: var(--olive); }
.badge.planned { background: var(--gray-150); color: var(--gray-500); }
.pill { border-radius: var(--radius-pill); padding: 2px 10px; font-family: var(--mono);
        font-size: 11px; background: var(--oat); }
.callout { background: rgba(217,119,87,0.06); border-left: 3px solid var(--clay);
           border-radius: var(--radius-row); padding: 14px 16px; margin: 14px 0; }
.code { font-family: var(--mono); background: var(--slate); color: #E8E6DF;
        padding: 2px 8px; border-radius: 6px; font-size: 12.5px; }
.diagram { overflow-x: auto; }
.diagram svg { min-width: 760px; width: 100%; height: auto; }
.node rect { fill: var(--white); stroke: var(--gray-300); stroke-width: 1.5; }
.node.planned rect { fill: var(--gray-150); stroke: var(--gray-500);
                     stroke-dasharray: 5 4; stroke-width: 1.5; }
.node text { font-family: var(--mono); }
.node .th { font-size: 13px; font-weight: 500; fill: var(--slate); }
.node .ts { font-size: 10.5px; fill: var(--gray-500); }
.edge { stroke: var(--gray-500); stroke-width: 1.5; fill: none; marker-end: url(#arrow); }
.edge.back { stroke: var(--clay); stroke-dasharray: 5 4; }
.lbl { font-family: var(--mono); font-size: 10.5px; fill: var(--gray-500); }
table { width: 100%; border-collapse: collapse; background: var(--white);
        border: var(--border); border-radius: var(--radius-panel); overflow: hidden; }
th { background: var(--gray-150); font-family: var(--mono); font-size: 11px;
     letter-spacing: 0.06em; text-transform: uppercase; color: var(--gray-500);
     text-align: left; padding: 10px 14px; }
td { padding: 10px 14px; border-top: 1px solid var(--gray-150);
     font-size: 13.5px; vertical-align: top; }
td.mono, .mono { font-family: var(--mono); font-size: 12.5px; }
.toolgrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 14px; }
.toolcard { background: var(--white); border: var(--border);
            border-radius: var(--radius-panel); padding: 18px; }
.toolcard .name { font-family: var(--mono); font-size: 15px; font-weight: 500; color: var(--slate); }
.toolcard .path { font-family: var(--mono); font-size: 11px; color: var(--gray-500); margin: 2px 0 10px; }
.toolcard .desc { font-size: 13.5px; margin-bottom: 12px; }
.toolcard .params { font-family: var(--mono); font-size: 12px; color: var(--gray-500); }
.toolcard .params b { color: var(--slate); font-weight: 500; }
.steps { list-style: none; counter-reset: s; }
.steps li { counter-increment: s; position: relative; padding: 8px 0 8px 44px;
            font-size: 14px; border-bottom: 1px dashed var(--gray-150); }
.steps li:last-child { border-bottom: none; }
.steps li::before { content: counter(s); position: absolute; left: 0; top: 10px;
  width: 26px; height: 26px; border-radius: 50%; background: var(--oat);
  color: var(--slate); font-family: var(--mono); font-size: 12px;
  display: flex; align-items: center; justify-content: center; }
.foot { margin-top: 64px; color: var(--gray-500); font-size: 12.5px; }
.mstone { display: flex; align-items: center; gap: 10px; padding: 6px 0;
          font-size: 14px; border-bottom: 1px dashed var(--gray-150); }
.mstone:last-child { border-bottom: none; }
.mstone .tick { width: 18px; height: 18px; border: 1.5px solid var(--gray-300);
                border-radius: 5px; flex: 0 0 auto; }
.mstone.done .tick { background: var(--olive); border-color: var(--olive);
                     position: relative; }
.mstone.done .tick::after { content: ""; position: absolute; left: 5px; top: 1px;
  width: 5px; height: 9px; border: solid white; border-width: 0 2px 2px 0;
  transform: rotate(45deg); }
.mstone.done { color: var(--gray-500); }
"""


def _fmt_params(parameters: dict[str, Any]) -> str:
    """把工具的参数 schema 渲染成紧凑的 HTML（属性名 + required 标记）。"""
    props = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
    required = set(parameters.get("required", [])) if isinstance(parameters, dict) else set()
    if not props:
        return '<span class="params">无参数</span>'
    chips = []
    for name, spec in props.items():
        ptype = spec.get("type", "?") if isinstance(spec, dict) else "?"
        req = " *" if name in required else ""
        chips.append(f"<b>{name}{req}</b>: {ptype}")
    return '<span class="params">' + " · ".join(chips) + "</span>"


def _render_tools(tools: list[dict[str, Any]], order: list[str]) -> str:
    cards = []
    # 按注册顺序排列；没注册到的工具（新文件）排在最后，并标记「未注册」
    by_class = {t["class"]: t for t in tools}
    ordered = [by_class[c] for c in order if c in by_class]
    extras = [t for t in tools if t["class"] not in order]
    for t in ordered + extras:
        reg_badge = (
            '<span class="badge planned" style="margin-left:8px">未注册</span>'
            if t["class"] not in order
            else ""
        )
        cards.append(
            f'<div class="toolcard">'
            f'<div class="name">{t["name"]}{reg_badge}</div>'
            f'<div class="path">{t["file"]}:{t["line"]}</div>'
            f'<div class="desc">{t["description"]}</div>'
            f'{_fmt_params(t["parameters"])}'
            f"</div>"
        )
    return f'<div class="toolgrid">{"".join(cards)}</div>'


def _render_modules(modules: list[dict[str, Any]]) -> str:
    rows = []
    for m in modules:
        classes = "".join(
            f'<span class="pill" style="margin-right:4px">{c["name"]}</span>'
            for c in m["classes"]
        )
        imports = "".join(
            f'<span class="pill" style="background:var(--gray-150);margin-right:4px">{i}</span>'
            for i in m["imports"]
        )
        rows.append(
            f"<tr><td class='mono'>{m['path']}</td>"
            f"<td>{m['lines']}</td>"
            f"<td>{classes or '—'}</td>"
            f"<td>{imports or '—'}</td>"
            f"<td>{m['doc']}</td></tr>"
        )
    return (
        "<table><thead><tr><th>文件</th><th>行数</th><th>类</th>"
        "<th>依赖</th><th>职责</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_milestones(milestones: list[dict[str, Any]]) -> str:
    items = []
    for ms in milestones:
        cls = "mstone done" if ms["done"] else "mstone"
        items.append(
            f'<div class="{cls}"><span class="tick"></span><span>{ms["text"]}</span></div>'
        )
    return '<div class="card">' + "".join(items) + "</div>"


def _render_flow(flow: list[str]) -> str:
    if not flow:
        return '<p class="sub">DESIGN.md 里还没写数据流说明。</p>'
    items = "".join(f"<li>{step}</li>" for step in flow)
    return f'<ol class="steps">{items}</ol>'


def render_html(data: dict[str, Any]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    d = data
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>our-agent · 架构图</title>
<style>{CSS}</style>
</head>
<body>
<div class="page">

  <header>
    <div class="eyebrow">our-agent · 架构图</div>
    <h1>Agent = LLM + 上下文 + 工具</h1>
    <p class="sub">由 scripts/gen_arch.py 自动生成 · {now} · 改代码后跑 <span class="code">python scripts/gen_arch.py</span> 即可更新</p>
  </header>

  <section>
    <h2>核心公式</h2>
    <div class="grid3">
      <div class="card"><h3>LLM · 大脑</h3><p style="font-size:13.5px;margin-top:6px">理解意图、思考规划、做出判断（Policy）。对应 llm/client.py。</p></div>
      <div class="card"><h3>上下文 · 眼睛</h3><p style="font-size:13.5px;margin-top:6px">每个决策点能看到的全部信息（Observation Space）。M2 深化。</p></div>
      <div class="card"><h3>工具 · 手脚</h3><p style="font-size:13.5px;margin-top:6px">能做的所有事情（Action Space）。对应 tools/。</p></div>
    </div>
  </section>

  <section>
    <h2>架构总览</h2>
    <div class="card diagram">{_render_svg(d["modules"])}</div>
    <div class="callout">虚线 = 计划中（文件还没写），实线 = 已实现。写完对应代码再跑一次脚本，虚线会自动变实线。</div>
  </section>

  <section>
    <h2>内置工具 <span class="pill">{len(d["tools"])}</span></h2>
    {_render_tools(d["tools"], d["register_order"])}
  </section>

  <section>
    <h2>数据流（一次任务怎么走）</h2>
    <div class="card">{_render_flow(d["flow"])}</div>
  </section>

  <section>
    <h2>模块清单 <span class="pill">{len(d["modules"])}</span></h2>
    {_render_modules(d["modules"])}
  </section>

  <section>
    <h2>里程碑进度</h2>
    {_render_milestones(d["milestones"])}
  </section>

  <div class="foot">
    <p>更新方式：改完代码（新增工具、新模块、勾选里程碑）后执行 <span class="code">python scripts/gen_arch.py</span>，覆盖生成本文件。</p>
    <p>生成原理：脚本用 Python 标准库 AST 扫描 agent/ llm/ tools/ 的源码与 DESIGN.md，不 import 项目代码，项目暂时跑不起来也能出图。</p>
  </div>

</div>
</body>
</html>
"""


# ------------------------------------------------------------------ 主流程 --

def collect() -> dict[str, Any]:
    """收集所有架构数据。"""
    return {
        "tools": _collect_tools(),
        "register_order": _collect_register_order(),
        "modules": _collect_modules(),
        "milestones": _collect_milestones(),
        "flow": _collect_flow(),
    }


def main() -> int:
    data = collect()
    OUTPUT.write_text(render_html(data), encoding="utf-8")

    # 摘要输出：让跑脚本的人一眼看到「图上有什么」，
    # 同时验证提取结果（这也是证明契约的验证手段）
    tools = ", ".join(t["name"] for t in data["tools"]) or "(无)"
    planned = [
        p
        for p in ("cli.py", "agent/loop.py", "llm/client.py", "tools/registry.py")
        if not any(m["path"] == p for m in data["modules"])
    ]
    print(f"生成 {OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"  工具 ({len(data['tools'])}): {tools}")
    print(f"  模块 ({len(data['modules'])} 个 .py)")
    print(f"  里程碑 ({len(data['milestones'])} 项)")
    print(f"  计划中（虚线节点）: {', '.join(planned) or '无'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
