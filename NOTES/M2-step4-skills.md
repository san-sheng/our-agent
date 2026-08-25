# M2 学习笔记 —— 第 4 步：Skills 按需加载（渐进式披露）

> 日期：2026-08-25（生活日；git 提交在 8-26 凌晨，按日期规则归前一天）
> 内容：`skills/` 技能目录 + `tools/builtin/load_skill.py` 按需加载工具 + `agent/loop.py` 元数据注入（【可用技能】meta 消息）
> 状态：✅ 完成（代码 + 测试 49 绿 + 真实 API 冒烟）
> 对应书：Ch2 §动态提示词与 Agent Skills（渐进式披露；元数据/核心流程/细则三层）
> 落点：DESIGN.md §9 的 M2 拆步「Skills」在此落实——**M2 上下文工程整体完成**

---

## 0. 一句话总结

M2 第 4 步做一件事：**Agent Skills 按需加载**——把能力模块化成
`skills/<name>/SKILL.md`，框架只把 `name + description` 元数据
（几百 token）常驻上下文，模型判断需要时通过 `load_skill` 工具
加载完整内容。这就是书的**渐进式披露（Progressive Disclosure）**：
先给一份目录摘要，需要时再取整本手册。

**M2 至此全部完成**：多轮对话 → 状态栏 → 提示注入 → Skills，四步闭环。

---

## 1. 为什么需要 Skills：提示词会膨胀

系统提示词不断塞业务知识会带来两个问题（书 710 行）：

1. **浪费 token**：大部分内容与当前任务无关
2. **注意力被稀释**：无关信息过多稀释模型对关键内容的注意力（上下文腐化）

Skills 的解法：**不是把所有知识一次性塞给 Agent，而是让它按需加载**。
每个 Skill = 一套包含专业领域指导的提示词集合，像新员工的专项任务手册。

## 2. 三层渐进式披露（书的核心结构）

| 层 | 内容 | 何时进上下文 |
|----|------|-------------|
| **第一层（元数据）** | name + description（几百 token） | 常驻——框架扫描所有 skill，注入摘要 |
| **第二层（核心流程）** | 完整 SKILL.md | 模型判断需要时，通过专用工具加载（作为 tool result） |
| **第三层（细则）** | 子文档（reference.md 等） | 按具体需求选择性深入 |

**description 是路由决策的关键**（书 725 行）：要写成**路由条件**而非
功能介绍——"Use when / Don't use when"+ 反例。缺反例的描述会在不相关的
任务上频繁误触发；"何时该用我"比"我能做什么"重要得多。

## 3. 三种实现方式与权衡（书 739-763 行）

| 方式 | 做法 | 优点 | 代价 |
|------|------|------|------|
| 一 | 注入 system prompt | 指令遵循最强（训练大量用这个位置） | 每次加载新 skill 改变 system 消息 → KV Cache 反复失效 |
| 二 | 普通文件读取（内容在上下文中间） | 不破坏缓存 | 要求模型在长上下文中间遵循指令，不同模型差异大 |
| **三（生产实现）** | **元数据动态提供 + 专用工具按需加载** | 兼顾上下文开销、缓存复用、指令遵循 | 首次 emit 要付一次写入代价（一次性写入、永久受益） |

我们选**方式三**，与 Claude Code 同思路。书里特别厘清：
**「元数据需要提前可见」是机制要求，『以什么消息角色注入』是实现细节**
（Claude Code 就用过 user-role 的 `<system-reminder>`）——所以我们复用
状态栏的 user-role meta 消息机制，不碰 system 前缀，KV Cache 最友好。

## 4. 我们的实现

### 4.1 目录结构与示例 skill

```
skills/
└── notes-writer/
    └── SKILL.md        # frontmatter(name+description) + 主体规范
```

示例 skill 挑了 `notes-writer`（写 NOTES 学习笔记的格式规范）——对我们
实际有用（以后写笔记可调），又天然演示渐进式披露：元数据只有一句路由
描述，完整规范（元数据块/章节/验证结果/文件清单）必须 load_skill 才有。

frontmatter 的 description 按书的要求写路由条件 + 反例：
```
description: 写 our-agent 的 NOTES/ 学习笔记时使用。Use when：需要新写或
回填 M1/M2 里程碑学习笔记、步骤笔记；Don't use：日常对话、普通 markdown
文档写作、DESIGN.md 回填、代码修改。加载后提供笔记格式规范……
```

### 4.2 load_skill 工具（tools/builtin/load_skill.py）

```python
class LoadSkill(Tool):
    name = "load_skill"
    def __init__(self, skills_dir: Path | None = None): ...
    def run(self, skill_name: str) -> str:
        # 只从 skills_dir/<name>/SKILL.md 找——不接受任意路径，
        # 技能加载不是任意文件读取（M2 第 3 步注入防御的延伸）
```

- **为什么不用 read_file 直接读**：模型不知道 skill 文件路径（内部细节）；
  专用工具能返回「技能不存在」标准错误（错误是模型的输入）；语义清晰
- **安全边界**：只从技能目录按约定找文件，不接受任意路径——skill 内容
  也是外部数据，加载结果同样走 `_wrap_tool_result` 来源标记
- **skills_dir 可注入**（测试用临时目录，不耦合 repo 实际文件）

### 4.3 元数据注入（agent/loop.py）

```python
req = list(messages) + [
    {"role": "user", "content": self._skill_metadata()},   # 【可用技能】
    {"role": "user", "content": self._status_bar(...)},    # 【状态栏】最末尾
]
```

- **`_skill_metadata()`**：扫描 skills_dir 下每个目录的 SKILL.md，极简解析
  frontmatter（不引 PyYAML，学习项目零依赖），输出：
  ```
  【可用技能】
  - notes-writer: 写 our-agent 的 NOTES/ 学习笔记时使用。……
  ```
- **不进 history**：与状态栏同机制（meta 消息只拼请求副本）——技能目录
  变化时旧目录不污染多轮记忆，守住「历史只追加不改写」
- **位置**：状态栏前面。技能元数据相对静态，状态栏每轮变化最频繁，
  保持最末尾对缓存最友好
- **SYSTEM_PROMPT 补一条**：「【可用技能】是框架注入的技能目录……任务需要
  相关专业技能时，用 load_skill 工具加载完整内容再执行」（模型才知道
  这个 meta 消息是什么、该怎么用）

### 4.4 KV Cache 视角

元数据追加在上下文末尾、不动 system 前缀——首次 emit 付一次写入代价，
之后整个会话不再重复（书 763 行：「一次性写入、永久受益」）。对比把
技能塞 system prompt，每次更新会让下游整条 trajectory 失效（数万到
数十万 token 的 cache_creation），那才叫不友好。

## 5. 测试策略（49 绿 = 45 + 4 新增）

| 用例 | 验证点 |
|------|--------|
| `test_skill_metadata_injected` | 请求末尾注入【可用技能】：含 name + description（路由条件语气），**不含 SKILL.md 主体细节**（渐进式披露断言）；缺 SKILL.md 的目录提示占位不崩 |
| `test_load_skill_success` | load_skill 返回完整 SKILL.md，带 `<tool_result tool="load_skill">` 来源标记（M2 第 3 步兼容） |
| `test_load_skill_unknown` | 加载不存在的技能 → 错误 + 计入失败统计（错误是模型的输入） |
| `test_skill_metadata_not_persisted` | 技能元数据不进 history（与状态栏同机制） |

**改造**：`_strip_status()` 增加滤掉【可用技能】——技能元数据与状态栏同为
meta 消息，断言轨迹结构时都要滤掉（否则 8 个既有断言会破）。

`_make_agent` 增加 `skills_dir` 参数，同时传给 Agent 和 registry——
**元数据扫描与技能加载必须指向同一目录**（否则模型看到目录里列着、
却加载不出来的技能）。

## 6. 验证结果（真实 API 冒烟）

任务：`帮我写一份简短的学习笔记，总结 Agent Skills 的渐进式披露机制，
存到 /tmp/skill-smoke-note.md`

```text
[usage] prompt=1373 cache_hit=0 miss=1373 命中率=0%
[Agent] 第 1 轮调用工具: load_skill          ← 看到元数据 → 识别需要 → 加载
[usage] prompt=2031 cache_hit=256 miss=1775 命中率=13%
[Agent] 第 2 轮调用工具: write_file          ← 按规范写笔记
[usage] prompt=2750 cache_hit=1152 miss=1598 命中率=42%
[Agent] 完成（第 3 轮）
产物: /tmp/skill-smoke-note.md（42 行，2.1K）
```

四个验证点：

1. **自动路由生效**：模型没被提示就自己调了 load_skill（书实验 2-6 流程：
   看到元数据 → 识别需要 → 加载 SKILL.md → 执行）——description 写得
   像路由条件的效果
2. **产物符合 notes-writer 规范**：元数据块（> 日期 / > 主题）+「## 0.
   一句话总结」+ 结构化章节（为什么需要 / 机制要点 / 设计取舍 / 类比）
3. **来源标记兼容**：load_skill 结果走 `_wrap_tool_result`，tool 角色
   清晰，外部内容与指令分离（M2 第 3 步防御没被新工具绕过）
4. **缓存逐步恢复**：0% → 13% → 42%（技能元数据 + 工具定义进入缓存后
   开始命中——「一次性写入、永久受益」的早期阶段）

## 7. 学习点自查

- [x] Skills 解决什么问题 → 提示词膨胀：浪费 token + 稀释注意力（§1）
- [x] 渐进式披露三层是什么 → 元数据（常驻）/ 核心流程（按需加载）/ 细则（深入阅读）（§2）
- [x] description 为什么是路由关键 → 写成「Use when + 反例」而非功能介绍；缺反例频繁误触发（§2）
- [x] 三种实现方式权衡 → system 注入（缓存反复失效）/ 普通文件读（模型遵循打折）/
      元数据 + 专用工具（生产实现，兼顾三者）（§3）
- [x] 元数据注入角色为什么可以用 user → 「元数据可见」是机制要求，「角色」是实现细节（§3）
- [x] 为什么不用 read_file 读 skill → 路径是内部细节；专用工具给标准错误；注入防御延伸（§4.2）
- [x] 「一次性写入、永久受益」是什么意思 → KV Cache 首次 emit 付一次代价，
      之后整个会话不再重复；对比塞 system prompt 的反复失效（§4.4）
- [x] Skills 与提示注入的关系 → Skill 是把外部内容当指令加载的制度化形式，
      安装来源不明 Skill 前必须审查（像审查代码）（书 685 行，M2 第 3 步笔记 §3 已记）

## 8. 下一步

- **M2 上下文工程整体完成** → 进入 **M3 记忆与知识库**（书 Ch3）：
  会话记忆 → 长期记忆 → 简单 RAG，跨会话召回测试
- Skills 的已知边界：description 写不好会误触发（路由质量依赖 skill 作者）；
  skill 内容安全（来源审查）在 M4 安全深化时补执行层
- 工具定义设计（书 §工具定义的设计 + Ch4「主动工具发现」）M4 展开

---

## 附：文件清单

```
skills/notes-writer/SKILL.md        # 示例技能：NOTES 学习笔记格式规范
tools/builtin/load_skill.py         # 按需加载工具（skills_dir 可注入）
tools/builtin/__init__.py           # default_registry 注册 LoadSkill
agent/loop.py                       # SKILLS_HEADER + _skill_metadata() + req 拼接 + SYSTEM_PROMPT 说明
tests/test_loop.py                  # 4 新用例 + _strip_status 滤技能元数据 + _make_agent 支持 skills_dir
NOTES/M2-step4-skills.md            # 本文
DESIGN.md                           # §4.2/§9/进度 回填（M2 完成）
```