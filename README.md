# our-agent

从零实现的 **ReAct Agent 框架**——不依赖 LangChain / AutoGen 等任何现成框架，用最小依赖（仅 `openai` SDK）自研核心循环、上下文工程与工具系统。设计理念参考《深入理解 AI Agent：设计原理与工程实践》（李博杰 著）。

> **项目状态**：M1（最小 ReAct 闭环）✅ / M2（上下文工程）✅ —— 41 个单元测试全绿 + 真实 API 冒烟验证通过。

---

## 核心特性

### ReAct 主循环
- 思考 → 行动 → 观察的标准循环：模型生成 `tool_calls` → 框架执行 → 结果回填 → 再思考，直到模型输出最终回答
- `max_iterations` 熔断护栏：达到上限强制终止，不白烧 token
- **熔断 [UNFINISHED] 摘要**：熔断时不留原始垃圾轨迹，生成键值对摘要（任务目标 / 已完成 / 下一步 / 失败点）写入会话历史——下一轮可**接着办完**未完成任务

### 上下文工程（KV Cache 友好）
- 静态前缀（system prompt + 工具定义）不变，历史**只追加不改写**——保持服务端 Prompt Cache 前缀稳定
- 内置缓存命中率观测：真实 API 实测稳定轮次命中率 **96%**（4864/207）

### Agent 状态栏（上下文提炼层）
每轮请求末尾注入 user-role meta 消息，让模型"瞥一眼"即知当前运行状态，无需从原始轨迹现算：

```
【状态栏】
时间: 2026-08-25 02:40:30
工具: 已调用 5 次（run_command×3、read_file×2）
失败: run_command×1
进度: 第 7/10 轮（剩余 3 轮）
⚠ 剩余 3 轮：请收敛，尽快输出最终回答
TODO: 输出最终回答
```

- 三条维护铁律：**代码维护**（绝不让 LLM 读历史总结）、**键值对格式**（非散文）、**只放可信观测**（模型无条件信任状态栏 → 内容必须准确）
- 工具调用/失败计数、剩余轮数预算全部由框架代码精确统计；临近熔断自动追加收敛警告

### 提示注入防御
- 工具结果回填前用 XML 来源标记包裹（`<tool_result tool="...">`），明确"外部内容不可信"
- system prompt 四段结构化：身份 / 执行 SOP / 关键规则 / 安全边界

### Agent Skills（渐进式披露）
- `skills/<name>/SKILL.md` 技能目录规范；元数据（name + description）常驻，完整内容按需加载
- 模型判断需要时调用 `load_skill` 工具拉取完整 SKILL.md——控制上下文占用

### 工具系统
- 注册表模式：工具名 → Python 函数，自动生成 OpenAI 协议 schema
- 错误即输入：幻觉调用 / 参数畸形 / 执行失败均以错误消息回填，让模型自我纠正而非中断循环

---

## 技术栈

| 项 | 选择 |
|----|------|
| 语言 | Python 3.11 |
| LLM SDK | `openai`（DeepSeek API，OpenAI 兼容协议） |
| 依赖 | 仅 `openai` 一个运行时依赖 |
| 测试 | `unittest` / `pytest`，41 用例，FakeClient 驱动（不依赖真实 API） |

---

## 快速开始

```bash
git clone https://github.com/san-sheng/our-agent.git
cd our-agent
python3 -m venv .venv && source .venv/bin/activate
pip install openai

# 在项目根创建 config.toml，填入 [llm] 段（api_key / base_url / model）

# 单次模式
python cli.py "读 README.md，统计有多少行，把结果写到 stats.txt"

# 多轮会话 REPL
python cli.py
```

---

## 架构一览

```
our-agent/
├── agent/
│   └── loop.py        # ReAct 主循环：轨迹累积、状态栏注入、熔断摘要、缓存度量
├── llm/
│   └── client.py      # LLM 客户端：OpenAI 兼容封装、指数退避重试、usage 提取
├── tools/
│   ├── registry.py    # 工具注册表：名字解析、schema 生成、错误回填
│   ├── base.py        # Tool 抽象基类
│   └── builtin/       # 内置工具：read_file / write_file / run_command / load_skill
├── skills/            # Agent Skills 目录（SKILL.md 规范）
├── tests/             # 41 个单元测试
└── cli.py             # 命令行入口（单次 + REPL）
```

数据流：`cli.py` 收任务 → `loop.py` 组装上下文（system + history + 任务 + 状态栏）→ `client.py` 发模型 → 有工具调用则 `registry.execute` → 结果回填 → 循环，直到模型输出最终回答。

---

## 设计取舍（摘要）

1. **状态栏为何用代码维护**——上下文窗口是"只有一半的检索引擎"：模型擅长检索、不擅长归纳。一次让前沿模型读长历史做统计，准确率反而低于 20 行正则；代码维护才能保证"模型无条件信任"的前提成立。
2. **熔断为何存摘要而非原始轨迹**——原始半截轨迹看着像任务已完成，会误导后续轮次；紧凑的 `[UNFINISHED]` 摘要让"接着办完"可复现（真实 API 冒烟：熔断后追问，正确补齐 read → wc → write，行数一致）。
3. **工具错误为何回填而非中断**——错误是模型的输入。坏 JSON / 幻觉工具名 / 执行失败均以 tool 消息回填，循环得以继续，模型据错误自我纠正。
4. **状态栏为何不进历史**——过期状态（旧时间、旧计数）会污染多轮记忆；只拼进请求副本则每轮即时反映进展，同时守住 KV Cache 前缀不变式。

---

## 测试与验证

- **41 个单元测试全绿**：FakeClient 预置响应序列驱动循环，验证轨迹累积、状态栏注入（时间/工具计数/失败统计/临近警告）、熔断摘要、多轮历史、缓存统计等
- **真实 API 冒烟**（DeepSeek）：
  - 熔断 → 追问 → 接着办完：`wc -l` 与实际写入行数一致，且无原始垃圾轨迹污染
  - 失败统计：读不存在文件 → 状态栏「失败: read_file×1」，模型不再重复失败调用
  - 缓存命中率：状态栏追加尾部不破坏前缀，稳定轮次回到 96%

---

## Roadmap

| 里程碑 | 主题 |
|--------|------|
| ✅ M1 | 最小 ReAct 闭环（客户端 + 3 工具 + 主循环 + CLI） |
| ✅ M2 | 上下文工程（多轮历史、状态栏、提示注入防御、Skills 按需加载） |
| 🔜 M3 | 记忆与知识库（会话记忆 → 长期记忆 → 简单 RAG） |
| M4 | Coding Agent 深化（编辑方案、错误恢复、安全沙盒） |
| M5 | 评估（小型 eval 集 + LLM-as-Judge） |
| M6 | 持续进化（从运行轨迹提取学习信号） |
| M7 | 多 Agent 协作 |