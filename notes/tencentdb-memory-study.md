# 学习笔记：TencentDB Agent Memory 的记忆蒸馏设计

> 来源：克隆 `TencentCloud/TencentDB-Agent-Memory`（v2.0.0-beta.1，MIT，9.9k⭐）
> 目的：为 our-agent 的 M3（记忆）阶段提供可借鉴的分层蒸馏 + 检索预算控制设计
> 日期：2026-07-31
> 结论先行：**不部署**（团队级 hub，部署重、功能与 Mnemosyne 重叠），**只偷师两个设计**——L0→L3 蒸馏管线、recall 预算控制。

---

## 1. 架构总览

```
四类记忆资产：Chat Memory / Skill / Wiki / CodeGraph
服务拆分：memory-core（记忆核心）+ memory-hub（管理面板）+ memory-proxy（LLM 代理）
```

我们的关注点只在 **MemoryCore**（记忆核心，`MemoryCore/src/core/`，~34k 行 TS）里的两条线：

1. **写管线（蒸馏）**：L0 原始对话 → L1 原子记忆 → L2 场景 → L3 persona
2. **读管线（召回）**：hybrid 检索 + RRF 融合 + 预算封顶 → 注入上下文

---

## 2. 写管线：L0 → L3 分层蒸馏

### 2.1 分层定义

| 层 | 存什么 | 谁负责写 | 物理形式 |
|---|---|---|---|
| L0 | 原始对话（完整上下文） | `l0-recorder.ts`，agent_end 时记录 | JSONL |
| L1 | 原子记忆（persona/episodic/instruction） | `l1-extractor.ts`，LLM 单次调用提取 | JSONL + FTS5/向量索引 |
| L2 | 场景块（围绕项目/话题的知识块） | `scene-extractor.ts`，LLM agent 用工具读写 | `scene_blocks/*.md` |
| L3 | persona（长期画像） | 从提取信号更新 | `persona.md` |

### 2.2 L1 提取（最值得抄的一段）

`l1-extractor.ts` 的核心设计——**一次 LLM 调用同时做两件事**：

```
1. 读最近 L0 消息（切分为 background + new）
2. 单次 LLM 调用：情境切分 + 记忆提取（JSON 结构化输出）
3. 批量冲突检测（l1-dedup.ts，基于 embedding top-K）
4. 写入 L1 JSONL
```

**提取 prompt 的精华**（`prompts/l1-extraction.ts`）：

- **三大类型**，每种有明确句式 + 触发词 + priority 打分区间：
  - `persona`：用户稳定属性/偏好/价值观（80-100 健康禁忌；50-70 一般喜好；<50 丢）
  - `episodic`：客观事件/决定/计划（80-100 重要事件；60-70 一般；<60 丢）
  - `instruction`：对 AI 的长期规则（**-1 全局死命令**；90-100 核心规则；<70 丢）
- **提取原则**：
  - 宁缺毋滥：琐碎闲聊、临时性指令、一次性操作不提取
  - 独立完整：记忆必须"跳出当前对话依然成立"（"用户（王小明）30岁，是一名软件工程师"而非"他说他30岁"）
  - 归纳合并：强关联/因果消息合并为一条，不碎片化
  - 主体必须以"用户"或"AI"为核心
- **场景切分**：话题切换条件（明确指令、意图转变、独立新目标）；命名"我（AI）在和xxx（用户身份）做xxx（目标活动）"，30-50 字符，全局唯一
- **时间语义**：episodic 尽量从 timestamp 推算绝对时间，metadata 里带 `activity_start_time`/`activity_end_time`（ISO8601）
- **容错**：无有效记忆也必须输出 scene 分割结果，memories 为空数组

### 2.3 L2 场景提取（工程上最狠的一段）

`scene-extractor.ts` 用 **LLM agent 而非单次调用**：

- 用 `CleanContextRunner`（工具启用的 LLM 环境）自主读写场景文件
- **沙箱**：workspaceDir 设为 `scene_blocks/`，系统文件（checkpoint、scene_index、persona.md）对 LLM 物理不可见
- 每次提取前备份（backupCount=10），带 checkpoint 可回滚
- maxScenes=15；timeoutMs=300_000（5 分钟，因为 LLM 会做多次工具调用）
- 顺带解析 persona 更新信号：LLM 输出 `[PERSONA_UPDATE_REQUEST]reason:...[/PERSONA_UPDATE_REQUEST]` → 驱动 L3 更新

### 2.4 管线调度（异步、防抖）

```ts
pipeline: {
  everyNConversations: 5,        // 每 5 次对话触发一次蒸馏
  l1IdleTimeoutSeconds: 600,     // L1 空闲 10 分钟
  l2DelayAfterL1Seconds: 10,     // L1 完成后延迟 10s 再跑 L2
  l2MinIntervalSeconds: 900,     // L2 最小间隔 15 分钟
  l2MaxIntervalSeconds: 3600,    // L2 最大间隔 1 小时
  sessionActiveWindowHours: 24,  // 活跃会话窗口
}
```

关键取舍：**蒸馏是异步后台管线**，不阻塞对话；L1 轻量高频，L2 重但低频（用间隔封顶防止无限累积）。

---

## 3. 读管线：召回 + 预算控制（第二个最值得抄的设计）

### 3.1 Hybrid 检索（auto-recall.ts）

```
strategy: hybrid（默认）
  ├─ keyword: FTS5 BM25（buildFtsQuery 构造查询）
  └─ embedding: 向量余弦相似度
  └─ 融合: RRF（Reciprocal Rank Fusion），k=60
      score(id) = Σ 1/(k + rank + 1)，出现在两个列表里则分数相加
```

- `candidateK = maxResults * 3`（多取候选再融合，防漏）
- 查询前先 `sanitizeText` 剥掉网关注入的元数据（Sender、时间戳、base64 图片等）
- 查询过短（<2 字符）直接跳过搜索
- embedding 不可用时**快速失败**而非静默降级（结构化错误，H-15 契约）

### 3.2 预算控制（recall 配置）

```ts
recall: {
  enabled: true,
  maxResults: 5,              // 注入条数上限
  maxCharsPerMemory: 0,       // 单条记忆字符上限（0=不限制）
  maxTotalRecallChars: 0,     // 全部记忆总字符上限（0=不限制）
  scoreThreshold: 0.3,        // 分数阈值
  strategy: "hybrid",
  timeoutMs: 5000,            // 召回总超时
}
```

**超时实现**：`Promise.race([核心逻辑, 5s定时器])`——超时返回结构化 `RecallResult.error`（code=20001），不是静默 undefined，让上层能区分"无结果"与"超时"。

**字符预算实现**（`applyRecallBudget`）：
- 单条超限 → 截断 + 尾部追加"…（已截断；可用 tdai_memory_search 查看详情）"
- 总字符超限 → 从前往后装，装不下就丢弃剩余（dropped 计数）
- 截断/丢弃都有 debug 日志（input=N, output=M, truncated=T, dropped=D）

### 3.3 注入分区（Prompt Caching 优化）

**这是很聪明的一段**——把注入内容按"稳定性"分两路，配合 prompt caching：

```
appendSystemContext（系统 prompt 尾部，稳定，可缓存）：
  persona（L3） + scene navigation（L2 索引） + memory tools guide

prependContext（用户 prompt 前缀，每轮变化，不缓存）：
  L1 相关记忆（不同轮次不同）
```

→ 稳定内容放系统侧吃缓存，动态内容放用户侧不污染缓存。这跟我们 Hermes 的 prompt caching 纪律（"never change context mid-conversation"）一个思路。

### 3.4 记忆行格式化（时间语义）

```
- [persona] 用户叫王小明，30岁，是一名软件工程师。
- [episodic|旅行计划] 用户计划五月去日本旅行。(活动时间: 2025-05-01 ~ 2025-05-10)
- [instruction] 用户要求回答时使用中文，保持简洁。
```

tag 带 `type|scene_name`，时间带"段时间优先、点时间兜底"。

### 3.5 工具指南注入（防无限检索）

注入 `<memory-tools-guide>`：告诉 agent 有 `tdai_memory_search` / `tdai_conversation_search` / `read_file` 三个工具可主动深入，但**每轮合计最多调用 3 次**，3 次无结果就放弃。防的是"记忆不足时 agent 无限检索"的失控。

---

## 4. 对 our-agent M3 的启示

### 值得抄的

1. **L1 提取 prompt 模板**——三大类型 + priority 打分区间 + "宁缺毋滥/独立完整/归纳合并"原则。这是经过 Kenty 验证的实战 prompt，直接可改造成我们的记忆提取器。
2. **召回预算的四个闸门**：条数（maxResults）+ 单条字符 + 总字符 + 超时（Promise.race）。防记忆淹没上下文的完整方案。
3. **稳定/动态分区注入**——稳定内容（画像、索引）放系统侧，动态内容（当轮相关记忆）放用户侧，服务 prompt caching。
4. **工具指南 + 调用次数限制**——把"记忆不够怎么办"交给模型主动检索，但设硬上限防失控。
5. **沙箱化 LLM agent 写记忆**——L2 场景用工具化 LLM 写文件，但 workspaceDir 物理隔离系统文件。安全边界清晰。

### 不抄的 / 注意的

- **不部署整个 hub**——团队级、Docker 三服务、两组 LLM key，我们不需要。
- L2 场景块每 15min-1h 才更新一次，且用 LLM agent 自主写文件——对我们单用户场景偏重，M3 先做 L1 原子记忆 + L3 画像即可。
- 它的存储是 SQLite + 可选 TCVDB（腾讯向量库）；我们用 Mnemosyne 已有的向量 + FTS 混合，不引入新存储。

### 落地建议

M3 记忆模块按这个顺序做：
1. L1 原子记忆提取器（抄 prompt 模板 + priority 打分）
2. recall 预算四闸门（条数/单条字符/总字符/超时）
3. 稳定/动态分区注入（对齐 prompt caching）
4. L3 画像用 `[PERSONA_UPDATE_REQUEST]` 信号机制延迟做

---

*仓库位置：`~/TencentDB-Agent-Memory/`，核心代码 `MemoryCore/src/core/{record,scene,hooks,prompts}/`*
