# M2 学习笔记 —— 第 1 步：多轮会话 + 缓存度量

> 日期：2026-08-23（生活日；git 提交在 8-24 凌晨，按日期规则归前一天）
> 内容：`agent/loop.py` 多轮历史 + `llm/client.py` usage 透传（缓存命中率度量）
> 状态：✅ 完成（代码 + 测试 36 绿 + 真实 API 验证）
> 对应书：Ch2 §2.2 上下文结构（静态前缀 + 轨迹）、§2.3 KV Cache / Prompt Cache

---

## 0. 一句话总结

M2 第 1 步做两件事：

1. **多轮对话**：Agent 从「一问一答」变成「有会话记忆」——每次任务完成后的轨迹
   存进 `self._history`，下一轮自动带上（会话内记忆，长期记忆是 M3）。
2. **缓存度量尺子**：`chat()` 透传 usage（含 DeepSeek 的
   `prompt_cache_hit/miss_tokens`），Agent 累计命中率——把「缓存好不好」
   从感觉变成数字。

对应书的落点：**静态前缀（system + 工具定义）不变量 + 动态轨迹追加末尾**。
我们的 system 和 tools 每轮原样发送（KV Cache 友好），历史消息只追加不改写。

---

## 1. 多轮对话：轨迹从「局部变量」变成「会话字段」

M1：每次 `run()` 全新构造轨迹（system + user），REPL 每行 = 新任务 = 新轨迹。
M2：轨迹分成两层——

```
messages = [system] + self._history + [新 user]
```

`self._history` 是 Agent 实例字段，跨 `run()` 存活。任务完成时把本次新增的
轨迹（新 user 起，到最终 assistant 回答止）存回 history，下一轮就带上了。

类比 C++：M1 的轨迹是函数内的局部 `std::vector`，M2 变成成员变量
`std::vector` 的尾部追加——状态从「栈」搬到「堆」上持续。

## 2. 关键设计点（每个都是取舍）

### 2.1 成功才保存，熔断不保存

`run()` 正常返回（模型给出最终回答）→ 轨迹存入 history。
达到 max_iterations 熔断 → **不保存**。理由：

1. 未完成的任务轨迹会误导下一轮——模型看到半截子轨迹，可能以为任务做完了
2. 熔断轨迹往往是一堆重复失败的工具调用，保存只会膨胀上下文、稀释注意力

> ⚠️ **来源说明**：这是我们的设计取舍，**书 Ch2 没有直接建议**「熔断不保存」。
> 书的相关论述反而偏向保留失败信息：§2.7 压缩保留优先级明确「未解决的
> TODO / 回滚笔记 / 验证状态必须保留」；§2.3 实验 2-3 批评滑动窗口丢了关键
> 结果导致 Agent 重蹈覆辙。本取舍的边界：若用户在同一会话里追问「刚才那个
> 任务帮我办完」，不保存会让模型看不到失败路径。
>
> ✅ **已拍板（2026-08-24 少爷）**：熔断「失败可回溯」与 **M2 第 2 步状态栏
> 一起实现**——熔断时复用状态栏格式生成 `[UNFINISHED]` 摘要写入 history
> （任务目标 / 已完成步骤 / 下一步 / 失败点），不死记原始垃圾轨迹。
> ✅ **已实现（2026-08-25，M2 第 2 步）**：见 `NOTES/M2-step2-statusbar.md`——
> 熔断现在会把 `[UNFINISHED]` 摘要存进 history，真实冒烟验证模型能靠它
> 「接着办完」；本段「状态栏步完成前，维持现状（不保存）」的条款作废。

### 2.2 历史只追加，不改写（KV Cache 前缀稳定）

对 history 的消息只 append、永不修改或删除——这直接对应 Ch2 §2.3 铁律第 2 条
「动态信息永远追加到末尾」：前缀（system + tools）不变 → 服务商 Prompt Cache
持续命中；history 是轨迹（后半段），追加不影响已缓存前缀。

（滑动窗口是反面教材：删除旧消息既破坏前缀一致性、又丢关键工具结果——
Ch2 实验 2-3 点名批评，M2 的压缩策略刻意避开它。）

### 2.3 usage 度量 = 缓存机制唯一可验证的契约

DESIGN.md §5 定 M2 验证方式是「观察 token 用量」。落地为：
`llm/client.py` 把每个响应的 usage 透传出来，`agent/loop.py` 累加
`cache_stats = {"hit", "miss"}`，命中率 = hit/(hit+miss)。

意义：缓存机制没有「客户端缓存代码」可写，它的工作全在「设计上不破坏前缀」。
命中率是把「有没有破坏」变成可复现的证据——哪天命中率掉下来，
就是哪里改了 system prompt / 工具定义 / 历史顺序。

### 2.4 DeepSeek 缓存字段藏在 pydantic 的 model_extra 里

openai SDK 的 `CompletionUsage` 只声明了标准字段
（prompt/completion/total_tokens）。DeepSeek 额外返回的
`prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` 是自定义字段，
pydantic 默认把它们存进 `model_extra` 而不是命名属性——

```python
extra = getattr(usage, "model_extra", None) or {}
d["prompt_cache_hit_tokens"] = extra.get("prompt_cache_hit_tokens", 0)
```

标准字段用 `getattr`，自定义字段从 `model_extra` 取，两手都要。

---

## 3. 踩坑：off-by-one（base_len → user_start）

第一版保存历史写的是：

```python
base_len = len(messages)      # 初始长度：system + history + 新 user
...
self._history.extend(messages[base_len:])   # ❌
```

`base_len` 是**初始**消息数量，但保存发生在循环 append 之后——
`messages[base_len:]` 切出来的是「第一个 append 的消息（assistant）起」，
**把新 user 漏了**。历史只剩 assistant，下次对话模型看不到用户这次说了什么。

修复：记住**新 user 的位置**而不是初始长度：

```python
user_start = len(messages) - 1   # 新 user 是初始 messages 的最后一个元素
...
self._history.extend(messages[user_start:])
```

**被 `test_multi_turn_history` 当场抓住**——断言「第二次请求必须包含
第一次的 user + assistant」一跑就红了。这类 bug 在真实对话里很难肉眼发现
（模型照样能回答，只是记忆残缺），测试的价值就在这：把「轨迹累积」这种
结构性不变式变成断言，写错立刻现形。

---

## 4. 测试策略（延续 M1 的假 LLM 驱动）

还是 `_FakeClient` 预置响应 + 记录每次请求。新增 6 个用例：

| 用例 | 验证点 |
|------|--------|
| `test_multi_turn_history` | 第二次 run 的请求 = system + 上轮轨迹(user+assistant) + 新 user |
| `test_reset_clears_history` | reset() 后回到 [system, user] 干净轨迹 |
| `test_break_does_not_save_history` | 熔断后 history 为空，不污染下轮 |
| `test_usage_stats_accumulate` | usage 跨 run 累计（hit/miss 求和）|
| `test_usage_passthrough` | chat() 透传标准 + model_extra 缓存字段 |
| `test_usage_none_returns_empty` | 没 usage 不报错，返回空 dict |

## 5. 验证结果（真实 API）

```text
=== R1：让 Agent 记住一个事实 ===
好的，我已记住：您的幸运数字是 42。
=== R2：问它记住没有（验证多轮记忆） ===
您的幸运数字是 42。           ← 记忆生效

缓存统计: hit=1536 miss=208 命中率=88%
```

- 多轮记忆验证：R1 让记住「幸运数字 42」，R2 问出 42——history 生效
- 缓存命中率 88%：前缀稳定，DeepSeek Prompt Cache 正常工作
  （第一次调用 miss 掉了 208 token——没有历史可命中；后续 1536 全命中）

CLI 里每轮结束会打印累计命中率，`verbose` 打开时每轮还打明细。

## 6. 学习点自查

- [x] 静态前缀 + 动态轨迹的结构具体长什么样 → system/tools 不变，
      history 及新消息 append（本节 §2.2）
- [x] 「前面不能动、后面可以追加」的实践含义 → 改 system prompt 一个字符
      缓存全废；历史只追加不改写（§2.2）
- [x] Prompt Cache 怎么度量 → usage 的 hit/miss 字段，命中率作为契约（§2.3）
- [x] 多轮记忆的边界 → 会话内记忆（M2），跨会话长期记忆（M3 记忆系统）

## 7. 下一步

- M2 第 2 步：Agent 状态栏（时间戳 + 工具计数 + TODO，末尾 user 消息注入）——
  Ch2 §2.6「上下文窗口是只有一半的检索引擎：给模型补上提炼层」
- M2 第 3/4 步：system prompt 结构化 + 提示注入测试、Skills 按需加载
- context.py / state.py 独立模块的提取（当前 history 内联在 loop.py）

---

## 附：文件清单

```
agent/loop.py       # self._history 多轮记忆 + reset() + _record_usage（缓存统计）
llm/client.py       # _usage_dict() 透传 usage（标准字段 + model_extra 缓存字段）
cli.py              # REPL 多轮会话（「新会话」命令）+ 每轮显示缓存命中率
tests/test_loop.py      # +4 用例：多轮历史/reset/熔断不保存/usage 累计
tests/test_llm_client.py # +2 用例：usage 透传/无 usage 容错
NOTES/M2-step1-context.md  # 本文
```