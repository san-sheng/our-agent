# M1 学习笔记 —— 第 1 步：LLM 客户端

> 日期：2026-08-02 ～ 08-03
> 内容：`llm/client.py` 从零理解（C++ 背景视角）
> 状态：✅ 完成（代码 + 测试 + 真实 API 验证）

---

## 0. 一句话总结

`llm/client.py` 是 Agent 的「大脑」接口：**把消息列表发给模型，拿回回复**。
它不关心循环、不关心工具——只做一件事，并处理好「对话可能失败」。

核心公式：**Agent = LLM + 上下文 + 工具**（书 Ch1）
直觉版：**大脑 + 眼睛 + 手脚**

---

## 1. Python 与 C++ 的思维转换（今天最大的认知升级）

### 1.1 `import` ≠ `#include`

| C++ | Python |
|---|---|
| `#include` 是预处理器**文本粘贴**，编译期定死 | `import` 是**运行时加载**，执行那个模块的全部代码 |
| 头文件只有声明，链接器找实现 | `.py` 文件声明+实现一体 |
| 需要 `#pragma once` 防重复粘贴 | 模块只加载一次，缓存进 `sys.modules` |

### 1.2 `__init__.py` 不是头文件

它是**包标记文件**：目录 + `__init__.py` = 包（namespace）。
三种用途：标记（空也行）/ 简化导入（提升子模块符号）/ 初始化（少用）。
我们项目里三个都是注释占位——**先最小化，需要时再加**。

### 1.3 `class` 是可执行语句

C++ 的类编译期定死；Python 的 `class` 是运行时语句，**执行到才创建类对象**。
openai 的 `APITimeoutError` 就是 import 时执行 `class APITimeoutError(...)` 创建的——没有注册表，纯运行时机制。

### 1.4 万物皆对象（连类都是）

`class LLMClient:` 执行后创建的是 `type` 的实例（metaclass 机制）——类对象本身也占内存、也有地址。
C++ 里类只是编译期概念，Python 里类是运行时对象。

---

## 2. 对象模型

### 2.1 `__init__` / `self`（vs C++ `this`）

- `self` = C++ 的 `this` 指针，但**显式写成第一个参数**（约定名，非关键字）
- `LLMClient(...)` 自动触发：`__new__`（分配）→ `__init__`（初始化）→ 返回实例
- `__init__` 不是构造函数！分配内存的是 `__new__`，`__init__` 只做初始化（≈构造函数体）
- 属性不是类里声明的，是 `__init__` 里 `self.xxx = ...` **动态挂上去的**

### 2.2 所有对象都在堆上

- Python 没有「栈对象」选项——全是堆分配，生命周期由引用计数决定
- 引用归零立即回收（≈ shared_ptr），循环引用由分代 GC 兜底
- 没有手动 delete，`__del__` 不保证时机（不是析构函数）
- 小整数 -5~256 是预分配单例（`x=1; y=1; x is y` → True）
- `__slots__` 去掉每个实例的 `__dict__`（56B → 40B）——M4 工具多起来再考虑

---

## 3. 异常系统（错误恢复的地基）

### 3.1 层次结构

```
BaseException                 ← 所有异常的根
├── Exception                 ← 我们用的：普通错误
│   ├── ZeroDivisionError / OSError / ...
│   └── openai.RateLimitError → APIStatusError → APIError → OpenAIError
├── SystemExit                ← 不是 Exception！
└── KeyboardInterrupt         ← 不是 Exception！
```

`except Exception` 故意接不住 SystemExit/KeyboardInterrupt——让退出/中断信号畅通。
C++ 类比：`std::exception` vs 信号（SIGINT）的隔离。

### 3.2 关键：`APIStatusError` 和 `APITimeoutError` 是**兄弟**不是父子

```
APITimeoutError → APIConnectionError → APIError
APIStatusError  → APIError（平级！）
```

语义：超时/连接错误 = 根本没等到 HTTP 响应（没有 status_code）；
`APIStatusError` = 服务器返回了 4xx/5xx（有 status_code）。
这就是 `_is_retryable` 里对 APIStatusError 查 `exc.status_code` 的原因。

### 3.3 `isinstance` 不是对比，是**沿类型链查找**

```
isinstance(err, APIStatusError)
= APIStatusError in type(err).__mro__   ← 在实例的类型链里找
```

`type(exc) == X` 是精确匹配（≈ `typeid`），`isinstance` 是层级匹配（≈ `dynamic_cast`）——面向未来，SDK 加子类也能覆盖。

### 3.4 异常对象和类的关系：**raise 之前就焊死了**

```
raise APITimeoutError(request=req) from err
     └─ ① APITimeoutError(...) 构造表达式 → 创建实例，__class__ 指向 APITimeoutError
       ② raise 把实例抛出（解释器只传递，不匹配、不注册）
```

实例是类「生」出来的，`type(exc)` 由创建它的类决定。isinstance 只是**查看**这个先天关系。

---

## 4. 错误恢复：重试（书 Ch1）

### 分层判断

```python
if isinstance(exc, APITimeoutError):    return True    # 特化在前
if isinstance(exc, APIConnectionError): return True    # 父类在后
if isinstance(exc, APIStatusError):
    return exc.status_code in {429, 500, 502, 503, 504}
```

- 可重试：超时/断连/限流/5xx（服务端可能恢复）
- 不可重试：401/400（请求本身错了，重试一万次一样）→ 直接 raise

### 指数退避 + 抖动

```python
delay = (2**attempt) + random.random()
```

- 指数退避：1s → 2s → 4s……
- 抖动（random）：多个请求不会「共振」同时重试，否则退避失效

### `raise ... from` 异常链

`from err` 把原始异常挂到 `__cause__`——调试时能看到「最终错误」+「最初原因」两层（C++ 得用 exception_ptr 自己包）。

---

## 5. 调用模型 API

### 5.1 同步阻塞

`client.chat.completions.create(**kwargs)` 是**同步阻塞**的（httpx 同步客户端）：
整个线程停在这一行直到响应/超时。C++ 类比：同步 socket recv。

- `APITimeoutError` 就是这一行等太久抛的
- M1 不做流式/异步（DESIGN.md §6.2 明确）——先跑通主线，再增强

### 5.2 `choices` 为什么是数组

协议允许 `n` 参数一次生成多个候选，所以响应格式固定为数组——即使 n=1 也是数组。
`Choice` 里有 `index` 字段（数组元素序号）。调用方永远写 `resp.choices[0]`。
**格式稳定，语义扩展靠参数**——协议设计原则。

### 5.3 返回统一格式（防腐层）

```python
return {"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls}
```

把 SDK 对象转成**自己的统一格式**，跟 openai 解耦——以后换 SDK 只改这个函数。
调用方（agent/loop.py）只认这三个字段。

---

## 6. 工程实践（少爷今天纠正的两个点 + 一个原理）

### 6.1 配置不进源码（v1 → v4 演化）

```
v1  key 硬编码源码        → 泄露风险 ❌
v2  key 进 config.toml    → 但 base_url/model 仍写死 ❌
v3  三项全进 config + 兜底 → 有隐藏默认值，新人困惑 ❌
v4  三项全进 config + 报错 → 配置即真相，缺什么说什么 ✅（当前）
```

**原则：可变的东西不进代码。** key/base_url/model 都在 `config.toml`（已 gitignore）。

### 6.2 fail-fast：白板报错，不设静默兜底

配置缺失 → `ValueError: 缺少配置: base_url。请在 ...config.toml 的 [llm] 段补齐...`
配置错误在**启动时**暴露，而不是带病运行。新人照着错误信息补就行。

### 6.3 `__file__` 资源定位

```python
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"
```

- 两个 parent：`client.py` 在 `llm/` 子目录，上两级才到项目根
- 用 `__file__` 而不是相对 cwd：**资源跟代码走，不跟运行目录走**（从 /tmp 跑也能找到）
- `.resolve()` 解析符号链接（本机 /home → /var/home 就是软链）

---

## 7. 学习点自查（DESIGN.md §6.1）

- [x] Agent 核心公式三件套对应代码里的哪个模块 → LLM=client.py，上下文=loop.py（M1 第 3 步），工具=tools/（M1 第 2 步）
- [x] ReAct 循环为什么是循环 → 工具结果必须回填给模型（M1 第 3 步验证）
- [x] 上下文五组件长什么样 → system/tools/user/assistant/tool（M1 第 3 步）
- [x] 工具描述为什么重要 → M1 第 2 步 tools/base.py 讲
- [x] 错误恢复：指数退避 + 抖动 → ✅ 已实现（llm/client.py）

## 8. 下一步

- M1 第 2 步：工具系统 `tools/base.py`（Tool 抽象 + registry + 3 内置工具）
- 书 Ch4：工具分类、描述的艺术、参数保真性
