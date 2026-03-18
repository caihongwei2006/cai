<p align="center">
  <img src="logo.png" alt="CAI Logo" width="200"/>
</p>

<h1 align="center">⚡ CAI — Concurrent Agent Intelligence<br>⚡ CAI — 并发智能体框架</h1>

<p align="center">
  <strong>
    High-performance Go agent framework: SOTA models plan, fast models execute.<br>
    高性能 Go 智能体框架：大模型规划，小模型执行，分脑协作。
  </strong>
</p>

---

## Why CAI? ／ 为什么选 CAI？

**The Problem:** SOTA LLMs (Claude Opus, GPT-5, Gemini Ultra) are powerful but slow (~30 tok/s) and expensive. Fast/local models (Qwen 2.5, LLaMA 3 8B) run at 100+ tok/s but lack complex reasoning. Every existing agent framework treats the model as a monolith — every step burns expensive tokens on trivial shell commands.

**现实问题：** SOTA 大模型能力强但慢且贵，快速小模型速度快但缺乏复杂推理。现有所有 Agent 框架都把模型当铁板一块——每一步都在用昂贵的 token 执行琐碎命令。

**CAI's Answer: Split the brain. ／ CAI 的方案：分脑架构。**

```
┌──────────────────────────────────────────────────────────┐
│                    CAI Agent ／ CAI 智能体                │
│                                                          │
│  ┌────────────┐    Intent (~20 tok)   ┌───────────────┐  │
│  │ Brain      │──────────────────────▶│ Cerebellum    │  │
│  │ 大模型规划  │                       │ 小模型执行     │  │
│  └────────────┘                       └───────────────┘  │
│        │                                    │            │
│        ▼                                    ▼            │
│  ┌────────────┐                       ┌───────────────┐  │
│  │ Triage     │◀──── error ───────────│ Engine        │  │
│  │ 大模型优化  │───── retry ─────────▶│ bash/py/js/as │  │
│  └────────────┘                       └───────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ IntentMemory (SQLite)                              │  │
│  │ JIT compile → cache → AOT reuse ／ 越用越快，越用越稳 │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

SOTA decomposes objectives into ~20-token Intent JSON. Fast model translates into scripts. Go manages concurrency, caching, lifecycle. **Terminal-native, not cloud-native.**

大模型将目标分解为 ~20 token 的 Intent JSON，小模型翻译为可执行脚本，Go 框架管理并发、缓存和生命周期。**面向终端，不面向云端。**

---

## Core Contracts ／ 核心接口

CAI is a **framework**, not a product. Four interfaces define the entire surface.

CAI 是**框架**而非产品，四个接口定义了全部表面积。

| Interface ／ 接口 | Role ／ 职责 | Implementor ／ 谁来实现 |
|-------------------|-------------|------------------------|
| `Planner` | Decompose objectives → `[]Intent` ／ 目标分解为意图 | You ／ 你 (or wrap SOTA LLM) |
| `ExecutorFunc` | Prompt + Intent → script output ／ 提示词+意图→脚本 | You ／ 你 (or wrap fast LLM) |
| `OptimizerFunc` | Analyze failures → better prompts ／ 分析失败→优化提示词 | You ／ 你 (or wrap SOTA LLM) |
| `HITLResolver` | Human-in-the-loop decisions ／ 人机协作决策 | You ／ 你 (CLI, Slack, auto…) |

---

## Execution Model ／ 执行模型

```
Trace (one objective ／ 一个目标)
  └── Span (one intent ／ 一个意图)
       ├── Epoch 1 [DEAD_END] ← failed, invisible to Brain ／ 失败，对 Brain 不可见
       ├── Epoch 2 [DEAD_END] ← retry with optimized prompt ／ 优化后重试
       └── Epoch 3 [SUCCESS]  ← Brain sees only this ／ Brain 只看到这个
```

| Feature ／ 特性 | Description ／ 描述 |
|-----------------|-------------------|
| **Trace → Span → Epoch** | Linear, append-only log — not a DAG ／ 线性只追加日志，不是 DAG |
| **N(1) Visibility** | Brain sees exactly 1 epoch, no O(N²) growth ／ Brain 只看 1 个 Epoch，无 O(N²) 膨胀 |
| **Dead-end spurs** | Failed epochs preserved for audit, invisible to agent ／ 失败保留但对 Agent 不可见 |
| **Forward-only Markov** | No branching, no graph GC, no Merkle trees ／ 无分支、无图 GC、无 Merkle 树 |

---

## Concurrency ／ 并发模型

Built on Go CSP — goroutines + typed channels, zero shared state.

基于 Go CSP 构建——协程 + 类型化 channel，零共享状态。

```
Brain ──▶ intentCh ──▶ Worker pool (N goroutines ／ N 个协程)
                            │
                            ├──▶ Engine (bash/py/js)
                            │
                       errorCh ──▶ Triage (independent goroutine ／ 独立协程)
                                        │
                                   hitlCh ──▶ HITL resolver
```

Workers execute intents concurrently. Triage runs independently — **never blocks the Brain**.

Worker 并发执行意图，Triage 独立运行——**永远不会阻塞 Brain**。

---

## JIT → AOT Prompt Compilation ／ JIT → AOT 提示词编译

```
1st run (JIT ／ 首次)：  Brain (SOTA) ──▶ compile prompt ──▶ save to SQLite
                                                                │
2nd run (AOT ／ 再次)：  Cache HIT ──▶ skip Brain ──▶ 0 SOTA tokens ／ 零大模型消耗
```

The system gets **faster with use**. Cache eviction by consecutive failures — not TTL.

系统**越用越快**。缓存淘汰由连续失败触发，而非基于时间。

---

## CAI vs LangChain-Go ／ 设计哲学对比

| Dimension ／ 维度 | CAI | LangChain-Go (tmc/langchaingo) |
|-------------------|-----|-------------------------------|
| **Model Arch ／ 模型架构** | Dual-tier: SOTA plans, FAST executes ／ 双层：大模型规划，小模型执行 | Monolithic: same model for all ／ 单层：所有步骤同一模型 |
| **Concurrency ／ 并发** | Structural: goroutines + channels ／ 结构性：协程+channel | Incidental: sequential chains ／ 附带性：顺序链 |
| **Context ／ 上下文** | O(1): 1 epoch visible ／ O(1)：只看 1 个 Epoch | O(N): history accumulates ／ O(N)：历史累积 |
| **Prompt Evolution ／ 提示词进化** | Auto: JIT→cache→AOT ／ 自动：JIT→缓存→AOT | Manual: dev manages templates ／ 手动：开发者管模板 |
| **Error Recovery ／ 错误恢复** | Built-in Triage goroutine ／ 内置 Triage 协程 | Manual retry logic ／ 手动重试 |
| **HITL ／ 人机协作** | First-class interface ／ 一等公民接口 | Not built-in ／ 未内置 |
| **Memory ／ 记忆** | Terminal-native SQLite, evolves ／ 终端原生 SQLite，自动进化 | Pluggable, stateless ／ 可插拔，无状态 |
| **Token Economics ／ Token 经济** | Intent ~20 tok, AOT amortizes to 0 ／ Intent ~20 tok，AOT 摊零 | Full cost per call ／ 每次全量成本 |
| **Philosophy ／ 哲学** | Compiler ／ 编译器 | Interpreter ／ 解释器 |

**The key insight ／ 核心差异：**

```
LangChain-Go:  SOTA → SOTA → SOTA → SOTA          (4× expensive ／ 4×昂贵)
CAI:           SOTA → FAST → FAST → FAST           (1× expensive ／ 1×昂贵 + 3×cheap)
               ↓ (cache ／ 缓存)
Next run:      FAST → FAST → FAST → FAST           (0× expensive ／ 0×昂贵)
```

---

## Benchmark ／ 基准测试

Real benchmark vs LangGraph (Python) & PicoClaw (Go serial), identical 2-intent scenario:

与 LangGraph (Python) 和 PicoClaw (Go 串行) 在相同场景下的实测对比：

| Framework ／ 框架 | Total (ms) | LLM Calls | Retry | Memory ／ 内存 |
|-------------------|-----------|-----------|-------|----------------|
| LangGraph (Python) | 5394 | 4 | 1 | ~5 GB |
| PicoClaw (Go serial) | 8746 | 5 | 1 | ~10 MB |
| **CAI (Go concurrent)** | **5549** | **4** | **1** | **~10 MB** |

> Single-request latency is dominated by LLM API (95%+). CAI's advantages compound at scale: parallel dispatch, non-blocking triage, O(1) context, 50x memory efficiency vs Python.

> 单请求延迟由 LLM API 主导 (95%+)。CAI 优势在规模化时复合叠加：并行分发、非阻塞 Triage、O(1) 上下文、内存效率比 Python 高 50 倍。

---

## Project Structure ／ 项目结构

```
cai/
├── cai.go              # Core runtime ／ 核心运行时
├── types.go            # Intent, Epoch, Span, Trace …
├── interfaces.go       # Planner, Executor, Optimizer, HITL
├── options.go          # Functional options ／ 函数式选项
├── tool_loop.go        # Self-iteration loop ／ 自迭代循环
├── defaults.go         # Default impls ／ 默认实现
├── cmd/                # CLI & demos ／ 命令行与示例
├── engine/             # Engines: bash, python, js, applescript
├── handler/            # Gin-inspired handler chain
├── hydrator/           # Template hydration ／ 模板注水
├── llm/                # OpenAI-compatible client
├── memory/             # SQLite IntentMemory + SystemEvolution
├── triage/             # Error analysis ／ 错误分析
├── prompt/             # Prompt scaffolding ／ 提示词脚手架
├── skill/              # Skill pack loader ／ 技能包加载器
├── envelope/           # Message classification ／ 消息分类
├── ipc/                # Inter-process communication ／ 进程间通信
└── data/               # Data collection ／ 数据采集
```

---

## Configuration ／ 配置

```go
agent, err := cai.New(ctx,
    cai.WithPlanner(myPlanner),             // Required ／ 必填
    cai.WithExecutor(myExecutor),            // Required ／ 必填
    cai.WithOptimizer(myOptimizer),          // Optional ／ 选填
    cai.WithHITL(myResolver),                // Optional ／ 选填
    cai.WithWorkers(4),                      // Worker pool ／ 工作池
    cai.WithMaxIterations(5),                // Iteration limit ／ 迭代上限
    cai.WithEvictionThreshold(3),            // Cache eviction ／ 缓存淘汰
    cai.WithMemoryDB(db),                    // SQLite backend
    cai.WithInitialPrompts(map[string]string{
        "list_files": "Output ONLY bash. No explanation.",
    }),
)
```

---

## License & Open Source Requirement ／ 许可证与开源要求

CAI is released under **AGPL-3.0**.

CAI 采用 **AGPL-3.0** 许可证发布。

> ⚠️ **Open Source Mandate ／ 开源强制条款：**
>
> Any product, service, or application built using CAI — whether distributed, deployed as SaaS, or offered commercially — **must be open-sourced** under a compatible license. This includes derivative works and internal tools serving external users. By using CAI, you agree to this requirement.
>
> 使用 CAI 框架构建的任何产品、服务或应用——无论分发、SaaS 部署还是商业化——**必须开源**，并采用兼容的开源许可证。包括衍生作品及面向外部用户的内部工具。使用 CAI 即代表同意此条款。

---

## Roadmap

- [ ] DAG-based intent dependency scheduling ／ 基于 DAG 的意图依赖调度
- [ ] Streaming execution output ／ 流式执行输出
- [ ] Built-in tool registry with schema validation ／ 内置工具注册与 Schema 校验
- [ ] Distributed agent coordination via IPC ／ 基于 IPC 的分布式 Agent 协调
- [ ] Plugin system for custom engines ／ 自定义引擎插件系统
- [ ] Metrics export (Prometheus/OpenTelemetry) ／ 指标导出

---

<p align="center">
  <strong>CAI — 越用越快，越用越稳</strong><br>
  <em>The more you use it, the faster and more stable it gets.</em>
</p>
