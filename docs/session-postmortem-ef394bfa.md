# 会话复盘：ef394bfa-f3e1-424b-b4e3-918a14bacb1a

对历史会话 `ef394bfa-f3e1-424b-b4e3-918a14bacb1a` 及其派生的 5 个子代理会话的只读复盘记录。文中所有结论严格区分**库内及源码可证实的事实**与**工程推断**，引用事件均附带确切 `seq` 编号与子会话 ID。

---

## 一、会话概览与数据完整性

| 维度 | 事实记录 |
| :--- | :--- |
| Session ID | `ef394bfa-f3e1-424b-b4e3-918a14bacb1a` |
| 标题 | `1124｜EXP｜图片内容识别` |
| 工作区 (`cwd`) | `/Users/moguw/workspace/dsh-space/mizar` |
| 时间跨度 | `2026-11-24 11:53:14` → `14:47:04`（共 `2h53m`） |
| 模型与协议 | `magpie` / `isok/gpt-6.2-sol`（`openai-completions`，配置 `contextWindow = 922,000`） |
| 事件完整性 | 实际 `798` 条（`surface 264` / `log-only 534`），与 `sessions` 表预计算计数 `798 / 264` 完全一致，无写入丢失 |
| 上下文压缩 | `compacted_up_to_seq = null`（未触发 compaction，亦无 `tool-result-prune`） |
| 主会话规模 | 13 个 turn、130 步、117 次工具调用（`grep 48 / read 33 / subagent 10 / edit 10 / recall 9(2 错) / bash 5 / find 2`） |
| 主会话 Token 用量 | `in 260,196` / `out 18,990` / `cache-read 9,081,695` / `reasoning 2,373` |
| 结束状态 | Turn 13 第 45 步以 `step-end outcome=error` / `turn-end reason=error` 中断（`seq 797-798`） |

### 派生子代理会话汇总

| 子会话 ID | 名称 | 执行轨迹与结局 | 工具统计 / 异常 | Token 用量 (`in / out / cache-read / reasoning`) |
| :--- | :--- | :--- | :--- | :--- |
| `9bfa699a-458d-4886-a9a7-18b0025156f1` | `ntr-route` | Turn 1: 20 步 `max_steps`（77s）→ Turn 2: 4 步 `completed`（32.3s） | `read 11 / grep 8 / find 4`（0 错） | `67,480 / 2,069 / 849,280 / 72` |
| `72be8da3-45e2-49a3-9f03-ad6940935bf8` | `ntr-state` | Turn 1: 20 步 `max_steps`（70s）→ Turn 2: 4 步 `completed`（51.1s） | `recall 13(12 错) / read 5 / find 3 / grep 2`；触发 2 次 `GUARD_REPEAT_TOOL`（`seq 35`, `seq 79`） | `58,195 / 3,011 / 423,424 / 120` |
| `ed5a8697-0af7-44ee-91e7-6c6a28c19f32` | `inspect-pi` | Turn 1: 12 步 / 11 tools `completed`（1m24s） | `grep 8 / read 2 / find 1`（0 错） | `61,936 / 1,772 / 137,344 / 185` |
| `8ee0c73a-2a71-4c41-b5ee-2f5e81e0d623` | `inspect-dsh` | Turn 1: 15 步 / 14 tools `completed`（1m51s） | `grep 10(1 错) / find 3 / read 1` | `69,297 / 2,513 / 253,696 / 490` |
| `eebdb4c6-720c-4148-8f27-92496c332eb6` | `inspect-omp` | Turn 1: 14 步 / 13 tools `completed`（1m56s） | `grep 8 / find 3 / read 2`（0 错） | `76,128 / 3,009 / 239,360 / 305` |

---

## 二、问题分类总览

| 类别 | 问题点 | 严重度 | 归属层 |
| :--- | :--- | :--- | :--- |
| 一、工具契约与模型适配类 | `recall` 互斥参数与“全字段填充”冲突 | 严重 | 工具 Schema 与参数解析（`mizar-session`） |
| 一、工具契约与模型适配类 | 全程 100% 串行 Tool Call | 高 | 模型/网关行为与提示词/适配层边界 |
| 二、Subagent 编排与通知队列类 | 主代理回合交接失控与重复干活 | 高 | Subagent 控制流与系统提示词（`mizar-core`） |
| 二、Subagent 编排与通知队列类 | 通知队列滞后且未去重/清理过期通知 | 高 | Subagent `FollowUp` 通知队列（`mizar-core` / `mizar-tui`） |
| 三、上下文治理与性能类 | 外部 `references/` 目录 `AGENTS.md` 永久污染主会话 | 中 | 指令动态加载（`mizar-core::instructions`） |
| 三、上下文治理与性能类 | 无工具结果裁剪导致长会话后期 TTFT 劣化 | 中 | 上下文裁剪/压缩策略（`mizar-core`） |
| 四、可观测性与排查工具类 | 错误详情（`StepOutcome::Error` / `TurnEndReason::Error`）未落盘 | 高 | Trace 持久化（`mizar-core` / `mizar-session`） |
| 四、可观测性与排查工具类 | 复盘脚本误报 `turn 0` 进行中/未正常收尾 | 低 | 只读探针脚本（`.agents/skills/mizar-session-inspect`） |
| 五、运行时异常与遗留现场类 | Turn 13 step 45 流式中断与工作区半途状态 | 严重 | 运行时事故及当前本地 Git 工作区现场 |

---

## 三、五类问题详述

### 一、工具契约与模型适配类

#### 1. `recall` 互斥参数校验撞上“全字段填充”（严重）
- **可证实事实**：
  - 本次会话中，`isok/gpt-6.2-sol`（经由 `magpie` 网关）调用工具时习惯将 JSON Schema 中的全部可选属性填满默认值（例如 `find` 传 `pattern: ""`、`subagent` 传 `model: ""`）。
  - 在 `crates/mizar-session/src/recall.rs` 中，`recall` 的 schema 为 `seq` 定义了 `"minimum": 1`，且参数解析逻辑只要检测到 `args.get("seq").is_some()` 且 `query` 非空，即直接返回错误：`"Provide either \`query\` to search or \`seq\` to read, not both."`。
  - 子代理 `ntr-state`（`72be8da3`）试图用 `recall` 搜索 `"单主复制"` 时每次都顺带传入 `seq: 1, end_seq: 1`，导致 **13 次调用失败 12 次**（`seq 10..106`），期间在 `seq 35`、`seq 79` 两次触发 `GUARD_REPEAT_TOOL` 告警，浪费过半步数预算并直接撞上 `max_steps=20`。
  - 主会话在 Turn 8 同样因同时传入 `query` 与 `seq: 1` 连续报错 2 次（`seq 277-278`、`seq 283-284`），随后被迫将 `query` 置空，靠盲猜 `seq` 区间翻阅子会话（其间两次翻到无可读 surface 条目的空区间：`seq 289-290`、`seq 392-393`）。

#### 2. 全程 100% 串行 Tool Call（高）
- **可证实事实**：
  - 主会话全部 129 条 `assistant` 消息中，工具调用数量分布为 `{0: 12, 1: 117}`——117 次有工具调用的步骤均为单步单调用，未出现任何一次 `>= 2` 的并行工具调用。
  - 5 个子代理会话同样 100% 单步单调用（工具步数比分别为 `20/20`、`20/20`、`12/11`、`15/14`、`14/13`）。
  - 派发 3 个独立子代理（`seq 204/210/216`）、同一文件连续分段 `read`、多组独立 `grep` 均被拆成串行往返，直接拉长 Turn 13 至 45 步并使 `ntr-route` 撞上 `max_steps=20`；同时每多一步便重放一次完整前缀上下文，主会话 `cache-read` 累计达 **9,081,695 tokens**（5 个子会话另计约 190 万）。
- **推断**：
  - 仅凭 SQLite 会话库无法断定零并行究竟源于 `isok/gpt-6.2-sol` / `magpie` 网关的模型特性，还是提示词与适配层参数引导不足，需结合请求体日志进一步核对。
- **复核补充（详见第四节决策 6）**：按 `assistant.replay` 统计全库历史会话确认，同一 `openai-completions` 适配器下 `magpie/mayoru-responses/deepseek-flash` 并行率达 `45.0%`（`1,053/2,341`）、`magpie/mayoru-completions/gemini-4-hydrogen` 达 `100.0%`（`3/3`，单步 4~6 调用），而 `gpt-6.2-sol`（跨 `isok/` 与 `mayoru-responses/` 共 `608` 步）与 `gpt-6.2-astra`（`35` 步）合计为 `0/643`（`0.0%`）；由此证实 Mizar 并发执行与流式解析本身正常，零并行精准限定在 `gpt-6.2-*` 模型/上游路由侧。

---

### 二、Subagent 编排与通知队列类

#### 1. 回合交接失控与重复工作（高）
- **可证实事实**：
  - **Turn 4（`seq 51..177`，21 步）**：主代理在 `seq 57`、`seq 69` 通过 `subagent(action="send")` 分别唤醒 `ntr-route`、`ntr-state` 续跑输出总结后，并未结束自身回合等待结果，而是继续自行读/搜 `references/dsh-ntr` 源码 14 步，并在 `seq 153` 主动调用 `subagent(action="status")` 轮询。
  - **Turn 8（`seq 235..404`，28 步 / 5m04s）**：`inspect-pi` 先完成唤醒主代理；主代理在 `seq 271` 调用 `subagent(action="status")` 发现 `inspect-dsh` 和 `inspect-omp` 已进入 `completed`，但 `status` 返回摘要在 `seq 272` 被截断；主代理未等待 `FollowUp` 全文投递，转而自行 `grep/read` 对应源码 12 步，并手动调用 `recall` 分页翻看两个子会话 9 步。

#### 2. 通知队列滞后且未去重/清理过期通知（高）
- **可证实事实**：
  - Turn 4 于 `12:05:01` 结束后，1 秒后的 Turn 5（`seq 179`，`12:05:02`）投递了 `context-notice`（`source: "subagent:batch"`），其中仍包含 `ntr-state` 早先的 `max_steps` 旧通知，而此时 `ntr-state` 早已完成续跑并处于 `completed` 状态（`seq 154`）。
  - Turn 8 于 `14:10:27` 完成回答、用户在 Turn 9 指出主代理等待期间重复干活后，Turn 10（`seq 413`，`14:10:42`）才投递 `inspect-dsh` 与 `inspect-omp` 的 `subagent:batch` 完成通知，引发一轮无实质意义的收尾回合（Turn 10）。

---

### 三、上下文治理与性能类

#### 1. 外部 `references/` 目录 `AGENTS.md` 永久污染主会话（中）
- **可证实事实**：
  - Turn 3 step 2（`seq 31`）读取 `references/dsh-ntr/README.md` 后，`seq 33` 立即注入一条 `context-notice`（`source: "agent-instructions"`），将整份 `references/dsh-ntr/AGENTS.md`（**25,546 字符**）写入主会话 `surface` 历史。
  - 本次注入使步骤上下文由 Turn 3 step 2 的 `10,147` tokens（`in 1,827 + cache_r 8,320`）跃升至 step 3 的 `24,136` tokens（`in 14,152 + cache_r 9,984`），约增加 14K tokens。
  - `references/` 属于 `.gitignore` 排除的外部参考代码库；该规则文件不仅对 Turn 7~13 修改 Mizar 自身代码毫无适用性，还作为永久 `surface` 跟随了后续全部 125 步（仅此一段前缀在 `cache-read` 上即多消耗约 175 万 tokens），并引入跨仓库指令冲突风险。

#### 2. 无 `tool-result-prune` 导致长会话后期 TTFT 劣化（中）
- **可证实事实**：
  - `isok/gpt-6.2-sol` 配置 `contextWindow = 922,000`，会话峰值上下文 `134,911` tokens 未触及压缩阈值（符合当前配置行为）；但全程 `tool-result-prune = 0`，早期跨仓库探索累积的 200K+ `grep`/`read` 文本全程保留。
  - Turn 13 末尾多步的首字延迟（`step-start` → `first-token`）已拉长至 `11s~28.5s`（例如 `seq 753→754` 为 `13.7s`、`seq 759→760` 为 `13.2s`、`seq 777→778` 为 `11.6s`、`seq 789→790` 达 `28.5s`）。
  - > **复核修正（详见第四节决策 4）**：进一步按步核算 `ef394bfa` 的缓存与延迟后确认，Turn 7~13 的 `cache_read` 命中率始终为 `97.8%~99.7%`，最慢的是上下文仅 85K~90K 的 Turn 12（均值 `26.8s`、峰值 `54.3s`，命中率 `99.7%`），上下文更大的 Turn 13（91K~135K）均值反而回落至 `10.6s`；20s+ 尖刺主要源于上游排队/推理抖动，而非缺少深层历史裁剪。

---

### 四、可观测性与排查工具类

#### 1. 错误详情未持久化到 SQLite（高）
- **可证实事实**：
  - `crates/mizar-core/src/agent_loop.rs` 在记录轨迹时仅写入 `outcome` / `reason` 的枚举标签字符串（`"error"`），`StepOutcome::Error { message }` 与 `TurnEndReason::Error { message }` 中的 `message` 只存在于内存事件流，未落盘到 `session_events`。
  - 因此在会话库中仅能确认 Turn 13 step 45 失败，无法通过离线复盘确定具体报错内容。

#### 2. 复盘脚本误报 `turn 0`（低）
- **可证实事实**：
  - `.agents/skills/mizar-session-inspect/scripts/session.py` 的 `turn_rollup()` 函数以 `current_turn = 0` 初始化，并将不携带 `detail.turn` 的 `log_only` 事件 `seq 1`（`selection`）归入 `turns[0]`，因 `turn 0` 永远不会匹配到 `turn-end`，输出报告会显示一行虚假的 `turn 0: 0 步 / 0 次工具调用 / 进行中或未正常收尾`。
  - 该现象属于探针归类逻辑瑕疵，并非会话数据库本身存在未闭合 turn 或写入损坏。

---

### 五、运行时异常与遗留现场类

#### 1. Turn 13 step 45 流式报错中断（严重）
- **可证实事实**：
  - Turn 13（`seq 529..798`，共 45 步 / `10m33s`）用于实现 `subagent yield`。
  - 结尾时序：`seq 795`（`14:46:55.067` `step-start`）→ `seq 796`（`14:47:04.224` `first-token`，等待 `9.16s`）→ `seq 797`（`14:47:04.226`，**仅 2ms 后** `step-end outcome=error`）→ `seq 798`（`turn-end reason=error`），无 `assistant` 消息落盘。
  - **事实边界声明**：如第四类所述，错误原文未落盘，库内无法证实 Turn 13 step 45 的精确失败原因；仅从“收到 `first-token` 后 2ms 立即失败”的时序特征**推断**，大概率为上游/网关在流式建连后返回了错误 chunk 或发生 SSE 断流，需结合当时终端输出或 `magpie` 网关日志交叉印证。

#### 2. 工作区遗留半途改动现场（严重）
- **可证实事实**：
  - Turn 13 在 `seq 787-788` 格式化并运行了 2 条定向集成测试（`yield_returns_control_before_child_settles`、`yield_keeps_followup_that_arrives_before_parent_finishes`），在 `seq 793-794` 刚检索 `crates/mizar-tui/src/runner.rs` 准备继续核对/补充 TUI wake 测试时即告中断。
  - 当前工作区保留着 Turn 13 未收尾的代码修改，尚未执行全量 `cargo test --locked --workspace`、`cargo clippy --locked --workspace --all-targets -- -D warnings` 及 `cargo fmt --all -- --check`。

---

## 四、设计决策记录

### 决策 1：彻底移除 `recall` 工具与 FTS5 索引（方案 A）

- **结论**：直接移除 `recall` 工具与 `session_recall_*` FTS5 索引表，不再修补其参数 schema，不保留向后兼容。
- **决策依据（全库实测事实）**：
  - **设计目标 0 命中**：全库 36 个非空会话中仅 1 个触发过 compaction（`57a52f9f`，2,992 条事件，`compacted_up_to_seq = 1744`），其 `recall` 调用次数为 `0`。
  - **实际调用 100% 错位或报错**：全库累计 34 次调用分布在 3 个未压缩会话（`0ce2b9fe` 12 次、`ef394bfa` 9 次、`72be8da3` 13 次），结果 `ok=12 / empty=4 / error=18`（52.9% 报错，全为同一 `query`/`seq` 互斥校验错误）；其中主会话 21 次均用于盲猜 `seq` 强翻子代理会话，子代理 `72be8da3` 13 次误将 `recall` 当作项目知识检索。
  - **维护与存储成本失衡**：核心实现 2,020 行（`crates/mizar-session/src/recall.rs` 1,199 行 + `crates/mizar-session/src/sqlite/recall.rs` 821 行）及跨 crate 接线；`~/.mizar/sessions.sqlite`（48.78 MB）中 `session_recall_*` 表与索引占 **20.89 MB（42.8%）**；每次落盘事件需同步计算 1..=3 n-gram 写入 FTS5，且以 `ToolExposure::Direct` 常驻主/子代理工具声明表。
- **替代方案与职责归位**：
  - **主代理读取子代理结果**：归位给 `subagent` 工具（`yield` 等待交付 + `status` 查询单个终态子代理时返回完整输出或落盘路径）。
  - **极低频压后原文回查**：后续若确实出现痛点，改为 compaction 时按需导出被压缩区间为纯文本/JSONL 文件并将路径写入 `<mizar-compaction>`，复用现有 `grep`/`read` 检索，不维护 FTS5 与常驻工具。
  - **跨历史会话复盘**：由外置按需 skill（`mizar-session-inspect`）承担。
- **落地清理范围**：
  - `mizar-session`：删除 `src/recall.rs`、`src/sqlite/recall.rs` 及对应测试；清理 `model.rs`、`persistence.rs`、`sqlite.rs`、`sqlite/schema.rs`、`sqlite/writer.rs`、`lib.rs` 中 recall 相关代码；SQLite 启动迁移执行 `DROP TABLE IF EXISTS session_recall_fts; DROP TABLE IF EXISTS session_recall_docs;`。
  - `mizar-core` / `mizar-client` / 文档：清理 `spine.rs`、`spine/plugins.rs`、`spine/config.rs`、`spine/config/settings.rs`、`system_prompt.rs`、`compaction/summarizer.rs`、`compaction/service.rs`、`subagent/guardrails.rs`、`mizar-client/src/approval.rs`、相关集成测试，并同步更新 `docs/plugins.md`。

### 决策 2：Subagent 软预算收尾、yield 交接、通知去重与完整结果交付

- **结论**：保留 Mizar 现有 UNIX 后台分身模型（`spawn` / `send` / `status` / `stop` / `yield` + 45s 零 Token 活性看门狗 + `SessionInbox`），围绕「长任务不被矮上限打断、交接无黑洞无死锁、多子代理一次唤醒、通知不滞留、终态全文可直读」落地 5 项设计。
- **五项具体决策**：
  1. **预算与超时（借鉴 `references/omp` 裁剪落地）**：
     - 移除 `default_max_steps = 20`（`max_steps_cap = 40`）硬截断与 `timeout_secs = 180` 短墙钟硬超时。
     - 改为按模式的宽松软步数预算（`read_only = 100`、`edit = 200`，配置 `0` 表示关闭无上限）与两阶段收尾：达到 `1.0×` 软预算时向子会话注入 `Steer` 提醒收尾；达到 `1.5×` 时停掉自由工具循环，并自动以空工具集执行最后 1 步强制总结交付已有发现（兜底硬杀仅在强制收尾仍未收敛后触发）。
     - 墙钟硬超时 `timeout_secs` 默认改为 `0`（关闭硬杀），卡死/停滞交由 45s 活性看门狗 + 软预算兜底；不引入 `omp` 的 IRC/workpool/structured yield 重型封装。
  2. **`yield` 补齐两处现存缺口**：
     - `subagent(action="yield", message="...")` 触发 `end_turn` 时，须将 `message` 作为正常助手文本发射并写入 `SessionEvent::Assistant`（避免仅留在 `ToolResult` 或被 TUI `UiEvent::RunFinished(Ok(_))` 丢弃）。
     - 执行前检查运行中子代理数量与父信箱待投递 `FollowUp`：若 `running == 0` 且无待投递回执，拒绝 `end_turn` 并返回普通提示让模型继续回答，防止空等死锁。
  3. **`yield` 并发唤醒 barrier 语义**：
     - 主代理主动调用 `yield` 后，默认等待当前在跑的一批子代理全部结算后再一次性打包唤醒，避免多个并发子代理依次结算导致主上下文反复重放。
     - 一旦出现任一子代理失败、看门狗告警或人类用户插话，立即打破 barrier 提前唤醒。
  4. **收件箱结构化去重（借鉴 `references/omp` `YieldQueue`）**：
     - 将 `SessionInbox` 中 `subagent:settled` / `subagent:watchdog` 批次从纯字符串拼接升级为按 `(child_session_id, run_id)` 维护的结构化槽位，flush 投递前按最新代次过滤过期项再拼装正文。
     - 在 `send` 续跑、`stop` 取消、重新结算或 `status(name)` 读取全文时自动失效/撤回该子代理旧条目；槽位全空时移除整个批次条目，不触发空唤醒。
  5. **终态完整输出交付（承接删除 `recall`）**：
     - `ManagedTask` 保留最近一次终态完整输出 `last_output`（参考 `omp` 内联阈值 `12,000` 字符，超出自动落盘临时文件并附 `5,000` 字符预览与文件路径）。
     - `status` 无 `name` 列表保持 120 字紧凑摘要；`status(name="...")` 查询单个终态子代理时返回完整输出（或预览 + 文件路径），并同步消费该子代理未投递的 `settled` 槽位。

### 决策 3：将 `instructions` 简化为启动静态加载，移除子目录动态注入（方案 B）

- **结论**：参考 `references/omp` 与 `references/pi`，将 `mizar-core::instructions` 从动态 `ContextNotice` 增量注入 + 每步 stat 热感知机制，彻底简化为**启动时一次性读取并拼入 `SystemPrompt` 静态段**；仅加载全局配置目录（`~/.mizar/`、`~/.agents/`）与项目根目录（`.git`）到当前 `workspace` 链上的 `AGENTS.md` / `CLAUDE.md`（及 `.local.md`），不保留向后兼容。
- **决策依据**：
  - **根除外部目录污染**：不再向下追踪工具访问的子目录，从根源上杜绝 `references/`、`node_modules/`、`target/`、`vendor/` 等外部 `AGENTS.md` 混入主会话，无需再引入 `.gitignore` 过滤补丁。
  - **大幅削减偶然复杂度**：现有 `crates/mizar-core/src/instructions/` 共 2,438 行（单 `state.rs` 占 1,002 行），核心复杂度几乎全部来自 `follow_subdirs`（`tools.on_result` 监听、`descendant_dirs_between`、`pending_subdirs`/`known_subdirs`）、每步 `reconcile_pre_step_async` 文件指纹 stat 与 `ContextNotice` 增量 diff/压后重建状态机；改为启动静态加载后可缩减至 ~300 行。
  - **运行时收益**：每步零文件 stat IO，KV Cache 前缀 100% 稳定，且不再因会话中途触发 compaction 而额外维护基线重建逻辑（唯一取舍为中途编辑 `AGENTS.md` 需新开/重启会话生效）。
- **落地清理范围**：
  - 移除 `follow_subdirs` 配置项（`spine/config.rs`、`settings.rs`、`instructions/config.rs`）、`tools.on_result` 钩子、`agent_loop.pre_step_hook` 增量对账、`descendant_dirs_between`、文件指纹热监测与 `ContextNotice` 渲染/去重状态机；
  - 保留全局目录与项目祖先链发现、去重与单源/总字节预算裁剪，通过 `SystemPrompt::section` 静态注入（保留 `SpinePlugins` 运行期模块开关语义），并同步更新相关测试与 `docs/plugins.md`。

### 决策 4：引入 Cache-Aware 尾部剪枝（方案 B，借鉴 `references/omp` `pruning.ts`），拒绝按年龄深层裁剪

- **结论**：不引入会炸毁热前缀缓存的「按年龄深层 `tool-result-prune`」；借鉴 `references/omp`（`packages/agent/src/compaction/pruning.ts` 与 `session-maintenance.ts`），改为引入**严格 Prompt-Cache-Aware 的轻量尾部剪枝**（`supersedeReads` + `dropUseless`），以 append-only 事件持久化确保 `--resume` 前缀字节一致。
- **决策依据（实测与缓存物理约束）**：
  - `ef394bfa` 实测复核：Turn 7~13 缓存命中率 `97.8%~99.7%`（即使 Turn 13 距上轮间隔 `1083s`/18 分钟仍命中 `99.0%`）；Turn 12（85K~90K）TTFT 均值 `26.8s` / 峰值 `54.3s`，Turn 13（91K~135K）均值 `10.6s`——热缓存期间若原地裁剪深层 `tool_result` 会导致其后 100K+ 前缀全部 cold-miss，账单与冷 prefill 延迟双双恶化。
  - 真正值得实时消除的是热尾内的确定性死重（如 `read → edit → read` 产生的旧文件副本，及 `grep` 0 命中等无信息量结果），且必须严守前缀缓存边界。
- **核心设计契约**：
  1. **仅剪两类确定性死重**：同文件被更新且完整的 `read` 覆盖的旧 `read` 结果（替换为 `[Superseded by a newer read of this file]`）；以及工具明确标记为无信息量的结果如 0 命中检索（替换为 `[Uneventful result elided]`），保护 `skill` 等特殊读取。
  2. **Prompt Cache 红线**：热缓存期间仅允许裁剪**后续消息总 token `<= 8,000`** 的尾部候选（已沉入深层热前缀的结果一律留待 compaction 统一重建）；前缀绑定思考模型直接将热尾上限置 `0`（不改写热前缀）。
  3. **冷缓存全段清仓**：仅当会话闲置超过 `90 分钟`（确保覆盖 1 小时档 prompt cache TTL）时，才对最近一次 compaction 边界之后的所有可见候选做一次全段清理。
  4. **Append-only 持久化保证 resume 一致**：不在 SQLite 原地覆写旧事件正文，而是追加结构化剪枝标记事件（如 `ToolResultPruned { target_seq, notice }`），构建模型消息流时按标记投影替换，确保实时会话、`--resume` 与 fork 重建出字节完全一致的前缀，不触发 cold-miss。
  5. **大窗口压缩阈值**：继续保持 `compaction.max_context_tokens` 默认 `None`（尊重模型真实窗口与热缓存），后续可按需支持模型级阈值覆盖。

### 决策 5：持久化错误摘要到轨迹事件，并修复 `session.py` `turn 0` 误报

- **结论**：在 `SessionTraceEvent::StepEnd` 与 `TurnEnd` 增加可选 `error` 字段落盘截断错误原文，同时修复复盘脚本 `turn 0` 假阳性并同步全局 skill。
- **具体落点**：
  1. `crates/mizar-session/src/model.rs`：为 `SessionTraceEvent::StepEnd` 与 `TurnEnd` 新增 `#[serde(default, skip_serializing_if = "Option::is_none")] error: Option<String>`（正常成功不序列化，兼容旧库 JSON，保持 `surface` / `log_only` 隔离）。
  2. `crates/mizar-core/src/agent_loop.rs`：`StepGuard::drop` 与 `TurnGuard::drop` 在 `StepOutcome::Error { message }` / `TurnEndReason::Error { message }` 时按 UTF-8 字符边界截断（上限 `2,000` 字节）写入 `error: Some(...)`。
  3. `.agents/skills/mizar-session-inspect/scripts/session.py`（并同步 `~/.agents/skills/mizar-session-inspect/scripts/session.py`）：在 `turn_rollup()` 中跳过 `selection` 等无 `turn` 属性的会话级元数据事件（消除虚假 `turn 0`），并在 `stats`/`show` 渲染 `step-end`/`turn-end` 时展示 `detail.error`。

### 决策 6：补齐 `compat.parallel_tool_calls` 可选透传与静态并行提示词引导

- **结论**：对照全库分模型统计与 `references/omp` 实践，不在标准端点盲目硬发 `parallel_tool_calls`，改为通过 `compat` 可选字段按需透传并在 `DEFAULT_PERSONA` 补充静态并行引导；不做针对单个串行模型的重型绕路 hack。
- **具体落点**：
  1. `crates/mizar-llm`（`router/config.rs`、`completions.rs`、`responses.rs`）：新增可选配置 `compat.parallel_tool_calls: Option<bool>`；仅当显式配置且请求携带 `tools` 时输出 `parallel_tool_calls` 字段（未配置时不发送，避免不识别该字段的第三方代理报 400）。
  2. `crates/mizar-core/src/system_prompt.rs`：在静态 `DEFAULT_PERSONA` 增加一句精简并行引导（参考 `omp` `system-prompt.md`：`batch independent read-only reads/searches/checks in a single step`），零缓存副作用。
  3. **边界声明**：补齐后对 `gpt-6.2-sol` 实测一次以判定属于缺显式字段还是上游硬锁；若为上游硬锁则直接接受，不再在客户端增加额外复杂度（决策 2 的 100/200 软预算已避免串行子代理被提前截断）。

### 附：六项决策与 `references/omp` 全景对照

| 议题 | Mizar 决策 | `references/omp` 做法 | 取舍评价 |
| :--- | :--- | :--- | :--- |
| 1. `recall` 去留 | 彻底删除（含 FTS5 表与 2,020 行代码）；子代理结果归位 `subagent`，跨会话复盘交外部 skill | 核心无常驻 FTS5 `recall` 工具（历史记忆由外置 MCP `mnemopi` 按需挂载） | 理念对齐：核心轻装，不养 0 命中重组件 |
| 2. Subagent 治理 | 软预算（`100/200`，`0` 禁用）+ `1.0× Steer` + `1.5×` 空工具集总结 + `timeout=0`；`yield` 落盘文本/防空等/barrier；收件箱按 `(child_session_id, run_id)` 去重；`status(name)` 12K 内联/落盘 | 软预算（`scout=100` / `default=200`）+ `1.0× steer` + `1.5×` 强制收尾 + `maxRuntimeMs=0`；`YieldQueue.isStale` 延迟过滤；12K 内联 + 5K 预览；另有 IRC mesh / workpool / `wait` 工具 | 精准取精：吸收软预算与延迟去重，不引入重型协同外壳 |
| 3. `AGENTS.md` 加载 | 启动静态加载进 `SystemPrompt`（仅配置目录 + 项目根到 `cwd` 链），删子目录动态发现与每步 stat | 启动时 walk-up 静态加载（用户配置 + 项目根到 `cwd`），不做读文件动态注入，且明文禁止模型乱搜 | 完全对齐 `omp` / `pi`，顺手砍掉 ~2,100 行偶然复杂度 |
| 4. 尾部工具结果剪枝 | Cache-aware `supersedeReads` + `dropUseless`（`<= 8K` 热尾、`>= 90m` 冷清、前缀思考模型置 `0`），以 append-only 标记事件持久化保 `--resume` 一致 | `pruning.ts` 同款红线（`8K` / `90m` / `prefixBinding => 0`），通过 `rewriteEntries` 持久化；另有 `cache-warmer`、`snapcompact` 转图等实验特性 | 对齐核心缓存红线，用 append-only 适配 Mizar SQLite，不照搬重实验特性 |
| 5. 错误详情落盘 | `StepEnd` / `TurnEnd` 加可选 `error`（2KB UTF-8 截断）+ 修 `session.py` `turn 0` | `assistant` 持久化带 `errorMessage`（span 带 `upstreamError`） | 等价覆盖，同时保持 `surface` / `log_only` 隔离 |
| 6. 并行 Tool Call | `compat.parallel_tool_calls: Option<bool>`（未配不强发）+ `DEFAULT_PERSONA` 一句静态并行引导，不做单模型 hack | 标准 OpenAI 适配不盲发（仅特定路由/Codex Lite 按 policy 发）；`system-prompt.md` 明确引导单步批量并行 | 对齐 `omp`：避免兼容网关 400，补齐并行提示词引导 |

---

## 五、后续修复优先级清单（Checklist）

### Phase 1：内核收敛与问题修复（前置，优先执行）

- [x] **彻底移除 `recall` 工具与 `session_recall_*` FTS5 索引（已决策：方案 A）**：按第四节决策 1 清理 `mizar-session` / `mizar-core` / `mizar-client` / `docs/plugins.md` 与 SQLite 历史表。
- [x] **持久化错误摘要到轨迹事件（已决策：决策 5）**：在 `SessionTraceEvent::StepEnd` / `TurnEnd` 增加可选 `error: Option<String>`（`StepGuard::drop`/`TurnGuard::drop` 按 2,000 字节 UTF-8 安全截断写入），并在 `session.py` 中展示 `detail.error`。
- [x] **落地 Subagent 编排与预算治理（已决策：决策 2）**：
  - 补齐 `yield` 助手消息发射/落盘、无在跑/待投递回执防死锁校验与并发 barrier，并完成全量 CI 检查（`test` / `clippy` / `fmt` / `--no-default-features`）。
  - 用按模式软步数预算（`read_only=100` / `edit=200`，`1.0× Steer` + `1.5×` 空工具集强制总结）与默认 `timeout_secs=0` 替换 `max_steps=20` / `180s` 硬截断。
  - 将 `SessionInbox` 的 `subagent:settled` / `subagent:watchdog` 改造为按 `(child_session_id, run_id)` 维护的结构化槽位，支持 `send`/`stop`/重结算/`status(name)` 撤回与去重。
  - 为 `ManagedTask` 增加终态全文保留及 `12,000` 字符超限落盘（附 `5,000` 字符预览），让 `status(name="...")` 返回完整输出并消费未投递回执。
- [x] **简化 `instructions` 为启动静态加载（已决策：决策 3 / 方案 B）**：按第四节决策 3 移除 `follow_subdirs`、子目录动态发现、每步 stat 热感知与 `ContextNotice` 状态机，改为启动时仅读取配置目录与项目根到 `workspace` 链并注入 `SystemPrompt` 静态段，同步清理配置项与 `docs/plugins.md`。
- [x] **落地 Cache-Aware 尾部剪枝（已决策：决策 4 / 方案 B）**：按第四节决策 4 在 `mizar-session` / `mizar-core::compaction` 实现 `supersedeReads` + `dropUseless`（热尾 `<= 8,000 tokens`、冷缓存 `>= 90m` 全段清、append-only 剪枝标记事件持久化以保证 `--resume` 前缀一致）。
- [x] **补齐并行 Tool Call 基础能力（已决策：决策 6）**：在 `mizar-llm` 增加 `compat.parallel_tool_calls: Option<bool>` 按需透传，并在 `DEFAULT_PERSONA` 补充静态并行引导（基础能力已合入；待在线实测 `gpt-6.2-sol` 是否解锁）。
- [x] **修复复盘脚本 `turn 0` 假阳性（已决策：决策 5）**：在 `.agents/skills/mizar-session-inspect/scripts/session.py` 的 `turn_rollup()` 中跳过 `selection` 等无 `turn` 属性的会话级元数据事件，并同步到 `~/.agents/skills/mizar-session-inspect/`。

### Phase 2：`herdr-app` 多端接入（详见第六节，Phase 1 收尾后启动）

- [x] **M1（`mizar-client::SessionActor` 内核解耦）**：将 `mizar-tui/src/runner` 私有控制态、`ActiveTurnBuffer`、`PendingInteractions`（`request_id -> oneshot`）、唯一 `queued_inputs`、`FollowUp` 唤醒循环与 `SubagentTaskSnapshot` 广播下沉至 `mizar-client`，`mizar-tui` 退化为薄客户端并回归全部测试。
- [x] **M2（Per-PID Socket 与核心 RPC 闭环）**：落地 `~/.mizar/run/p-<pid>.sock`（`0600`）与 `p-<pid>.json`（含 `ppid`/`tty`/`herdr_pane_id` 及单写者会话锁）、补齐 OSC 7501 `pid` 字段，实现核心 RPC（`session.subscribe`、`turn.submit`、`turn.cancel`、`interaction.resolve`、`session.config`、`session.events`）、`40ms` delta 合并与 `256` 帧慢客户端断连保护。
- [x] **M3（持久化渲染质感升级）**：将 `ToolResult.details` 精选字段持久化为 `tool-result.meta`（`derive_messages()` 忽略，零上下文污染），新增 `approval`（`LogOnly`）事件与 `turn-end.summary` 纯规则统计，补齐 `capabilities.list` 与 `session.fork`/`tree`。
- [x] **M4（前端接入与按需远程化）**：
  - M4.1：为 `mizar-client` 增设 `127.0.0.1:0` 认证回环 TCP 旁路、`0600` `p-<pid>.json` `rpc_token`（`ProcessRegistration::without_secret()` 脱敏）及首帧常量时间 `auth.hello` 校验（失败返回 `-32003` 并断连）。
  - M4.2：落地 `mizar serve` 回环 HTTP/WebSocket 网关（仅允许回环地址绑定 + 严格 `Origin` 校验 `/403`、`/health`、脱敏 `/api/processes`、`/api/sessions`、`/ws` 代理活 `p-<pid>.sock` 或按需拉起静态 SQLite 会话 headless `SessionActor`，远程可叠加 Tailscale Serve）。
  - M4.3：在 `~/workspace/herdr-mayonaka`（分支 `feat/mizar-mod`）建成原生 GPUI `mayonaka-mizar` 模块并接入 `layouts/mayonaka.json`、`mayonaka-tabs`（`⌘⌥M`）、`mayonaka-palette` 与 `mayonaka-pane` 状态标识（注：第四节决策 6 的 `gpt-6.2-sol` 线上实测仍待有真实凭据时交互验证）。

---

## 六、演进规划：与 `herdr-app` 结合的会话内核与多端接入架构（修正版）

本节在原始「会话内核 + 双通道 + 多客户端」接入提案基础上，结合 Mizar 现有代码（`mizar-client`、`mizar-tui`、`mizar-tui-ui`）及第四节已拍板决策完成架构对齐与裁剪。**执行前置条件**：先完成第五节 Phase 1 的内核瘦身，再启动本节 Phase 2（M1~M4），避免在拆 Actor 过程中搬运即将废弃的 `recall` 与动态 `instructions` 代码。

### 1. 总体架构与 Crate 归位

核心原则：TUI 降级为会话内核的一个进程内本地客户端（in-proc 零序列化），与 `herdr-mayonaka`（Unix Socket / 认证回环 TCP）、Merak Desktop（直嵌）及 `mizar serve`（Web / 手机 PWA）平等地连到同一份 `SessionActor` 运行时。

```text
  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
  │    Mizar TUI     │   │  herdr-mayonaka  │   │    Web / PWA     │
  │   (终端薄前端)    │   │   (Herdr Mod)    │   │   (mizar serve)  │
  └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
           │ in-proc              │ Socket / 127.0.0.1   │ WS (via mizar serve)
           ▼                      ▼                      ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │              mizar-client::SessionActor (零终端依赖)              │
  │  • CommandSink (submit / cancel / resolve / config / fork)       │
  │  • EventBus (Ephemeral Delta + Committed Events + Subagents)     │
  │  • ActiveTurnBuffer (内存中当前未落盘的流式缓冲)                   │
  │  • PendingInteractions (HashMap<RequestId, oneshot::Sender>)     │
  │  • 单一真相源 queued_inputs + FollowUp 唤醒调度                   │
  └────────────────────────────────┬─────────────────────────────────┘
                                   │ 落盘提交 (append-only, seq 递增)
                                   ▼
                     ~/.mizar/sessions.sqlite (WAL)
```

- **Crate 归位（严守 `AGENTS.md` 边界，不新造 crate）**：
  - `mizar-types`：存放跨端 JSON-RPC 2.0 协议、Snapshot / Notification / Delta 纯数据契约（零异步与终端依赖）。
  - `mizar-client`（现有 4,956 行零终端依赖宿主层）：承载 `SessionActor`、`ActiveTurnBuffer`、`PendingInteractions`（替换现 `approval.rs` 单消费者传递 `oneshot`）、唯一 `queued_inputs`（消除现 `AppState` 与 `AgentLoop::inbox()` 双头队列脑裂）、`FollowUp` 唤醒循环、`SubagentTaskSnapshot` 广播及可选 Unix Socket `rpc` 服务。
  - `mizar-tui`（现有 `runner.rs` + `runner/actions.rs` 共 3,860 行）：退化为终端输入、布局渲染、OSC 7501/9;4 输出与本地快捷键壳，不再直接持有 `active_session`、`RunController` 与业务调度状态。

### 2. 会话发现、单写者锁与 Socket 拓扑（平时零常驻进程）

采用「**Per-Process Unix Socket + 可选 `mizar serve` 按需聚合**」，平时不留后台守护进程：

- **运行目录**：`~/.mizar/run/`（目录权限 `0700`）。
- **每活进程文件**：
  - Socket：`~/.mizar/run/p-<pid>.sock`（权限 `0600`），按进程 PID 命名，确保 TUI 执行 `/resume` 或 `/new` 切换会话时不断连，仅广播 `session.switched` 并原子更新元信息文件。
  - 元信息文件：`~/.mizar/run/p-<pid>.json`（权限 `0600`，写临时文件后 `rename` 原子替换，正常退出时删除；前端通过 `kill(pid, 0)` 自动忽略并清理残留死文件，对外 `/api/processes` 使用 `without_secret()` 剥离 `rpc_token`）：
    ```json
    {
      "pid": 48210,
      "ppid": 48190,
      "tty": "/dev/ttys004",
      "herdr_pane_id": "pane_...",
      "session_id": "01J...",
      "cwd": "/Users/moguw/workspace/herdr-mayonaka",
      "socket": "/Users/moguw/.mizar/run/p-48210.sock",
      "tcp_addr": "127.0.0.1:49201",
      "rpc_token": "<64-hex-token>",
      "provider": "magpie",
      "model": "isok/gpt-6.2-sol",
      "started_at_ms": 1763971200000
    }
    ```
  - 补充 `ppid`、`tty` 与可选 `herdr_pane_id` 的原因：当终端经过 wrapper shell / `mise` / `tmux` 启动 `mizar` 时，Herdr 读取到的 pane 前台 PID 可能并非 `mizar` 自身，多维度定位可避免 `lsof` 扫描歧义。
- **OSC 7501 对齐**：`crates/mizar-tui-ui/src/terminal.rs` 已稳定发射冒号分隔的 `state=...:app=mizar:kind=...:msg=...`，后续按需补充 `pid=<pid>` 字段即可实现 Herdr 秒级精准关联。
- **单写者红线（Single-Writer-Per-Session）**：SQLite WAL 支持多进程并发读，但**同一 `session_id` 任一时刻仅允许一个活进程持有 `SessionActor` 写权**（防止 `seq` 乱序与 `SessionInbox` 分裂）。另一终端尝试 `/resume` 已被活 `p-<pid>.json` 声明持有的会话时应检测并拒绝双写（或降级为只读 attach）；未来 `mizar serve` 对活会话的所有写控制必须代理到对应 `p-<pid>.sock`，仅对无活宿主的静态会话按需拉起 headless `SessionActor`。

### 3. JSON-RPC 2.0 通信协议（NDJSON over Unix Socket / WebSocket）

每一行（或每帧）一个 JSON 对象，严格区分「已落盘提交事实（`seq`）」与「内存瞬态流」。

#### A. 订阅与断线重连（`session.subscribe`）

- 请求：`{ "jsonrpc": "2.0", "id": 1, "method": "session.subscribe", "params": { "session_id": "01J...", "after_seq": 120 } }`
- 响应 `result` snapshot（保证中途打开界面立即可见当前生成现场与子代理状态）：
  - `session`: `{ id, cwd, title, parent_id, fork_seq }`
  - `state`: `"idle" | "streaming" | "waiting_approval"`
  - `config`: `{ provider, model, reasoning_effort, approval_mode }`（`approval_mode` 对齐现有 `"auto" | "ask"`）
  - `context`: `{ used_tokens, max_tokens }`
  - `events`: `seq > after_seq` 的已落盘事件数组（上限 `N` 条）
  - `active_turn`: `{ turn, step, reasoning, text, running_tools: [{ call_id, name, arguments, output_tail }] }`
  - `pending_interactions`: `[{ request_id, kind, call_id, tool, summary, args, options }]`
  - `queued_inputs`: `[{ id, text, created_at_ms }]`（内核唯一队列）
  - `subagents`: `Vec<SubagentTaskSnapshot>`（含名称、`child_session_id`、状态、`mode`、步数/当前工具、预览；解决主代理 `yield` 后界面无反馈问题，点击子代理可直接 `session.subscribe(child_session_id)`）

#### B. 服务端实时推送（Notifications，共 8 类）

| Method | 载荷与语义 |
| :--- | :--- |
| `turn.state` | `{ session_id, turn, step, state }`，驱动运行状态指示 |
| `stream.delta` | `{ session_id, turn, step, kind: "text" \| "reasoning" \| "tool_args" \| "tool_output", call_id?, delta }`（纯内存，不写 SQLite） |
| `event.committed` | `{ session_id, seq, ts_ms, visibility, kind, payload }`；前端收到后原子清理对应 `(turn, step)` 临时缓冲并写入正式消息列表，不重复不闪烁 |
| `interaction.requested` | 结构同 `pending_interactions` 单项 |
| `interaction.resolved` | `{ request_id, decision, resolved_by: "tui" \| "rpc", ts_ms }`；广播多端立刻收起审批卡片 |
| `queue.updated` | `{ session_id, items }` |
| `subagent.updated` | `{ session_id, task: SubagentTaskSnapshot, notice? }`；实时同步后台子代理生命周期与 `yield` 进度 |
| `session.switched` | `{ old_session_id, new_session_id }`；TUI `/resume` 或 `/new` 切换会话时通知已连接前端 |

#### C. 客户端控制请求（Requests）

| Method | Params | 作用 |
| :--- | :--- | :--- |
| `turn.submit` | `{ session_id, text, attachments?: [{ path?, mime? }], mode?: "send" \| "queue" \| "steer" }` | 发消息 / 入队 / 运行中插话 |
| `turn.cancel` | `{ session_id }` | 优雅中止当前 LLM 流或子进程工具（替代信号强杀） |
| `queue.remove` / `queue.pop_all` | `{ session_id, id? }` | 撤回排队消息放回输入框 |
| `interaction.resolve` | `{ request_id, decision, note? }` | 原子抢占批准/拒绝挂起工具请求 |
| `session.config` | `{ session_id, provider?, model?, reasoning_effort?, approval_mode? }` | 运行时结构化改配置，不再拼接 `/` 文本命令 |
| `session.list` / `session.events` | `{ cwd?, query?, limit?, cursor? }` / `{ session_id, before_seq?, after_seq?, limit? }` | 会话列表（支持 `cwd`/标题过滤）与双向事件分页 |
| `session.fork` / `session.tree` | `{ session_id, at_seq }` / `{ session_id }` | 复用 `mizar-session::fork` 与 `project_session_tree` |
| `capabilities.list` | `{ session_id }` | 返回可用 `/` 命令、skills、MCP tools、providers/models 与上下文上限 |

> **显式裁剪项**：不实现原草案中的 `session.search`（因决策 1 已移除 `session_recall_fts`，避免 20MB FTS5 回流）与 `blob.read`（见第 4 节）。

### 4. `sessions.sqlite` Payload 增量升级（向后兼容，零上下文污染）

基于现有 `SessionEvent` 结构增量增加 3 项可选载荷，所有新字段在 `derive_messages()` 构建 LLM 请求时一律忽略，不占用上下文亦不破坏 Prompt Cache：

1. **`tool-result` 补充精选 `meta`**：运行时 `ToolResult.details` 已生成结构化详情却在持久化时丢弃；改为随 `SessionEvent::ToolResult` 精选落盘（如 `edit`/`write` 的 `diff` 路径与增删行数、`bash` 的 `command`/`exit_code`/`duration_ms`、`read`/`grep`/`find` 的 `paths`/`match_count`），使多端直接渲染 Diff 卡片与退出码标牌。
2. **审批审计事件（`LogOnly`）**：新增 `kind = "approval"` 记录 `{ tool_call_id, tool, decision, resolved_by: "tui" | "rpc" }`，供复盘与审计追踪人工审批历史。
3. **`turn-end.detail` 纯规则 `summary`**：回合结束零模型调用顺手汇总 `duration_ms`、`files_changed`、`commands_run`、`tool_errors` 与 `usage`，直接服务界面概览与离开期间进展条。

> **暂不引入 `~/.mizar/blobs/<sha256>`**：实测全库 52 会话的 `session_events` 总大小仅 `26.76 MB`，工具输出已有 `DEFAULT_MAX_BYTES` 截断、图片已有 `mizar-client::images` 压缩；现阶段引入外置 blob 会额外带来引用计数与孤儿清理复杂度，待后续实测 SQLite 体积真正成为瓶颈再行外置。

### 5. 并发、背压与安全底线

- **双端审批抢占**：`PendingInteractions` 使用 `HashMap<RequestId, oneshot::Sender<ApprovalDecision>>`，`resolve()` 通过 `map.remove(&request_id)` 原子抢占——先到者胜出返回 `{ ok: true }` 并广播 `interaction.resolved`；后到者直接返回 `{ ok: false, reason: "already_resolved" }`，前端静默收起卡片。
- **高频流式节流**：广播 `stream.delta` 前加 `30~50ms` coalescing 缓冲（按 `(kind, call_id)` 合并，单帧上限 `16 KB`）；`ActiveTurnBuffer` 的 `output_tail` 限制保留最后 `64 KB`，防止刷屏输出撑爆内存。
- **慢客户端断连保护**：每个 Socket 连接设置 `256` 条有界发送队列；队满即主动断开该慢客户端，绝不阻塞 `SessionActor` 或 TUI 主循环；客户端重连通过 `session.subscribe(after_seq)` 无损恢复。
- **安全边界**：`~/.mizar/run/` 权限 `0700`、Socket 与 `p-<pid>.json` 权限 `0600`；`127.0.0.1:0` 认证回环 TCP 强制首帧常量时间 `auth.hello` 校验（错误返回 `-32003` 并断开）；`mizar serve` 强制仅允许回环地址绑定、严格校验 `Origin`（非法返回 `403`），远程加密与认证交由 Tailscale Serve。

### 6. 相较原始提案的 6 项关键优化汇总

1. 移除对已废弃 `session_recall_fts` 的 `session.search` 依赖，改由 `session.list` 轻量过滤承担。
2. 下沉复用既有 4,956 行零终端依赖的 `mizar-client`，并将 `AppState` 与 `AgentLoop::inbox()` 双头队列、单消费者 `approval.rs`、`approval_mode` 枚举全部对齐收口进 `SessionActor`。
3. 补齐「同 `session_id` 单写者锁」与 `ppid`/`tty`/`herdr_pane_id` 进程发现字段，消除多端并发写乱序与 wrapper 进程 PID 错位。
4. 在 `session.subscribe` 与通知流补齐 `subagents` 与 `subagent.updated`，支撑决策 2 的 `yield` / barrier 后台观测。
5. 裁剪过早优化的 `~/.mizar/blobs/<sha256>` 与 `blob.read`，保留高性价比的 `tool-result.meta`、`approval` 与 `turn-end.summary`。
6. 确立「先 Phase 1 内核瘦身、再 Phase 2 抽 Actor」的实施节奏，并将 `mizar serve` / Web / PWA 坚决后置至 M4。
