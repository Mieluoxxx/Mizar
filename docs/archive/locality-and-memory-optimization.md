# Mizar 内存剖析与局部性原理优化设计

> **状态**：诊断与优化设计（Proposed / 尚未落地代码）  
> **诊断基线**：基于 2026-11-20 本地多开实测采样（4~8 个常驻 `mizar` 进程，会话 `44c1df53-f47c-4b62-bea0-01e09dbdae25`）  
> **关联文档**：Crate 拓扑与服务装配见 [plugins.md](../plugins.md)；外部插件与 `SessionInbox` 设计见 [plugin-runtime-and-inbox.md](plugin-runtime-and-inbox.md)；架构与边界约束见 [AGENTS.md](../../AGENTS.md)。

---

## 一、背景与实测基线

在 16 核 Apple M4 Max / 128 GB 内存（macOS ARM64，物理页大小 `16,384` 字节）环境下，对长时间多开运行的 `mizar` 进程进行连续追踪采样（`footprint` / `vmmap` / `heap -s -H` / `ps`），得到如下演进基线：

| 采样时刻 | 活跃主进程数 | 真实物理内存 (`phys_footprint`) | 活堆对象 (`heap`) | 可回收碎片空页 (`Reclaimable`) | 堆碎片率 (`vmmap`) | `ps` RSS | 线程总数 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **00:10** | 4 个 | **64.0 MB** | **29.6 MB** | **30.3 MB** | 59%–66% | 147.7 MiB | ~82 |
| **00:53** | 5 个 | **79.0 MB** | **26.2 MB** | **~41.0 MB** | 61%–73% | 186.2 MiB | ~103 |
| **11:40** | 8 个 *(含 3 个过夜)* | **122.8 MB** | **~54.4 MB** | **~69.5 MB** | 56%–69% | 300.5 MiB | ~160 |
| **11:53** | 7 个 *(含 2 个过夜)* | **118.3 MB** | **~58.4 MB** | **~77.0 MB** | 49%–71% | 288.2 MiB | 141 |

典型单进程追踪（`PID 4640`）：
- 运行 **47 分钟**（会话原始数据 `1.91 MiB`，1,255 事件）：活堆 `12.6 MB`，`Reclaimable` 空页 `17.0 MB`，碎片率 `59%`，`footprint` `24.0 MB`，`RSS` `53.9 MiB`。
- 运行 **90 分钟**（会话原始数据 `2.33 MiB`，1,828 事件）：活堆反而回落到 **`9.1 MB`**，但 `Reclaimable` 空页暴涨到 **`26.0 MB`**，碎片率恶化至 **`73%`**（32.8 MB 堆脏页中 `23.7 MB` 为空洞），`RSS` 推高至 **`65.5 MiB`**。
- 运行 **10.8~11.6 小时过夜**的 3 个进程（`45712` / `42343` / `23856`）：真实物理内存稳定在 **`17.9–22.6 MB`**。

**核心判断**：
1. **没有无界内存泄漏**，整体物理占用在 128 GB 主机上仍属轻量（7 开仅占已用内存约 `0.18%`）。
2. **核心矛盾在于“倍率放大 + 页碎片只涨不退”**：每 `1 MB` 会话原始 JSON 会在堆上放大成约 `5 MB` 活堆与约 `15 MB` RSS，且可回收碎片页体积超过活堆本身（`77.0 MB` vs `58.4 MB`）。
3. **根因在局部性失配**：`heap -s -H 4640` 显示堆内存在 **28,165 个节点**，主体是 `16B`（10,959 个）、`32B`（5,409 个）、`48B`（1,171 个）、`64B`（1,384 个）、`640B`（1,673 个）的碎片小块，同时伴随 `2048KB[1]`、`512KB[1]` 的脉冲大块。**朝生暮死的短命渲染小对象与长期驻留的冷历史对象混杂在同一批 16 KB 物理页里，破坏了 CPU 缓存、内存工作集与分配器页的局部性**。

---

## 二、局部性原理在 Mizar 中的四层映射

要理解 `mizar` 的 CPU 抖动与内存碎片，需将局部性原理（时间局部性、空间局部性、生命周期局部性）映射到系统的四个缓存层级：

| 层级 | 缓存载体 | 典型单位 / 容量 | Mizar 中的热点数据 | 违背局部性时的直接代价 |
| :--- | :--- | :--- | :--- | :--- |
| **1. CPU 硬件缓存** | L1 / L2 / L3 Cache | Cache Line（常见 64B） | 行高数组、当前视口条目、活跃 Turn 尾部 | 每帧扫全历史冲刷 L1/L2，导致 Cache Thrashing 与 CPU 抬升 |
| **2. 内存分配器页** | `libmalloc` 区域 / OS 物理页 | 16 KB 页（macOS ARM64） | 同生命周期对象聚集（短命临时块 vs 长寿历史） | 几个长寿小块钉死整张 16 KB 页，使 `Reclaimable` 空洞无法归还 |
| **3. 应用层工作集** | `AppState` / `Session` 堆缓存 | 视口邻域几十条记录 | 视口可见 `RenderedLine`、正在流式输出的消息 | 95% 冷历史与 5% 热视口共驻堆内，引发 5 倍放大 |
| **4. 磁盘 / 页缓存** | SQLite `cache_size` + OS Page Cache | 4 KB B-Tree 页 | `(session_id, seq)` 聚簇事件、WAL 末尾 | 8 开各囤双连接独立页缓存，削弱内核统一 Page Cache 复用 |

---

## 三、现有架构中已做对的局部性设计

重构时应保留并强化以下三处已经契合局部性的既有设计：

1. **条目级排版缓存（时间局部性）**  
   [AppState::item_cache](../../crates/mizar-tui-ui/src/app.rs#L397) 与 [get_or_build_cached](../../crates/mizar-tui-ui/src/ui.rs#L1019-L1042) 以 `ItemRenderKey` 为键缓存每条消息的 Markdown 折行与高亮结果，未发生宽度/主题/长度变化时不重跑词法高亮。
2. **连续容器与事件批处理排空（空间 + 帧级时间局部性）**  
   [SessionInner::log](../../crates/mizar-session/src/lib.rs#L44-L47)（`Vec<SessionEntry>`）与 [AppState::items](../../crates/mizar-tui-ui/src/app.rs#L395)（`Vec<ChatItem>`）采用连续数组保存主时间线；[runner.rs](../../crates/mizar-tui/src/runner.rs#L1085-L1099) 在渲染前通过 `while let Ok(more) = ui_rx.try_recv()` 批量排空积压的流式 delta，避免每个 token 单独触发重绘。
3. **SQLite 联合主键聚簇与语句缓存（磁盘页空间局部性）**  
   [schema.rs](../../crates/mizar-session-sqlite/src/schema.rs#L61-L70) 为 `session_events` 定义 `PRIMARY KEY (session_id, seq)`，使同一会话的事件按 `seq` 顺序物理聚集在连续 B-Tree 页上；[load_session_from_conn](../../crates/mizar-session-sqlite/src/schema.rs#L158-L165) 使用 `prepare_cached` 并启用 `WAL`，让写路径高度集中在 WAL 尾部。

---

## 四、违背局部性原理的四大源码病灶

### 1. 渲染热路径每 33ms 扫全量历史并深拷贝丢弃（破坏空间/时间局部性 + 制造碎片）

* **源码定位**：[runner.rs](../../crates/mizar-tui/src/runner.rs#L1017-L1047)（`reveal_tick` `33ms` / `spinner_tick` `90ms`）→ [render_transcript](../../crates/mizar-tui-ui/src/ui.rs#L667-L981)。
* **行为分析**：终端同一时刻只展示视口内几十行（`row_off in sticky_rows..view_h`），且活跃流式输出时只有最末尾 1 条在更新。但当前 `render_transcript` 每帧都遍历 `start_item..total_items` **全部条目**：
  1. 即使命中 `item_cache`，[get_or_build_cached](../../crates/mizar-tui-ui/src/ui.rs#L1031) 仍执行 `hit.lines.clone()`，将全会话的 [RenderedLine](../../crates/mizar-tui-ui/src/markdown.rs#L18-L21)（内部 `Line<'static>` 含多段堆分配 `Span` / `String`）深拷贝进临时 `all_lines: Vec<RenderedLine>`，最终在 [ui.rs](../../crates/mizar-tui-ui/src/ui.rs#L965-L981) 只绘制视口几十行后将整张 `all_lines` 释放。
  2. 每帧为全部历史 `ChatItem::User` 分配 `preview: String`（[ui.rs](../../crates/mizar-tui-ui/src/ui.rs#L744-L753)）。
  3. 每帧为全部历史 `ChatItem::ToolCall` 执行 `app.tool_extras.get(id).cloned()`（[ui.rs](../../crates/mizar-tui-ui/src/ui.rs#L827)），连同未展开、未显示在屏幕上的 `full_content` / `diff_raw` / `full_args` 一并克隆。
  4. [start_local_bash](../../crates/mizar-tui-ui/src/app.rs#L684) 及主题/侧栏拖拽（[app.rs](../../crates/mizar-tui-ui/src/app.rs#L1926)）等路径直接调用 `self.item_cache.clear()`，一次性击穿所有冷历史缓存（尽管 `ItemRenderKey` 已包含 `width` 与 `theme_id`）。
* **局部性后果**：本可局限在几 KB 热数据里的渲染工作集，被迫每 33ms 扫过几 MB 冷数据并批量制造上万个 `16B/32B` 短命堆块，造成 L1/L2 Cache 抖动与高强度 `MADV_FREE` 空洞堆积。

### 2. 冷热数据混居且缓存只增不减（破坏时间局部性与页聚合）

* **源码定位**：[SessionInner::log](../../crates/mizar-session/src/lib.rs#L44-L47)、[AppState::items / tool_extras / item_cache](../../crates/mizar-tui-ui/src/app.rs#L395-L397)。
* **行为分析**：当对话推进到上千事件后，早期历史条目与已压缩（compacted）区间的工具正文访问频率趋近于零，但内存中同时保留三份完整副本：
  1. `SessionInner::log`（只追加，compaction 后不裁剪旧正文）；
  2. `AppState::items` 与 `tool_extras`（保留全量 `full_content` / `diff_raw` / `full_args`）；
  3. `AppState::item_cache`（无容量上限，保存整场会话展开高亮后的 `Vec<RenderedLine>` 词元树）。
* **局部性后果**：冷历史长寿对象占着物理页，使与之交错分配的短命块释放后页面无法整页回收，`4640` 呈现“活堆下降、空页反升”的典型碎片锁死形态。

### 3. 轨迹检测每步全量重建事件流（忽略尾部追加局部性）

* **源码定位**：[TrajectoryRuntime::evaluate_and_emit_new_findings](../../crates/mizar-trajectory/src/collector.rs#L341-L391) → [trace_events_from_session](../../crates/mizar-trajectory/src/collector.rs#L255-L257)。
* **行为分析**：会话事件只追加在 `SessionInner::log` 尾部，但每次 `ToolResult` / `StepEnd` / `TurnEnd` 触发异常检测时，程序都调用 `inspect()` → `trace_events_from_session(&session)`，重新加锁遍历整份日志并克隆出一个完整的 `Vec<TraceEvent>`，全量重跑 `fold_trajectory`、`derive_timeline` 与 `detect_anomalies`，最后仅过滤出未发射的新 `finding`。
* **局部性后果**：把 $O(1)$ 的尾部增量判定变成随会话长度线性增长的 $O(N)$ 全量扫描与脉冲堆分配。

### 4. 过多 Worker 线程与重复 HTTP/SQLite 状态（削弱核内缓存亲和性与多进程共享）

* **源码定位**：[main.rs](../../crates/mizar-tui/src/main.rs#L162)（`#[tokio::main]`）；[completions](../../crates/mizar-llm-completions/src/lib.rs#L623) / [responses](../../crates/mizar-llm-responses/src/lib.rs#L571) / [messages](../../crates/mizar-llm-messages/src/lib.rs#L754)（各自 `reqwest::Client::new()`）；[sqlite open](../../crates/mizar-session-sqlite/src/lib.rs#L59-L76) 与 [schema.rs](../../crates/mizar-session-sqlite/src/schema.rs#L13-L31)。
* **行为分析**：
  * `#[tokio::main]` 在 16 核机器上拉起 16+ 个 `tokio-rt-worker`，单 TUI 进程常态持有 20–21 线程（41–45 个虚拟栈区、脏栈页 `656–672 KB`）。低并发交互任务在过多 worker 间唤醒迁移，降低 CPU 核私有 L1/L2 Cache 命中率。
  * 三个协议适配器各自创建独立 `reqwest::Client`，重复驻留 `rustls` 根证书、系统代理探测与连接池状态（堆中固定出现 `224KB[3]` / `240KB[3]` 区域）。
  * SQLite 读写双连接均未设置 `PRAGMA cache_size`，每连接默认占用最高 ~2 MB 页缓存；多进程并发时各进程自建缓存，削弱 OS 统一 Page Cache 效率。

---

## 五、基于局部性原理的重构方案（按 ROI 排序）

### P0-1：视口空间局部性改造（消除全量克隆与渲染分配风暴）

* **目标文件**：[app.rs](../../crates/mizar-tui-ui/src/app.rs#L167-L171)、[ui.rs](../../crates/mizar-tui-ui/src/ui.rs#L667-L1042)。
* **设计要点**：
  1. **`Arc` 零拷贝共享行缓冲**：将 `CachedItemLines.lines` 由 `Vec<RenderedLine>` 改为 `Arc<Vec<RenderedLine>>`；`get_or_build_cached` 命中时只做指针引用计数递增（`Arc::clone`），彻底消灭整表 `Vec<RenderedLine>` 与内部 `Span`/`String` 的每帧克隆。
  2. **L1 Cache 友好的紧凑行高索引**：将 `render_transcript` 拆为「高度累计 + 视口绘制」两阶段。维护紧凑的 `item_heights: Vec<u32>`（或前缀和 `line_offsets: Vec<usize>`）：2,000 条消息仅占 `8–16 KB`，可完整装入 L1 Data Cache。定位 `[scroll_top .. scroll_top + view_h]` 时使用二分查找；当 `auto_scroll == true` 时甚至可直接从末尾向前累加，利用 99% 场景停留在底部的强时间局部性。
  3. **按需访问 `tool_extras` 与 `User.preview`**：移除循环顶部的无条件 `app.tool_extras.get(id).cloned()`，仅在缓存未命中、实际调用 `build_tool_lines` 时传入借用 `&ToolExtraMeta`；`TimelineTurn.preview` 仅对用户可见 turn 缓存或延迟截断，不再每帧为历史全部 `User` 消息 `.to_string()`。
  4. **局部失效代替全表 `clear()`**：插入新工具卡时仅失效新条目及前一个同组工具尾条目；主题或宽度切换无需急切清表（`ItemRenderKey` 已含 `width` 与 `theme_id`，未命中的可见条目自然重建）。

### P0-2：核亲和性优化与 SQLite 页缓存收敛

* **目标文件**：[main.rs](../../crates/mizar-tui/src/main.rs#L162)、[schema.rs](../../crates/mizar-session-sqlite/src/schema.rs#L13-L31)。
* **设计要点**：
  * 将 `#[tokio::main]` 替换为 `tokio::runtime::Builder::new_multi_thread().worker_threads(2..=4).enable_all()`（可用环境变量覆盖），使异步任务集中在少数核心调度，保留 L1/L2 Cache 亲和性并减少 80% 闲置线程栈。
  * 在 `configure_connection` 与 `configure_reader_connection` 中显式设置 `PRAGMA cache_size = -512;`（单连接封顶 `512 KiB`），避免 7–8 开多进程各自囤积冷 B-Tree 页，将磁盘缓存管理统一收敛到操作系统 Page Cache。

### P0-3：全局共享 `reqwest::Client`

* **目标文件**：[mizar-llm-http](../../crates/mizar-llm-http/src/lib.rs#L30-L37)、[mizar-llm-router](../../crates/mizar-llm-router/src/lib.rs#L198-L265) 及三个协议适配器。
* **设计要点**：在 `mizar-llm-http` 提供共享 `Client` 构造函数（由 `LlmRouter` 或插件初始化时持有一份 `reqwest::Client` 注入各适配器），使连接池、DNS 缓存与 TLS 会话票据聚集在单一热内存块中复用，消除 `224KB[3]` 冗余驻留。

### P1-4：`item_cache` 视口 LRU 工作集封顶

* **目标文件**：[AppState::item_cache](../../crates/mizar-tui-ui/src/app.rs#L397)、[ui.rs](../../crates/mizar-tui-ui/src/ui.rs#L1019-L1042)。
* **设计要点**：将“每个条目在当前宽度下的总行数”与“完整 `Arc<Vec<RenderedLine>>` 高亮结果”分离。行数元数据仅数十字节/条，可全量保留以支持瞬时滚动条与时间轴定位；完整高亮缓存仅保留**视口上下缓冲窗口（如最近/邻近 64~128 条）**，滑出工作集即通过 LRU 淘汰，切断长会话活堆与 `item_cache` 的线性耦合。

### P1-5：`TrajectoryRuntime` 尾部增量局部计算

* **目标文件**：[SessionTraceBuffer](../../crates/mizar-trajectory/src/collector.rs#L260-L268)、[collector.rs](../../crates/mizar-trajectory/src/collector.rs#L341-L391)。
* **设计要点**：在 `SessionTraceBuffer` 中记录 `last_folded_seq`（或 `last_entry_index`）及增量折叠状态机。`evaluate_and_emit_new_findings` 每次只读取 `seq > last_folded_seq` 的新增尾部切片推进状态与规则检查，不再全量克隆 `Vec<TraceEvent>`；全量投影仅在打开轨迹场景或 `/export` 时按需构造。

### P1-6：生命周期局部性与全局分配器评估（`mimalloc`）

* **目标文件**：根 [Cargo.toml](../../Cargo.toml)、[main.rs](../../crates/mizar-tui/src/main.rs)。
* **设计要点**：在完成 P0-1 与 P1-5 消除周期性短命分配风暴后，针对中长寿命对象（会话记录、滚动淘汰后的缓存块）仍可能产生的跨页混合，评估引入 `mimalloc` 作为 Rust `#[global_allocator]`。利用其按尺寸桶（size-class）和线程本地堆隔离的特性，把相近生命周期对象聚拢在相邻页内，方便整页回收。（注：C 侧 `rusqlite` / `rquickjs` 仍走系统 `libmalloc`，需实测 macOS release 构建下的 `vmmap` 碎片率收益。）

### P2-7：冷热分层懒加载（严格遵守 `mizar-tui-ui` 纯前端边界）

* **架构红线**（同 [AGENTS.md](../../AGENTS.md)）：`mizar-tui-ui` **严禁**直接引入 `rusqlite`、`tokio` 或 `cordis`，因此不可在 UI crate 内直连 SQLite。
* **分层方案**：
  1. **UI 冷字段卸载**：在 `mizar-tui-ui` 侧，对已滑出最近窗口且处于折叠状态的历史 `ToolCall`，保留 `output_preview` 与对应的 `(session_id, seq)` 定位标记，释放 `ToolExtraMeta.full_content` / `diff_raw` 大字符串；
  2. **按需请求宿主回读**：当用户展开该历史卡片或切换 `verbose_expanded` 时，若本地已卸载，由 `mizar-tui-ui` 发出轻量 `AppAction`（如按需补全工具详情），经由 `mizar-tui`（[runner.rs](../../crates/mizar-tui/src/runner.rs)）查询 `SessionPersistence`（命中 SQLite `(session_id, seq)` 聚簇主键与 OS Page Cache），再通过 `UiEvent` 回填展示。
  3. **会话日志冷裁剪**：仅在超长会话（数十 MB 级别）场景下再考虑为 `SessionInner::log` 引入“仅内存保留 compaction 窗口 + 热尾”，避免过早触动 `mizar-compaction` / `mizar-recall` 等上游调用点。

---

## 六、预期收益与验证口径

完成上述改造后，建议按以下可观测指标在真机多开场景对比回归（改造前后均需运行 `AGENTS.md` 规定的全量门禁，并同步采集运行时采样）：

1. **正确性与边界门禁**：
   * `cargo check --locked -p mizar-agent-spine --no-default-features --all-targets`
   * `cargo test --locked --workspace --no-fail-fast`
   * `cargo clippy --locked --workspace --all-targets -- -D warnings`
2. **局部性与内存回归对比（macOS 实测工具链）**：
   * **渲染分配风暴**：对比流式输出期间 `heap -s -H <pid>` 中 `16B/32B/640B` 节点增量与峰值大块（原 `2048KB/512KB`）是否消失；
   * **碎片与空页率**：连续运行 90 分钟的进程对比 `footprint <pid>` 的 `Reclaimable` 与 `vmmap <pid>` 的 `% FRAG`（基线 `4640` 为 `26 MB` 空页 / `73%` 碎片）；
   * **多开常驻基线**：对比 7~8 开场景下总线程数（基线 `141–160` 线程）与单进程空载/稳态 `phys_footprint`（基线 `10.4–23.3 MB`）。
