---
name: mizar-session-inspect
description: Read mizar's persisted session database (~/.mizar/sessions.sqlite) to answer "what did that session do?", "why did it stop?", "which tools did it call?", or "where did this bug show up before?" — without resuming it. Read-only, no process side effects.
when-to-use: 主人要求查看/复盘/排查某个历史 mizar 会话（给出 session id 或前缀、或说「刚才那个会话」「昨天那个报错的会话」）；需要确认会话为何中断（turn-end reason=error）、统计每次 turn 的步数与耗时、看某个工具调用到底传了什么参数、跨会话搜索一段报错文本或路径。不适用：当前正在运行的会话（直接看终端）、需要改数据库或删会话（本技能只读）。
---

# mizar 会话复盘

`mizar` 的所有会话落在一个 SQLite 库里，**不是文件**。本技能提供只读探针 `scripts/session.py`，
把「50 条事件、26 条 log-only 轨迹」压缩成人能读的格式，回答「这个会话做了什么、卡在哪」。

## 数据在哪

```
数据库            ~/.mizar/sessions.sqlite          （WAL 模式，常有 -wal / -shm 伴生文件）
覆盖顺序          --db  >  MIZAR_SESSION_DB  >  MIZAR_HOME/sessions.sqlite  >  ~/.mizar/sessions.sqlite
MIZAR_SESSION_DB  设成 none / off / memory 表示不落盘——此时没有库可读
工作区过滤        sessions 表有 cwd 列；跨工作区共用同一个库
```

权威定义（改脚本前先看）：`crates/mizar-session-sqlite/src/schema.rs`（表结构）、
`crates/mizar-session/src/model.rs`（`payload_json` 的形状）。

三张表：

```
sessions             id, format_version, cwd, title, parent_id, fork_seq,
                     created_at_ms, updated_at_ms, event_count, surface_count, compacted_up_to_seq
session_events       session_id, seq, ts_ms, visibility, kind, payload_json
                     PRIMARY KEY (session_id, seq)
session_recall_docs  FTS5 回溯索引（preview 是截断到 4000 字的正文摘要，可当快速全文搜索用）
```

## 怎么用

脚本在本技能目录的 `scripts/session.py`。**相对路径按本技能的 base directory 解析**（加载技能时
`<skill_resources>` 会告诉你该目录的绝对路径），所以下面的 `$S` 用哪种写法都行：

```sh
# 按加载时给出的 base directory 定位（推荐，仓库级/用户级安装都适用）
S="<base directory>/scripts/session.py"

# 或者直接用已知路径之一
S=.agents/skills/mizar-session-inspect/scripts/session.py                    # 仓库级（在仓库根目录下）
S=~/.agents/skills/mizar-session-inspect/scripts/session.py                  # 用户级

python3 "$S" db                                    # 库路径、三件套大小、会话/事件总数
python3 "$S" list --cwd "$PWD" --limit 10          # 本工作区最近会话
python3 "$S" show 55ab3c1a --width 120             # 概览 + 事件大纲（会话 id 支持唯一前缀）
python3 "$S" show 55ab3c1a --text --thinking       # 展开对话正文，含 reasoning
python3 "$S" timeline 55ab3c1a                     # 只看轨迹：turn/step/tool-call 边界
python3 "$S" stats 55ab3c1a                        # turn×step 表 + token 用量 + 工具直方图
python3 "$S" search "error text" --limit 20        # 跨会话搜 payload 原文
python3 "$S" show 55ab3c1a --json                  # stdout 只有 JSON（概览走 stderr，可直接管道给 jq）
```

`--db` / `--width` / `--json` 写在子命令前后都行。`/usr/bin/python3` 是 macOS 自带的 3.9，脚本兼容它。

自检：`sh "<base directory>/scripts/selftest.sh"`
（校验 frontmatter、语法、建临时库跑通 7 个子命令、降级路径、错误退出、`--json` 纯净度）。
注意：`selftest.sh` 里的 `--db` 指临时库，不碰 `~/.mizar`。

## 两份副本

同一份技能同时存在于两处（内容应保持一致）：

```
<repo>/.agents/skills/mizar-session-inspect/     rank 200（项目级，高优先）
~/.agents/skills/mizar-session-inspect/          rank 500（用户级，兜底）
```

二者同名时 `crates/mizar-skills` 只保留 rank 数字更小的那份（见 `filesystem.rs` 的 `PROJECT_AGENTS_RANK` /
`USER_AGENTS_RANK`，去重逻辑在 `registry.rs`）。**改完一边记得同步另一边**，否则在仓库外工作时行为会不一样：

```sh
cp -R <repo>/.agents/skills/mizar-session-inspect ~/.agents/skills/
```

## 关键概念：surface 与 log-only 是两套并行叙事

`visibility` 决定一条事件是否参与模型上下文：

```
surface       user / context-notice / assistant / tool-result / compaction / bash-execution(默认)
              → 还原「它说了什么、看到什么」
log-only      turn-start / step-start / first-token / step-end / tool-call / tool-outcome /
              turn-end / anomaly / compaction-start / compaction-end / tool-result-prune
              → 还原「它卡在第几步、每次工具调用传了什么」
```

`show` 默认打印两边（log-only 行首带 `·`），`--surface` 只看对话侧。**单看任何一边都会漏**：
surface 里 assistant 可能全是 tool-call 零文本，而 log-only 的 `tool-call` 才是带完整参数的版本
（assistant 的 `blocks[].tool-call` 同源，但 `tool-call` 轨迹还记了 `turn`/`step`）。

## 判断「会话为什么停了」

按这个顺序查，因为**错误文本本身没有落盘**：

```
1. show <id>                    → 看 trace 行的 turn 汇总：哪个 turn 是 error / cancelled / 进行中
2. stats <id>                   → 看每个 turn 的耗时：末步耗时异常长（几十秒~几分钟）= 请求超时/失败
3. timeline <id> --gap 10       → 看最后一段的间隔标注，定位「哪一步之后就没动静了」
4. show <id> --json | 看最后几条 step-end / turn-end 的 outcome 字段
```

`TurnEndReason::Error { message }` 里的 message **只进内存事件，不进 SQLite**——
`crates/mizar-agent-loop/src/lib.rs` 的 `StepGuard` / `TurnGuard` 只序列化 `outcome.as_str()` /
`reason.as_str()`（即 `"error"` 这个词）。`crates/mizar-session/src/model.rs` 里
`SessionTraceEvent::StepEnd` 的 `outcome: String` 同理。所以库里能查到「哪个 turn 失败了」，
查不到「失败原因」。要拿原因只能：① 看终端当时输出；② 对同一 prompt 重跑；
③ 查 provider 的请求日志。**不要在报告里编造错误原因**。

其他常见结局：

```
completed   正常收尾
max_steps   撞到步数上限
cancelled   用户 Esc / Ctrl-C，或 fork 掉
无 turn-end  进程被 kill / 崩溃；反过来「有 turn-start 没 step-end」= 卡在某步中途
```

## 几条反直觉的坑

**1. `-readonly` 在只有主库文件时会失败。** SQLite 打开 WAL 库需要能建 `-shm`，
文件不在（SIGKILL 后 WAL 被清、或只拷了主库）就会报 `unable to open database file (14)`。
脚本的处理：先试 `mode=ro`；失败则把主库 + 存在的 `-wal/-shm` 复制到临时目录；
副本仍失败才对**副本**用 `immutable=1`。对活库用 `immutable=1` 是错的——绕过锁会读到撕裂状态。

**2. 不要 `cp` 活库后读副本当作等价。** 活库的 `-wal` 里有未 checkpoint 的页；
只拷主库会丢最后一批事件（实测同一时刻主库 2422 事件、活库 2439 事件）。脚本优先直读活库。

**3. `event_count` / `surface_count` 是预计算计数器**，可与 `count(*)` 交叉验证：
对不上说明有写入丢失或副本不完整。

**4. `title` 不是 LLM 生成的。** 目前是首条 user 文本首行、空白折叠、72 字符截断
（`crates/mizar-session/src/model.rs::derive_session_title`），所以标题会带路径、带换行折痕。
`compacted_up_to_seq` 非空表示发生过压缩，`seq <= 它` 的 surface 事件已折叠进 `compaction` 摘要——
复盘时要意识到早期历史不在模型窗口里了，但**原始事件仍在库里**（`--surface` 不筛这个）。

**5. 0 事件的会话是启动即退出的空壳**，`list` 默认隐藏，`--all` 才显示。

**6. `references/` 下的会话不属于本仓库**：那些 summary/设计文档是 agent 读过的外部材料，
`tool-result` 里的内容是当时的快照，不代表仓库现在的代码。

## 输出纪律

- 报告里区分**库里能证实的事实**与**推断**。库里查不到错误原因就直说查不到，不要补全。
- 引用工具输出时给 `seq` 号，方便主人自己核。
- `events` 行里的「库内计数 vs 实际条数」不一致要讲出来，别默默忽略。
- 会话正在被写时（`show` 与 `list` 的数字每次都在变）优先看 `timeline` 的稳定边界，别看总数。

## 扩展

脚本只读、单文件、无第三方依赖（只用标准库，兼容 Python 3.9）。
加子命令的规矩：新的 `cmd_xxx(conn, args)` + `sub.add_parser(..., parents=[common])`，
SQL 一律参数化，绝不拼字符串；**新子命令要同步加进 `selftest.sh` 的冒烟列表**。

`--json` 子命令若也要打概览，必须像 `show` 一样把概览写到 stderr，否则污染管道。