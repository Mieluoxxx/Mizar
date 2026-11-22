#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mizar 会话库（SQLite）只读探针。

设计约束：
- 全程只读：`mode=ro` 打开；活库带 `-shm` / `-wal` 时直接读，绝不 checkpoint。
- 零依赖：只用标准库，兼容 /usr/bin/python3（3.9）。
- 不改动任何文件：唯一写盘是活库打不开时把「三件套」复制到临时目录。

数据来源见 `crates/mizar-session-sqlite/src/schema.rs`（schema）
与 `crates/mizar-session/src/model.rs`（payload 形状）。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path

SURFACE_KINDS = {
    "user",
    "context-notice",
    "assistant",
    "tool-result",
    "compaction",
    "bash-execution",
}

HEADER_FIELDS = (
    "id, format_version, cwd, title, parent_id, fork_seq, "
    "created_at_ms, updated_at_ms, event_count, surface_count, compacted_up_to_seq"
)


# ---------------------------------------------------------------- 基础设施


def die(msg: str) -> "None":
    sys.stderr.write("error: %s\n" % msg)
    raise SystemExit(2)


def one_line(text: str, width: int = 0) -> str:
    """折叠空白为单空格，超过 width 时截断并补省略号。"""
    flat = " ".join(str(text).split())
    if width and len(flat) > width:
        return flat[: max(0, width - 1)] + "…"
    return flat


def resolve_db_path(explicit: "str | None") -> Path:
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("MIZAR_SESSION_DB", "").strip()
    if env:
        if env.lower() in ("none", "off", "memory"):
            die("MIZAR_SESSION_DB=%s：该环境变量禁用了会话落盘，没有库可读" % env)
        return Path(env).expanduser()
    home = os.environ.get("MIZAR_HOME", "").strip()
    base = Path(home).expanduser() if home else Path.home() / ".mizar"
    return base / "sessions.sqlite"


def _open(uri: str):
    conn = sqlite3.connect(uri, uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    # 真正碰一下表，把「文件能打开但读不了」也归入失败分支
    conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
    return conn


def connect(db: Path):
    """只读打开会话库，返回 (连接, 降级说明)。

    WAL 库的三种可读情形：
    1. 活库带 `-shm` / `-wal`：`mode=ro` 直接读（本轮反复验证过的路径）。
    2. 只有主库文件（SIGKILL 后 WAL 被删、或只拷了主文件）：SQLite 拒绝以
       `mode=ro` 打开（它需要建 `-shm`）。此时把主库 + 存在的 `-wal/-shm`
       复制到临时目录再读；副本仍打不开（说明 WAL 头标记残留但没有 wal 文件），
       才对**副本**用 `immutable=1`。
    3. 绝不对活库用 `immutable=1`：那会绕过文件锁，可能读到撕裂状态。
    """
    if not db.is_file():
        die("找不到会话库：%s" % db)
    first_err = None
    try:
        return _open("file:%s?mode=ro" % db), None
    except sqlite3.Error as exc:
        first_err = str(exc)

    tmpdir = tempfile.mkdtemp(prefix="mizar-session-inspect-")
    copy = Path(tmpdir) / db.name
    shutil.copy2(db, copy)
    copied = ["main"]
    for suffix in ("-wal", "-shm"):
        src = Path(str(db) + suffix)
        if src.is_file():
            shutil.copy2(src, Path(str(copy) + suffix))
            copied.append(suffix)

    try:
        return _open("file:%s?mode=ro" % copy), "活库只读打开失败（%s），已复制 %s 到 %s 读取" % (
            one_line(first_err, 120),
            "+".join(copied),
            tmpdir,
        )
    except sqlite3.Error as exc:
        second_err = str(exc)

    # 主库 WAL 头残留但无 -wal 文件：副本是静态快照，对它用 immutable=1 是安全的。
    try:
        return _open("file:%s?immutable=1" % copy), (
            "活库只有主库文件、无 `-wal`，已按不可变快照（immutable=1）读取副本 %s；"
            "该库最后一次 checkpoint 之后的事件可能缺失" % tmpdir
        )
    except sqlite3.Error as exc:
        die("打不开会话库：%s\n只读失败：%s\n副本失败：%s\n不可变副本失败：%s" % (db, first_err, second_err, exc))
    raise AssertionError("unreachable")


def fmt_ts(ms, pattern: str = "%H:%M:%S") -> str:
    if ms is None:
        return "-"
    try:
        return datetime.fromtimestamp(int(ms) / 1000.0).strftime(pattern)
    except (ValueError, OSError, OverflowError):
        return "?"


def fmt_duration(ms: float) -> str:
    seconds = ms / 1000.0
    if seconds < 1:
        return "%dms" % int(ms)
    if seconds < 60:
        return "%.1fs" % seconds
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return "%dm%02ds" % (minutes, seconds)
    hours, minutes = divmod(minutes, 60)
    return "%dh%02dm" % (hours, minutes)


def fmt_size(n: int) -> str:
    if n < 1024:
        return "%dB" % n
    if n < 1024 * 1024:
        return "%.1fK" % (n / 1024.0)
    return "%.1fM" % (n / (1024.0 * 1024.0))


def fmt_count(n) -> str:
    return "{:,}".format(int(n)).replace(",", " ")


def resolve_session_id(conn: sqlite3.Connection, needle: str) -> str:
    """接受完整 id 或唯一前缀。"""
    row = conn.execute("SELECT id FROM sessions WHERE id = ?", (needle,)).fetchone()
    if row:
        return row["id"]
    rows = conn.execute(
        "SELECT id FROM sessions WHERE id LIKE ? ORDER BY updated_at_ms DESC LIMIT 20",
        (needle + "%",),
    ).fetchall()
    if not rows:
        die("没有匹配的会话 id：%s" % needle)
    if len(rows) > 1:
        die(
            "前缀 %s 命中多个会话，请补全：\n  %s"
            % (needle, "\n  ".join(r["id"] for r in rows))
        )
    return rows[0]["id"]


def load_header(conn: sqlite3.Connection, sid: str):
    row = conn.execute(
        "SELECT %s FROM sessions WHERE id = ?" % HEADER_FIELDS, (sid,)
    ).fetchone()
    if not row:
        die("会话不存在：%s" % sid)
    return row


def load_entries(conn: sqlite3.Connection, sid: str, include_log: bool):
    sql = "SELECT seq, ts_ms, visibility, kind, payload_json FROM session_events WHERE session_id = ?"
    if not include_log:
        sql += " AND visibility = 'surface'"
    sql += " ORDER BY seq"
    out = []
    for row in conn.execute(sql, (sid,)):
        try:
            payload = json.loads(row["payload_json"])
        except json.JSONDecodeError:
            payload = {"_parse_error": True, "_raw": row["payload_json"][:200]}
        out.append(
            {
                "seq": row["seq"],
                "ts": row["ts_ms"],
                "visibility": row["visibility"],
                "kind": row["kind"],
                "payload": payload,
            }
        )
    return out


# ---------------------------------------------------------------- 摘要计算


def summarize_usage(entries):
    totals = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "reasoning": 0}
    models = []
    for entry in entries:
        usage = entry["payload"].get("usage") or {}
        totals["input"] += int(usage.get("inputTokens") or 0)
        totals["output"] += int(usage.get("outputTokens") or 0)
        totals["cache_read"] += int(usage.get("cacheReadTokens") or 0)
        totals["cache_write"] += int(usage.get("cacheWriteTokens") or 0)
        totals["reasoning"] += int(usage.get("reasoningTokens") or 0)
        replay = entry["payload"].get("replay") or {}
        provider = replay.get("provider")
        model = replay.get("model")
        if provider and model:
            label = "%s/%s" % (provider, model)
            if label not in models:
                models.append(label)
        elif model and model not in models:
            models.append(model)
    return totals, models


def turn_rollup(entries):
    """按 turn 汇总 step / 工具 / 时长 / 结局，用于判断「卡在哪」。

    归类以轨迹事件自身的 `detail.turn` 为准（turn 号单调递增，但 surface 事件
    夹在中间、turn-end 之后也可能再出现同号 turn-start），不看「当前上下文」，
    否则重跑/续跑的 turn 号会互相污染。缺失 `turn` 字段时回退到上次见过的号。
    """
    turns = {}
    order = []
    current_turn = 0

    def bucket_for(turn_no, ts):
        if turn_no not in turns:
            turns[turn_no] = {
                "turn": turn_no,
                "start": ts,
                "end": None,
                "reason": None,
                "steps": [],
                "tools": 0,
                "errors": 0,
                "unclosed": 0,
            }
            order.append(turn_no)
        turn = turns[turn_no]
        if turn["start"] is None:
            turn["start"] = ts
        return turn
    for entry in entries:
        kind = entry["kind"]
        if kind in SURFACE_KINDS:
            continue
        detail = entry["payload"].get("detail") or {}
        turn_no = detail.get("turn")
        if turn_no is None:
            turn_no = current_turn
        else:
            current_turn = turn_no
        turn = bucket_for(turn_no, entry["ts"])
        if kind == "turn-start":
            if turn["reason"] is not None or turn["end"] is not None:
                # 同号 turn 重跑：单独记一截步数，避免把上一轮的 steps 吞掉
                turn["unclosed"] += 1
                turn["reason"] = None
                turn["end"] = None
            if turn["start"] is None:
                turn["start"] = entry["ts"]
        elif kind == "step-start":
            turn["steps"].append(
                {"step": detail.get("step"), "start": entry["ts"], "end": None, "outcome": None}
            )
        elif kind == "step-end":
            for step in reversed(turn["steps"]):
                if step["end"] is None:
                    step["end"] = entry["ts"]
                    step["outcome"] = detail.get("outcome")
                    if detail.get("outcome") == "error":
                        turn["errors"] += 1
                    break
        elif kind == "tool-call":
            turn["tools"] += 1
        elif kind == "turn-end":
            turn["end"] = entry["ts"]
            turn["reason"] = detail.get("reason")
    return turns, order


# ---------------------------------------------------------------- 渲染


def render_header(conn, header, entries, all_entries, out=None):
    """打印会话概览。`out` 传入时（`--json`）改写 stderr，保持 stdout 只剩 JSON。"""
    stream = out or sys.stdout
    surface = [e for e in entries if e["visibility"] == "surface"]
    totals, models = summarize_usage(all_entries)
    turns, order = turn_rollup(all_entries)
    last_ts = max((e["ts"] for e in entries), default=None)

    def emit(line):
        print(line, file=stream)

    emit("id        %s" % header["id"])
    emit("title     %s" % (one_line(header["title"], 120) if header["title"] else "(无标题)"))
    emit("cwd       %s" % (header["cwd"] or "-"))
    emit(
        "time      %s → %s (%s)"
        % (
            fmt_ts(header["created_at_ms"], "%Y-%m-%d %H:%M:%S"),
            fmt_ts(last_ts or header["updated_at_ms"], "%H:%M:%S"),
            fmt_duration(max(0, (last_ts or header["updated_at_ms"]) - header["created_at_ms"])),
        )
    )
    emit(
        "events    %d（surface %d / log-only %d，库内计数 %s / %s）"
        % (
            len(entries),
            len(surface),
            len(entries) - len(surface),
            fmt_count(header["event_count"]),
            fmt_count(header["surface_count"]),
        )
    )
    lineage = ["format v%d" % header["format_version"]]
    if header["parent_id"]:
        lineage.append("parent %s" % header["parent_id"][:8])
    if header["fork_seq"] is not None:
        lineage.append("fork_seq %s" % header["fork_seq"])
    if header["compacted_up_to_seq"] is not None:
        lineage.append("compacted_up_to_seq %s" % header["compacted_up_to_seq"])
    emit("lineage   %s" % " · ".join(lineage))

    if order:
        parts = []
        for turn_no in order:
            turn = turns[turn_no]
            steps = len(turn["steps"])
            opened = any(step["end"] is None for step in turn["steps"])
            if turn["reason"]:
                outcome = turn["reason"]
            elif opened or turn["end"] is None:
                outcome = "进行中或未正常收尾"
            else:
                outcome = "无 turn-end"
            parts.append(
                "turn %s: %s 步 / %s 次工具调用 / %s"
                % (turn_no, steps, turn["tools"], outcome)
            )
        emit("trace     %s" % " · ".join(parts))
        errored = [t for t in order if turns[t]["reason"] == "error"]
        if errored:
            emit(
                "          注意：%d 个 turn 以 error 结束，但 turn-end / step-end 只落盘结局字符串，"
                "错误文本本身不在库里" % len(errored)
            )
    else:
        emit("trace     (无轨迹事件)")

    if models:
        emit("model     %s" % ", ".join(models))
    if totals["input"] or totals["output"]:
        emit(
            "usage     in %s / out %s / cache-read %s / reasoning %s"
            % (
                fmt_count(totals["input"]),
                fmt_count(totals["output"]),
                fmt_count(totals["cache_read"]),
                fmt_count(totals["reasoning"]),
            )
        )
    resume = "mizar"
    if header["cwd"]:
        resume += " -C %s" % header["cwd"]
    resume += " --resume %s" % header["id"]
    emit("resume    %s" % resume)
    emit("")


def describe_assistant(payload, width):
    blocks = payload.get("blocks") or []
    texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
    reasoning = [b.get("text", "") for b in blocks if b.get("type") == "reasoning"]
    calls = [b for b in blocks if b.get("type") == "tool-call"]
    bits = []
    if texts:
        bits.append("text %s" % fmt_size(sum(len(t) for t in texts)))
    if reasoning:
        bits.append("thinking %s" % fmt_size(sum(len(t) for t in reasoning)))
    if calls:
        bits.append("calls %d（%s）" % (len(calls), ", ".join(c.get("name", "?") for c in calls)))
    if not bits:
        bits.append("(空)")
    return " · ".join(bits)


def describe_entry(entry, width):
    """一行式概要，供 show 使用。"""
    kind = entry["kind"]
    payload = entry["payload"]
    if payload.get("_parse_error"):
        return "(payload 无法解析：%s)" % one_line(payload.get("_raw", ""), max(20, width - 26))
    if kind == "user":
        return one_line(payload.get("text", ""), width)
    if kind == "assistant":
        return describe_assistant(payload, width)
    if kind == "tool-result":
        flag = "ERR" if payload.get("is_error") else "ok"
        content = payload.get("content") or ""
        return "%s %s %s｜%s" % (
            payload.get("name", "?"),
            flag,
            fmt_size(len(content)),
            one_line(content, max(20, width - 30)),
        )
    if kind == "context-notice":
        return "[%s] %s" % (
            payload.get("source", "?"),
            one_line(payload.get("text", ""), max(20, width - 20)),
        )
    if kind == "compaction":
        return "summary %s · first_kept_seq=%s · %s → %s tok" % (
            fmt_size(len(payload.get("summary") or "")),
            payload.get("first_kept_seq"),
            payload.get("tokens_before", "?"),
            payload.get("tokens_after", "?"),
        )
    if kind == "bash-execution":
        return "! %s (exit %s)" % (
            one_line(payload.get("command", ""), width),
            payload.get("exit_code"),
        )
    if kind == "tool-result-prune":
        return "pruned %d 条" % len(payload.get("pruned") or [])
    detail = payload.get("detail") or {}
    if kind in ("step-start", "first-token"):
        return "turn %s step %s%s" % (
            detail.get("turn"),
            detail.get("step"),
            " · prompt %s chars · %s tools"
            % (detail.get("system_prompt_chars"), detail.get("tools_count"))
            if kind == "step-start"
            else "",
        )
    if kind == "step-end":
        return "turn %s step %s → %s" % (detail.get("turn"), detail.get("step"), detail.get("outcome"))
    if kind == "tool-call":
        return "%s %s" % (detail.get("name"), one_line(detail.get("arguments", ""), max(20, width - 20)))
    if kind == "tool-outcome":
        return "%s → %s" % (
            detail.get("tool_call_id", "?")[:16],
            "被拦截" if detail.get("blocked") else "放行",
        )
    if kind == "turn-end":
        return "turn %s → %s" % (detail.get("turn"), detail.get("reason"))
    if kind == "turn-start":
        return "turn %s" % detail.get("turn")
    if kind == "compaction-start":
        return "trigger %s" % detail.get("trigger")
    if kind == "compaction-end":
        return "%s · first_kept_seq=%s · %s → %s tok" % (
            detail.get("status"),
            detail.get("first_kept_seq"),
            detail.get("tokens_before"),
            detail.get("tokens_after"),
        )
    if kind == "anomaly":
        return "%s [%s] %s" % (detail.get("rule_id"), detail.get("severity"), detail.get("title"))
    return one_line(json.dumps(detail, ensure_ascii=False), width)


def render_outline(entries, width, gap_ms):
    print("%4s  %-12s %-13s %s" % ("seq", "time", "kind", "detail"))
    prev_ts = None
    for entry in entries:
        stamp = fmt_ts(entry["ts"], "%H:%M:%S")
        if prev_ts is not None and gap_ms and entry["ts"] - prev_ts >= gap_ms:
            print(
                "%4s  %-12s %-13s [间隔 %s]"
                % ("", "", "", fmt_duration(entry["ts"] - prev_ts))
            )
        prev_ts = entry["ts"]
        mark = "" if entry["visibility"] == "surface" else "·"
        print(
            "%4d  %-12s %-12s %s %s"
            % (entry["seq"], stamp, entry["kind"], mark, describe_entry(entry, width))
        )


def render_text(entries, width, show_thinking):
    """对话正文视角：只展开 human 可读内容，工具输出取首行。"""
    for entry in entries:
        kind = entry["kind"]
        payload = entry["payload"]
        if kind == "user":
            print("【用户】")
            print(payload.get("text", ""))
            print("")
        elif kind == "assistant":
            blocks = payload.get("blocks") or []
            texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
            reasoning = [b.get("text", "") for b in blocks if b.get("type") == "reasoning"]
            calls = [b for b in blocks if b.get("type") == "tool-call"]
            if show_thinking and reasoning:
                for chunk in reasoning:
                    print("【思考】" + one_line(chunk, width))
            body = "\n".join(texts).strip()
            if body:
                print("【助手】")
                print(body)
                print("")
            if calls:
                print("【工具调用】")
                for call in calls:
                    print("  %s %s" % (call.get("name"), one_line(call.get("arguments", ""), width)))
                print("")
        elif kind == "tool-result":
            flag = "错误" if payload.get("is_error") else "结果"
            content = payload.get("content") or ""
            print(
                "【工具%s】%s  %s"
                % (flag, payload.get("name", "?"), fmt_size(len(content)))
            )
            print("  " + one_line(content, max(40, width - 2)))
            print("")
        elif kind == "compaction":
            print("【压缩】保留 seq >= %s" % payload.get("first_kept_seq"))
            print(one_line(payload.get("summary", ""), width))
            print("")
        elif kind == "bash-execution":
            print("【Shell】%s" % payload.get("command", ""))
            print("  " + one_line(payload.get("output", ""), max(40, width - 2)))
            print("")


# ---------------------------------------------------------------- 子命令


# ---------------------------------------------------------------- 子命令


def flatten_text(node, out):
    """递归收集 payload 里所有字符串字段，用于搜索命中定位。"""
    if isinstance(node, str):
        out.append(node)
    elif isinstance(node, dict):
        for value in node.values():
            flatten_text(value, out)
    elif isinstance(node, list):
        for value in node:
            flatten_text(value, out)


def search_snippet(payload_json: str, needle: str, width: int) -> str:
    """在反序列化后的字段里定位命中处，返回带上下文的可读片段。"""
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError:
        idx = payload_json.find(needle)
        start = max(0, idx - 40)
        return one_line(payload_json[start : start + width], width)
    texts = []
    flatten_text(payload, texts)
    lower = needle.lower()
    for text in texts:
        flat = " ".join(text.split())
        idx = flat.lower().find(lower)
        if idx < 0:
            continue
        start = max(0, idx - width // 3)
        return one_line(flat[start : start + width], width)
    return one_line(" ".join(texts), max(20, width))


def cmd_db(conn, args):
    path = args._db_path
    print("db        %s" % path)
    for suffix in ("", "-wal", "-shm"):
        side = Path(str(path) + suffix)
        state = fmt_size(side.stat().st_size) if side.is_file() else "缺失"
        print("  %-8s %s" % (side.name if suffix else "main", state))
    sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    events = conn.execute("SELECT COUNT(*) FROM session_events").fetchone()[0]
    docs = conn.execute("SELECT COUNT(*) FROM session_recall_docs").fetchone()[0]
    empty = conn.execute("SELECT COUNT(*) FROM sessions WHERE event_count = 0").fetchone()[0]
    print("")
    print("sessions  %d（其中 0 事件的空壳 %d）" % (sessions, empty))
    print("events    %d" % events)
    print("recall    %d docs" % docs)
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    print("schema    user_version=%s" % version)
    if args.note:
        print("")
        print("note      %s" % args.note)


def cmd_list(conn, args):
    conditions = []
    params = []
    if args.cwd:
        conditions.append("cwd = ?")
        params.append(os.path.abspath(os.path.expanduser(args.cwd)))
    if not args.all:
        conditions.append("event_count > 0")
    sql = "SELECT %s FROM sessions" % HEADER_FIELDS
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY updated_at_ms DESC"
    if args.limit:
        sql += " LIMIT ?"
        params.append(args.limit)

    rows = conn.execute(sql, params).fetchall()
    if not rows:
        print("(没有匹配的会话)")
        return
    print(
        "%-36s %-16s %5s %5s  %s"
        % ("id", "updated", "事件", "surface", "title")
    )
    for row in rows:
        print(
            "%-36s %-16s %5d %5d  %s"
            % (
                row["id"],
                fmt_ts(row["updated_at_ms"], "%m-%d %H:%M:%S"),
                row["event_count"],
                row["surface_count"],
                one_line(row["title"] or "(无标题)", args.width - 70 if args.width > 90 else 40),
            )
        )


def cmd_show(conn, args):
    sid = resolve_session_id(conn, args.session)
    header = load_header(conn, sid)
    entries = load_entries(conn, sid, include_log=True)
    all_entries = entries
    if args.surface:
        entries = [e for e in entries if e["visibility"] == "surface"]
    render_header(conn, header, entries, all_entries, sys.stderr if args.json else None)
    if args.json:
        print(
            json.dumps(
                {
                    "session": dict(header),
                    "entries": entries,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.text:
        render_text(
            [e for e in entries if e["visibility"] == "surface"],
            args.width,
            args.thinking,
        )
        return
    render_outline(entries, args.width, args.gap * 1000)


def cmd_timeline(conn, args):
    sid = resolve_session_id(conn, args.session)
    header = load_header(conn, sid)
    entries = load_entries(conn, sid, include_log=True)
    turn_entries = [e for e in entries if e["kind"] not in SURFACE_KINDS]
    if not turn_entries:
        print("(该会话没有轨迹事件)")
        return
    print("%s  %s  %s 个 turn" % (header["id"], one_line(header["title"] or "-", 60), len(turn_rollup(entries)[1])))
    print("")
    render_outline(turn_entries, args.width, args.gap * 1000)


def cmd_stats(conn, args):
    sid = resolve_session_id(conn, args.session)
    header = load_header(conn, sid)
    entries = load_entries(conn, sid, include_log=True)
    totals, models = summarize_usage(entries)
    turns, order = turn_rollup(entries)

    print("%s  %s" % (header["id"], one_line(header["title"] or "-", 70)))
    print("")
    print("%-6s %-10s %-10s %6s %6s %7s  %s" % ("turn", "start", "end", "steps", "tools", "耗时", "结局"))
    for turn_no in order:
        turn = turns[turn_no]
        if turn["start"] and turn["end"]:
            span = fmt_duration(turn["end"] - turn["start"])
        elif turn["start"]:
            span = "进行中"
        else:
            span = "-"
        print(
            "%-6s %-10s %-10s %6d %6d %7s  %s"
            % (
                turn_no,
                fmt_ts(turn["start"]),
                fmt_ts(turn["end"]),
                len(turn["steps"]),
                turn["tools"],
                span,
                turn["reason"] or "未结束",
            )
        )
    print("")
    if models:
        print("model     %s" % ", ".join(models))
    print(
        "usage     in %s / out %s / cache-read %s / cache-write %s / reasoning %s"
        % (
            fmt_count(totals["input"]),
            fmt_count(totals["output"]),
            fmt_count(totals["cache_read"]),
            fmt_count(totals["cache_write"]),
            fmt_count(totals["reasoning"]),
        )
    )

    histogram = {}
    for entry in entries:
        if entry["kind"] == "tool-result":
            name = entry["payload"].get("name") or "?"
            bucket = histogram.setdefault(name, {"n": 0, "bytes": 0, "err": 0})
            bucket["n"] += 1
            bucket["bytes"] += len(entry["payload"].get("content") or "")
            if entry["payload"].get("is_error"):
                bucket["err"] += 1
    if histogram:
        print("")
        print("%-14s %5s %10s %5s" % ("tool", "次数", "输出量", "错误"))
        for name, bucket in sorted(histogram.items(), key=lambda kv: -kv[1]["n"]):
            print("%-14s %5d %10s %5d" % (name, bucket["n"], fmt_size(bucket["bytes"]), bucket["err"]))

    anomalies = [e for e in entries if e["kind"] == "anomaly"]
    if anomalies:
        print("")
        print("anomaly")
        for entry in anomalies:
            detail = entry["payload"].get("detail") or {}
            print(
                "  seq %s [%s] %s — %s"
                % (
                    entry["seq"],
                    detail.get("severity"),
                    detail.get("rule_id"),
                    one_line(detail.get("summary", ""), args.width - 40),
                )
            )


def cmd_text(conn, args):
    sid = resolve_session_id(conn, args.session)
    header = load_header(conn, sid)
    print("# %s" % one_line(header["title"] or "(无标题)", 100))
    print("")
    entries = [
        e for e in load_entries(conn, sid, include_log=False) if e["visibility"] == "surface"
    ]
    render_text(entries, args.width, args.thinking)


def cmd_search(conn, args):
    needle = args.query
    sql = (
        "SELECT e.session_id, e.seq, e.ts_ms, e.kind, e.payload_json, s.title "
        "FROM session_events e JOIN sessions s ON s.id = e.session_id "
        "WHERE e.payload_json LIKE ? ESCAPE '\\'"
    )
    # 字面量搜索：% 与 _ 必须转义，否则会变成 LIKE 通配符误命中
    escaped = needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    params = ["%" + escaped + "%"]
    if args.session:
        sql += " AND e.session_id = ?"
        params.append(resolve_session_id(conn, args.session))
    if args.cwd:
        sql += " AND s.cwd = ?"
        params.append(os.path.abspath(os.path.expanduser(args.cwd)))
    sql += " ORDER BY e.ts_ms DESC LIMIT ?"
    params.append(args.limit)

    rows = conn.execute(sql, params).fetchall()
    if not rows:
        print("(没有命中)")
        return
    for row in rows:
        snippet = search_snippet(row["payload_json"], needle, args.width - 50)
        print(
            "%s  seq %-5d %-12s %-14s %s"
            % (
                row["session_id"][:8],
                row["seq"],
                fmt_ts(row["ts_ms"], "%m-%d %H:%M"),
                row["kind"],
                snippet,
            )
        )
    print("")
    print("命中 %d 条（按时间倒序，上限 %d）" % (len(rows), args.limit))


# ---------------------------------------------------------------- 入口


def build_parser():
    # 子命令用 SUPPRESS 默认值，避免 argparse 用子解析器的 None 覆盖主解析器已解析的值
    # （`session.py --db X list` 里父层的 X 不会被 `list` 的空默认值抵消）。
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", default=argparse.SUPPRESS, help="会话库路径")
    common.add_argument("--width", type=int, default=argparse.SUPPRESS, help="单行截断宽度")
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="输出原始 JSON")

    parser = argparse.ArgumentParser(
        prog="session.py",
        description="mizar 会话库只读探针（list / show / timeline / text / stats / search / db）",
        epilog="--db / --width / --json 写在子命令前后都行。",
    )
    parser.add_argument(
        "--db", help="会话库路径，默认按 MIZAR_SESSION_DB → MIZAR_HOME → ~/.mizar 解析"
    )
    parser.add_argument("--width", type=int, default=140, help="单行截断宽度，默认 140")
    parser.add_argument("--json", action="store_true", help="show 输出原始 JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    p_db = sub.add_parser("db", parents=[common], help="打印库路径、文件状态与总量")
    p_db.set_defaults(func=cmd_db)

    p_list = sub.add_parser("list", parents=[common], help="会话列表")
    p_list.add_argument("--cwd", help="只列该工作区的会话")
    p_list.add_argument("--limit", type=int, default=15)
    p_list.add_argument("--all", action="store_true", help="包含 0 事件的空壳会话")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", parents=[common], help="会话概览 + 事件大纲（默认含轨迹）")
    p_show.add_argument("session", help="会话 id 或其唯一前缀")
    p_show.add_argument("--surface", action="store_true", help="只看 surface 事件（对模型可见的）")
    p_show.add_argument("--text", action="store_true", help="改为展开对话正文")
    p_show.add_argument("--thinking", action="store_true", help="--text 时一并展开 reasoning")
    p_show.add_argument("--gap", type=float, default=30.0, help="超过该秒数的间隔单独标注，0 关闭")
    p_show.set_defaults(func=cmd_show)

    p_timeline = sub.add_parser("timeline", parents=[common], help="只看轨迹事件（turn/step/tool-call 边界）")
    p_timeline.add_argument("session")
    p_timeline.add_argument("--gap", type=float, default=30.0)
    p_timeline.set_defaults(func=cmd_timeline)

    p_text = sub.add_parser("text", parents=[common], help="纯正文：user 全文、assistant 文本、工具摘要")
    p_text.add_argument("session")
    p_text.add_argument("--thinking", action="store_true")
    p_text.set_defaults(func=cmd_text)

    p_stats = sub.add_parser("stats", parents=[common], help="turn×step 汇总、token 用量、工具直方图")
    p_stats.add_argument("session")
    p_stats.set_defaults(func=cmd_stats)

    p_search = sub.add_parser("search", parents=[common], help="跨会话搜索 payload 原文")
    p_search.add_argument("query")
    p_search.add_argument("--session", help="限定单个会话")
    p_search.add_argument("--cwd", help="限定工作区")
    p_search.add_argument("--limit", type=int, default=40)
    p_search.set_defaults(func=cmd_search)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    db = resolve_db_path(args.db)
    conn, note = connect(db)
    args._db_path = db
    args.note = note
    if note:
        sys.stderr.write("note: %s\n" % note)
    try:
        args.func(conn, args)
    finally:
        conn.close()


if __name__ == "__main__":
    main()