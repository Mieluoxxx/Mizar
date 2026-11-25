#!/bin/sh
# mizar-session-inspect 自检：验证 SKILL.md 能被 mizar-skills 的解析规则接受，
# 并对探针脚本做冒烟测试（构造一个最小 SQLite 库，跑通各子命令）。
#
# 用法：sh ~/.agents/skills/mizar-session-inspect/scripts/selftest.sh
set -eu

SKILL_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PY=${PYTHON:-/usr/bin/python3}
SESSION_PY="$SKILL_DIR/scripts/session.py"
WORK=$(mktemp -d "${TMPDIR:-/tmp}/mizar-session-inspect-selftest.XXXXXX")
trap 'rm -rf "$WORK"' EXIT INT TERM

fail() { printf 'FAIL: %s\n' "$1" >&2; exit 1; }
ok() { printf '  ok  %s\n' "$1"; }

printf '== frontmatter ==\n'
"$PY" - "$SKILL_DIR/SKILL.md" <<'PY' || fail "SKILL.md frontmatter 不合规"
import sys
path = sys.argv[1]
raw = open(path, encoding="utf-8").read()
lines = raw.splitlines()
assert lines and lines[0] == "---", "首行必须是 ---"
end = lines.index("---", 1)
fields = {}
key = None
for line in lines[1:end]:
    if line.startswith((" ", "\t")):
        continue
    if ":" in line:
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
name = fields.get("name", "")
assert name, "缺少 name"
assert name == name.lower() and all(c.isalnum() or c == "-" for c in name) and "--" not in name and not name.startswith("-"), "name 必须是 kebab-case: %r" % name
assert fields.get("description"), "缺少 description"
assert fields.get("when-to-use"), "缺少 when-to-use"
print("  name: %s" % name)
PY
ok "frontmatter 合规（name / description / when-to-use）"

printf '== python 语法 ==\n'
"$PY" -m py_compile "$SESSION_PY" || fail "python 语法错误"
ok "session.py 可编译"

printf '== 冒烟测试 ==\n'
DB="$WORK/sessions.sqlite"
"$PY" - "$DB" <<'PY' || fail "构造测试库失败"
import json, sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
# 用 WAL 模式建库：关闭连接时 SQLite 会 checkpoint 并删除 -wal/-shm，
# 但主库头部仍保留 WAL 标记——这正是「只有主库文件」的降级场景。
conn.execute("PRAGMA journal_mode = WAL")
conn.executescript("""
CREATE TABLE sessions (id TEXT PRIMARY KEY, format_version INTEGER NOT NULL, cwd TEXT, title TEXT,
  parent_id TEXT, fork_seq INTEGER, created_at_ms INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL,
  event_count INTEGER NOT NULL DEFAULT 0, surface_count INTEGER NOT NULL DEFAULT 0, compacted_up_to_seq INTEGER);
CREATE TABLE session_events (session_id TEXT NOT NULL, seq INTEGER NOT NULL, ts_ms INTEGER NOT NULL,
  visibility TEXT NOT NULL, kind TEXT NOT NULL, payload_json TEXT NOT NULL, PRIMARY KEY (session_id, seq));
CREATE TABLE session_recall_docs (doc_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
  seq INTEGER NOT NULL, ts_ms INTEGER NOT NULL, kind TEXT NOT NULL, preview TEXT NOT NULL, UNIQUE(session_id, seq));
PRAGMA user_version = 1;
""")
sid = "11111111-2222-3333-4444-555555555555"
base = 1795000000000
rows = [
    (1, "log_only", "selection", {"type": "selection", "provider": "openai", "model": "gpt-test"}),
    (2, "log_only", "turn-start", {"type": "trace", "detail": {"kind": "turn-start", "turn": 1}}),
    (3, "surface", "user", {"type": "user", "text": "冒烟测试输入"}),
    (4, "log_only", "step-start", {"type": "trace", "detail": {"kind": "step-start", "turn": 1, "step": 1, "system_prompt_chars": 10, "tools_count": 1}}),
    (5, "surface", "assistant", {"type": "assistant", "blocks": [{"type": "tool-call", "id": "c1", "name": "read", "arguments": "{}"}]}),
    (6, "surface", "tool-result", {"type": "tool-result", "tool_call_id": "c1", "name": "read", "content": "x", "is_error": False}),
    (7, "log_only", "tool-result-prune", {"type": "trace", "detail": {"kind": "tool-result-prune", "target_seq": 6, "notice": "[Superseded by a newer read of this file]"}}),
    (8, "log_only", "step-end", {"type": "trace", "detail": {"kind": "step-end", "turn": 1, "step": 1, "outcome": "error", "error": "upstream stream reset"}}),
    (9, "log_only", "turn-end", {"type": "trace", "detail": {"kind": "turn-end", "turn": 1, "reason": "error", "error": "upstream stream reset"}}),
    (10, "log_only", "turn-start", {"type": "trace", "detail": {"kind": "turn-start", "turn": 2}}),
    (11, "surface", "user", {"type": "user", "text": "继续修复"}),
    (12, "log_only", "step-start", {"type": "trace", "detail": {"kind": "step-start", "turn": 2, "step": 1, "system_prompt_chars": 10, "tools_count": 1}}),
    (13, "log_only", "tool-call", {"type": "trace", "detail": {"kind": "tool-call", "turn": 2, "step": 1, "call_id": "c2", "name": "edit", "arguments": "{}"}}),
    (14, "log_only", "approval", {"type": "trace", "detail": {"kind": "approval", "tool_call_id": "c2", "tool": "edit", "decision": "allow_once", "resolved_by": "tui"}}),
    (15, "surface", "tool-result", {"type": "tool-result", "tool_call_id": "c2", "name": "edit", "content": "ok", "is_error": False, "meta": {"kind": "diff", "path": "src/lib.rs", "added": 2, "removed": 1, "diff": "--- a\n+++ b\n-old\n+new\n+more"}}),
    (16, "log_only", "step-end", {"type": "trace", "detail": {"kind": "step-end", "turn": 2, "step": 1, "outcome": "completed"}}),
    (17, "log_only", "turn-end", {"type": "trace", "detail": {"kind": "turn-end", "turn": 2, "reason": "completed", "summary": {"durationMs": 420, "filesChanged": ["src/lib.rs"], "commandsRun": 1, "toolErrors": 0, "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}}}}),
]
conn.executemany("INSERT INTO session_events VALUES (?,?,?,?,?,?)", [
    (sid, seq, base + seq * 1000, vis, kind, json.dumps(payload, ensure_ascii=False))
    for seq, vis, kind, payload in rows
])
conn.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
             (sid, 1, "/tmp/smoke", "冒烟会话", None, None, base + 8000, base + 17000, len(rows), 5, None))
conn.commit()
PY
ok "测试库已生成（mode=$(sqlite3 "$DB" 'PRAGMA journal_mode;' 2>/dev/null || echo unknown)）"

ls "$DB"-wal >/dev/null 2>&1 \
    && ok "-wal 存在（降级测试直接拷主库）" \
    || ok "-wal 已 checkpoint 删除（主库头部仍留 WAL 标记，正好是降级场景）"

for cmd in "db" "list" "show 11111111" "timeline 11111111" "stats 11111111" "text 11111111" "search 冒烟"; do
    # shellcheck disable=SC2086
    if ! out=$("$PY" "$SESSION_PY" --db "$DB" --width 100 $cmd 2>&1); then
        printf '%s\n' "$out" >&2
        fail "子命令失败: $cmd"
    fi
    [ -n "$out" ] || fail "子命令无输出: $cmd"
done
ok "7 个子命令全部有输出"

stats_out=$("$PY" "$SESSION_PY" --db "$DB" --width 120 stats 11111111)
case $stats_out in
    *"turn 0"*|*"0      "*) fail "stats 不应出现虚假 turn 0: $stats_out" ;;
esac
case $stats_out in
    *"upstream stream reset"*) ok "stats 跳过 selection/approval 且展示 detail.error" ;;
    *) fail "stats 未展示 detail.error: $stats_out" ;;
esac
case $stats_out in
    *"turn-summary"*) ;;
    *) fail "stats 未展示 turn-summary: $stats_out" ;;
esac
case $stats_out in
    *"files=1 [src/lib.rs]"*) ;;
    *) fail "stats turn-summary 缺少 filesChanged: $stats_out" ;;
esac
case $stats_out in
    *"cmds=1"*) ;;
    *) fail "stats turn-summary 缺少 cmds=1: $stats_out" ;;
esac
case $stats_out in
    *"allow_once via tui"*) ok "stats 展示 turn-end.summary 与 approval" ;;
    *) fail "stats 未展示 approval: $stats_out" ;;
esac
show_out=$("$PY" "$SESSION_PY" --db "$DB" --width 120 show 11111111)
case $show_out in
    *"upstream stream reset"*) ;;
    *) fail "show 未展示 detail.error: $show_out" ;;
esac
case $show_out in
    *"allow_once via tui"*) ;;
    *) fail "show 未展示 approval: $show_out" ;;
esac
case $show_out in
    *"1 files"*) ;;
    *) fail "show 未展示 turn-end.summary files: $show_out" ;;
esac
case $show_out in
    *"1 cmds"*) ;;
    *) fail "show 未展示 turn-end.summary cmds: $show_out" ;;
esac
case $show_out in
    *"seq #6"*) ok "show 展示 detail.error、tool-result-prune、approval 与 turn-end.summary" ;;
    *) fail "show 未展示 tool-result-prune: $show_out" ;;
esac

printf '== 降级路径 ==\n'
cp "$DB" "$WORK/mainonly.sqlite"
out=$("$PY" "$SESSION_PY" --db "$WORK/mainonly.sqlite" list 2>&1) || { printf '%s\n' "$out" >&2; fail "主库单独拷贝时打不开"; }
case $out in
    *"immutable=1"*) ok "只有主库文件时降级为 immutable 快照读取" ;;
    *) fail "预期走 immutable 降级路径，实际输出: $out" ;;
esac

printf '== 错误处理 ==\n'
if "$PY" "$SESSION_PY" --db "$WORK/nope.sqlite" list >/dev/null 2>&1; then
    fail "库不存在时应失败"
fi
ok "库不存在时非零退出"
if "$PY" "$SESSION_PY" --db "$DB" show nosuchid >/dev/null 2>&1; then
    fail "会话不存在时应失败"
fi
ok "会话不存在时非零退出"

printf '== 搜索是字面量语义 ==\n'
# `%` 与 `_` 如果没转义，LIKE 会把它们当通配符；这里断言不会误命中。
if ! out=$("$PY" "$SESSION_PY" --db "$DB" search '100%' 2>&1); then
    printf '%s\n' "$out" >&2
    fail "search 包含 % 时崩溃"
fi
case $out in
    *"没有命中"*) ok "搜 100% 不误命中（% 已转义）" ;;
    *) fail "搜 100% 不应命中，实际: $out" ;;
esac
if ! out=$("$PY" "$SESSION_PY" --db "$DB" search '冒烟测试' 2>&1); then
    printf '%s\n' "$out" >&2
    fail "search 正常词失败"
fi
case $out in
    *"没有命中"*) fail "搜 冒烟测试 应命中 user 事件" ;;
    *) ok "搜正常词仍有命中" ;;
esac

printf '\n== --json 输出是纯 JSON ==\n'
if ! out=$("$PY" "$SESSION_PY" --db "$DB" show 11111111 --json 2>/dev/null); then
    fail "--json 运行失败"
fi
printf '%s' "$out" | "$PY" -c 'import json,sys; d=json.load(sys.stdin); assert d["session"]["id"], "missing session"; assert d["entries"], "missing entries"' \
    || fail "--json 的 stdout 不是纯 JSON（概览文字泄漏到 stdout？）"
ok "--json 的 stdout 只有 JSON（概览走 stderr）"

printf '\n全部通过\n'