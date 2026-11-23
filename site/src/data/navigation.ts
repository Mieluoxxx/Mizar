import { getUiStrings } from './i18n';
import { splitLocaleContentId, type Locale } from '../lib/url';

export interface NavItem {
  slug: string;
  fileSlug: string;
  groupId: string;
  groupTitle: string;
  label: string;
  title: string;
  description: string;
}

export interface NavGroup {
  id: string;
  title: string;
  items: NavItem[];
}

interface BilingualText {
  zh: string;
  en: string;
}

interface NavItemDef {
  slug: string;
  fileSlug: string;
  label: BilingualText;
  title: BilingualText;
  description: BilingualText;
}

interface NavGroupDef {
  id: string;
  title: BilingualText;
  items: NavItemDef[];
}

const NAV_GROUP_DEFS: NavGroupDef[] = [
  {
    id: 'quickstart',
    title: {
      zh: '快速开始',
      en: 'Quickstart',
    },
    items: [
      {
        slug: 'intro',
        fileSlug: 'intro',
        label: {
          zh: '简介与安装',
          en: 'Intro',
        },
        title: {
          zh: '简介与快速开始',
          en: 'Introduction & Quick Start',
        },
        description: {
          zh: '安装 Mizar、选择免 Key 演示或直连大模型，完成终端里的第一次对话与代码改动。',
          en: 'Install Mizar, run the keyless mock demo or connect a live LLM, and make your first terminal edit.',
        },
      },
    ],
  },
  {
    id: 'configuration',
    title: {
      zh: '配置与模型',
      en: 'Config & Models',
    },
    items: [
      {
        slug: 'config',
        fileSlug: 'config',
        label: {
          zh: '配置文件',
          en: 'Config',
        },
        title: {
          zh: '配置与优先级',
          en: 'Config & Precedence',
        },
        description: {
          zh: '全局与项目级 settings.json 合并规则、/settings 实时配置、上下文压缩与模块原子开关。',
          en: 'Global and project settings.json merge rules, live /settings modal, compaction, and module toggles.',
        },
      },
      {
        slug: 'providers',
        fileSlug: 'providers',
        label: {
          zh: '模型与凭据',
          en: 'Providers',
        },
        title: {
          zh: '模型与凭据',
          en: 'Models & Credentials',
        },
        description: {
          zh: '配置 DeepSeek、OpenAI、Anthropic 与本地模型，使用环境变量和系统钥匙串管理密钥。',
          en: 'Configure DeepSeek, OpenAI, Anthropic, and local models with env vars and keychain commands.',
        },
      },
    ],
  },
  {
    id: 'tui',
    title: {
      zh: 'TUI & CLI',
      en: 'TUI & CLI',
    },
    items: [
      {
        slug: 'tui',
        fileSlug: 'tui/index',
        label: {
          zh: 'TUI 界面',
          en: 'TUI',
        },
        title: {
          zh: '终端界面与双模视口',
          en: 'TUI & Dual Viewports',
        },
        description: {
          zh: 'Fullscreen 与 Inline 双视口切换、分栏诊断面板、鼠标划词放行与 Launchpad 快捷入口。',
          en: 'Switch between Fullscreen and Inline viewports, inspect the split side panel, copy text, and use Launchpad shortcuts.',
        },
      },
      {
        slug: 'tui/bash-mode',
        fileSlug: 'tui/bash-mode',
        label: {
          zh: '本地 Bash',
          en: 'Bash Mode',
        },
        title: {
          zh: '本地 Bash 模式 (! / !!)',
          en: 'Local Bash Mode (! / !!)',
        },
        description: {
          zh: '用 ! 把测试输出交给模型继续修，用 !! 零 Token 消耗查看状态，以及中断与输出落盘机制。',
          en: 'Feed test output to the model with !, inspect state at zero token cost with !!, plus interrupts and spill files.',
        },
      },
      {
        slug: 'tui/commands',
        fileSlug: 'tui/commands',
        label: {
          zh: '斜杠命令',
          en: 'Commands',
        },
        title: {
          zh: '斜杠命令',
          en: 'Commands',
        },
        description: {
          zh: '按模型视图、会话分叉、视口诊断与工具规则分组的斜杠命令参考。',
          en: 'Grouped slash command reference across models, sessions, viewports, diagnostics, tools, and security.',
        },
      },
      {
        slug: 'tui/keybinds',
        fileSlug: 'tui/keybinds',
        label: {
          zh: '快捷键',
          en: 'Keybinds',
        },
        title: {
          zh: '快捷键速查',
          en: 'Keybinds',
        },
        description: {
          zh: '全局、编辑器、侧边栏、补全弹窗与会话轨迹视图的快捷键速查表。',
          en: 'Quick reference for global, editor, side panel, completion popup, and /trace keybindings.',
        },
      },
      {
        slug: 'tui/sessions',
        fileSlug: 'tui/sessions',
        label: {
          zh: '会话与轨迹',
          en: 'Sessions',
        },
        title: {
          zh: '会话、分叉与性能轨迹',
          en: 'Sessions, Forking & /trace',
        },
        description: {
          zh: '恢复历史会话、/tree 分叉探索、/compact 上下文压缩与 FTS5 回溯，及 /trace 三泳道排查热点。',
          en: 'Resume sessions, fork with /tree, compact context via /compact + FTS5 recall, and profile latency in /trace.',
        },
      },
      {
        slug: 'tui/cli',
        fileSlug: 'tui/cli',
        label: {
          zh: 'CLI 参数',
          en: 'CLI',
        },
        title: {
          zh: 'CLI 参数',
          en: 'CLI Flags',
        },
        description: {
          zh: '按场景组合命令行启动参数，覆盖模型、工作区、视口、工具集、无头 exec 与 --trace-out。',
          en: 'Combine CLI startup flags across workspace, models, headless exec, --trace-out, and approvals.',
        },
      },
      {
        slug: 'tui/themes',
        fileSlug: 'tui/themes',
        label: {
          zh: '终端主题',
          en: 'Themes',
        },
        title: {
          zh: '终端主题',
          en: 'Terminal Themes',
        },
        description: {
          zh: 'auto / mist / sakura 等内置终端主题配置、真彩色支持与终端背景自适应探测。',
          en: 'Built-in auto, mist, sakura, and other terminal themes with truecolor and background detection.',
        },
      },
    ],
  },
  {
    id: 'capabilities',
    title: {
      zh: '工具与扩展',
      en: 'Tools & Extensions',
    },
    items: [
      {
        slug: 'tools',
        fileSlug: 'tools',
        label: {
          zh: '内置工具',
          en: 'Tools',
        },
        title: {
          zh: '内置工具',
          en: 'Built-in Tools',
        },
        description: {
          zh: '6 个内置编程工具、recall 情景回溯、常驻 subagent、输出截断与 ToolExposure 层级。',
          en: 'The 6 coding tools, static recall, persistent subagent, truncation limits, and ToolExposure tiers.',
        },
      },
      {
        slug: 'mcp',
        fileSlug: 'mcp',
        label: {
          zh: 'MCP 服务器',
          en: 'MCP Servers',
        },
        title: {
          zh: 'MCP 服务器接入',
          en: 'MCP Server Integration',
        },
        description: {
          zh: '通过 mcp.json 接入本地 Stdio 与远程 HTTP 服务，并在项目内局部覆写暴露策略。',
          en: 'Connect local Stdio and remote HTTP MCP servers via mcp.json with per-project exposure overrides.',
        },
      },
      {
        slug: 'codemode',
        fileSlug: 'codemode',
        label: {
          zh: 'Codemode 沙箱',
          en: 'Codemode',
        },
        title: {
          zh: '按需发现与沙箱编排',
          en: 'Tool Search & Codemode',
        },
        description: {
          zh: '利用工具搜索与隔离 JS 沙箱，避免海量 MCP 工具撑爆上下文与 Token 费用。',
          en: 'Use BM25 tool_search and the isolated QuickJS sandbox to avoid bloating context with large MCP catalogs.',
        },
      },
    ],
  },
  {
    id: 'rules-security',
    title: {
      zh: '规则、权限与排障',
      en: 'Rules & Security',
    },
    items: [
      {
        slug: 'instructions',
        fileSlug: 'instructions',
        label: {
          zh: '项目规则',
          en: 'Rules',
        },
        title: {
          zh: '项目规则 (AGENTS.md)',
          en: 'Project Rules (AGENTS.md)',
        },
        description: {
          zh: '沿目录树自动加载 AGENTS.md 与本地覆盖规则，了解 64 KiB 上下文预算的保近弃远策略。',
          en: 'Auto-discover AGENTS.md and local overrides up the directory tree under a 64 KiB proximity-first budget.',
        },
      },
      {
        slug: 'skills',
        fileSlug: 'skills',
        label: {
          zh: '技能 (Skills)',
          en: 'Skills',
        },
        title: {
          zh: '可复用技能 (Skills)',
          en: 'Reusable Skills',
        },
        description: {
          zh: '编写 SKILL.md 封装团队工作流，通过 /技能名 一键调用并在编辑后热更新。',
          en: 'Package team workflows into SKILL.md files, invoke them with /skill-name, and hot-reload edits.',
        },
      },
      {
        slug: 'security',
        fileSlug: 'security',
        label: {
          zh: '权限与安全',
          en: 'Permissions',
        },
        title: {
          zh: '权限与安全',
          en: 'Permissions & Security',
        },
        description: {
          zh: '工具预设、黑白名单并集、工作区限制与执行审批，及三套场景化加固配置。',
          en: 'Tool presets, cumulative allow/exclude lists, workspace boundaries, --approve prompts, and three hardening profiles.',
        },
      },
      {
        slug: 'troubleshooting',
        fileSlug: 'troubleshooting',
        label: {
          zh: '排障指南',
          en: 'Troubleshooting',
        },
        title: {
          zh: '排障指南',
          en: 'Troubleshooting',
        },
        description: {
          zh: '按现象定位演示模式回退、MCP 不被调用、JSON 报错、只读环境与鼠标划词问题。',
          en: 'Diagnose mock-mode fallbacks, skipped MCP tools, strict JSON errors, read-only workspaces, and mouse selection.',
        },
      },
    ],
  },
  {
    id: 'dev-arch',
    title: {
      zh: '架构与开发',
      en: 'Development',
    },
    items: [
      {
        slug: 'dev',
        fileSlug: 'dev/index',
        label: {
          zh: '架构分层',
          en: 'Architecture',
        },
        title: {
          zh: 'Workspace 分层与隔离边界',
          en: 'Workspace Layers & Cores',
        },
        description: {
          zh: '10 个 crate 的静态 DAG 职责分层，以及纯核心（mizar-types/mizar-tui-ui）与 Merak 零终端隔离边界。',
          en: 'Responsibilities across 10 workspace crates, pure-core rules (mizar-types/mizar-tui-ui), and zero-TUI client bounds.',
        },
      },
      {
        slug: 'dev/runtime',
        fileSlug: 'dev/runtime',
        label: {
          zh: '运行时与钩子',
          en: 'Runtime & Hooks',
        },
        title: {
          zh: 'MizarRuntime、强类型钩子与新增工具',
          en: 'MizarRuntime, Typed Hooks & Tools',
        },
        description: {
          zh: 'MizarRuntime::build 静态装配、RAII HookGuard 清理、强类型钩子与事件总线，以及从零注册新工具。',
          en: 'MizarRuntime::build static assembly, RAII HookGuard cleanup, typed hooks/event buses, and registering new tools.',
        },
      },
      {
        slug: 'dev/conventions',
        fileSlug: 'dev/conventions',
        label: {
          zh: '开发规范',
          en: 'Conventions',
        },
        title: {
          zh: '工程规范、门控与测试',
          en: 'Conventions, Gates & Tests',
        },
        description: {
          zh: '仓库约定、文案双语规范、文件写队列、feature 门控验证与提交流程。',
          en: 'Repo conventions, bilingual text rules, file mutation queues, feature-gate checks, and commit workflow.',
        },
      },
    ],
  },
];

function buildNavGroups(locale: Locale): NavGroup[] {
  return NAV_GROUP_DEFS.map((group) => {
    const groupTitle = group.title[locale];
    return {
      id: group.id,
      title: groupTitle,
      items: group.items.map((item) => ({
        slug: item.slug,
        fileSlug: item.fileSlug,
        groupId: group.id,
        groupTitle,
        label: item.label[locale],
        title: item.title[locale],
        description: item.description[locale],
      })),
    };
  });
}

const GROUPS_BY_LOCALE: Record<Locale, NavGroup[]> = {
  zh: buildNavGroups('zh'),
  en: buildNavGroups('en'),
};

const FLAT_BY_LOCALE: Record<Locale, NavItem[]> = {
  zh: GROUPS_BY_LOCALE.zh.flatMap((group) => group.items),
  en: GROUPS_BY_LOCALE.en.flatMap((group) => group.items),
};

const BY_SLUG_BY_LOCALE: Record<Locale, Map<string, NavItem>> = {
  zh: new Map(FLAT_BY_LOCALE.zh.map((item) => [item.slug, item])),
  en: new Map(FLAT_BY_LOCALE.en.map((item) => [item.slug, item])),
};

export function getNavGroups(locale: Locale = 'zh'): NavGroup[] {
  return GROUPS_BY_LOCALE[locale] ?? GROUPS_BY_LOCALE.zh;
}

export function getFlatNavItems(locale: Locale = 'zh'): NavItem[] {
  return FLAT_BY_LOCALE[locale] ?? FLAT_BY_LOCALE.zh;
}

export const NAV_GROUPS: NavGroup[] = GROUPS_BY_LOCALE.zh;
export const FLAT_NAV_ITEMS: NavItem[] = FLAT_BY_LOCALE.zh;

function normalizeLookupSlug(slug: string): string {
  return splitLocaleContentId(slug).slug;
}

export function getPageMeta(
  slug: string,
  locale: Locale = 'zh',
): NavItem | undefined {
  const map = BY_SLUG_BY_LOCALE[locale] ?? BY_SLUG_BY_LOCALE.zh;
  return map.get(normalizeLookupSlug(slug));
}

export function getGroupTitleForSlug(
  slug: string,
  locale: Locale = 'zh',
): string {
  return (
    getPageMeta(slug, locale)?.groupTitle ?? getUiStrings(locale).brandWordmark
  );
}

/** 统一使用手册内线性前后翻页（PREVIOUS / NEXT）。 */
export function getPagination(
  slug: string,
  locale: Locale = 'zh',
): {
  prev: NavItem | null;
  next: NavItem | null;
} {
  const items = getFlatNavItems(locale);
  const clean = normalizeLookupSlug(slug);
  const idx = items.findIndex((item) => item.slug === clean);
  if (idx === -1) return { prev: null, next: null };
  return {
    prev: idx > 0 ? items[idx - 1] : null,
    next: idx < items.length - 1 ? items[idx + 1] : null,
  };
}
