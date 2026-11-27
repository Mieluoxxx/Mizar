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
    id: 'getting-started',
    title: {
      zh: '快速开始',
      en: 'Getting Started',
    },
    items: [
      {
        slug: 'intro',
        fileSlug: 'intro',
        label: {
          zh: '简介与安装',
          en: 'Quick Start',
        },
        title: {
          zh: '简介与快速开始',
          en: 'Introduction & Quick Start',
        },
        description: {
          zh: '安装 Mizar、配置模型凭据或离线演示，完成终端与无头模式的首次调用。',
          en: 'Install Mizar, configure model credentials or keyless mock mode, and run your first interactive or headless session.',
        },
      },
      {
        slug: 'workbench',
        fileSlug: 'workbench',
        label: {
          zh: '终端工作台',
          en: 'Workbench',
        },
        title: {
          zh: '终端工作台、快捷键与斜杠命令',
          en: 'Workbench, Keybindings & Slash Commands',
        },
        description: {
          zh: 'Fullscreen 与 Inline 双视口、右侧分栏、!/!! 本地 Bash、快捷键与全部 24 个斜杠命令。',
          en: 'Fullscreen and Inline viewports, split side panel, !/!! local Bash, keybindings, and all 24 slash commands.',
        },
      },
    ],
  },
  {
    id: 'core-concepts',
    title: {
      zh: '核心机制',
      en: 'Core Concepts',
    },
    items: [
      {
        slug: 'tools',
        fileSlug: 'tools',
        label: {
          zh: '工具与扩展',
          en: 'Tools & Extensions',
        },
        title: {
          zh: '工具体系、按需 Codemode、MCP 与技能',
          en: 'Tools, On-Demand Codemode, MCP & Skills',
        },
        description: {
          zh: '五级 ToolExposure 模型、MinimalCoding 默认四件套、按需 QuickJS codemode、tool_search、MCP 与 SKILL.md 目录注入。',
          en: 'Five-tier ToolExposure model, MinimalCoding default 4 tools, on-demand QuickJS codemode, tool_search, MCP, and SKILL.md <location> catalog.',
        },
      },
      {
        slug: 'sessions',
        fileSlug: 'sessions',
        label: {
          zh: '会话与轨迹',
          en: 'Sessions & Tracing',
        },
        title: {
          zh: '会话持久化、压缩剪枝与轨迹分析',
          en: 'Sessions, Compaction & Trajectory Tracing',
        },
        description: {
          zh: '宽容 JSONL 单会话存储与 SHA-256 Blob 外置、/tree 分叉、热缓存感知剪枝与 B-5 压缩、/trace 与 --trace-out。',
          en: 'Tolerant JSONL persistence + SHA-256 blobs, /tree forking, hot-cache-aware pruning & B-5 compaction, /trace, and --trace-out.',
        },
      },
      {
        slug: 'context',
        fileSlug: 'context',
        label: {
          zh: '上下文与缓存',
          en: 'Context & Caching',
        },
        title: {
          zh: 'KV Cache 前缀稳定、静态提示词与指令发现',
          en: 'KV Cache Stability, Static Prompts & Instructions',
        },
        description: {
          zh: '精简静态英文系统提示词、Turn-0 严格顺序、启动期静态 AGENTS.md 加载、按需工具表与协议层缓存断点保持。',
          en: 'Lean static English system prompt, Turn-0 ordering, startup-static AGENTS.md, on-demand tool schemas, and wire cache breakpoints.',
        },
      },
      {
        slug: 'security',
        fileSlug: 'security',
        label: {
          zh: '安全与审批',
          en: 'Security & Approvals',
        },
        title: {
          zh: '安全边界、路径沙箱与交互审批',
          en: 'Security Boundaries, Path Scope & Approval Gate',
        },
        description: {
          zh: 'restrictToWorkspace 符号链接逃逸防护、bash 非沙箱说明、--approve 审批闸门与纵深防御组合。',
          en: 'restrictToWorkspace symlink escape protection, bash shell caveats, --approve gate, and defense-in-depth recipes.',
        },
      },
    ],
  },
  {
    id: 'reference',
    title: {
      zh: '参考手册',
      en: 'Reference',
    },
    items: [
      {
        slug: 'reference/cli',
        fileSlug: 'reference/cli',
        label: {
          zh: 'CLI 与环境变量',
          en: 'CLI & Env Vars',
        },
        title: {
          zh: 'CLI 参数、Exec 输出契约与环境变量',
          en: 'CLI Flags, Exec Contract & Environment Variables',
        },
        description: {
          zh: 'mizar / mizar exec / mizar update 全量命令行参数、JSON 输出 Schema、退出码与全部环境变量。',
          en: 'Complete CLI flags for mizar, mizar exec, and mizar update, JSON output schema, exit codes, and environment variables.',
        },
      },
      {
        slug: 'reference/config',
        fileSlug: 'reference/config',
        label: {
          zh: '配置参考',
          en: 'Config Reference',
        },
        title: {
          zh: 'settings.json、模块控制与 mcp.json',
          en: 'settings.json, Plugin Toggles & mcp.json',
        },
        description: {
          zh: '配置层级合并规则、settings.json 全字段、disabledPlugins 与 10 个宿主扩展 ID、以及 mcp.json Schema。',
          en: 'Config merge precedence, full settings.json schema, disabledPlugins, 10 host extension IDs, and mcp.json schema.',
        },
      },
      {
        slug: 'reference/models',
        fileSlug: 'reference/models',
        label: {
          zh: '模型与兼容参数',
          en: 'Models & Compat',
        },
        title: {
          zh: 'models.json 路由协议与 16 项 compat 开关',
          en: 'models.json Routing & 16 Compat Flags',
        },
        description: {
          zh: '内置提供方解析顺序、models.json Schema、三协议适配器、全部 16 项 CompatConfig 参数与凭据模板。',
          en: 'Built-in provider resolution, models.json schema, protocol adapters, all 16 CompatConfig flags, and provider recipes.',
        },
      },
    ],
  },
  {
    id: 'ops-internals',
    title: {
      zh: '排障与架构',
      en: 'Ops & Internals',
    },
    items: [
      {
        slug: 'troubleshooting',
        fileSlug: 'troubleshooting',
        label: {
          zh: '排障指南',
          en: 'Troubleshooting',
        },
        title: {
          zh: '常见故障排查与运行诊断',
          en: 'Troubleshooting & Runtime Diagnostics',
        },
        description: {
          zh: '路由失败、代理证书、兼容网关报错、JSONL 持久化错误、MCP 断连、Codemode 超时与终端渲染问题定位。',
          en: 'Diagnose routing errors, TLS roots, gateway compat 400s, JSONL persistence errors, MCP disconnects, codemode timeouts, and terminal glitches.',
        },
      },
      {
        slug: 'architecture',
        fileSlug: 'architecture',
        label: {
          zh: '架构与开发',
          en: 'Architecture & Dev',
        },
        title: {
          zh: '33-Crate Cordis 微内核架构与开发指南',
          en: '33-Crate Cordis Architecture & Development',
        },
        description: {
          zh: 'Cordis 微内核、33 个 Crate 分层（28 运行时 + 5 纯核心）、三段工具管线、feature 门控与静态/用户故事验收。',
          en: 'Cordis microkernel, 33-crate layered topology (28 runtime + 5 pure core), tool waterfall, Cargo features, and static/story gates.',
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
