import type { Locale } from '../lib/url';

export type CalloutType =
  | 'note'
  | 'info'
  | 'tip'
  | 'important'
  | 'caution'
  | 'warning'
  | 'danger';

export interface UiStrings {
  siteTitle: string;
  brandWordmark: string;
  brandHomeAria: string;
  skipToContent: string;
  langSwitchAria: string;
  searchTrigger: string;
  searchOpenAria: string;
  searchModalTitle: string;
  searchPlaceholder: string;
  searchInputAria: string;
  searchCloseAria: string;
  searchResultsAria: string;
  searchRecommendedHeading: string;
  searchResultsPrefix: string;
  searchEmptyQueryTemplate: string;
  searchEmptyDefault: string;
  topbarGithubAria: string;
  cornerGithubAria: string;
  sidebarAsideAria: string;
  sidebarNavAria: string;
  sidebarToggleAriaPrefix: string;
  asideNavAria: string;
  tocNavAria: string;
  contentsHeading: string;
  paginationAria: string;
  prevLabel: string;
  nextLabel: string;
  editOnGithub: string;
  opensInNewTabSr: string;
  copyMarkdownIdle: string;
  copyMarkdownCopied: string;
  copyMarkdownFailed: string;
  copyMarkdownAria: string;
  copyMarkdownTitle: string;
  copyStatusCopied: string;
  copyStatusCopiedMarkdown: string;
  copyStatusFailed: string;
  markdownExportGroupAria: string;
  footerNavAria: string;
  footerDevLabel: string;
  footerTroubleshootingLabel: string;
  footerStatusTitle: string;
  footerStatusText: string;
  themeGroupAria: string;
  themeSystemAria: string;
  themeSystemTitle: string;
  themeLightAria: string;
  themeLightTitle: string;
  themeDarkAria: string;
  themeDarkTitle: string;
  codeCopyIdle: string;
  codeCopyCopied: string;
  codeCopyFailed: string;
  codeCopyAria: (label: string) => string;
  codeTabsAria: string;
  codeTabsCopyAria: string;
  copyHeadingAnchorAria: string;
  tableRegionAria: string;
  diagramMetaDesktop: string;
  diagramMetaMobile: string;
  diagramDefaultTitle: string;
  calloutTitles: Record<CalloutType, string>;
}

const UI_STRINGS: Record<Locale, UiStrings> = {
  zh: {
    siteTitle: 'Mizar 使用手册',
    brandWordmark: '使用手册',
    brandHomeAria: 'Mizar 使用手册首页',
    skipToContent: '跳转到正文',
    langSwitchAria: '语言切换',
    searchTrigger: 'Search',
    searchOpenAria: '搜索文档',
    searchModalTitle: '搜索 Mizar 使用手册',
    searchPlaceholder: '搜索文档、配置字段、CLI 参数或命令…',
    searchInputAria: '搜索文档关键词',
    searchCloseAria: '关闭搜索面板',
    searchResultsAria: '搜索结果',
    searchRecommendedHeading: 'RECOMMENDED',
    searchResultsPrefix: 'RESULTS',
    searchEmptyQueryTemplate: '未找到与 “{query}” 相关的文档条目。',
    searchEmptyDefault: '暂无推荐条目。',
    topbarGithubAria: 'GitHub 仓库',
    cornerGithubAria: 'Mizar GitHub 仓库',
    sidebarAsideAria: '使用手册导航栏',
    sidebarNavAria: '使用手册目录',
    sidebarToggleAriaPrefix: '切换目录',
    asideNavAria: '页面导航',
    tocNavAria: '本页目录',
    contentsHeading: 'CONTENTS',
    paginationAria: '文档前后页',
    prevLabel: 'PREVIOUS',
    nextLabel: 'NEXT',
    editOnGithub: '在 GitHub 上编辑',
    opensInNewTabSr: '（在新标签页打开）',
    copyMarkdownIdle: '复制为 Markdown',
    copyMarkdownCopied: '已复制 Markdown',
    copyMarkdownFailed: '复制失败',
    copyMarkdownAria: '复制当前页为 Markdown',
    copyMarkdownTitle: '复制当前页 Markdown',
    copyStatusCopied: '已复制',
    copyStatusCopiedMarkdown: '已复制 Markdown',
    copyStatusFailed: '复制失败',
    markdownExportGroupAria: 'Markdown 导出',
    footerNavAria: '辅助导航',
    footerDevLabel: '架构与开发',
    footerTroubleshootingLabel: '排障指南',
    footerStatusTitle: '会话持久化、树形分叉与 /trace 轨迹分析',
    footerStatusText: 'Session · Trace Ready',
    themeGroupAria: '主题切换',
    themeSystemAria: '跟随系统主题 (System)',
    themeSystemTitle: 'System 主题',
    themeLightAria: '亮色主题 (Light)',
    themeLightTitle: 'Light 主题',
    themeDarkAria: '暗色主题 (Dark)',
    themeDarkTitle: 'Dark 主题',
    codeCopyIdle: 'Copy',
    codeCopyCopied: 'Copied',
    codeCopyFailed: 'Failed',
    codeCopyAria: (label) => `复制 ${label} 代码`,
    codeTabsAria: '代码示例选项卡',
    codeTabsCopyAria: '复制当前选项卡代码',
    copyHeadingAnchorAria: '复制本节锚点链接',
    tableRegionAria: '文档数据表格',
    diagramMetaDesktop: 'WIREFRAME // SPEC',
    diagramMetaMobile: '↔ 滑动全览',
    diagramDefaultTitle: '结构图',
    calloutTitles: {
      note: '说明',
      info: '信息',
      tip: '提示',
      important: '重要',
      caution: '注意',
      warning: '警告',
      danger: '危险',
    },
  },
  en: {
    siteTitle: 'Mizar Docs',
    brandWordmark: 'Docs',
    brandHomeAria: 'Mizar Docs Home',
    skipToContent: 'Skip to content',
    langSwitchAria: 'Language switcher',
    searchTrigger: 'Search',
    searchOpenAria: 'Search docs',
    searchModalTitle: 'Search Mizar Docs',
    searchPlaceholder: 'Search docs, config fields, CLI flags, or commands…',
    searchInputAria: 'Search documentation keywords',
    searchCloseAria: 'Close search modal',
    searchResultsAria: 'Search results',
    searchRecommendedHeading: 'RECOMMENDED',
    searchResultsPrefix: 'RESULTS',
    searchEmptyQueryTemplate: 'No results found for "{query}".',
    searchEmptyDefault: 'No recommended entries.',
    topbarGithubAria: 'GitHub repository',
    cornerGithubAria: 'Mizar GitHub repository',
    sidebarAsideAria: 'Docs sidebar',
    sidebarNavAria: 'Docs navigation',
    sidebarToggleAriaPrefix: 'Toggle navigation',
    asideNavAria: 'Page navigation',
    tocNavAria: 'On this page',
    contentsHeading: 'CONTENTS',
    paginationAria: 'Pagination',
    prevLabel: 'PREVIOUS',
    nextLabel: 'NEXT',
    editOnGithub: 'Edit on GitHub',
    opensInNewTabSr: '(opens in a new tab)',
    copyMarkdownIdle: 'Copy as Markdown',
    copyMarkdownCopied: 'Copied Markdown',
    copyMarkdownFailed: 'Copy failed',
    copyMarkdownAria: 'Copy page as Markdown',
    copyMarkdownTitle: 'Copy page as Markdown',
    copyStatusCopied: 'Copied',
    copyStatusCopiedMarkdown: 'Copied Markdown',
    copyStatusFailed: 'Copy failed',
    markdownExportGroupAria: 'Markdown export',
    footerNavAria: 'Footer navigation',
    footerDevLabel: 'Architecture & Dev',
    footerTroubleshootingLabel: 'Troubleshooting',
    footerStatusTitle: 'Session persistence, /tree branching & /trace profiler',
    footerStatusText: 'Session · Trace Ready',
    themeGroupAria: 'Theme switcher',
    themeSystemAria: 'System theme',
    themeSystemTitle: 'System theme',
    themeLightAria: 'Light theme',
    themeLightTitle: 'Light theme',
    themeDarkAria: 'Dark theme',
    themeDarkTitle: 'Dark theme',
    codeCopyIdle: 'Copy',
    codeCopyCopied: 'Copied',
    codeCopyFailed: 'Failed',
    codeCopyAria: (label) => `Copy ${label} code`,
    codeTabsAria: 'Code example tabs',
    codeTabsCopyAria: 'Copy active tab code',
    copyHeadingAnchorAria: 'Copy link to section',
    tableRegionAria: 'Documentation table',
    diagramMetaDesktop: 'WIREFRAME // SPEC',
    diagramMetaMobile: '↔ Scroll',
    diagramDefaultTitle: 'Diagram',
    calloutTitles: {
      note: 'NOTE',
      info: 'INFO',
      tip: 'TIP',
      important: 'IMPORTANT',
      caution: 'CAUTION',
      warning: 'WARNING',
      danger: 'DANGER',
    },
  },
};

export function getUiStrings(locale: Locale = 'zh'): UiStrings {
  return UI_STRINGS[locale] ?? UI_STRINGS.zh;
}

export function getCalloutDefaultTitle(
  type: string = 'note',
  locale: Locale = 'zh',
): string {
  const titles = getUiStrings(locale).calloutTitles;
  return titles[type as CalloutType] ?? titles.note;
}
