export type Locale = 'zh' | 'en';

/**
 * 安全拼接 `import.meta.env.BASE_URL` 与站内路径，
 * 保证在 `/` 或 `/<repo>/` 任意子路径下均与 `trailingSlash: 'always'` 严格匹配。
 */
export function getNormalizedBase(): string {
  const raw = import.meta.env.BASE_URL || '/';
  const withLead = raw.startsWith('/') ? raw : `/${raw}`;
  return withLead.length > 1 && withLead.endsWith('/')
    ? withLead.slice(0, -1)
    : withLead;
}

/** 剥离站点 base 前缀，返回以 `/` 开头的站内路径。 */
function stripBase(pathname: string): string {
  const base = getNormalizedBase();
  if (base === '/') return pathname || '/';
  if (pathname === base) return '/';
  if (pathname.startsWith(`${base}/`)) {
    return pathname.slice(base.length) || '/';
  }
  return pathname || '/';
}

/** 从 URL 路径判断当前语言（剥离 base 后 `/en` 或 `/en/...` 为 `'en'`，其余为 `'zh'`）。 */
export function getLocaleFromUrl(url: URL): Locale {
  const path = stripBase(url.pathname);
  return path === '/en' || path.startsWith('/en/') ? 'en' : 'zh';
}

/** 规范化内容条目 ID 为站内路由 slug（将 `foo/index` 收敛为 `foo`）。 */
export function normalizeContentSlug(entryId: string): string {
  const withoutExt = entryId.replace(/\.mdx?$/, '').replace(/^\/+|\/+$/g, '');
  if (withoutExt === '' || withoutExt === 'index' || withoutExt === 'intro') {
    return 'intro';
  }
  return withoutExt.replace(/\/index$/, '');
}

/** 拆分内容条目 ID 为 `{ locale, slug }`（统一将首页与目录页收敛为规范 slug）。 */
export function splitLocaleContentId(entryId: string): {
  locale: Locale;
  slug: string;
} {
  const withoutExt = entryId.replace(/\.mdx?$/, '').replace(/^\/+|\/+$/g, '');
  if (withoutExt === 'en' || withoutExt.startsWith('en/')) {
    const rest = withoutExt === 'en' ? '' : withoutExt.slice(3);
    return {
      locale: 'en',
      slug: normalizeContentSlug(rest),
    };
  }
  return {
    locale: 'zh',
    slug: normalizeContentSlug(withoutExt),
  };
}

/** 构造带 base 前缀、语言前缀与尾斜杠的文档页面链接（支持 `#hash`）。 */
export function docHref(slug: string, locale: Locale = 'zh'): string {
  const base = getNormalizedBase();
  const [rawPath, hash] = slug.split('#');
  const clean =
    locale === 'en'
      ? splitLocaleContentId(rawPath).slug
      : normalizeContentSlug(rawPath);
  const path =
    locale === 'en'
      ? clean === 'intro'
        ? '/en/'
        : `/en/${clean}/`
      : clean === 'intro'
        ? '/'
        : `/${clean}/`;
  const joined = base === '/' ? path : `${base}${path}`;
  return hash ? `${joined}#${hash}` : joined;
}

/** 构造静态资源/脚本路径（不强加尾斜杠）。 */
export function assetHref(path: string): string {
  const base = getNormalizedBase();
  const clean = path.replace(/^\/+/, '');
  return base === '/' ? `/${clean}` : `${base}/${clean}`;
}

/** 规范化 pathname（剥离 base 前缀、语言前缀与首尾斜杠）用于激活态比对。 */
export function normalizeRouteSlug(pathname: string): string {
  return splitLocaleContentId(stripBase(pathname)).slug;
}

export function getGithubRepoUrl(): string {
  const proc = (
    globalThis as { process?: { env?: Record<string, string | undefined> } }
  ).process;
  const envRepo = proc?.env?.GITHUB_REPOSITORY?.trim() || '';
  if (envRepo && envRepo.includes('/')) {
    return `https://github.com/${envRepo}`;
  }
  return 'https://github.com/Mieluoxxx/mizar';
}

export function getEditPageUrl(
  slug: string,
  filePath?: string,
  locale?: Locale,
): string {
  const repo = getGithubRepoUrl();
  if (filePath) {
    const normalized = filePath.replace(/\\/g, '/').replace(/^\/+/, '');
    const docsMarker = 'src/content/docs/';
    const markerIdx = normalized.indexOf(docsMarker);
    const repoRelative =
      markerIdx !== -1
        ? `site/${normalized.slice(markerIdx)}`
        : normalized.startsWith('site/')
          ? normalized
          : `site/${normalized}`;
    return `${repo}/blob/main/${repoRelative}`;
  }
  const split = splitLocaleContentId(slug);
  const activeLocale = locale ?? split.locale;
  const rel =
    split.slug === 'tui'
      ? 'tui/index'
      : split.slug === 'dev'
        ? 'dev/index'
        : split.slug;
  const localePrefix = activeLocale === 'en' ? 'en/' : '';
  return `${repo}/blob/main/site/src/content/docs/${localePrefix}${rel}.mdx`;
}
