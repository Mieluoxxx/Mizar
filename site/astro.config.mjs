import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';

function resolveSiteAndBase() {
  const ghRepo = process.env.GITHUB_REPOSITORY?.trim() || '';
  const slashIdx = ghRepo.indexOf('/');
  const ghOwner = slashIdx > 0 ? ghRepo.slice(0, slashIdx) : '';
  const ghName = slashIdx > 0 ? ghRepo.slice(slashIdx + 1) : '';

  const envSite = process.env.ASTRO_SITE?.trim();
  const envBase = process.env.ASTRO_BASE?.trim();

  const site =
    envSite ||
    (ghOwner ? `https://${ghOwner}.github.io` : 'https://mizar.local');

  let base = '/';
  if (envBase) {
    base = envBase;
  } else if (ghOwner && ghName) {
    base =
      ghName.toLowerCase() === `${ghOwner.toLowerCase()}.github.io`
        ? '/'
        : `/${ghName}`;
  }

  if (!base.startsWith('/')) base = `/${base}`;
  if (base.length > 1 && base.endsWith('/')) base = base.slice(0, -1);

  return { site, base };
}

const { site, base } = resolveSiteAndBase();

/** 将 MDX 中的站内绝对路径链接自动补全 base 前缀（兼容 GitHub Pages 子路径）。 */
function rehypeBaseLinks() {
  const prefixHref = (href) => {
    if (
      typeof href === 'string' &&
      href.startsWith('/') &&
      !href.startsWith('//') &&
      !href.startsWith(`${base}/`) &&
      !href.startsWith(`${base}#`) &&
      href !== base
    ) {
      return `${base}${href}`;
    }
    return href;
  };
  return (tree) => {
    if (base === '/') return;
    const visit = (node) => {
      if (!node || typeof node !== 'object') return;
      if (
        node.type === 'element' &&
        node.tagName === 'a' &&
        typeof node.properties?.href === 'string'
      ) {
        node.properties.href = prefixHref(node.properties.href);
      } else if (
        (node.type === 'mdxJsxFlowElement' || node.type === 'mdxJsxTextElement') &&
        node.name === 'a' &&
        Array.isArray(node.attributes)
      ) {
        for (const attr of node.attributes) {
          if (attr?.type === 'mdxJsxAttribute' && attr.name === 'href' && typeof attr.value === 'string') {
            attr.value = prefixHref(attr.value);
          }
        }
      }
      if (Array.isArray(node.children)) {
        for (const child of node.children) visit(child);
      }
    };
    visit(tree);
  };
}

export default defineConfig({
  site,
  base,
  trailingSlash: 'always',
  integrations: [mdx()],
  markdown: {
    rehypePlugins: [rehypeBaseLinks],
    shikiConfig: {
      theme: 'css-variables',
      wrap: false,
      transformers: [
        {
          pre(node) {
            const rawMeta = this.options.meta?.__raw;
            if (!rawMeta) return;
            const titleMatch = rawMeta.match(/\btitle="([^"]*)"|\btitle='([^']*)'/);
            const kindMatch = rawMeta.match(/\bkind="(file|terminal)"/);
            const title = titleMatch?.[1] ?? titleMatch?.[2];
            if (title) node.properties['data-title'] = title;
            if (kindMatch?.[1]) node.properties['data-kind'] = kindMatch[1];
          },
        },
      ],
    },
  },
});
