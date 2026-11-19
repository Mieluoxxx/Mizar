import { getCalloutDefaultTitle, getUiStrings } from '../data/i18n';
import type { Locale } from './url';

/**
 * 将 MDX 原始正文清理为可直接复制或检索的纯正 Markdown / 纯文本。
 */
export function cleanMdxToMarkdown(
  title: string,
  rawBody: string,
  locale: Locale = 'zh',
): string {
  const ui = getUiStrings(locale);
  // 1. 移除顶层 import 块，并提取顶层 CodeTabs export 数组以供还原
  let body = rawBody.replace(/^import\s+[\s\S]*?;\s*\n?/gm, '');
  const exportedTabs = new Map<string, string>();
  body = body.replace(
    /^export\s+const\s+(\w+)\s*=\s*(\[[\s\S]*?\]);\s*\n?/gm,
    (_m, name: string, arrSrc: string) => {
      exportedTabs.set(name, arrSrc);
      return '';
    },
  );

  // 1b. 将带 title/kind meta 的 Markdown fenced code block 转为标准 Markdown 标题 + 围栏
  body = body.replace(
    /^(`{3,4})([^\s`]+)[ \t]+(?=[^\n]*(?:\btitle=|\bkind=))([^\n]+)$/gm,
    (_m, fence: string, lang: string, meta: string) => {
      const titleMatch = meta.match(/\btitle="([^"]*)"|\btitle='([^']*)'/);
      const blockTitle = titleMatch?.[1] ?? titleMatch?.[2];
      const prefix = blockTitle ? `**${blockTitle}**\n\n` : '';
      return `${prefix}${fence}${lang}`;
    },
  );

  // 2. 将 <DiagramEmbed ... /> 转为引用摘要
  body = body.replace(/<DiagramEmbed\b([\s\S]*?)\/>/g, (_m, attrs: string) => {
    const idMatch = attrs.match(/\bid="([^"]+)"/);
    const titleMatch = attrs.match(/\btitle="([^"]+)"/);
    const captionMatch = attrs.match(/\bcaption="([^"]+)"/);
    const diagramTitle = titleMatch
      ? titleMatch[1]
      : idMatch?.[1] || ui.diagramDefaultTitle;
    const captionLine = captionMatch ? `\n> ${captionMatch[1]}` : '';
    const prefix = locale === 'en' ? 'Diagram:' : '图解：';
    return `> **${prefix}${locale === 'en' ? ' ' : ''}${diagramTitle}**${captionLine}`;
  });

  // 3. 将 <DocCard ... /> 转为 Markdown 列表项
  body = body.replace(/^[ \t]*<DocCard\b([\s\S]*?)\/?>/gm, (_m, attrs: string) => {
    const titleMatch = attrs.match(/\btitle="([^"]+)"/);
    const descMatch = attrs.match(/\bdescription="([^"]+)"/);
    if (!titleMatch && !descMatch) return '';
    const sep = locale === 'en' ? ': ' : '：';
    return `- **${titleMatch?.[1] || ''}**${sep}${descMatch?.[1] || ''}`;
  });

  // 4. 将 <CodeTabs ... tabs={[...]} /> 或 <CodeTabs tabs={varName} /> 拆分为带标题的 fenced code blocks
  const formatTabArray = (arrSrc: string): string => {
    const blocks: string[] = [];
    const tabRegex = /\{\s*label:\s*'([^']+)',\s*lang:\s*'([^']+)',\s*code:\s*`([\s\S]*?)`,?\s*\}/g;
    let match: RegExpExecArray | null;
    while ((match = tabRegex.exec(arrSrc)) !== null) {
      const [, label, lang, code] = match;
      const normalizedCode = code.replace(/\\\$\{/g, '${').replace(/\\`/g, '`').trim();
      const fence = normalizedCode.includes('```') ? '````' : '```';
      blocks.push(`**${label}**\n\n${fence}${lang}\n${normalizedCode}\n${fence}`);
    }
    return blocks.join('\n\n');
  };
  body = body.replace(/<CodeTabs\b[\s\S]*?tabs=\{(\[[\s\S]*?\])\}\s*\/>/g, (_m, arrSrc: string) =>
    formatTabArray(arrSrc),
  );
  body = body.replace(/<CodeTabs\b[\s\S]*?tabs=\{(\w+)\}\s*\/>/g, (_m, name: string) => {
    const arrSrc = exportedTabs.get(name);
    return arrSrc ? formatTabArray(arrSrc) : '';
  });

  // 保护 fenced code blocks，避免后续 JSX/HTML 清理误伤示例代码
  const codePlaceholders: string[] = [];
  body = body.replace(/^(`{3,4})[^\n]*\n[\s\S]*?^\1[ \t]*$/gm, (block) => {
    const token = `__MIZAR_CODE_BLOCK_${codePlaceholders.length}__`;
    codePlaceholders.push(block);
    return token;
  });

  const decodeInlineHtml = (text: string): string =>
    text
      .replace(/<br\s*\/?>/gi, '\n')
      .replace(/<strong>([\s\S]*?)<\/strong>/g, '**$1**')
      .replace(/<code>([\s\S]*?)<\/code>/g, '`$1`')
      .replace(/<\/?(?:p|div)[^>]*>/g, '')
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .replace(/&#36;/g, '$')
      .replace(/&#123;/g, '{')
      .replace(/&#125;/g, '}');

  // 5. 将 <Callout> 转为完整的多行 Markdown blockquote
  body = body.replace(
    /<Callout(?:\s+type="([^"]+)")?(?:\s+title="([^"]+)")?\s*>([\s\S]*?)<\/Callout>/g,
    (_m, type: string | undefined, calloutTitle: string | undefined, inner: string) => {
      const label = calloutTitle || getCalloutDefaultTitle(type || 'note', locale);
      const lines = decodeInlineHtml(inner)
        .trim()
        .split('\n')
        .map((line) => line.trim())
        .map((line) => (line ? `> ${line}` : '>'));
      return `> **${label}**\n>\n${lines.join('\n')}`;
    },
  );

  // 6. 将 <Steps> 转为标准有序列表（去除 JSX 缩进，避免变成缩进代码块）
  body = body.replace(/<Steps\b[^>]*>([\s\S]*?)<\/Steps>/g, (_m, inner: string) => {
    const items: string[] = [];
    let stepIdx = 0;
    for (const liMatch of inner.matchAll(/<li\b[^>]*>([\s\S]*?)<\/li>/g)) {
      stepIdx += 1;
      const text = decodeInlineHtml(liMatch[1])
        .split('\n')
        .map((line) => line.trim())
        .filter(Boolean)
        .join(' ');
      if (text) items.push(`${stepIdx}. ${text}`);
    }
    return items.join('\n');
  });

  // 7. 移除剩余包装标签并还原内联 HTML / 实体
  body = decodeInlineHtml(body.replace(/<\/?CardGroup[^>]*>/g, ''));

  // 还原受保护的代码块
  body = body.replace(/__MIZAR_CODE_BLOCK_(\d+)__/g, (_m, idx: string) => codePlaceholders[Number(idx)] ?? '');

  const cleaned = body.replace(/\n{3,}/g, '\n\n').trim();
  return title ? `# ${title}\n\n${cleaned}\n` : `${cleaned}\n`;
}

/** 提取搜索用纯文本片段。 */
export function extractPlainText(
  rawBody: string,
  locale: Locale = 'zh',
): string {
  return cleanMdxToMarkdown('', rawBody, locale)
    .replace(/^#+\s*/gm, '')
    .replace(/```[\s\S]*?```/g, (block) => block.replace(/```[^\n]*/g, ''))
    .replace(/<[^>]+>/g, ' ')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/[`*_>|#{}]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}
