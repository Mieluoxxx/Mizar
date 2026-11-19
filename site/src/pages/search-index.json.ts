import type { APIRoute } from 'astro';
import { getCollection } from 'astro:content';
import { getGroupTitleForSlug } from '../data/navigation';
import { extractPlainText } from '../lib/markdown';
import { docHref, splitLocaleContentId } from '../lib/url';

export const GET: APIRoute = async () => {
  const entries = await getCollection('docs');

  const items = entries
    .sort((a, b) => a.data.order - b.data.order)
    .map((entry) => {
      const { locale, slug } = splitLocaleContentId(entry.id);
      const groupTitle = getGroupTitleForSlug(slug, locale);
      const headings = Array.from(
        (entry.body || '').matchAll(/^#{2,3}\s+(.+)$/gm),
      ).map((m) => m[1].trim());
      return {
        locale,
        slug,
        title: entry.data.title,
        group: groupTitle,
        description: entry.data.description,
        headings,
        text: extractPlainText(entry.body || '', locale),
        href: docHref(slug, locale),
      };
    });

  return new Response(JSON.stringify(items), {
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
    },
  });
};
