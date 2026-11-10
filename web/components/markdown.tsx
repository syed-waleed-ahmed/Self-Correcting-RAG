import type { ReactNode } from "react";

// Minimal Markdown for model answers: paragraphs, bullet/numbered lists,
// headings, **bold**, `code` and [S1]-style citations. Builds React elements
// (never raw HTML), so model output cannot inject markup.
function inline(text: string): ReactNode[] {
  return text.split(/(\*\*[^*\n]+\*\*|`[^`\n]+`|\[S\d+\])/g).map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return (
        <code key={i} className="rounded bg-slate-100 px-1 py-0.5 font-mono text-[0.85em] dark:bg-slate-800">
          {part.slice(1, -1)}
        </code>
      );
    }
    if (/^\[S\d+\]$/.test(part)) {
      return (
        <sup key={i} className="ml-0.5 rounded bg-brand-50 px-1 font-mono text-[0.7em] text-brand-700 dark:bg-slate-800 dark:text-brand-100">
          {part.slice(1, -1)}
        </sup>
      );
    }
    return part;
  });
}

export function Markdown({ text }: { text: string }) {
  const blocks = text
    .split(/\n{2,}/)
    .map((b) => b.trim())
    .filter(Boolean);
  return (
    <div className="space-y-3 leading-relaxed text-slate-800 dark:text-slate-200">
      {blocks.map((block, i) => {
        const lines = block
          .split("\n")
          .map((l) => l.trim())
          .filter(Boolean);
        if (lines.every((l) => /^[-*]\s+/.test(l))) {
          return (
            <ul key={i} className="list-disc space-y-1 pl-5">
              {lines.map((l, j) => (
                <li key={j}>{inline(l.replace(/^[-*]\s+/, ""))}</li>
              ))}
            </ul>
          );
        }
        if (lines.every((l) => /^\d+[.)]\s+/.test(l))) {
          return (
            <ol key={i} className="list-decimal space-y-1 pl-5">
              {lines.map((l, j) => (
                <li key={j}>{inline(l.replace(/^\d+[.)]\s+/, ""))}</li>
              ))}
            </ol>
          );
        }
        if (/^#{1,6}\s+/.test(block)) {
          return (
            <p key={i} className="font-semibold">
              {inline(block.replace(/^#{1,6}\s+/, ""))}
            </p>
          );
        }
        return (
          <p key={i} className="whitespace-pre-line">
            {inline(block)}
          </p>
        );
      })}
    </div>
  );
}
