// Helpers for rendering agent runs.
import type { AgentRunResult } from "./api";

export interface Evidence {
  id: string; // "S1", "S2", ... as cited in the answer
  source: string;
  text: string;
}

/**
 * Rebuild the numbered evidence the writer saw, in the same order as the
 * server (src/scrag/orchestration/workflow.py, `_evidence`): each document
 * search hit is one item; any other successful tool output is one item.
 */
export function evidenceFromSteps(steps: AgentRunResult["steps"]): Evidence[] {
  const items: Evidence[] = [];
  for (const step of steps) {
    const call = step.call;
    if (call.status !== "ok") continue;
    if (call.tool === "document_search" && Array.isArray(call.output)) {
      for (const hit of call.output as { source?: unknown; text?: unknown }[]) {
        items.push({ id: `S${items.length + 1}`, source: String(hit?.source ?? ""), text: String(hit?.text ?? "") });
      }
    } else {
      items.push({
        id: `S${items.length + 1}`,
        source: `tool:${call.tool}`,
        text: JSON.stringify(call.output ?? null),
      });
    }
  }
  return items;
}
