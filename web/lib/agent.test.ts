import assert from "node:assert/strict";
import { test } from "node:test";
import { evidenceFromSteps } from "./agent.ts";

const call = (tool: string, status: string, output: unknown) => ({
  index: 0,
  purpose: "",
  call: { tool, status, output, input: {}, error: null, latency_ms: 1 },
});

test("numbers evidence like the server: one per search hit, one per other tool", () => {
  const steps = [
    call("document_search", "ok", [
      { source: "a.md", text: "alpha" },
      { source: "b.md", text: "beta" },
    ]),
    call("calculator", "ok", { expression: "60*3", result: 180 }),
    call("document_search", "timeout", null),
  ] as Parameters<typeof evidenceFromSteps>[0];
  assert.deepEqual(evidenceFromSteps(steps), [
    { id: "S1", source: "a.md", text: "alpha" },
    { id: "S2", source: "b.md", text: "beta" },
    { id: "S3", source: "tool:calculator", text: '{"expression":"60*3","result":180}' },
  ]);
});
