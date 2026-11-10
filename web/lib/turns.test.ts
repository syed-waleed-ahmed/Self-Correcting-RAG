import assert from "node:assert/strict";
import { test } from "node:test";
import { turnsFromMessages } from "./turns.ts";

const msg = (id: number, role: "user" | "assistant", content: string, metadata = {}) => ({
  id,
  role,
  content,
  metadata,
  created_at: "2026-11-10T00:00:00Z",
});

test("pairs saved messages into question/answer turns", () => {
  const turns = turnsFromMessages([
    msg(1, "user", "What is RAG?"),
    msg(2, "assistant", "Retrieval + generation.", { evaluation_status: "passed", evaluation_score: 0.9, sources: ["a.md"] }),
    msg(3, "user", "And BM25?"),
    msg(4, "assistant", "Keyword ranking.", { sources: [] }),
  ]);
  assert.deepEqual(
    turns.map((t) => [t.question, t.answer, t.status, t.score, t.sources]),
    [
      ["What is RAG?", "Retrieval + generation.", "passed", 0.9, ["a.md"]],
      ["And BM25?", "Keyword ranking.", undefined, null, []],
    ],
  );
});

test("keeps a question whose answer is missing", () => {
  const turns = turnsFromMessages([msg(1, "user", "Unanswered?")]);
  assert.equal(turns.length, 1);
  assert.equal(turns[0]?.answer, "");
});
