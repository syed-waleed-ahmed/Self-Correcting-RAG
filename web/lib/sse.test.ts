import assert from "node:assert/strict";
import { test } from "node:test";
import { createSseParser } from "./sse.ts";

test("parses events split across arbitrary chunk boundaries", () => {
  const parser = createSseParser();
  const stream = 'event: progress\ndata: {"stage":"retrieve"}\n\nevent: result\ndata: {"answer":"ok"}\n\n';
  const events = [];
  for (let i = 0; i < stream.length; i += 7) events.push(...parser.push(stream.slice(i, i + 7)));
  assert.deepEqual(events, [
    { event: "progress", data: '{"stage":"retrieve"}' },
    { event: "result", data: '{"answer":"ok"}' },
  ]);
});

test("ignores heartbeat comments and blank blocks", () => {
  const parser = createSseParser();
  assert.deepEqual(parser.push(": ping\n\n\n\nevent: error\ndata: {}\n\n"), [{ event: "error", data: "{}" }]);
});

test("joins multi-line data, defaults the event name, and handles CRLF", () => {
  const parser = createSseParser();
  assert.deepEqual(parser.push("data: line 1\r\ndata: line 2\r\n\r\n"), [
    { event: "message", data: "line 1\nline 2" },
  ]);
});

test("flush returns a final event without a trailing blank line", () => {
  const parser = createSseParser();
  assert.deepEqual(parser.push("event: result\ndata: {}"), []);
  assert.deepEqual(parser.flush(), [{ event: "result", data: "{}" }]);
  assert.deepEqual(parser.flush(), []);
});
