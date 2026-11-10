// Incremental parser for Server-Sent Events read from a fetch() body.
// EventSource only supports GET, and the streaming endpoints are POSTs, so the
// stream is parsed by hand. Only the `event:` and `data:` fields are used.

export interface SseEvent {
  event: string;
  data: string;
}

function parseBlock(block: string): SseEvent | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (!line || line.startsWith(":")) continue; // blank or comment (heartbeat)
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }
  return data.length > 0 ? { event, data: data.join("\n") } : null;
}

export interface SseParser {
  /** Feed a decoded chunk; returns the events it completed. */
  push(chunk: string): SseEvent[];
  /** Parse whatever is left once the stream has ended. */
  flush(): SseEvent[];
}

export function createSseParser(): SseParser {
  let buffer = "";
  return {
    push(chunk) {
      buffer += chunk.replace(/\r/g, "");
      const events: SseEvent[] = [];
      let end = buffer.indexOf("\n\n");
      while (end !== -1) {
        const parsed = parseBlock(buffer.slice(0, end));
        if (parsed) events.push(parsed);
        buffer = buffer.slice(end + 2);
        end = buffer.indexOf("\n\n");
      }
      return events;
    },
    flush() {
      const parsed = parseBlock(buffer);
      buffer = "";
      return parsed ? [parsed] : [];
    },
  };
}
