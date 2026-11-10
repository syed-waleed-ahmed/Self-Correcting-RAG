import assert from "node:assert/strict";
import { test } from "node:test";
import { clearSession, loadSession, saveSession } from "./session.ts";

function memoryStorage() {
  const map = new Map<string, string>();
  return {
    getItem: (k: string) => map.get(k) ?? null,
    setItem: (k: string, v: string) => void map.set(k, v),
    removeItem: (k: string) => void map.delete(k),
    size: () => map.size,
  };
}

const session = { token: "scs_abc", sessionId: "s1", expiresAt: "2026-11-11T08:00:00Z" };

test("round-trips a live session", () => {
  const storage = memoryStorage();
  saveSession(session, storage);
  assert.deepEqual(loadSession(storage, new Date("2026-11-10T08:00:00Z")), session);
  clearSession(storage);
  assert.equal(loadSession(storage), null);
});

test("drops an expired session", () => {
  const storage = memoryStorage();
  saveSession(session, storage);
  assert.equal(loadSession(storage, new Date("2026-11-12T00:00:00Z")), null);
  assert.equal(storage.size(), 0);
});

test("survives corrupt data and unavailable storage", () => {
  const storage = memoryStorage();
  storage.setItem("scrag.session.v1", "{not json");
  assert.equal(loadSession(storage), null);
  const throwing = {
    getItem: () => {
      throw new Error("blocked");
    },
    setItem: () => {
      throw new Error("blocked");
    },
    removeItem: () => {
      throw new Error("blocked");
    },
  };
  assert.equal(loadSession(throwing), null);
  assert.doesNotThrow(() => saveSession(session, throwing));
  assert.doesNotThrow(() => clearSession(throwing));
  assert.equal(loadSession(null), null);
});
