// The visitor's anonymous session, kept in this browser only.
//
// The token is a bearer secret for a 24-hour anonymous session that holds no
// account data. localStorage can come back empty or throw (private windows,
// blocked storage), so every access is guarded and the app works without it.

export interface StoredSession {
  token: string;
  sessionId: string;
  expiresAt: string;
}

const KEY = "scrag.session.v1";

type StorageLike = Pick<Storage, "getItem" | "setItem" | "removeItem">;

function defaultStorage(): StorageLike | null {
  try {
    return typeof window === "undefined" ? null : window.localStorage;
  } catch {
    return null;
  }
}

export function loadSession(
  storage: StorageLike | null = defaultStorage(),
  now: Date = new Date(),
): StoredSession | null {
  try {
    const raw = storage?.getItem(KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<StoredSession>;
    const expires = new Date(parsed.expiresAt ?? "");
    if (!parsed.token || !parsed.sessionId || Number.isNaN(expires.getTime()) || expires <= now) {
      storage?.removeItem(KEY);
      return null;
    }
    return { token: parsed.token, sessionId: parsed.sessionId, expiresAt: parsed.expiresAt! };
  } catch {
    return null;
  }
}

export function saveSession(session: StoredSession, storage: StorageLike | null = defaultStorage()): void {
  try {
    storage?.setItem(KEY, JSON.stringify(session));
  } catch {
    /* storage unavailable: the session lasts until the page is closed */
  }
}

export function clearSession(storage: StorageLike | null = defaultStorage()): void {
  try {
    storage?.removeItem(KEY);
  } catch {
    /* nothing to clear */
  }
}
