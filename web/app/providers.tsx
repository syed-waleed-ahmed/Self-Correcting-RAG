"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { ApiError, api, type DemoInfo } from "@/lib/api";
import { clearSession, loadSession, saveSession, type StoredSession } from "@/lib/session";

export type Connection = "connecting" | "waking" | "online" | "offline";

interface AppState {
  connection: Connection;
  demo: DemoInfo | null;
  refreshDemo: () => Promise<void>;
  /** The visitor's anonymous session (null: none yet). `sessionLoaded` is false until read from storage. */
  session: StoredSession | null;
  sessionLoaded: boolean;
  sessionLimits: Record<string, number> | null;
  startSession: () => Promise<StoredSession>;
  /** Forget a session the server no longer accepts (expired or deleted elsewhere). */
  forgetSession: () => void;
  /** Delete the session and everything in it on the server, then forget it. */
  deleteSession: () => Promise<void>;
  /** Call with any API error: forgets the session if the server rejected it. */
  handleSessionError: (err: unknown) => void;
}

const AppContext = createContext<AppState | null>(null);

export function useApp(): AppState {
  const state = useContext(AppContext);
  if (!state) throw new Error("useApp must be used inside <Providers>");
  return state;
}

export function Providers({ children }: { children: ReactNode }) {
  const [connection, setConnection] = useState<Connection>("connecting");
  const [demo, setDemo] = useState<DemoInfo | null>(null);
  const [session, setSession] = useState<StoredSession | null>(null);
  const [sessionLoaded, setSessionLoaded] = useState(false);
  const [sessionLimits, setSessionLimits] = useState<Record<string, number> | null>(null);
  const started = useRef(false);

  const refreshDemo = useCallback(async () => {
    try {
      setDemo(await api.demo());
    } catch {
      /* keep the last known info */
    }
  }, []);

  // The session lives in this browser only; read it after mount (no SSR mismatch).
  useEffect(() => {
    setSession(loadSession());
    setSessionLoaded(true);
  }, []);

  // Free-tier instances sleep when idle: poll /health until the API is awake.
  useEffect(() => {
    if (started.current) return;
    started.current = true;
    let cancelled = false;
    const deadline = Date.now() + 150_000;
    const poll = async (attempt: number) => {
      try {
        await api.health();
        if (cancelled) return;
        setConnection("online");
        void refreshDemo();
      } catch {
        if (cancelled) return;
        if (Date.now() > deadline) {
          setConnection("offline");
          return;
        }
        setConnection(attempt === 0 ? "connecting" : "waking");
        setTimeout(() => void poll(attempt + 1), 5_000);
      }
    };
    void poll(0);
    return () => {
      cancelled = true;
    };
  }, [refreshDemo]);

  // Learn the session's limits (and drop it if the server no longer knows it).
  useEffect(() => {
    if (!session || connection !== "online") return;
    let cancelled = false;
    api
      .session(session.token)
      .then((info) => !cancelled && setSessionLimits(info.limits))
      .catch((err) => {
        if (!cancelled && err instanceof ApiError && err.code === "session_required") {
          clearSession();
          setSession(null);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [session, connection]);

  const startSession = useCallback(async () => {
    const created = await api.createSession();
    const stored = { token: created.token, sessionId: created.session_id, expiresAt: created.expires_at };
    saveSession(stored);
    setSession(stored);
    setSessionLimits(created.limits);
    return stored;
  }, []);

  const forgetSession = useCallback(() => {
    clearSession();
    setSession(null);
    setSessionLimits(null);
  }, []);

  const deleteSession = useCallback(async () => {
    if (session) {
      try {
        await api.deleteSession(session.token);
      } catch (err) {
        if (!(err instanceof ApiError && err.code === "session_required")) throw err;
      }
    }
    forgetSession();
  }, [session, forgetSession]);

  const handleSessionError = useCallback(
    (err: unknown) => {
      if (err instanceof ApiError && err.code === "session_required") forgetSession();
    },
    [forgetSession],
  );

  const value = useMemo<AppState>(
    () => ({
      connection,
      demo,
      refreshDemo,
      session,
      sessionLoaded,
      sessionLimits,
      startSession,
      forgetSession,
      deleteSession,
      handleSessionError,
    }),
    [connection, demo, refreshDemo, session, sessionLoaded, sessionLimits, startSession, forgetSession, deleteSession, handleSessionError],
  );
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}
