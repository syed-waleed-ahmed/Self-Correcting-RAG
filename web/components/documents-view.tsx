"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { useApp } from "@/app/providers";
import { api, describeError, type DocumentRecord, type DocumentStatus } from "@/lib/api";
import { Banner } from "./shell";
import { Card, Pill, Spinner, buttonClass, formatBytes, timeUntil } from "./ui";

const ACCEPT = [".txt", ".md", ".markdown", ".html", ".htm", ".csv", ".json", ".pdf", ".docx"];

const FAILURE_TEXT: Record<string, string> = {
  empty_document: "no text could be extracted",
  document_too_long: "longer than an upload may be",
  processing_failed: "processing failed",
  lease_expired: "processing was interrupted",
};

const STATUS_STYLE: Record<DocumentStatus, string> = {
  queued: "border-slate-200 text-slate-600 dark:border-slate-700 dark:text-slate-300",
  processing: "border-amber-200 bg-amber-50 text-amber-800 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200",
  ready: "border-emerald-200 bg-emerald-50 text-emerald-800 dark:border-emerald-900 dark:bg-emerald-950 dark:text-emerald-300",
  failed: "border-red-200 bg-red-50 text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-300",
};

export function DocumentsView() {
  const { connection, demo, session, sessionLoaded, sessionLimits, startSession, deleteSession, handleSessionError } =
    useApp();
  const [docs, setDocs] = useState<DocumentRecord[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [uploading, setUploading] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const token = session?.token ?? null;
  const maxDocs = sessionLimits?.max_documents ?? 5;
  const maxBytes = sessionLimits?.upload_max_bytes ?? 1_000_000;

  const load = useCallback(async () => {
    if (!token) return;
    try {
      setDocs(await api.documents(token));
    } catch (err) {
      handleSessionError(err);
      setError(describeError(err));
    }
  }, [token, handleSessionError]);

  useEffect(() => {
    setDocs(null);
    void load();
  }, [load]);

  // Refresh while anything is still being ingested.
  const pending = docs?.some((d) => d.status === "queued" || d.status === "processing") ?? false;
  useEffect(() => {
    if (!pending) return;
    const timer = setTimeout(() => void load(), 1500);
    return () => clearTimeout(timer);
  }, [pending, docs, load]);

  if (demo && !demo.features?.sessions) {
    return (
      <Banner tone="slate">
        Uploads and history need a database, and this server runs without one. Questions about the sample documents
        still work on the <Link href="/" className="text-brand-600 underline">Ask</Link> page.
      </Banner>
    );
  }

  const begin = async () => {
    setStarting(true);
    setError(null);
    try {
      await startSession();
    } catch (err) {
      setError(describeError(err));
    } finally {
      setStarting(false);
    }
  };

  const upload = async (files: File[]) => {
    if (!token || files.length === 0) return;
    setError(null);
    setNotice(null);
    let count = docs?.length ?? 0;
    for (const file of files) {
      const suffix = file.name.includes(".") ? `.${file.name.split(".").pop()!.toLowerCase()}` : "";
      if (!ACCEPT.includes(suffix)) {
        setError(`${file.name}: unsupported type. Use ${ACCEPT.join(", ")}.`);
        continue;
      }
      if (file.size > maxBytes) {
        setError(`${file.name} is ${formatBytes(file.size)}; the limit is ${formatBytes(maxBytes)}.`);
        continue;
      }
      if (count >= maxDocs) {
        setError(`This session holds at most ${maxDocs} documents. Delete one to add more.`);
        break;
      }
      setUploading(file.name);
      try {
        const result = await api.upload(token, file);
        if (result.duplicate) setNotice(`${file.name} has the same content as ${result.document.filename}; kept the existing one.`);
        else count += 1;
      } catch (err) {
        handleSessionError(err);
        setError(`${file.name}: ${describeError(err)}`);
      }
    }
    setUploading(null);
    await load();
  };

  const remove = async (doc: DocumentRecord) => {
    if (!token) return;
    try {
      await api.deleteDocument(token, doc.id);
      setDocs((prev) => prev?.filter((d) => d.id !== doc.id) ?? null);
    } catch (err) {
      handleSessionError(err);
      setError(describeError(err));
    }
  };

  const wipe = async () => {
    setDeleting(true);
    try {
      await deleteSession();
      setDocs(null);
      setConfirmDelete(false);
      setNotice("Your session and everything in it were deleted.");
    } catch (err) {
      setError(describeError(err));
    } finally {
      setDeleting(false);
    }
  };

  const online = connection === "online";
  const ready = docs?.filter((d) => d.status === "ready").length ?? 0;

  return (
    <div className="space-y-6">
      {error && (
        <div role="alert" className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-900 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
          {error}
        </div>
      )}
      {notice && <Banner tone="slate">{notice}</Banner>}

      {!sessionLoaded ? null : !session ? (
        <Card title="Your documents">
          <p className="max-w-2xl text-sm text-slate-600 dark:text-slate-300">
            Start a private session to upload your own documents (TXT, Markdown, HTML, CSV, JSON, PDF or DOCX) and ask
            questions about them. Your session, uploads and history are deleted automatically after 24 hours, or
            immediately when you choose. Only the extracted text is stored. Please don&apos;t upload personal or
            confidential data: this is a public demo, and text is sent to the LLM provider to answer questions.
          </p>
          <button type="button" onClick={() => void begin()} disabled={starting || !online} className={`mt-4 ${buttonClass.primary}`}>
            {starting ? "Starting…" : "Start a private session"}
          </button>
        </Card>
      ) : (
        <>
          <Card
            title="Session"
            aside={<Pill>expires {timeUntil(session.expiresAt)}</Pill>}
          >
            <div className="flex flex-wrap items-center gap-3 text-sm text-slate-600 dark:text-slate-300">
              <span>
                {docs?.length ?? 0} of {maxDocs} documents · up to {formatBytes(maxBytes)} each
              </span>
              <span className="ml-auto flex gap-2">
                {ready > 0 && (
                  <Link href="/?scope=session" className={buttonClass.secondary}>
                    Ask about my documents →
                  </Link>
                )}
                {confirmDelete ? (
                  <>
                    <button type="button" onClick={() => void wipe()} disabled={deleting} className={buttonClass.danger}>
                      {deleting ? "Deleting…" : "Confirm: delete everything"}
                    </button>
                    <button type="button" onClick={() => setConfirmDelete(false)} disabled={deleting} className="text-xs text-slate-500 hover:underline">
                      Cancel
                    </button>
                  </>
                ) : (
                  <button type="button" onClick={() => setConfirmDelete(true)} className={buttonClass.danger}>
                    Delete all my data
                  </button>
                )}
              </span>
            </div>
          </Card>

          <Card title="Upload">
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
                void upload(Array.from(e.dataTransfer.files));
              }}
              className={`flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-8 text-center text-sm transition ${
                dragging ? "border-brand-500 bg-brand-50 dark:bg-slate-800" : "border-slate-300 dark:border-slate-700"
              }`}
            >
              {uploading ? (
                <p className="flex items-center gap-2 text-slate-600 dark:text-slate-300">
                  <Spinner label="Uploading" /> Uploading {uploading}…
                </p>
              ) : (
                <>
                  <p className="text-slate-600 dark:text-slate-300">Drop files here, or</p>
                  <button
                    type="button"
                    onClick={() => fileInput.current?.click()}
                    disabled={!online || (docs?.length ?? 0) >= maxDocs}
                    className={buttonClass.secondary}
                  >
                    Choose files
                  </button>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    {ACCEPT.join(" ")} · max {formatBytes(maxBytes)} · PDFs up to 50 pages
                  </p>
                </>
              )}
              <input
                ref={fileInput}
                type="file"
                multiple
                accept={ACCEPT.join(",")}
                className="sr-only"
                aria-label="Choose files to upload"
                onChange={(e) => {
                  void upload(Array.from(e.target.files ?? []));
                  e.target.value = "";
                }}
              />
            </div>
          </Card>

          <Card title="Documents" aside={pending ? <Spinner label="Processing" /> : null}>
            {docs === null ? (
              <p className="text-sm text-slate-500">Loading…</p>
            ) : docs.length === 0 ? (
              <p className="text-sm text-slate-500 dark:text-slate-400">No documents yet.</p>
            ) : (
              <ul className="divide-y divide-slate-100 dark:divide-slate-800">
                {docs.map((doc) => (
                  <li key={doc.id} className="flex flex-wrap items-center gap-3 py-2.5 text-sm">
                    <span className="min-w-0 flex-1 truncate font-medium text-slate-800 dark:text-slate-100" title={doc.filename}>
                      {doc.filename}
                    </span>
                    <span className="text-xs text-slate-500">{formatBytes(doc.size_bytes)}</span>
                    <span className={`rounded-full border px-2 py-0.5 text-xs ${STATUS_STYLE[doc.status]}`}>
                      {doc.status === "ready"
                        ? `ready · ${doc.chunk_count} chunk${doc.chunk_count === 1 ? "" : "s"}`
                        : doc.status === "failed"
                          ? `failed: ${FAILURE_TEXT[doc.error ?? ""] ?? doc.error ?? "unknown"}`
                          : doc.status}
                    </span>
                    <button
                      type="button"
                      onClick={() => void remove(doc)}
                      aria-label={`Delete ${doc.filename}`}
                      className="text-xs text-slate-400 hover:text-red-600"
                    >
                      Delete
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </>
      )}
    </div>
  );
}
