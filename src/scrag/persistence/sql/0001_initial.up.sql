-- Phase 4 initial schema: anonymous sessions, uploaded documents and their
-- chunks, conversation history, a durable job queue and usage counters.
--
-- Everything lives in the `scrag` schema. Supabase's Data API only exposes
-- `public`, and row level security is enabled below without any policy as a
-- second barrier, so these tables are reachable only through the API server's
-- own database role (the table owner, which RLS does not restrict).
--
-- Every row a visitor owns hangs off scrag.sessions with ON DELETE CASCADE:
-- deleting a session (on request, or when it expires) deletes all of its data.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE scrag.sessions (
    id          uuid PRIMARY KEY,
    token_hash  bytea NOT NULL UNIQUE,  -- sha256 of the bearer token; the token is never stored
    kind        text NOT NULL DEFAULT 'anonymous' CHECK (kind IN ('anonymous', 'user')),
    subject     text,                   -- identity-provider user id for kind = 'user'
    created_at  timestamptz NOT NULL DEFAULT now(),
    expires_at  timestamptz NOT NULL
);
CREATE INDEX sessions_expires_at_idx ON scrag.sessions (expires_at);

CREATE TABLE scrag.documents (
    id            uuid PRIMARY KEY,
    session_id    uuid NOT NULL REFERENCES scrag.sessions (id) ON DELETE CASCADE,
    filename      text NOT NULL,
    content_hash  text NOT NULL,
    size_bytes    integer NOT NULL CHECK (size_bytes >= 0),
    content       text NOT NULL,       -- extracted text; the original bytes are not kept
    status        text NOT NULL DEFAULT 'queued'
                  CHECK (status IN ('queued', 'processing', 'ready', 'failed')),
    error         text,                -- stable error code, never a stack trace
    chunk_count   integer NOT NULL DEFAULT 0,
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now(),
    UNIQUE (session_id, content_hash)
);
CREATE INDEX documents_session_idx ON scrag.documents (session_id, created_at);

-- `embedding` has no fixed dimension, and rows record the model that produced
-- them: searches filter on the current model, so changing the embedding model
-- can never compare vectors of different spaces. Per-session data is small, so
-- search is an exact scan over one session's rows (an HNSW index would
-- post-filter by session and silently return too few results).
CREATE TABLE scrag.chunks (
    id               bigserial PRIMARY KEY,
    document_id      uuid NOT NULL REFERENCES scrag.documents (id) ON DELETE CASCADE,
    session_id       uuid NOT NULL REFERENCES scrag.sessions (id) ON DELETE CASCADE,
    position         integer NOT NULL,
    text             text NOT NULL,
    embedding        vector NOT NULL,
    embedding_model  text NOT NULL,
    tsv              tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    UNIQUE (document_id, position)
);
CREATE INDEX chunks_session_idx ON scrag.chunks (session_id, embedding_model);
CREATE INDEX chunks_tsv_idx ON scrag.chunks USING gin (tsv);

CREATE TABLE scrag.conversations (
    id          uuid PRIMARY KEY,
    session_id  uuid NOT NULL REFERENCES scrag.sessions (id) ON DELETE CASCADE,
    title       text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX conversations_session_idx ON scrag.conversations (session_id, updated_at DESC);

CREATE TABLE scrag.messages (
    id               bigserial PRIMARY KEY,
    conversation_id  uuid NOT NULL REFERENCES scrag.conversations (id) ON DELETE CASCADE,
    session_id       uuid NOT NULL REFERENCES scrag.sessions (id) ON DELETE CASCADE,
    role             text NOT NULL CHECK (role IN ('user', 'assistant')),
    content          text NOT NULL,
    metadata         jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX messages_conversation_idx ON scrag.messages (conversation_id, id);
CREATE INDEX messages_session_idx ON scrag.messages (session_id);

-- Durable job queue. Workers claim with FOR UPDATE SKIP LOCKED and hold a
-- lease (locked_until); a job whose worker died is claimable again once its
-- lease expires, which is how interrupted work resumes after a restart.
CREATE TABLE scrag.jobs (
    id            uuid PRIMARY KEY,
    kind          text NOT NULL,
    payload       jsonb NOT NULL DEFAULT '{}'::jsonb,
    session_id    uuid REFERENCES scrag.sessions (id) ON DELETE CASCADE,
    status        text NOT NULL DEFAULT 'queued'
                  CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
    attempts      integer NOT NULL DEFAULT 0,
    max_attempts  integer NOT NULL DEFAULT 3 CHECK (max_attempts >= 1),
    run_after     timestamptz NOT NULL DEFAULT now(),
    locked_by     text,
    locked_until  timestamptz,
    last_error    text,                -- operator-facing; not returned by the API
    result        jsonb,
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX jobs_claimable_idx ON scrag.jobs (run_after) WHERE status IN ('queued', 'running');

-- Quota counters that must survive restarts (e.g. the demo's daily live-run budget).
CREATE TABLE scrag.usage_counters (
    subject       text NOT NULL,
    window_start  date NOT NULL,
    used          integer NOT NULL DEFAULT 0 CHECK (used >= 0),
    PRIMARY KEY (subject, window_start)
);

ALTER TABLE scrag.sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE scrag.documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE scrag.chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE scrag.conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE scrag.messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE scrag.jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE scrag.usage_counters ENABLE ROW LEVEL SECURITY;
