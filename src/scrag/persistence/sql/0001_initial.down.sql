-- Reverts 0001_initial. The `vector` extension is left installed: other
-- schemas in the same database may use it.
DROP TABLE IF EXISTS
    scrag.usage_counters,
    scrag.jobs,
    scrag.messages,
    scrag.conversations,
    scrag.chunks,
    scrag.documents,
    scrag.sessions;
