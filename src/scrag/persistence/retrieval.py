"""Retrieval over a session's uploads, optionally fused with the sample corpus.

The session's chunks are ranked twice in Postgres (pgvector cosine distance and
full-text ``ts_rank_cd``) and fused with Reciprocal Rank Fusion, mirroring the
in-process hybrid retriever. With ``include_samples`` that fused list is fused
once more with the sample-corpus ranking, so each source counts as one ranking
and neither drowns out the other.
"""

from __future__ import annotations

import uuid

from ..embeddings import Embedder, embed_checked
from ..models import Chunk, RetrievedChunk
from ..retriever import SupportsRetrieve
from ..sparse import reciprocal_rank_fusion
from .documents import ChunkHit, DocumentStore


def _to_retrieved(hit: ChunkHit, *, score: float, keyword: float | None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            id=f"upload:{hit.chunk_id}",
            source=hit.filename,
            text=hit.text,
            position=hit.position,
            metadata={"origin": "upload", "document_id": str(hit.document_id)},
        ),
        score=score,
        keyword_score=keyword,
    )


class SessionRetriever:
    def __init__(
        self,
        embedder: Embedder,
        documents: DocumentStore,
        session_id: uuid.UUID,
        *,
        embedding_model: str,
        embedding_dim: int,
        default_top_k: int = 5,
        candidate_pool: int = 20,
        rrf_k: int = 60,
        samples: SupportsRetrieve | None = None,
    ) -> None:
        self.embedder = embedder
        self.documents = documents
        self.session_id = session_id
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.default_top_k = default_top_k
        self.candidate_pool = candidate_pool
        self.rrf_k = rrf_k
        self.samples = samples
        self.mode = "session+samples" if samples is not None else "session"

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        k = top_k or self.default_top_k
        pool = max(k, self.candidate_pool)
        vector = embed_checked(self.embedder, [query], self.embedding_dim)[0]
        dense = self.documents.search_dense(
            self.session_id, vector, pool, embedding_model=self.embedding_model
        )
        keyword = self.documents.search_keyword(self.session_id, query, pool)

        cosine = {hit.chunk_id: hit.score for hit in dense}
        keyword_scores = {hit.chunk_id: hit.score for hit in keyword}
        hits = {hit.chunk_id: hit for hit in [*dense, *keyword]}
        fused = reciprocal_rank_fusion(
            [[h.chunk_id for h in dense], [h.chunk_id for h in keyword]], k=self.rrf_k
        )
        session_ranked = sorted(fused, key=lambda cid: (-fused[cid], -cosine.get(cid, 0.0)))
        mine = {
            f"upload:{cid}": _to_retrieved(
                hits[cid], score=cosine.get(cid, 0.0), keyword=keyword_scores.get(cid)
            )
            for cid in session_ranked
        }
        if self.samples is None:
            ranked = session_ranked[:k]
            return [
                mine[f"upload:{cid}"].model_copy(update={"fusion_score": fused[cid]})
                for cid in ranked
            ]

        sample_hits = self.samples.retrieve(query, pool)
        theirs = {rc.chunk.id: rc for rc in sample_hits}
        final = reciprocal_rank_fusion(
            [list(mine), [rc.chunk.id for rc in sample_hits]], k=self.rrf_k
        )
        merged = {**theirs, **mine}
        best = sorted(final, key=lambda cid: (-final[cid], -merged[cid].score))[:k]
        return [merged[cid].model_copy(update={"fusion_score": final[cid]}) for cid in best]
