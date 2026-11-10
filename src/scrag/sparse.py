"""In-process BM25 keyword index.

Dense embeddings blur exact identifiers (error codes, config names, version
numbers); BM25 matches them reliably. The index is small and dependency-free and
is rebuilt from the vector store's chunks whenever the store's revision changes.

Tokenization keeps compound identifiers whole *and* indexes their parts, so
``demo_budget_exhausted`` matches both the exact code and the word "budget".
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

# Words joined by '.', '_' or '-' form one compound token (e.g. "x-request-id").
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_PART_RE = re.compile(r"[._-]")

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "do",
        "does",
        "did",
        "for",
        "from",
        "has",
        "have",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "about",
        "after",
        "before",
        "between",
        "both",
        "each",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "too",
        "very",
        "just",
        "should",
        "could",
        "may",
        "might",
        "must",
        "also",
        "any",
        "all",
        "not",
        "no",
        "nor",
        "via",
        "per",
        "used",
        "use",
        "using",
    ]
)


def tokenize(text: str) -> list[str]:
    """Lowercase tokens without stopwords; compounds are kept whole plus split."""
    tokens: list[str] = []
    for match in _TOKEN_RE.findall(text.lower()):
        parts = [p for p in _PART_RE.split(match) if p]
        if len(parts) > 1:
            tokens.append(match)
        tokens.extend(p for p in parts if p not in STOPWORDS)
    return tokens


class BM25Index:
    """Okapi BM25 over a fixed list of texts (row-aligned with a vector store)."""

    def __init__(self, texts: Sequence[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.size = len(texts)
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._doc_len: list[int] = []
        for idx, text in enumerate(texts):
            counts = Counter(tokenize(text))
            self._doc_len.append(sum(counts.values()))
            for term, tf in counts.items():
                self._postings[term].append((idx, tf))
        self._avgdl = (sum(self._doc_len) / self.size) if self.size else 0.0
        self._idf = {
            term: math.log(1.0 + (self.size - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self._postings.items()
        }

    def scores(self, query: str) -> dict[int, float]:
        """BM25 score for every row containing at least one query term."""
        result: dict[int, float] = defaultdict(float)
        if not self.size:
            return result
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for idx, tf in self._postings[term]:
                norm = 1.0 - self.b + self.b * self._doc_len[idx] / (self._avgdl or 1.0)
                result[idx] += idf * tf * (self.k1 + 1.0) / (tf + self.k1 * norm)
        return result

    def top(self, query: str, n: int) -> list[tuple[int, float]]:
        """The ``n`` best ``(row, score)`` pairs with a positive score, best first."""
        ranked = sorted(self.scores(query).items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:n]


def reciprocal_rank_fusion(rankings: Iterable[Sequence[int]], k: int = 60) -> dict[int, float]:
    """Fuse ranked lists of row ids: each list adds ``1 / (k + rank)`` (rank from 1)."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] += 1.0 / (k + rank)
    return fused
