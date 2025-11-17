from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

# Lazy-load the model the first time we need it
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # Small, fast, general-purpose sentence embedding model
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Get embeddings for a list of texts using a local sentence-transformers model.
    Returns a 2D numpy array [len(texts), embedding_dim].
    """
    if not texts:
        # 384 is the dimension of all-MiniLM-L6-v2
        return np.zeros((0, 384), dtype="float32")

    model = _get_model()
    # convert_to_numpy=True already gives a NumPy array
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
    # ensure float32 for smaller size
    return embeddings.astype("float32")


def embed_text(text: str) -> np.ndarray:
    """Convenience wrapper for a single string."""
    return embed_texts([text])[0]
