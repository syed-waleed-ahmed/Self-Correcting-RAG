import os
from glob import glob
from typing import List, Tuple
import numpy as np

from .config import DATA_DIR, INDEX_PATH, META_PATH, TOP_K
from .embeddings import embed_texts, embed_text


def build_index() -> None:
    """
    Build an in-memory vector index and save it to disk (INDEX_PATH, META_PATH).
    Each row corresponds to one document chunk; here we use whole files as chunks
    for simplicity.
    """
    file_paths = sorted(glob(os.path.join(DATA_DIR, "*.txt")))
    if not file_paths:
        raise ValueError(f"No .txt files found in {DATA_DIR}")

    texts = []
    meta_lines = []

    for i, path in enumerate(file_paths):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            continue
        texts.append(text)
        meta_lines.append(f"{i}\t{os.path.basename(path)}")

    if not texts:
        raise ValueError("All documents were empty.")

    print(f"Building index for {len(texts)} documents...")
    embeddings = embed_texts(texts)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    embeddings = embeddings / norms

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    np.save(INDEX_PATH, embeddings)

    with open(META_PATH, "w", encoding="utf-8") as f:
        for line, text in zip(meta_lines, texts):
            # store index, filename, and the raw text
            f.write(f"{line}\t{text.replace('\n', ' ')}\n")

    print(f"Index built and saved to {INDEX_PATH} and {META_PATH}.")


def _load_index() -> Tuple[np.ndarray, List[dict]]:
    """
    Load embeddings and metadata.
    Returns (embeddings, metadata_list) where metadata_list[i] is a dict with keys:
    id, filename, text
    """
    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        raise FileNotFoundError(
            "Index files not found. Run build_index() first (e.g. via main.py --build-index)."
        )

    embeddings = np.load(INDEX_PATH)
    metadata: List[dict] = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", maxsplit=3)
            idx_str, filename, text = parts[0], parts[1], parts[2]
            metadata.append(
                {"id": int(idx_str), "filename": filename, "text": text}
            )

    return embeddings, metadata


def retrieve(query: str, top_k: int = TOP_K) -> List[dict]:
    """
    Retrieve top_k most similar documents (cosine similarity).
    Returns list of metadata dicts with extra "score" field.
    """
    embeddings, metadata = _load_index()

    query_vec = embed_text(query)
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)

    sims = embeddings @ query_vec  # cosine since vectors normalized
    top_indices = np.argsort(-sims)[:top_k]

    results = []
    for idx in top_indices:
        item = metadata[idx].copy()
        item["score"] = float(sims[idx])
        results.append(item)

    return results
