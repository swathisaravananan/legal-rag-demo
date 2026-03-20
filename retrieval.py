"""
retrieval.py — Vector search, Maximal Marginal Relevance (MMR), and cross-encoder reranking.

Design decisions:
  - MMR with λ=0.7 balances relevance (70%) and diversity (30%), reducing redundant
    chunks from long repetitive documents while keeping the most relevant passage first.
  - Cross-encoder reranker (ms-marco-MiniLM-L-6-v2) operates on (query, passage) pairs
    as a pointwise scorer — much higher precision than a bi-encoder at the cost of
    O(k) inference calls.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

RERANKER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


# ---------------------------------------------------------------------------
# Cross-encoder loader
# ---------------------------------------------------------------------------


def load_reranker() -> CrossEncoder:
    """Load and return the cross-encoder reranker model."""
    return CrossEncoder(RERANKER_MODEL_NAME)


# ---------------------------------------------------------------------------
# MMR retrieval
# ---------------------------------------------------------------------------


def retrieve_mmr(
    query: str,
    collection: Any,  # chromadb.Collection
    model: SentenceTransformer,
    k: int = 5,
    fetch_k: int | None = None,
    lambda_param: float = 0.7,
) -> list[dict[str, Any]]:
    """
    Retrieve ``k`` chunks using Maximal Marginal Relevance.

    Strategy
    --------
    1. Embed the query with the bi-encoder.
    2. Fetch ``fetch_k`` (≥ 2k) candidates from ChromaDB by cosine similarity.
    3. Apply MMR to the candidate set to select ``k`` diverse results:

       .. math::

           \\text{MMR}(d) = \\lambda \\cdot \\text{sim}(q, d)
                           - (1-\\lambda) \\cdot \\max_{d_j \\in S} \\text{sim}(d, d_j)

       where *S* is the set of already-selected documents.

    Parameters
    ----------
    query:
        Natural-language question string.
    collection:
        ChromaDB collection to search.
    model:
        Bi-encoder used for query embedding.
    k:
        Number of chunks to return.
    fetch_k:
        Candidate pool size before MMR; defaults to ``max(2*k, 20)``.
    lambda_param:
        MMR λ — higher values favour relevance, lower favour diversity.

    Returns
    -------
    List of chunk dicts with keys: ``text``, ``source``, ``page``,
    ``chunk_index``, ``strategy``, ``score`` (cosine similarity).
    """
    if fetch_k is None:
        fetch_k = max(2 * k, 20)

    # 1. Embed query
    query_emb: np.ndarray = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    )[0].astype(np.float32)

    # 2. Fetch candidates from ChromaDB
    n_available = collection.count()
    if n_available == 0:
        return []
    actual_fetch = min(fetch_k, n_available)

    results = collection.query(
        query_embeddings=[query_emb.tolist()],
        n_results=actual_fetch,
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    candidate_texts: list[str] = results["documents"][0]
    candidate_metas: list[dict] = results["metadatas"][0]
    candidate_dists: list[float] = results["distances"][0]
    candidate_embs: list[list[float]] = results["embeddings"][0]

    if not candidate_texts:
        return []

    # Convert distances → cosine similarities (ChromaDB cosine distance = 1 - sim)
    cos_sims = [1.0 - d for d in candidate_dists]
    emb_matrix = np.array(candidate_embs, dtype=np.float32)  # (fetch_k, D)

    # 3. MMR selection
    actual_k = min(k, len(candidate_texts))
    selected_indices: list[int] = []
    remaining: list[int] = list(range(len(candidate_texts)))

    while len(selected_indices) < actual_k and remaining:
        if not selected_indices:
            # First pick: highest cosine similarity to query
            best = max(remaining, key=lambda i: cos_sims[i])
        else:
            selected_embs = emb_matrix[selected_indices]  # (|S|, D)
            best = -1
            best_score = float("-inf")
            for i in remaining:
                relevance = cos_sims[i]
                # Max similarity to any already-selected doc
                inter_sims = cosine_similarity(
                    emb_matrix[i : i + 1], selected_embs
                )[0]
                diversity_penalty = float(inter_sims.max())
                mmr_score = lambda_param * relevance - (1.0 - lambda_param) * diversity_penalty
                if mmr_score > best_score:
                    best_score = mmr_score
                    best = i

        selected_indices.append(best)
        remaining.remove(best)

    # 4. Build output
    chunks: list[dict[str, Any]] = []
    for rank, idx in enumerate(selected_indices):
        meta = candidate_metas[idx]
        chunks.append(
            {
                "text": candidate_texts[idx],
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", 0),
                "chunk_index": meta.get("chunk_index", idx),
                "strategy": meta.get("strategy", "unknown"),
                "score": round(cos_sims[idx], 4),
                "mmr_rank": rank + 1,
            }
        )
    return chunks


# ---------------------------------------------------------------------------
# Cross-encoder reranking
# ---------------------------------------------------------------------------


def rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    reranker: CrossEncoder,
) -> list[dict[str, Any]]:
    """
    Rerank ``chunks`` using a cross-encoder model.

    The cross-encoder jointly encodes (query, passage) pairs, producing a
    calibrated relevance score that is significantly more precise than
    bi-encoder cosine similarity alone — at the cost of O(k) forward passes.

    Parameters
    ----------
    query:
        The user's question.
    chunks:
        Chunks returned by :func:`retrieve_mmr`.
    reranker:
        A loaded :class:`sentence_transformers.CrossEncoder` instance.

    Returns
    -------
    Chunks sorted by descending cross-encoder score, each with an added
    ``rerank_score`` key and a ``rerank_rank`` key.
    """
    if not chunks:
        return []

    pairs = [(query, c["text"]) for c in chunks]
    scores: list[float] = reranker.predict(pairs).tolist()

    reranked = []
    for chunk, score in zip(chunks, scores):
        reranked.append({**chunk, "rerank_score": round(score, 4)})

    reranked.sort(key=lambda c: c["rerank_score"], reverse=True)
    for rank, chunk in enumerate(reranked):
        chunk["rerank_rank"] = rank + 1

    return reranked
