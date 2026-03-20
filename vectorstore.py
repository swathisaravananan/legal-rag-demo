"""
vectorstore.py — Pure-numpy in-memory vector store.

Drop-in replacement for ChromaDB's Collection interface so the rest of the
codebase needs no changes. Uses pre-normalised dot-product for cosine
similarity (equivalent to cosine similarity when embeddings are L2-normalised,
which sentence-transformers produces by default with normalize_embeddings=True).

No external dependencies beyond numpy — avoids the opentelemetry/protobuf
incompatibility that ChromaDB introduces on Python 3.14.
"""

from __future__ import annotations

import numpy as np


class NumpyVectorStore:
    """Minimal ChromaDB-compatible in-memory vector store."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict] = []
        self._embeddings: np.ndarray | None = None  # shape (N, D)

    # ------------------------------------------------------------------
    # ChromaDB-compatible interface
    # ------------------------------------------------------------------

    def count(self) -> int:
        return len(self._ids)

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        emb_array = np.array(embeddings, dtype=np.float32)
        id_to_idx = {id_: i for i, id_ in enumerate(self._ids)}

        new_ids, new_docs, new_metas, new_embs = [], [], [], []

        for i, id_ in enumerate(ids):
            if id_ in id_to_idx:
                idx = id_to_idx[id_]
                self._documents[idx] = documents[i]
                self._metadatas[idx] = metadatas[i]
                self._embeddings[idx] = emb_array[i]
            else:
                new_ids.append(id_)
                new_docs.append(documents[i])
                new_metas.append(metadatas[i])
                new_embs.append(emb_array[i])

        if new_ids:
            self._ids.extend(new_ids)
            self._documents.extend(new_docs)
            self._metadatas.extend(new_metas)
            new_block = np.stack(new_embs).astype(np.float32)
            self._embeddings = (
                new_block
                if self._embeddings is None
                else np.vstack([self._embeddings, new_block])
            )

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str] | None = None,
    ) -> dict:
        if self._embeddings is None or len(self._ids) == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "embeddings": [[]], "ids": [[]]}

        q = np.array(query_embeddings[0], dtype=np.float32)
        # Dot-product == cosine similarity for L2-normalised vectors
        sims = self._embeddings @ q  # (N,)

        actual_n = min(n_results, len(self._ids))
        top_indices = np.argsort(sims)[::-1][:actual_n].tolist()

        return {
            "documents": [[self._documents[i] for i in top_indices]],
            "metadatas": [[self._metadatas[i] for i in top_indices]],
            # ChromaDB cosine distance = 1 − similarity
            "distances": [[float(1.0 - sims[i]) for i in top_indices]],
            "embeddings": [[self._embeddings[i].tolist() for i in top_indices]],
            "ids": [[self._ids[i] for i in top_indices]],
        }

    def delete_collection(self) -> None:
        """Reset the store (mirrors chromadb client.delete_collection)."""
        self.__init__()
