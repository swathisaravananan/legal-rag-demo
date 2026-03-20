"""
ingestion.py — Document loading, chunking, embedding, and vector store population.

Supports:
  - PDF extraction via pdfplumber (with PyPDF2 fallback)
  - Plain text extraction
  - Fixed-size chunking with configurable overlap
  - Semantic chunking by paragraph / section boundary
  - Sentence-transformer embeddings (all-MiniLM-L6-v2)
  - ChromaDB persistent vector store
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "legal_rag_demo"

# ---------------------------------------------------------------------------
# Model loading (expensive — call once and cache in Streamlit)
# ---------------------------------------------------------------------------


def load_embedding_model() -> SentenceTransformer:
    """Load and return the sentence-transformer embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def extract_text_from_pdf(
    file_path: str | Path,
    ocr_lang: str = "tam+eng",
    ocr_dpi: int = 300,
) -> list[dict[str, Any]]:
    """
    Extract text from a PDF file page by page.

    For each page, the function first tries pdfplumber (fast, lossless for
    text-based PDFs).  If a page yields fewer than 20 characters — indicating
    a scanned/image-only page — it falls back to Tesseract OCR via pdf2image.

    Parameters
    ----------
    file_path:
        Path to the PDF file.
    ocr_lang:
        Tesseract language string.  ``"tam+eng"`` handles mixed Tamil/English
        documents.  Use ``"tam"`` for Tamil-only, ``"eng"`` for English-only.
        Requires the corresponding Tesseract language packs to be installed:
            macOS:  ``brew install tesseract tesseract-lang``
            Linux:  ``apt-get install tesseract-ocr tesseract-ocr-tam``
    ocr_dpi:
        Resolution for rasterising PDF pages before OCR.  300 DPI is the
        standard minimum for acceptable OCR quality on legal documents.

    Returns
    -------
    List of dicts with keys:
        - ``page``: 1-indexed page number
        - ``text``: extracted or OCR-ed text
        - ``source``: filename stem
        - ``ocr_used``: True if OCR was used for this page
    """
    pages: list[dict[str, Any]] = []
    source = Path(file_path).stem
    file_path = Path(file_path)

    # Try pdfplumber first for each page; use OCR on pages that yield no text
    try:
        with pdfplumber.open(file_path) as pdf:
            num_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if len(text.strip()) >= 20:
                    pages.append(
                        {"page": i, "text": text, "source": source, "ocr_used": False}
                    )
                else:
                    # Page is image-based — attempt OCR
                    ocr_text = _ocr_page(file_path, page_index=i - 1, lang=ocr_lang, dpi=ocr_dpi)
                    if ocr_text.strip():
                        pages.append(
                            {"page": i, "text": ocr_text, "source": source, "ocr_used": True}
                        )
    except Exception as exc:  # pylint: disable=broad-except
        # Full fallback: OCR every page
        try:
            ocr_pages = _ocr_all_pages(file_path, lang=ocr_lang, dpi=ocr_dpi)
            pages.extend(
                {"page": i + 1, "text": t, "source": source, "ocr_used": True}
                for i, t in enumerate(ocr_pages)
                if t.strip()
            )
        except Exception as inner_exc:  # pylint: disable=broad-except
            raise RuntimeError(
                f"Could not extract text from PDF {file_path}: {inner_exc}"
            ) from exc

    return pages


def _ocr_page(
    file_path: Path,
    page_index: int,
    lang: str,
    dpi: int,
) -> str:
    """
    Rasterise a single PDF page and run Tesseract OCR on it.

    Requires ``pdf2image`` (which needs poppler) and ``pytesseract``
    (which needs tesseract + language packs).
    """
    try:
        from pdf2image import convert_from_path  # noqa: PLC0415
        import pytesseract  # noqa: PLC0415

        images = convert_from_path(
            str(file_path),
            dpi=dpi,
            first_page=page_index + 1,
            last_page=page_index + 1,
        )
        if not images:
            return ""
        return pytesseract.image_to_string(images[0], lang=lang)
    except ImportError as exc:
        raise ImportError(
            "OCR requires pdf2image and pytesseract.\n"
            "  pip install pdf2image pytesseract\n"
            "System deps:\n"
            "  macOS:  brew install poppler tesseract tesseract-lang\n"
            "  Linux:  apt-get install poppler-utils tesseract-ocr tesseract-ocr-tam"
        ) from exc


def _ocr_all_pages(file_path: Path, lang: str, dpi: int) -> list[str]:
    """Rasterise every page of a PDF and return a list of OCR strings."""
    from pdf2image import convert_from_path  # noqa: PLC0415
    import pytesseract  # noqa: PLC0415

    images = convert_from_path(str(file_path), dpi=dpi)
    return [pytesseract.image_to_string(img, lang=lang) for img in images]


def extract_text_from_txt(file_path: str | Path) -> list[dict[str, Any]]:
    """
    Extract text from a plain-text file.

    Splits on form-feed characters (``\\f``) or every ~3,000 characters to
    simulate pages for citation purposes.
    """
    source = Path(file_path).stem
    raw = Path(file_path).read_text(encoding="utf-8", errors="replace")

    # Split on explicit page breaks first
    raw_pages = raw.split("\f")
    if len(raw_pages) == 1:
        # No explicit page breaks — chunk into ~3 000-char pseudo-pages
        page_size = 3_000
        raw_pages = [raw[i : i + page_size] for i in range(0, len(raw), page_size)]

    pages = []
    for i, text in enumerate(raw_pages, start=1):
        if text.strip():
            pages.append({"page": i, "text": text, "source": source})

    return pages


def load_document(file_path: str | Path) -> list[dict[str, Any]]:
    """Dispatch to the appropriate extractor based on file extension."""
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(path)
    elif path.suffix.lower() in {".txt", ".md"}:
        return extract_text_from_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------


def chunk_fixed_size(
    pages: list[dict[str, Any]],
    chunk_size: int = 500,
    overlap: int = 75,
) -> list[dict[str, Any]]:
    """
    Fixed-size token-approximate chunking with overlap.

    Splits each page's text into chunks of roughly ``chunk_size`` words,
    with an ``overlap``-word sliding window so context is preserved at
    chunk boundaries (≈10–15% overlap by default).

    Each returned chunk dict contains:
        - ``chunk_id``: deterministic hash ID
        - ``text``: chunk text
        - ``source``: document name
        - ``page``: source page number
        - ``chunk_index``: sequential index within the document
        - ``strategy``: ``"fixed"``
    """
    chunks: list[dict[str, Any]] = []
    chunk_index = 0

    for page in pages:
        words = page["text"].split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunk_id = _make_chunk_id(page["source"], chunk_index)
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_index": chunk_index,
                    "strategy": "fixed",
                }
            )
            chunk_index += 1
            if end == len(words):
                break
            start = end - overlap  # slide back by overlap words

    return chunks


def chunk_semantic(
    pages: list[dict[str, Any]],
    min_chunk_words: int = 80,
    max_chunk_words: int = 600,
) -> list[dict[str, Any]]:
    """
    Semantic chunking by paragraph / section boundary.

    Splits text on double-newlines (paragraph breaks) and legal section
    headers (e.g., "ARTICLE I", "Section 1.", "I. ", numbered clauses).
    Adjacent paragraphs are merged until ``max_chunk_words`` is reached,
    preventing tiny orphan chunks while respecting natural boundaries.

    Each returned chunk dict contains the same keys as :func:`chunk_fixed_size`
    plus ``strategy="semantic"``.
    """
    # Regex to detect section/article headers
    header_re = re.compile(
        r"^(?:ARTICLE\s+[IVXLCDM]+|Section\s+\d+|[IVX]+\.\s+[A-Z]|\d+\.\s+[A-Z])",
        re.MULTILINE,
    )

    chunks: list[dict[str, Any]] = []
    chunk_index = 0

    for page in pages:
        # Split on paragraph breaks or detected headers
        raw_paras = re.split(r"\n{2,}", page["text"].strip())
        paragraphs: list[str] = []
        for para in raw_paras:
            # Further split on inline headers
            sub_splits = header_re.split(para)
            headers = header_re.findall(para)
            if headers:
                for j, part in enumerate(sub_splits):
                    if part.strip():
                        prefix = headers[j - 1] + " " if j > 0 and j - 1 < len(headers) else ""
                        paragraphs.append(prefix + part.strip())
            else:
                if para.strip():
                    paragraphs.append(para.strip())

        # Merge short paragraphs into chunks up to max_chunk_words
        current_parts: list[str] = []
        current_word_count = 0

        def _flush() -> None:
            nonlocal current_parts, current_word_count, chunk_index
            if current_parts:
                chunk_text = "\n\n".join(current_parts)
                chunk_id = _make_chunk_id(page["source"], chunk_index)
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "source": page["source"],
                        "page": page["page"],
                        "chunk_index": chunk_index,
                        "strategy": "semantic",
                    }
                )
                chunk_index += 1
                current_parts = []
                current_word_count = 0

        for para in paragraphs:
            word_count = len(para.split())
            if current_word_count + word_count > max_chunk_words and current_word_count >= min_chunk_words:
                _flush()
            current_parts.append(para)
            current_word_count += word_count

        _flush()

    return chunks


def _make_chunk_id(source: str, index: int) -> str:
    """Create a deterministic hex chunk ID from source name and index."""
    return hashlib.md5(f"{source}::{index}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def embed_chunks(
    chunks: list[dict[str, Any]],
    model: SentenceTransformer,
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Compute L2-normalised embeddings for a list of chunk dicts.

    Returns an ``(N, D)`` float32 array where D is the model's embedding
    dimension (384 for all-MiniLM-L6-v2).
    """
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# ChromaDB storage
# ---------------------------------------------------------------------------


def get_chroma_collection(reset: bool = False) -> chromadb.Collection:
    """
    Return (or create) the persistent ChromaDB collection.

    Parameters
    ----------
    reset:
        If ``True``, delete and recreate the collection — useful when the
        user re-processes documents with a different chunking strategy.
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:  # pylint: disable=broad-except
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def store_chunks(
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
    collection: chromadb.Collection,
) -> None:
    """
    Upsert chunks and their pre-computed embeddings into ChromaDB.

    Existing chunks with the same ID are overwritten, so calling this
    multiple times with the same document is idempotent.
    """
    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "source": c["source"],
            "page": c["page"],
            "chunk_index": c["chunk_index"],
            "strategy": c["strategy"],
        }
        for c in chunks
    ]
    embedding_list = embeddings.tolist()

    # ChromaDB has a batch limit; upsert in pages of 500
    batch = 500
    for start in range(0, len(ids), batch):
        collection.upsert(
            ids=ids[start : start + batch],
            embeddings=embedding_list[start : start + batch],
            documents=documents[start : start + batch],
            metadatas=metadatas[start : start + batch],
        )


def ingest_document(
    file_path: str | Path,
    model: SentenceTransformer,
    collection: chromadb.Collection,
    strategy: str = "fixed",
    chunk_size: int = 500,
    overlap: int = 75,
    ocr_lang: str = "tam+eng",
) -> list[dict[str, Any]]:
    """
    End-to-end ingestion: load → chunk → embed → store.

    Parameters
    ----------
    file_path:
        Path to a PDF or TXT file.
    model:
        A loaded SentenceTransformer instance.
    collection:
        Target ChromaDB collection.
    strategy:
        ``"fixed"`` for fixed-size chunking, ``"semantic"`` for
        paragraph/section-based chunking.
    chunk_size:
        Word count per chunk (fixed strategy only).
    overlap:
        Overlap in words between consecutive chunks (fixed strategy only).

    Returns
    -------
    The list of chunk dicts that were stored.
    """
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        pages = extract_text_from_pdf(path, ocr_lang=ocr_lang)
    else:
        pages = load_document(file_path)
    if strategy == "semantic":
        chunks = chunk_semantic(pages)
    else:
        chunks = chunk_fixed_size(pages, chunk_size=chunk_size, overlap=overlap)

    embeddings = embed_chunks(chunks, model)
    store_chunks(chunks, embeddings, collection)
    return chunks
