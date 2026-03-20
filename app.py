"""
app.py — LegalRAG: Streamlit front-end for the legal document RAG pipeline.

Run:
    streamlit run app.py

Environment:
    GOOGLE_API_KEY — required for answer generation; if absent the app
    degrades gracefully to retrieval-only mode.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from generation import generate_answer
from hallucination import build_highlighted_html, grounding_badge_html, score_answer
from ingestion import (
    EMBEDDING_MODEL_NAME,
    get_chroma_collection,
    ingest_document,
    load_embedding_model,
)
from retrieval import load_reranker, retrieve_mmr, rerank_chunks

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="LegalRAG — AI Legal Document Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Reset & base ── */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main .block-container { padding-top: 0 !important; max-width: 1400px; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #0d1f3c 100%);
        border-right: 1px solid #1e3a5f;
    }
    [data-testid="stSidebar"] * { color: #c8d6e8 !important; }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #7eb3f5 !important;
        font-size: 0.72em !important;
        font-weight: 700 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        margin-top: 1.2em !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #c9a84c, #e8c96a) !important;
        color: #0a1628 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        letter-spacing: 0.03em !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(201,168,76,0.4) !important;
    }
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stCheckbox label {
        font-size: 0.85em !important;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stMultiSelect > div > div {
        background: #0f2340 !important;
        border-color: #1e3a5f !important;
    }
    [data-testid="stSidebar"] hr { border-color: #1e3a5f !important; }

    /* ── Top header bar ── */
    .site-header {
        background: linear-gradient(135deg, #0a1628 0%, #102240 100%);
        padding: 18px 32px 16px;
        margin: -1rem -1rem 0;
        border-bottom: 2px solid #c9a84c;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .site-header-logo { font-size: 2em; }
    .site-header-title {
        font-size: 1.5em;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.02em;
    }
    .site-header-title span { color: #c9a84c; }
    .site-header-sub {
        font-size: 0.78em;
        color: #7eb3f5;
        margin-top: 2px;
        font-weight: 400;
    }
    .site-header-badges { margin-left: auto; display: flex; gap: 8px; align-items: center; }
    .hbadge {
        background: #0f2340;
        border: 1px solid #1e3a5f;
        color: #7eb3f5 !important;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.72em;
        font-weight: 500;
        white-space: nowrap;
    }
    .hbadge-gold {
        background: rgba(201,168,76,0.15);
        border-color: #c9a84c;
        color: #c9a84c !important;
    }

    /* ── Hero / empty state ── */
    .hero-wrap {
        text-align: center;
        padding: 60px 20px 40px;
    }
    .hero-icon { font-size: 4em; margin-bottom: 12px; }
    .hero-title {
        font-size: 2em;
        font-weight: 700;
        color: #0a1628;
        margin-bottom: 8px;
    }
    .hero-title span { color: #c9a84c; }
    .hero-sub { color: #556; font-size: 1em; margin-bottom: 40px; max-width: 520px; margin-left: auto; margin-right: auto; }
    .feature-grid { display: flex; gap: 16px; justify-content: center; flex-wrap: wrap; margin-bottom: 32px; }
    .feature-card {
        background: #fff;
        border: 1px solid #e8edf5;
        border-radius: 12px;
        padding: 20px 22px;
        width: 200px;
        text-align: left;
        box-shadow: 0 2px 8px rgba(10,22,40,0.06);
    }
    .feature-card-icon { font-size: 1.6em; margin-bottom: 8px; }
    .feature-card-title { font-weight: 600; font-size: 0.88em; color: #0a1628; margin-bottom: 4px; }
    .feature-card-desc { font-size: 0.78em; color: #667; line-height: 1.5; }

    /* ── Doc status bar ── */
    .doc-bar {
        background: linear-gradient(90deg, #f0f5ff, #f7f9ff);
        border: 1px solid #dce8ff;
        border-radius: 10px;
        padding: 10px 18px;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }
    .doc-bar-label { font-size: 0.78em; font-weight: 600; color: #334; text-transform: uppercase; letter-spacing: 0.06em; }
    .doc-pill {
        background: #0a1628;
        color: #c9a84c !important;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.76em;
        font-weight: 600;
        font-family: 'Inter', monospace;
    }

    /* ── Question input area ── */
    .question-area {
        background: #fff;
        border: 2px solid #e0e8f5;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 2px 12px rgba(10,22,40,0.05);
    }
    .question-label {
        font-size: 0.78em;
        font-weight: 700;
        color: #7eb3f5;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 8px;
    }

    /* ── Answer card ── */
    .answer-card {
        background: #fff;
        border: 1px solid #e0e8f5;
        border-top: 3px solid #0a1628;
        border-radius: 12px;
        padding: 24px 28px;
        box-shadow: 0 4px 20px rgba(10,22,40,0.07);
        margin-top: 4px;
    }
    .answer-card-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 14px;
        padding-bottom: 12px;
        border-bottom: 1px solid #f0f4f9;
    }
    .answer-card-title { font-weight: 700; font-size: 0.9em; color: #0a1628; }
    .answer-body { line-height: 1.85; font-size: 0.93em; color: #1a2a3a; }

    /* ── Citations ── */
    .citations-block {
        background: #f7f9ff;
        border: 1px solid #dce8ff;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 14px;
    }
    .citations-title { font-size: 0.72em; font-weight: 700; color: #334; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
    .citation-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.82em;
        color: #0a1628;
        padding: 3px 0;
    }
    .citation-dot { color: #c9a84c; font-size: 0.9em; }

    /* ── Grounding legend ── */
    .legend-bar {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 12px;
        align-items: center;
    }
    .legend-label { font-size: 0.72em; color: #667; font-weight: 500; }
    .legend-item { font-size: 0.72em; padding: 2px 8px; border-radius: 4px; font-weight: 500; }

    /* ── Chunk panel ── */
    .chunk-panel-header {
        font-size: 0.72em;
        font-weight: 700;
        color: #334;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 2px solid #f0f4f9;
    }
    .chunk-card {
        background: #fff;
        border: 1px solid #e8edf5;
        border-left: 3px solid #0a1628;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 10px;
        font-size: 0.83em;
        color: #223;
        box-shadow: 0 1px 4px rgba(10,22,40,0.04);
        transition: border-color 0.2s;
    }
    .chunk-card:hover { border-left-color: #c9a84c; }
    .chunk-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 7px;
    }
    .chunk-source { font-weight: 600; color: #0a1628; font-size: 0.88em; }
    .chunk-score {
        background: #f0f5ff;
        color: #334d7a;
        padding: 1px 7px;
        border-radius: 10px;
        font-size: 0.78em;
        font-weight: 600;
        font-family: monospace;
    }
    .chunk-score-rerank { background: rgba(201,168,76,0.15); color: #7a5a00; }
    .chunk-page { color: #778; font-size: 0.8em; margin-top: 1px; }
    .chunk-text { color: #334; line-height: 1.6; }

    /* ── Metric tiles ── */
    .metrics-row { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
    .metric-tile {
        background: #fff;
        border: 1px solid #e8edf5;
        border-radius: 10px;
        padding: 12px 18px;
        min-width: 120px;
        box-shadow: 0 1px 4px rgba(10,22,40,0.04);
    }
    .metric-tile-value { font-size: 1.4em; font-weight: 700; color: #0a1628; }
    .metric-tile-label { font-size: 0.7em; color: #778; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px; }

    /* ── Architecture tab ── */
    .arch-section-title {
        font-size: 0.72em;
        font-weight: 700;
        color: #7eb3f5;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin: 20px 0 10px;
    }
    .design-card {
        background: #fff;
        border: 1px solid #e8edf5;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 10px;
        box-shadow: 0 1px 4px rgba(10,22,40,0.04);
    }
    .design-card-title { font-weight: 700; font-size: 0.88em; color: #0a1628; margin-bottom: 6px; }
    .design-card-body { font-size: 0.82em; color: #445; line-height: 1.65; }
    .design-card-math {
        font-family: monospace;
        background: #f4f7ff;
        border: 1px solid #dce8ff;
        border-radius: 6px;
        padding: 6px 10px;
        font-size: 0.85em;
        color: #0a1628;
        margin: 6px 0;
        display: block;
    }
    .strategy-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 8px; }
    .strategy-card {
        background: #f7f9ff;
        border: 1px solid #dce8ff;
        border-radius: 8px;
        padding: 14px 16px;
        font-size: 0.82em;
        color: #334;
        line-height: 1.65;
    }
    .strategy-card-title { font-weight: 700; color: #0a1628; margin-bottom: 6px; font-size: 0.92em; }
    .ocr-badge {
        display: inline-block;
        background: rgba(201,168,76,0.15);
        color: #7a5a00;
        border: 1px solid #c9a84c;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.78em;
        font-weight: 600;
        margin-left: 6px;
        vertical-align: middle;
    }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #f4f7ff;
        border-radius: 10px;
        padding: 4px;
        gap: 2px;
        border: 1px solid #e0e8f5;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.85em !important;
        color: #556 !important;
        padding: 6px 18px !important;
    }
    .stTabs [aria-selected="true"] {
        background: #0a1628 !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached heavy model loading
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading embedding model…")
def _load_embedding_model():
    return load_embedding_model()


@st.cache_resource(show_spinner="Loading reranker…")
def _load_reranker():
    return load_reranker()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "processed_docs" not in st.session_state:
    st.session_state.processed_docs: list[str] = []
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks: list[dict] = []
if "collection" not in st.session_state:
    st.session_state.collection = None
if "qa_history" not in st.session_state:
    st.session_state.qa_history: list[dict] = []

# ---------------------------------------------------------------------------
# Sample document paths
# ---------------------------------------------------------------------------

SAMPLE_DOCS_DIR = Path(__file__).parent / "sample_docs"
SAMPLE_DOC_PATHS = sorted(SAMPLE_DOCS_DIR.glob("*.txt")) + sorted(SAMPLE_DOCS_DIR.glob("*.pdf"))
SAMPLE_DOC_NAMES = [p.name for p in SAMPLE_DOC_PATHS]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    # Logo
    st.markdown(
        """
        <div style="padding: 20px 4px 12px;">
            <div style="font-size:1.6em; font-weight:800; color:#fff; letter-spacing:-0.02em;">
                ⚖️ Legal<span style="color:#c9a84c;">RAG</span>
            </div>
            <div style="font-size:0.73em; color:#5a7fa8; margin-top:4px; line-height:1.4;">
                AI-powered legal document intelligence<br>
                Tamil &amp; English support
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # --- Documents ---
    st.markdown("### 📂 Documents")
    doc_mode = st.radio(
        "Source",
        ["Pre-loaded samples", "Upload PDF / TXT"],
        label_visibility="collapsed",
    )

    if doc_mode == "Pre-loaded samples":
        selected_samples = st.multiselect(
            "Select documents",
            SAMPLE_DOC_NAMES,
            default=SAMPLE_DOC_NAMES,
            label_visibility="collapsed",
        )
        upload_files = []
    else:
        selected_samples = []
        upload_files = st.file_uploader(
            "Upload files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

    st.divider()

    # --- Chunking ---
    st.markdown("### ✂️ Chunking")
    chunking_strategy = st.radio(
        "Strategy",
        ["Fixed-size with overlap", "Semantic (paragraph/section)"],
        label_visibility="collapsed",
    )
    strategy_key = "fixed" if chunking_strategy.startswith("Fixed") else "semantic"

    if strategy_key == "fixed":
        chunk_size = st.slider("Chunk size (words)", 100, 800, 500, step=50)
        overlap = st.slider("Overlap (words)", 10, 150, 75, step=5)
        st.markdown(
            f"<span style='font-size:0.75em;color:#5a7fa8;'>Overlap ratio: "
            f"<b style='color:#c9a84c;'>{overlap/chunk_size:.0%}</b></span>",
            unsafe_allow_html=True,
        )
    else:
        chunk_size, overlap = 500, 75
        st.markdown(
            "<span style='font-size:0.75em;color:#5a7fa8;'>Paragraph & section boundaries</span>",
            unsafe_allow_html=True,
        )

    st.divider()

    # --- Retrieval ---
    st.markdown("### 🔍 Retrieval")
    top_k = st.slider("Top-k chunks", 1, 10, 5)
    mmr_lambda = st.slider(
        "MMR λ  (relevance ↔ diversity)",
        0.0, 1.0, 0.7, step=0.05,
        help="λ=1.0 = pure relevance  ·  λ=0.0 = maximum diversity",
    )
    use_reranker = st.checkbox(
        "Cross-encoder reranking",
        value=False,
        help="Reranks bi-encoder candidates with multilingual mmarco cross-encoder.",
    )

    st.divider()

    # --- Language & OCR ---
    st.markdown("### 🌐 Language & OCR")
    ocr_lang = st.selectbox(
        "Tesseract language",
        ["tam+eng", "tam", "eng"],
        index=0,
        label_visibility="collapsed",
        help=(
            "Used for scanned/image PDFs only.\n"
            "macOS:  brew install tesseract tesseract-lang\n"
            "Linux:  apt-get install tesseract-ocr tesseract-ocr-tam"
        ),
    )
    st.markdown(
        "<span style='font-size:0.74em;color:#5a7fa8;'>Embeddings: multilingual-MiniLM-L12</span>",
        unsafe_allow_html=True,
    )

    st.divider()

    process_btn = st.button("⚙️  Process Documents", type="primary", use_container_width=True)

    # Footer info
    st.markdown(
        """
        <div style="padding: 16px 0 8px; font-size:0.7em; color:#3d5a7a; line-height:1.8;">
            <div>Vector store · NumpyVectorStore (in-memory)</div>
            <div>Generator · Groq / llama-3.3-70b-versatile</div>
            <div>Reranker · mmarco multilingual</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Site header
# ---------------------------------------------------------------------------

api_status = "🟢 API Ready" if os.environ.get("GROQ_API_KEY") else "🔴 No API Key"
doc_count = len(st.session_state.processed_docs)

st.markdown(
    f"""
    <div class="site-header">
        <div class="site-header-logo">⚖️</div>
        <div>
            <div class="site-header-title">Legal<span>RAG</span></div>
            <div class="site-header-sub">Retrieval-Augmented Generation · Tamil &amp; English Legal Documents</div>
        </div>
        <div class="site-header-badges">
            <span class="hbadge hbadge-gold">{doc_count} doc{"s" if doc_count != 1 else ""} indexed</span>
            <span class="hbadge">{api_status}</span>
            <span class="hbadge">Groq LLM</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------------

if process_btn:
    embed_model = _load_embedding_model()
    paths_to_ingest: list[Path] = [SAMPLE_DOCS_DIR / name for name in selected_samples]

    tmp_paths: list[Path] = []
    for uf in upload_files:
        suffix = Path(uf.name).suffix
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uf.read())
        tmp.close()
        tmp_paths.append(Path(tmp.name))
    paths_to_ingest.extend(tmp_paths)

    if not paths_to_ingest:
        st.sidebar.warning("No documents selected.")
    else:
        with st.spinner("Indexing documents…"):
            collection = get_chroma_collection(reset=True)
            all_chunks: list[dict] = []
            doc_names: list[str] = []
            progress = st.sidebar.progress(0.0)

            for i, path in enumerate(paths_to_ingest):
                chunks = ingest_document(
                    path, embed_model, collection,
                    strategy=strategy_key,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    ocr_lang=ocr_lang,
                )
                all_chunks.extend(chunks)
                doc_names.append(path.stem)
                progress.progress((i + 1) / len(paths_to_ingest))

            st.session_state.collection = collection
            st.session_state.all_chunks = all_chunks
            st.session_state.processed_docs = doc_names
            st.session_state.qa_history = []

            for tmp_path in tmp_paths:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        st.sidebar.success(
            f"✅ {len(paths_to_ingest)} doc(s) · {len(all_chunks)} chunks"
        )
        st.rerun()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_qa, tab_arch = st.tabs(["💬  Ask Documents", "🏗️  Architecture & Design"])

# ===========================================================================
# TAB 1 — Q&A
# ===========================================================================

with tab_qa:

    if not st.session_state.processed_docs:
        # ── Hero / landing state ──
        st.markdown(
            """
            <div class="hero-wrap">
                <div class="hero-icon">⚖️</div>
                <div class="hero-title">Legal<span>RAG</span></div>
                <div class="hero-sub">
                    Ask natural language questions over legal documents in Tamil and English.
                    Upload your own PDFs or use the pre-loaded samples to get started.
                </div>
                <div class="feature-grid">
                    <div class="feature-card">
                        <div class="feature-card-icon">🔍</div>
                        <div class="feature-card-title">MMR Retrieval</div>
                        <div class="feature-card-desc">Maximal Marginal Relevance for diverse, relevant results</div>
                    </div>
                    <div class="feature-card">
                        <div class="feature-card-icon">🌐</div>
                        <div class="feature-card-title">Tamil & English</div>
                        <div class="feature-card-desc">Multilingual embeddings + Tesseract OCR for scanned PDFs</div>
                    </div>
                    <div class="feature-card">
                        <div class="feature-card-icon">✅</div>
                        <div class="feature-card-title">Grounding Score</div>
                        <div class="feature-card-desc">Token-overlap hallucination detection per sentence</div>
                    </div>
                    <div class="feature-card">
                        <div class="feature-card-icon">📎</div>
                        <div class="feature-card-title">Citations</div>
                        <div class="feature-card-desc">Inline source citations with document and page number</div>
                    </div>
                    <div class="feature-card">
                        <div class="feature-card-icon">⚡</div>
                        <div class="feature-card-title">Reranking</div>
                        <div class="feature-card-desc">Cross-encoder precision boost over bi-encoder recall</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info("👈  Select documents in the sidebar and click **Process Documents** to begin.", icon="💡")

    else:
        # ── Doc status bar ──
        pills = " ".join(
            f'<span class="doc-pill">{d}</span>'
            for d in st.session_state.processed_docs
        )
        chunk_count = len(st.session_state.all_chunks)
        st.markdown(
            f'<div class="doc-bar">'
            f'<span class="doc-bar-label">Indexed</span>{pills}'
            f'<span style="margin-left:auto;font-size:0.76em;color:#778;">'
            f'{chunk_count} chunks · {strategy_key} chunking</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Question input ──
        st.markdown(
            '<div class="question-area">'
            '<div class="question-label">Ask a question</div>',
            unsafe_allow_html=True,
        )
        question = st.text_input(
            "question",
            placeholder="e.g.  சந்தா கட்டணம் என்ன?  /  What is the liability cap?",
            label_visibility="collapsed",
        )
        col_ask, col_clear = st.columns([6, 1])
        with col_ask:
            ask_btn = st.button("Ask  ➤", type="primary", disabled=not question.strip(), use_container_width=True)
        with col_clear:
            if st.button("Clear", use_container_width=True):
                st.session_state.qa_history = []
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Run pipeline ──
        if ask_btn and question.strip():
            embed_model = _load_embedding_model()
            collection = st.session_state.collection

            with st.spinner("Retrieving relevant chunks…"):
                chunks = retrieve_mmr(
                    question, collection, embed_model,
                    k=top_k, lambda_param=mmr_lambda,
                )

            chunks_before = chunks.copy()
            chunks_after = None

            if use_reranker and chunks:
                with st.spinner("Reranking with cross-encoder…"):
                    reranker = _load_reranker()
                    chunks_after = rerank_chunks(question, chunks, reranker)
                    chunks = chunks_after

            with st.spinner("Generating answer with Claude…"):
                result = generate_answer(question, chunks)

            scored_sentences, overall_score = score_answer(result["answer"], chunks)

            st.session_state.qa_history.append({
                "question": question,
                "result": result,
                "chunks": chunks,
                "chunks_before": chunks_before,
                "chunks_after": chunks_after,
                "scored_sentences": scored_sentences,
                "overall_score": overall_score,
            })

        # ── Display Q&A history (most recent first) ──
        for entry in reversed(st.session_state.qa_history):
            result = entry["result"]
            chunks = entry["chunks"]
            scored_sentences = entry["scored_sentences"]
            overall_score = entry["overall_score"]
            highlighted_html = build_highlighted_html(scored_sentences)
            badge_html = grounding_badge_html(overall_score)

            st.markdown(
                f'<div style="font-size:0.78em;color:#667;font-weight:600;'
                f'text-transform:uppercase;letter-spacing:0.08em;margin:16px 0 6px;">Question</div>'
                f'<div style="font-size:1.0em;font-weight:600;color:#0a1628;'
                f'margin-bottom:16px;">{entry["question"]}</div>',
                unsafe_allow_html=True,
            )

            col_ans, col_chunks = st.columns([3, 2], gap="large")

            with col_ans:
                # Metrics row
                pct = int(overall_score * 100)
                st.markdown(
                    f'<div class="metrics-row">'
                    f'<div class="metric-tile"><div class="metric-tile-value">{pct}%</div>'
                    f'<div class="metric-tile-label">Grounding</div></div>'
                    f'<div class="metric-tile"><div class="metric-tile-value">{len(chunks)}</div>'
                    f'<div class="metric-tile-label">Chunks used</div></div>'
                    + (
                        f'<div class="metric-tile"><div class="metric-tile-value">'
                        f'{result["input_tokens"]}</div>'
                        f'<div class="metric-tile-label">Tokens in</div></div>'
                        f'<div class="metric-tile"><div class="metric-tile-value">'
                        f'{result["output_tokens"]}</div>'
                        f'<div class="metric-tile-label">Tokens out</div></div>'
                        if not result["generation_skipped"] else ""
                    )
                    + '</div>',
                    unsafe_allow_html=True,
                )

                # Answer card
                if not result["generation_skipped"]:
                    citations_html = ""
                    if result["citations"]:
                        items = "".join(
                            f'<div class="citation-item">'
                            f'<span class="citation-dot">◆</span>'
                            f'<span><b>{c["source"]}</b> — Page {c["page"]}</span>'
                            f'</div>'
                            for c in result["citations"]
                        )
                        citations_html = (
                            f'<div class="citations-block">'
                            f'<div class="citations-title">Sources cited</div>'
                            f'{items}</div>'
                        )

                    st.markdown(
                        f'<div class="answer-card">'
                        f'<div class="answer-card-header">'
                        f'<span style="font-size:1.1em;">🤖</span>'
                        f'<span class="answer-card-title">Answer</span>'
                        f'<span style="margin-left:auto;">{badge_html}</span>'
                        f'</div>'
                        f'<div class="answer-body">{highlighted_html}</div>'
                        f'{citations_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # Grounding legend
                    st.markdown(
                        '<div class="legend-bar">'
                        '<span class="legend-label">Highlight key:</span>'
                        '<span class="legend-item" style="background:#d4edda;color:#155724;">well-grounded ≥30%</span>'
                        '<span class="legend-item" style="background:#fff3cd;color:#856404;">partial 10–30%</span>'
                        '<span class="legend-item" style="background:#f8d7da;color:#721c24;">low &lt;10%</span>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning(result["answer"])

            with col_chunks:
                # Reranker toggle
                display_chunks = chunks
                if use_reranker and entry["chunks_after"] is not None:
                    view = st.radio(
                        "View chunks",
                        ["After reranking", "Before reranking"],
                        horizontal=True,
                        label_visibility="collapsed",
                        key=f"view_{entry['question'][:20]}",
                    )
                    display_chunks = (
                        entry["chunks_after"] if view == "After reranking"
                        else entry["chunks_before"]
                    )

                st.markdown('<div class="chunk-panel-header">Retrieved Evidence</div>', unsafe_allow_html=True)

                for rank, chunk in enumerate(display_chunks, 1):
                    has_rerank = "rerank_score" in chunk
                    score_val = chunk["rerank_score"] if has_rerank else chunk["score"]
                    score_cls = "chunk-score-rerank" if has_rerank else "chunk-score"
                    score_label = "rerank" if has_rerank else "cosine"
                    excerpt = chunk["text"][:420] + ("…" if len(chunk["text"]) > 420 else "")

                    st.markdown(
                        f'<div class="chunk-card">'
                        f'<div class="chunk-meta">'
                        f'<div>'
                        f'<span class="chunk-source">#{rank} · {chunk["source"]}</span>'
                        f'<div class="chunk-page">Page {chunk["page"]} · {chunk["strategy"]} chunk</div>'
                        f'</div>'
                        f'<span class="chunk-score {score_cls}">{score_label} {score_val:.3f}</span>'
                        f'</div>'
                        f'<div class="chunk-text">{excerpt}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("<hr style='border:none;border-top:1px solid #eef2f9;margin:24px 0;'>", unsafe_allow_html=True)

# ===========================================================================
# TAB 2 — Architecture
# ===========================================================================

with tab_arch:
    st.markdown(
        """
        <div style="padding: 24px 0 8px;">
            <div style="font-size:1.3em;font-weight:700;color:#0a1628;margin-bottom:6px;">
                Pipeline Architecture
            </div>
            <div style="font-size:0.88em;color:#556;max-width:680px;line-height:1.6;">
                LegalRAG implements a production-grade RAG pipeline with multilingual support,
                scanned-PDF OCR, MMR diversity retrieval, and sentence-level grounding scores.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_diag, col_notes = st.columns([3, 2], gap="large")

    with col_diag:
        st.markdown('<div class="arch-section-title">System diagram</div>', unsafe_allow_html=True)
        st.graphviz_chart(
            """
            digraph LegalRAG {
                rankdir=TB;
                graph [fontname="Helvetica", bgcolor="transparent", pad="0.5", splines=ortho];
                node  [fontname="Helvetica", fontsize=10, style="filled,rounded",
                       shape=box, margin="0.22,0.12", penwidth=1.4];
                edge  [fontname="Helvetica", fontsize=8.5, penwidth=1.2];

                // ── Ingestion cluster ──────────────────────────────────────
                subgraph cluster_ingest {
                    label="  Ingestion Pipeline (offline)  ";
                    style=filled; fillcolor="#f0f5ff";
                    color="#0a1628"; fontcolor="#0a1628"; fontsize=10; fontname="Helvetica-Bold";

                    DOC  [label="PDF / TXT\\nDocument", fillcolor="#cfe2ff", color="#084298"];
                    EXT  [label="Text Extraction\\n(pdfplumber)", fillcolor="#cfe2ff", color="#084298"];
                    OCR  [label="Tesseract OCR\\n(tam+eng, 300 DPI)", fillcolor="#fff3cd", color="#856404",
                          style="filled,rounded,dashed"];
                    CHUNKF [label="Fixed-size Chunking\\n(chunk_size=500, overlap=75)", fillcolor="#cfe2ff", color="#084298"];
                    CHUNKS [label="Semantic Chunking\\n(paragraph / section)", fillcolor="#cfe2ff", color="#084298",
                            style="filled,rounded,dashed"];
                    EMB  [label="Multilingual Embedding\\nparaphrase-multilingual-MiniLM-L12", fillcolor="#cfe2ff", color="#084298"];
                    VDB  [label="NumpyVectorStore\\n(cosine similarity, in-memory)", fillcolor="#cfe2ff", color="#084298"];

                    DOC  -> EXT;
                    EXT  -> OCR  [label=" scanned page", style=dashed, color="#856404", fontcolor="#856404"];
                    EXT  -> CHUNKF [label=" text-based"];
                    OCR  -> CHUNKF [style=dashed, color="#856404"];
                    CHUNKF -> EMB [label=" fixed strategy"];
                    CHUNKS -> EMB [label=" semantic strategy", style=dashed];
                    EMB  -> VDB;
                }

                // ── Query cluster ──────────────────────────────────────────
                subgraph cluster_query {
                    label="  Query Pipeline (online)  ";
                    style=filled; fillcolor="#f0fdf4";
                    color="#198754"; fontcolor="#198754"; fontsize=10; fontname="Helvetica-Bold";

                    Q    [label="User Question\\n(Tamil or English)", fillcolor="#d1e7dd", color="#0f5132"];
                    QE   [label="Query Embedding\\n(same multilingual model)", fillcolor="#d1e7dd", color="#0f5132"];
                    MMR  [label="MMR Retrieval\\n(λ=0.7, top-k)", fillcolor="#d1e7dd", color="#0f5132"];
                    XE   [label="Cross-encoder Reranker\\nmmarco-mMiniLMv2 (optional)", fillcolor="#d1e7dd", color="#0f5132",
                          style="filled,rounded,dashed"];
                    GEN  [label="Groq Generation\\nllama-3.3-70b-versatile", fillcolor="#d1e7dd", color="#0f5132"];
                    GND  [label="Grounding Score\\n(token overlap per sentence)", fillcolor="#d1e7dd", color="#0f5132"];
                    OUT  [label="Answer + Citations\\n+ Highlighted Response", fillcolor="#d1e7dd", color="#0f5132",
                          penwidth=2.0];

                    Q -> QE -> MMR -> XE -> GEN -> GND -> OUT;
                }

                // ── Cross-path ─────────────────────────────────────────────
                VDB -> MMR [label="  ANN search  ", style=dashed, color="#0a1628",
                            fontcolor="#0a1628", penwidth=1.6];
            }
            """,
            use_container_width=True,
        )

        # Chunking comparison
        st.markdown('<div class="arch-section-title">Chunking strategies</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="strategy-grid">
                <div class="strategy-card">
                    <div class="strategy-card-title">Fixed-size with overlap</div>
                    Splits every page into windows of ~500 words with a 75-word
                    sliding overlap (~15%). Ensures no sentence vanishes at a
                    boundary. Best for uniform legal prose — SEC filings, contracts.
                    <br><br><b>Risk:</b> may split mid-clause.
                </div>
                <div class="strategy-card">
                    <div class="strategy-card-title">Semantic (paragraph / section)</div>
                    Splits on double newlines and detects legal section headers
                    (ARTICLE I, Section 2., etc.) to keep clauses intact.
                    Best for structured documents — court opinions, agreements.
                    <br><br><b>Risk:</b> variable chunk sizes.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_notes:
        st.markdown('<div class="arch-section-title">Design decisions</div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="design-card">
                <div class="design-card-title">1 · Tamil OCR support <span class="ocr-badge">NEW</span></div>
                <div class="design-card-body">
                    Each PDF page is tried with pdfplumber first. If fewer than
                    20 characters are extracted (scanned page), pdf2image rasterises
                    it at 300 DPI and Tesseract runs with <code>tam+eng</code>.
                    The multilingual embedding model then handles Tamil text natively.
                </div>
            </div>

            <div class="design-card">
                <div class="design-card-title">2 · Multilingual embeddings <span class="ocr-badge">NEW</span></div>
                <div class="design-card-body">
                    Switched from <code>all-MiniLM-L6-v2</code> (English-only) to
                    <code>paraphrase-multilingual-MiniLM-L12-v2</code> (50+ languages).
                    Tamil queries now match Tamil document chunks correctly. The
                    reranker was similarly upgraded to the multilingual mmarco model.
                </div>
            </div>

            <div class="design-card">
                <div class="design-card-title">3 · MMR retrieval (λ = 0.7)</div>
                <div class="design-card-body">
                    Balances relevance and diversity so repetitive boilerplate
                    paragraphs in filings don't crowd out unique evidence chunks.
                    <code class="design-card-math">MMR(d) = λ·sim(q,d) − (1−λ)·max_{dⱼ∈S} sim(d,dⱼ)</code>
                    λ=0.7 keeps the first result highly relevant while penalising
                    near-duplicate picks.
                </div>
            </div>

            <div class="design-card">
                <div class="design-card-title">4 · Bi-encoder → Cross-encoder</div>
                <div class="design-card-body">
                    Bi-encoder encodes query and passage <i>independently</i> — O(1)
                    at query time. Cross-encoder encodes <i>pairs</i> jointly —
                    much higher precision, O(k) forward passes.
                    Pipeline: bi-encoder for recall, cross-encoder for precision.
                </div>
            </div>

            <div class="design-card">
                <div class="design-card-title">5 · Grounding score</div>
                <div class="design-card-body">
                    Token-overlap Jaccard computed per sentence against the union
                    of all retrieved chunk tokens (after stop-word removal):
                    <code class="design-card-math">grounding(s) = |tok(s) ∩ tok(ctx)| / |tok(s)|</code>
                    Transparent, model-agnostic, no secondary LLM call needed.
                    Color-coded green / yellow / red inline in the answer.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
