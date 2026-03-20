# LegalRAG

A production-quality Retrieval-Augmented Generation demo for legal and corporate documents, built with Streamlit, ChromaDB, sentence-transformers, and the Anthropic API.

## Quick start

```bash
# 1. Clone / enter the project directory
cd legal-rag-demo

# 2. Create a virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Anthropic API key
cp .env.example .env
# Edit .env and replace "your-key-here" with your actual key

# 5. Launch
streamlit run app.py
```

The app runs fully locally except for the Anthropic API call. If no API key is provided it degrades gracefully to retrieval-only mode.

---

## Architecture overview

```
User Question
     │
     ▼
Bi-encoder embedding (all-MiniLM-L6-v2)
     │
     ▼
MMR Retrieval ──────────────── ChromaDB (cosine index)
  λ=0.7, top-k                      ▲
     │                              │
     │                    Ingestion pipeline
     │                    ┌─────────────────┐
     │                    │ PDF / TXT        │
     │                    │ → pdfplumber     │
     │                    │ → Chunking       │
     │                    │   fixed | sem.   │
     │                    │ → all-MiniLM     │
     │                    └─────────────────┘
     │
     ▼
Cross-encoder reranker (optional)
ms-marco-MiniLM-L-6-v2
     │
     ▼
LLM Generation
claude-sonnet-4-20250514
(cite-only-from-context prompt)
     │
     ▼
Grounding scorer (token overlap)
     │
     ▼
Highlighted answer + citations
```

---

## Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | 384-dim, fast, no API key, strong zero-shot recall on legal text |
| Chunking overlap | 10–15% | Preserves boundary context; prevents retrieval blind spots at splits |
| Retrieval | MMR with λ=0.7 | Balances relevance (70%) and diversity (30%); avoids boilerplate duplicates |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | Joint (query, passage) encoding gives ~10–15% nDCG improvement over bi-encoder alone |
| Grounding score | Token-overlap Jaccard | Transparent, fast, explainable — no secondary LLM call needed |
| Vector store | ChromaDB (local, persistent) | Zero-config, no server, HNSW cosine index |
| LLM | claude-sonnet-4-20250514 | Best instruction-following for citation-constrained prompts |

---

## Project structure

```
legal-rag-demo/
├── app.py              # Streamlit UI (Q&A tab + Architecture tab)
├── ingestion.py        # Text extraction, chunking, embedding, ChromaDB storage
├── retrieval.py        # MMR retrieval, cross-encoder reranking
├── generation.py       # Anthropic API wrapper, citation extraction
├── hallucination.py    # Token-overlap grounding scores, HTML highlighting
├── sample_docs/
│   ├── mock_sec_10k.txt        # Synthetic SEC Form 10-K (TechVault Systems)
│   ├── mock_court_opinion.txt  # Synthetic SCOTUS opinion (data privacy)
│   └── mock_contract.txt       # Synthetic software license agreement
├── .env.example        # API key template
├── requirements.txt
└── README.md
```

---

## Sample questions to try

**SEC 10-K (TechVault Systems)**
- What is TechVault's annual recurring revenue?
- What were TechVault's net losses in fiscal year 2023?
- Who are TechVault's main competitors?
- What is TechVault's gross margin?

**Court Opinion (DataStream v. FTC)**
- What was the Supreme Court's holding on the disgorgement issue?
- What accuracy did DataStream's model achieve in predicting medical diagnoses?
- Who dissented and why?
- What three-part test did the FTC use for unfairness?

**Software Contract (Nexus Intelligence / Meridian Capital)**
- What is the annual subscription fee?
- Under what conditions can the licensor increase fees?
- What is the liability cap?
- What happens to the software license upon termination?

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `anthropic` | LLM generation |
| `chromadb` | Local vector store |
| `sentence-transformers` | Bi-encoder + cross-encoder |
| `pdfplumber` | PDF text extraction |
| `PyPDF2` | PDF fallback extractor |
| `python-dotenv` | `.env` loading |
| `scikit-learn` | Cosine similarity for MMR |
| `numpy` | Numerical operations |
| `torch` / `transformers` | Model inference backend |
