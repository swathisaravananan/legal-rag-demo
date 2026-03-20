"""
generation.py — LLM answer generation with grounded citation extraction.

Uses LiteLLM as a unified provider interface, defaulting to Gemini
(gemini-2.0-flash) with automatic fallback to gemini-1.5-flash if rate
limited. LiteLLM supports 100+ providers so switching models requires
only a model name change.

If GOOGLE_API_KEY is absent or all retries fail, the function returns a
``generation_skipped`` sentinel so the UI degrades gracefully to
retrieval-only mode.
"""

from __future__ import annotations

import os
import re
from typing import Any

import litellm
from litellm import completion

# Suppress LiteLLM's verbose logging
litellm.set_verbose = False

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert legal document analyst. Your task is to answer questions \
using ONLY the context chunks provided below. Follow these rules strictly:

1. Answer solely from the provided context. If the answer is not present or \
   cannot be reasonably inferred from the context, respond with exactly: \
   "Insufficient information in the provided documents to answer this question."

2. For every factual claim, add an inline citation in this exact format: \
   [Source: <document_name>, Page <page_number>]

3. Be precise and concise. Do not fabricate facts, dates, or figures.

4. If multiple sources support a claim, cite all of them.

5. Structure longer answers with clear paragraph breaks.
"""

_USER_TEMPLATE = """\
Context chunks:
{context}

---
Question: {question}

Answer (cite every claim with [Source: ..., Page ...]):"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_context(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Chunk {i}] Source: {chunk['source']}, Page {chunk['page']}\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def _extract_citations(answer_text: str) -> list[dict[str, str]]:
    """Parse [Source: <name>, Page <n>] citations from the generated answer."""
    pattern = re.compile(
        r"\[Source:\s*([^,\]]+),\s*Page\s*(\d+)\]",
        re.IGNORECASE,
    )
    seen: set[tuple[str, str]] = set()
    citations: list[dict[str, str]] = []
    for match in pattern.finditer(answer_text):
        source = match.group(1).strip()
        page = match.group(2).strip()
        key = (source.lower(), page)
        if key not in seen:
            seen.add(key)
            citations.append({"source": source, "page": page})
    return citations


def _get_api_key() -> str:
    """Resolve GROQ_API_KEY from env var or Streamlit secrets."""
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        try:
            import streamlit as st  # noqa: PLC0415
            key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:  # pylint: disable=broad-except
            pass
    return key


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------


def generate_answer(
    question: str,
    chunks: list[dict[str, Any]],
    api_key: str | None = None,
    model: str = "groq/llama-3.3-70b-versatile",
) -> dict[str, Any]:
    """
    Generate a grounded answer from retrieved chunks via LiteLLM.

    Uses Groq's llama-3.3-70b-versatile as the primary model with automatic
    fallback to llama-3.1-8b-instant on rate limit errors. LiteLLM handles
    retries and provider normalisation transparently.

    Parameters
    ----------
    question:
        The user's natural-language question.
    chunks:
        Retrieved (and optionally reranked) context chunks.
    api_key:
        Google API key. Falls back to ``GOOGLE_API_KEY`` env var / Streamlit secrets.
    model:
        LiteLLM model string (e.g. ``"gemini/gemini-2.0-flash"``).

    Returns
    -------
    Dict with keys ``answer``, ``citations``, ``input_tokens``,
    ``output_tokens``, ``generation_skipped``, ``error``.
    """
    key = api_key or _get_api_key()

    if not key:
        return {
            "answer": (
                "⚠️ No Groq API key found. Set `GROQ_API_KEY` in your "
                "`.env` file or Streamlit secrets to enable answer generation."
            ),
            "citations": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "generation_skipped": True,
            "error": "missing_api_key",
        }

    if not chunks:
        return {
            "answer": "No relevant context was retrieved. Try rephrasing your question.",
            "citations": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "generation_skipped": True,
            "error": "no_chunks",
        }

    context = _format_context(chunks)
    prompt = _USER_TEMPLATE.format(context=context, question=question)

    # Set the key for LiteLLM's Groq provider
    os.environ["GROQ_API_KEY"] = key

    try:
        response = completion(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            # Fallback to smaller model if rate limited
            fallbacks=["groq/llama-3.1-8b-instant"],
            num_retries=3,
        )

        answer_text: str = response.choices[0].message.content
        citations = _extract_citations(answer_text)
        usage = response.usage

        return {
            "answer": answer_text,
            "citations": citations,
            "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "generation_skipped": False,
            "error": "",
        }

    except Exception as exc:  # pylint: disable=broad-except
        msg = str(exc)
        if "invalid" in msg.lower() or "api_key" in msg.lower():
            friendly = "⚠️ Invalid Google API key. Please check your `GOOGLE_API_KEY`."
        elif "rate" in msg.lower() or "quota" in msg.lower() or "429" in msg:
            friendly = "⚠️ Rate limit hit on all models. Please wait a minute and retry."
        else:
            friendly = f"⚠️ Generation failed: {exc}"
        return {
            "answer": friendly,
            "citations": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "generation_skipped": True,
            "error": msg,
        }
