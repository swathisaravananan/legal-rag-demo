"""
generation.py — LLM answer generation with grounded citation extraction.

Uses Google Gemini (gemini-2.0-flash) with a strict grounding prompt that
instructs the model to answer ONLY from provided context chunks and to cite
sources inline using [Source: <document>, Page <n>] notation.

If GOOGLE_API_KEY is absent or the API call fails, the function returns a
``generation_skipped`` sentinel so the UI can degrade gracefully to retrieval-
only mode.
"""

from __future__ import annotations

import os
import re
from typing import Any

import google.generativeai as genai

# ---------------------------------------------------------------------------
# System prompt
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
    """
    Parse inline citations from the generated answer.

    Expects the format: [Source: <name>, Page <n>]

    Returns a list of dicts with keys ``source`` and ``page``.
    """
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
    """Resolve GOOGLE_API_KEY from env var or Streamlit secrets."""
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        try:
            import streamlit as st  # noqa: PLC0415
            key = st.secrets.get("GOOGLE_API_KEY", "")
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
    model: str = "gemini-2.0-flash",
) -> dict[str, Any]:
    """
    Generate a grounded answer from retrieved chunks via the Gemini API.

    Parameters
    ----------
    question:
        The user's natural-language question.
    chunks:
        Retrieved (and optionally reranked) context chunks.
    api_key:
        Google API key. Falls back to the ``GOOGLE_API_KEY`` env var,
        then Streamlit secrets.
    model:
        Gemini model ID to use.

    Returns
    -------
    A dict with:
        - ``answer``: generated answer string (or error/sentinel message)
        - ``citations``: list of extracted citation dicts
        - ``input_tokens``: prompt token count (0 on error)
        - ``output_tokens``: completion token count (0 on error)
        - ``generation_skipped``: True if generation could not be performed
        - ``error``: error message string (empty on success)
    """
    key = api_key or _get_api_key()

    if not key:
        return {
            "answer": (
                "⚠️ No Google API key found. Set `GOOGLE_API_KEY` in your "
                "`.env` file or Streamlit secrets to enable answer generation. "
                "The retrieved chunks above are shown for inspection."
            ),
            "citations": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "generation_skipped": True,
            "error": "missing_api_key",
        }

    if not chunks:
        return {
            "answer": "No relevant context was retrieved. Try rephrasing your question or uploading more documents.",
            "citations": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "generation_skipped": True,
            "error": "no_chunks",
        }

    context = _format_context(chunks)
    prompt = _USER_TEMPLATE.format(context=context, question=question)

    try:
        genai.configure(api_key=key)
        gemini = genai.GenerativeModel(
            model_name=model,
            system_instruction=_SYSTEM_PROMPT,
        )
        response = gemini.generate_content(prompt)
        answer_text: str = response.text
        citations = _extract_citations(answer_text)

        # Gemini returns token counts in usage_metadata
        input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
        output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        return {
            "answer": answer_text,
            "citations": citations,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "generation_skipped": False,
            "error": "",
        }

    except Exception as exc:  # pylint: disable=broad-except
        msg = str(exc)
        if "API_KEY_INVALID" in msg or "invalid" in msg.lower():
            friendly = "⚠️ Invalid Google API key. Please check your `GOOGLE_API_KEY`."
        elif "quota" in msg.lower() or "rate" in msg.lower():
            friendly = "⚠️ Gemini rate limit exceeded. Please wait a moment and try again."
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
