"""
generation.py — LLM answer generation with grounded citation extraction.

Uses Anthropic's claude-sonnet-4-20250514 with a strict grounding prompt that
instructs the model to answer ONLY from provided context chunks and to cite
sources inline using [Source: <document>, Page <n>] notation.

If ANTHROPIC_API_KEY is absent or the API call fails, the function returns a
``generation_skipped`` sentinel so the UI can degrade gracefully to retrieval-
only mode.
"""

from __future__ import annotations

import os
import re
from typing import Any

import anthropic

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


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------


def generate_answer(
    question: str,
    chunks: list[dict[str, Any]],
    api_key: str | None = None,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """
    Generate a grounded answer from retrieved chunks via the Anthropic API.

    Parameters
    ----------
    question:
        The user's natural-language question.
    chunks:
        Retrieved (and optionally reranked) context chunks.
    api_key:
        Anthropic API key. Falls back to the ``ANTHROPIC_API_KEY`` env var.
    model:
        Anthropic model ID to use.
    max_tokens:
        Maximum tokens for the completion.

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
    # Check env var, then Streamlit secrets (for Streamlit Cloud deployment)
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        try:
            import streamlit as st  # noqa: PLC0415
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:  # pylint: disable=broad-except
            pass
    key = api_key
    if not key:
        return {
            "answer": (
                "⚠️ No Anthropic API key found. Set `ANTHROPIC_API_KEY` in your "
                "`.env` file or environment variables to enable answer generation. "
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
    user_message = _USER_TEMPLATE.format(context=context, question=question)

    try:
        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        answer_text: str = response.content[0].text
        citations = _extract_citations(answer_text)
        return {
            "answer": answer_text,
            "citations": citations,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "generation_skipped": False,
            "error": "",
        }

    except anthropic.AuthenticationError:
        return {
            "answer": "⚠️ Invalid Anthropic API key. Please check your `ANTHROPIC_API_KEY`.",
            "citations": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "generation_skipped": True,
            "error": "auth_error",
        }
    except anthropic.RateLimitError:
        return {
            "answer": "⚠️ Anthropic rate limit exceeded. Please wait a moment and try again.",
            "citations": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "generation_skipped": True,
            "error": "rate_limit",
        }
    except Exception as exc:  # pylint: disable=broad-except
        return {
            "answer": f"⚠️ Generation failed: {exc}",
            "citations": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "generation_skipped": True,
            "error": str(exc),
        }
