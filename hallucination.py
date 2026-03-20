"""
hallucination.py — Token-overlap grounding scores and HTML highlighting.

Approach
--------
For each sentence in the generated answer we compute a *grounding score* —
the fraction of its content tokens that appear in the retrieved context:

    grounding(s) = |tokens(s) ∩ tokens(context)| / |tokens(s)|

Tokens are lowercased alphabetic strings of length ≥ 3 after removing a
curated stop-word list.  This is intentionally simple and explainable — a
Cornell ML researcher will appreciate that we are not overclaiming here and
that the heuristic is transparent.

Color coding:
  - Green  (score ≥ 0.30) : sentence well-grounded in retrieved context
  - Yellow (score ≥ 0.10) : partially grounded / uncertain
  - Red    (score <  0.10) : potentially ungrounded

The overall *grounding score* is the macro-average across sentences,
weighted by sentence length (word count).
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Stop words (minimal set for legal text)
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "that", "this", "with", "from", "are", "was",
        "were", "been", "has", "have", "had", "not", "but", "its", "their",
        "they", "them", "these", "those", "shall", "will", "may", "can",
        "also", "any", "all", "each", "such", "than", "then", "when", "which",
        "who", "whom", "upon", "under", "above", "below", "into", "within",
        "without", "between", "through", "during", "before", "after", "must",
        "said", "does", "did", "being", "more", "other", "some", "only",
        "both", "here", "there", "where", "about", "however", "therefore",
        "thus", "pursuant", "including", "whether", "provided", "accordance",
    }
)

# Grounding thresholds
_THRESHOLD_GREEN = 0.30
_THRESHOLD_YELLOW = 0.10


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """
    Return a set of content tokens from ``text``.

    Extracts lowercase alphabetic runs of length ≥ 3 that are not stop words.
    """
    raw = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return {t for t in raw if t not in _STOP_WORDS}


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    """
    Split answer text into sentences.

    Uses a simple boundary regex that handles abbreviations poorly but is
    adequate for generated legal prose.  Preserves citation markers so they
    are not separated from their host sentence.
    """
    # Temporarily mask citation markers [Source: ..., Page N] to avoid
    # splitting on the period before "Page"
    masked = re.sub(
        r"\[Source:[^\]]+\]", lambda m: m.group().replace(".", "·"), text
    )
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\(])", masked)
    # Restore periods
    return [p.replace("·", ".").strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_sentence(sentence: str, context_tokens: set[str]) -> float:
    """
    Compute the grounding score for a single sentence.

    Parameters
    ----------
    sentence:
        One sentence from the generated answer.
    context_tokens:
        Union of content tokens across all retrieved chunks.

    Returns
    -------
    Float in [0, 1]; higher means better-grounded.
    """
    sent_tokens = _tokenize(sentence)
    if not sent_tokens:
        return 1.0  # e.g., citation-only sentence — treat as grounded
    overlap = sent_tokens & context_tokens
    return len(overlap) / len(sent_tokens)


def score_answer(
    answer: str,
    chunks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    """
    Score every sentence of ``answer`` against the retrieved ``chunks``.

    Parameters
    ----------
    answer:
        Full generated answer text.
    chunks:
        Retrieved context chunks (list of dicts with a ``"text"`` key).

    Returns
    -------
    sentences_scored:
        List of dicts with keys ``text``, ``score``, ``label``.
        ``label`` ∈ {``"green"``, ``"yellow"``, ``"red"``}.
    overall_score:
        Macro-weighted-average grounding score in [0, 1].
    """
    context_tokens: set[str] = set()
    for chunk in chunks:
        context_tokens |= _tokenize(chunk["text"])

    sentences = _split_sentences(answer)
    if not sentences:
        return [], 0.0

    scored: list[dict[str, Any]] = []
    total_weight = 0.0
    weighted_sum = 0.0

    for sent in sentences:
        score = score_sentence(sent, context_tokens)
        if score >= _THRESHOLD_GREEN:
            label = "green"
        elif score >= _THRESHOLD_YELLOW:
            label = "yellow"
        else:
            label = "red"

        weight = max(len(sent.split()), 1)
        weighted_sum += score * weight
        total_weight += weight

        scored.append({"text": sent, "score": score, "label": label})

    overall = weighted_sum / total_weight if total_weight > 0 else 0.0
    return scored, round(overall, 4)


# ---------------------------------------------------------------------------
# HTML highlighting
# ---------------------------------------------------------------------------

_COLOUR_MAP = {
    "green":  ("#d4edda", "#155724"),   # (bg, text)
    "yellow": ("#fff3cd", "#856404"),
    "red":    ("#f8d7da", "#721c24"),
}

_LABEL_TEXT = {
    "green":  "well-grounded",
    "yellow": "partially grounded",
    "red":    "potentially ungrounded",
}


def build_highlighted_html(
    scored_sentences: list[dict[str, Any]],
) -> str:
    """
    Build an HTML string with each sentence highlighted by grounding score.

    The output is safe to render with ``st.markdown(..., unsafe_allow_html=True)``.

    Parameters
    ----------
    scored_sentences:
        Output of :func:`score_answer` — list of ``{text, score, label}`` dicts.

    Returns
    -------
    HTML fragment (no ``<html>`` or ``<body>`` wrapper).
    """
    if not scored_sentences:
        return "<em>No answer to display.</em>"

    parts: list[str] = []
    for item in scored_sentences:
        bg, fg = _COLOUR_MAP[item["label"]]
        tip = f'{_LABEL_TEXT[item["label"]]} (score: {item["score"]:.2f})'
        # Inline span with tooltip via title attribute
        parts.append(
            f'<span style="background-color:{bg}; color:{fg}; '
            f'padding:2px 4px; border-radius:3px; margin:1px; '
            f'display:inline;" title="{tip}">{item["text"]}</span>'
        )

    return " ".join(parts)


def grounding_badge_html(overall_score: float) -> str:
    """
    Return a coloured HTML badge for the overall grounding score.

    Parameters
    ----------
    overall_score:
        Float in [0, 1] from :func:`score_answer`.
    """
    if overall_score >= _THRESHOLD_GREEN:
        bg, fg, label = "#d4edda", "#155724", "High"
    elif overall_score >= _THRESHOLD_YELLOW:
        bg, fg, label = "#fff3cd", "#856404", "Medium"
    else:
        bg, fg, label = "#f8d7da", "#721c24", "Low"

    pct = int(overall_score * 100)
    return (
        f'<span style="background:{bg}; color:{fg}; padding:4px 10px; '
        f'border-radius:12px; font-weight:bold; font-size:0.9em;">'
        f'{label} grounding — {pct}%</span>'
    )
