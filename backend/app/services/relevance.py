from __future__ import annotations

import logging

from app.services.llm import LLMService
from app.services.types import RetrievedChunk

logger = logging.getLogger(__name__)

# If the top PDF score is at or above this, trust the chunks without asking
# an LLM. Saves the Groq call for the clearly confident cases.
CONFIDENT_THRESHOLD = 0.75

# Judge score at or above this means "the chunks answer the question".
# Below this, fire the web fallback.
#   5 = passages directly answer with specific facts
#   4 = passages answer most of the question
#   3 = passages are topically relevant but do not answer
#   2 = passages are adjacent but irrelevant
#   1 = passages are unrelated
JUDGE_ANSWER_THRESHOLD = 4

# Max length of each chunk to include in the judge prompt.
CHUNK_PREVIEW_LEN = 350

_JUDGE_PROMPT = (
    "You are a relevance judge for a medical RAG system. "
    "Given a user question and 3 retrieved passages from a medical textbook, "
    "score how well the passages answer the question. "
    "\n\n"
    "Score on this exact 1-5 scale:\n"
    "5 = The passages directly answer the question with the specific facts "
    "requested.\n"
    "4 = The passages answer most of the question; some details may be "
    "missing.\n"
    "3 = The passages are topically relevant but do not actually answer the "
    "specific question asked.\n"
    "2 = The passages are about an adjacent topic but are not relevant to "
    "this question.\n"
    "1 = The passages are unrelated to the question.\n"
    "\n"
    "Examples:\n"
    "- Question: 'What are the side effects of aspirin?' with passages about "
    "aspirin side effects → score 5.\n"
    "- Question: 'What are the 2026 WHO recommendations for malaria?' with "
    "passages about malaria in general but no 2026 or WHO content → score 3.\n"
    "- Question: 'What is vitamin D intake for adults?' with passages about "
    "vitamin B12 → score 2.\n"
    "\n"
    "Rules:\n"
    "1. Focus on whether the passages contain the information needed to "
    "answer the question, not on whether the topic matches.\n"
    "2. If the passages discuss a specific year, drug, or organization not "
    "mentioned in the passages, use score 3 or lower.\n"
    "3. Return ONLY the single digit 1, 2, 3, 4, or 5. No explanation, no "
    "punctuation, no code fences.\n"
)


async def score_chunks_for_question(
    question: str,
    chunks: list[RetrievedChunk],
    llm: LLMService,
) -> int:
    """Return a 1-5 relevance score for how well chunks answer the question.

    Skips the LLM judge entirely if the top cosine score is above
    CONFIDENT_THRESHOLD (returns 5 without calling the LLM).

    Fails open: on judge errors or unparseable responses, returns
    JUDGE_ANSWER_THRESHOLD so the caller does not penalize the PDF path.
    """
    if not chunks:
        return 1

    top_score = chunks[0].score
    if top_score is not None and top_score >= CONFIDENT_THRESHOLD:
        logger.info(
            "relevance_skipped top_score=%.3f (>= %.2f)",
            top_score, CONFIDENT_THRESHOLD,
        )
        return 5

    # Build the judge input — 3 chunks max, ~350 chars each
    previews = []
    for i, c in enumerate(chunks[:3], 1):
        text = c.content.strip()
        if len(text) > CHUNK_PREVIEW_LEN:
            text = text[:CHUNK_PREVIEW_LEN] + "…"
        title = c.title or "unknown"
        page = c.page or "?"
        previews.append(
            f"Passage {i} (from {title}, page {page}):\n{text}"
        )
    passages_str = "\n\n".join(previews)

    user_prompt = (
        f"Question: {question}\n\n"
        f"Retrieved passages:\n{passages_str}\n\n"
        f"Score (1-5):"
    )

    try:
        result = await llm.complete(_JUDGE_PROMPT, user_prompt)
        raw = result.content.strip()
        # Extract the first digit from the response
        digit = next((ch for ch in raw if ch in "12345"), None)
        if digit is None:
            logger.warning(
                "relevance_judge_unparseable first_50=%r", result.content[:50]
            )
            return JUDGE_ANSWER_THRESHOLD
        score = int(digit)
        logger.info(
            "relevance_judged top_score=%.3f score=%d",
            top_score if top_score is not None else -1.0, score,
        )
        return score
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "relevance_judge_failed error_type=%s", type(exc).__name__
        )
        # Fail open — do not penalize the PDF path on a judge error
        return JUDGE_ANSWER_THRESHOLD