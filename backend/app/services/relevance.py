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
    "You are a strict relevance judge for a medical RAG system. "
    "Given a user question and 3 retrieved passages from a medical textbook, "
    "score how well the passages answer the question. "
    "\n\n"
    "Score on this exact 1-5 scale:\n"
    "5 = The passages DIRECTLY answer the question. If the question asks for a "
    "specific fact (a number, a year, an organization, a recommendation, a "
    "dose, a schedule), the passages must contain THAT EXACT FACT, not just "
    "the general topic.\n"
    "4 = The passages answer MOST of the question but miss one minor detail.\n"
    "3 = The passages are on the general topic but do NOT contain the specific "
    "fact requested.\n"
    "2 = The passages are about an adjacent topic, not this question.\n"
    "1 = The passages are unrelated.\n"
    "\n"
    "Critical rule — apply this BEFORE scoring: if the user question mentions "
    "a specific year, organization, guideline, dose, schedule, or recommendation "
    "that is NOT present in the passages, the score is 3 or lower, no matter how "
    "topically related the passages are. A page about malaria in general does "
    "NOT answer 'what are the 2026 WHO recommendations for malaria prevention'."
    "\n"
    "\n"
    "Examples:\n"
    "- Q: 'What are the side effects of aspirin?' Passages: aspirin side effects. "
    "→ Score 5 (direct answer).\n"
    "- Q: 'What is the recommended daily Vitamin D intake for adults?' Passages: "
    "general vitamins overview, no specific daily intake numbers. → Score 3 "
    "(topic matches, specific fact missing).\n"
    "- Q: 'What are the 2026 WHO recommendations for malaria prevention?' "
    "Passages: malaria is a mosquito-borne disease in tropical regions, no 2026, "
    "no WHO, no recommendations. → Score 3 (topic matches, specific fact missing).\n"
    "- Q: 'What is vitamin D intake for adults?' Passages: pages about vitamin "
    "B12. → Score 2 (adjacent topic).\n"
    "- Q: 'What is vitamin D intake for adults?' Passages: pages about hip "
    "fractures. → Score 1 (unrelated).\n"
    "\n"
    "Return ONLY the single digit 1, 2, 3, 4, or 5. "
    "No explanation, no punctuation, no code fences."
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