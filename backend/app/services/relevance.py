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
    "You will be given a user question and 3 retrieved passages from a medical textbook. "
    "Your job is to decide whether those passages actually contain the information needed to answer the question. "
    "\n\n"
    "MECHANICAL CHECK — do this FIRST, before assigning a score:\n"
    "1. Extract every specific 'anchor' from the question: any year, any organization "
    "name (WHO, CDC, NHS, FDA, etc.), any guideline, any dose, any schedule, any recommendation. "
    "2. For each anchor, check whether that EXACT anchor (or a direct synonym) appears "
    "in the passages. Do NOT accept 'related topic' as a match.\n"
    "3. If ANY anchor is missing from the passages, the maximum score is 3.\n"
    "\n"
    "Only after the mechanical check passes may you assign 4 or 5.\n"
    "\n"
    "Score scale:\n"
    "5 = Every anchor in the question appears in the passages AND the passages state the answer.\n"
    "4 = Every anchor appears, but one minor detail is missing.\n"
    "3 = An anchor is missing, or the passages are on the general topic but do not answer.\n"
    "2 = Different but adjacent topic.\n"
    "1 = Unrelated.\n"
    "\n"
    "Examples of the mechanical check:\n"
    "- Q: 'What are the 2026 WHO recommendations for malaria prevention?' "
    "Anchors: [2026, WHO, recommendations, malaria, prevention]. "
    "Passages that mention malaria prevention but not 'WHO' and not '2026' → score 3. "
    "Passages that mention 'WHO' but not '2026 recommendations' → score 3. "
    "Only passages that contain BOTH '2026' AND 'WHO' AND 'recommendations' → score 4 or 5.\n"
    "- Q: 'What is the recommended daily Vitamin D intake for adults?' "
    "Anchors: [daily, Vitamin D, intake, adults, recommended]. "
    "Passages about vitamins generally but no daily intake numbers → score 3.\n"
    "- Q: 'What are the side effects of aspirin?' "
    "Anchors: [side effects, aspirin]. "
    "Passages listing aspirin side effects → score 5. No anchor missing.\n"
    "\n"
    "Return ONLY the single digit 1, 2, 3, 4, or 5. No explanation, no punctuation, no code fences."
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