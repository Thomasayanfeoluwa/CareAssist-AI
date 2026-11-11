from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.services.llm import LLMService
from app.services.session_store import HistoryEntry

logger = logging.getLogger(__name__)

_REWRITE_PROMPT = (
    "You are a query rewriter for a medical RAG system. "
    "Your job: turn a follow-up message into a standalone search query "
    "that a medical textbook index can answer. "
    "\n\n"
    "CRITICAL RULE: The rewritten query MUST include the exact medical "
    "topic (drug name, condition, symptom, treatment) from the PREVIOUS "
    "USER MESSAGE, not from the assistant's reply. The assistant's reply "
    "may mention warnings, caveats, or unrelated topics — ignore those.\n\n"
    "Example that is correct:\n"
    "  Previous user message: 'How many milligrams of aspirin should I take?'\n"
    "  Assistant reply: 'I cannot give a dosage. Children under 12 need special care.'\n"
    "  New user message: 'What about children?'\n"
    "  Rewritten: 'What are the aspirin considerations for children?'  "
    "(contains 'aspirin' from the user, not 'children' from the reply)\n\n"
    "Example that is wrong:\n"
    "  Same inputs, rewritten as 'What about children\\'s health?' — this "
    "LOSES the aspirin topic and is NOT acceptable.\n\n"
    "If the follow-up does not refer to the previous topic at all, return "
    "the follow-up unchanged.\n"
    "Do NOT answer the question. Only rewrite it.\n"
    "Do NOT add medical facts not implied by the previous user message.\n"
    "Return ONLY the rewritten question — no quotes, no explanation.\n"
)

_FOLLOWUP_MARKERS = (
    " it ", " that ", " this ", " those ", " these ",
    " what about ", " how about ", " and ", " also ",
    " why ", " when ", " more ", " else",
)

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "what", "how", "why",
    "when", "where", "which", "who", "of", "to", "for", "in", "on", "at",
    "by", "with", "and", "or", "but", "not", "if", "than", "then", "so",
    "do", "does", "did", "have", "has", "had", "be", "been", "being",
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "this", "that", "these", "those", "am",
    "should", "would", "could", "can", "will", "may", "might", "must",
    "about", "some", "any", "all", "no", "yes", "there", "here",
})


@dataclass(frozen=True)
class RewriteResult:
    query: str
    was_rewritten: bool


def _is_followup(message: str) -> bool:
    trimmed = message.strip()
    lowered = f" {trimmed.lower()} "
    return (
        len(trimmed.split()) <= 8
        or any(m in lowered for m in _FOLLOWUP_MARKERS)
    )


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]{4,}", text.lower())


def _anchor_words(text: str) -> set[str]:
    return {w for w in _tokenize(text) if w not in _STOPWORDS}


def _find_topic_anchor(history: list[HistoryEntry]) -> str:
    """Most recent USER message that does not look like a follow-up."""
    candidates = [
        h.content.strip()
        for h in history
        if h.role == "user" and h.content.strip()
    ]
    if not candidates:
        return ""
    for c in reversed(candidates):
        if not _is_followup(c):
            return c
    return candidates[-1]


async def rewrite_query(
    user_message: str,
    history: list[HistoryEntry],
    llm: LLMService,
    max_history: int = 5,
) -> RewriteResult:
    """Turn a follow-up into a standalone retrieval query.

    Three safeguards against topic loss:
    1. Anchor on the previous USER message, not the assistant reply.
    2. Reject rewrites that drop the anchor's medical topic.
    3. Fall back to "{anchor} {followup}" concatenation when the LLM drifts.
    4. Prepend the anchor's primary term so Pinecone weights it first.
    """
    trimmed = user_message.strip()
    if not history or not trimmed:
        return RewriteResult(query=trimmed, was_rewritten=False)

    if not _is_followup(trimmed):
        return RewriteResult(query=trimmed, was_rewritten=False)

    anchor = _find_topic_anchor(history)
    if not anchor:
        return RewriteResult(query=trimmed, was_rewritten=False)

    concatenated = f"{anchor} {trimmed}"

    recent = history[-max_history:]
    history_str = "".join(
        f"{'User' if h.role == 'user' else 'Assistant'}: {h.content[:200]}\n"
        for h in recent
    )
    user_prompt = (
        f"Conversation history:\n{history_str}\n\n"
        f"Previous user topic (ANCHOR — the rewritten query must reference this):\n{anchor}\n\n"
        f"New user message:\n{trimmed}\n\n"
        f"Rewritten standalone question:"
    )

    try:
        result = await llm.complete(_REWRITE_PROMPT, user_prompt)
        rewritten = result.content.strip().strip('"').strip("'").strip()
        if not rewritten or len(rewritten) > 500:
            return RewriteResult(query=concatenated, was_rewritten=False)

        anchor_set = _anchor_words(anchor)
        rewrite_set = {w.lower() for w in _tokenize(rewritten)}

        if anchor_set and not (anchor_set & rewrite_set):
            logger.warning(
                "query_rewrite_lost_topic anchor_terms=%d rewritten_len=%d",
                len(anchor_set), len(rewritten),
            )
            return RewriteResult(query=concatenated, was_rewritten=False)

        if anchor_set:
            primary = max(anchor_set, key=len)
            if not rewritten.lower().startswith(primary.lower()):
                rewritten = f"{primary}: {rewritten}"
                logger.info(
                    "query_rewrite_anchored primary_len=%d rewritten_len=%d",
                    len(primary), len(rewritten),
                )

        logger.info(
            "query_rewritten original_len=%d rewritten_len=%d",
            len(trimmed), len(rewritten),
        )
        return RewriteResult(query=rewritten, was_rewritten=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "query_rewrite_failed error_type=%s (using concatenation)",
            type(exc).__name__,
        )
        return RewriteResult(query=concatenated, was_rewritten=False)