from __future__ import annotations

import logging
from dataclasses import dataclass

from app.services.llm import LLMService
from app.services.session_store import HistoryEntry

logger = logging.getLogger(__name__)

_REWRITE_PROMPT = (
    "You are a query rewriter for a medical RAG system. "
    "Given a short conversation history and a new user message, "
    "rewrite the new message into a complete, standalone question "
    "that contains all the context needed to search a medical textbook. "
    "\n\n"
    "Rules:\n"
    "1. Preserve the user's original intent exactly.\n"
    "2. If the new message is already standalone, return it unchanged.\n"
    "3. Do not answer the question. Only rewrite it.\n"
    "4. Do not add medical facts not implied by the history.\n"
    "5. Return only the rewritten question, no quotes, no explanation.\n"
    "6. If the message is a greeting or off-topic, return it unchanged.\n"
)


@dataclass(frozen=True)
class RewriteResult:
    query: str           # the query to send to retrieval
    was_rewritten: bool  # False if history was empty or rewriter returned unchanged


async def rewrite_query(
    user_message: str,
    history: list[HistoryEntry],
    llm: LLMService,
    max_history: int = 5,
) -> RewriteResult:
    """Turn a follow-up into a standalone retrieval query.

    - If history is empty: return message unchanged (no LLM call).
    - If message is short or contains an implicit reference ("it", "that",
      "those", "children", "why", "how"), rewrite using history.
    - Otherwise: return unchanged.

    Fails soft: if the rewrite LLM call errors, returns the original message.
    """
    trimmed = user_message.strip()
    if not history or not trimmed:
        return RewriteResult(query=trimmed, was_rewritten=False)

    # Heuristic: only rewrite if the message looks like a follow-up
    followup_markers = (
        " it ", " that ", " this ", " those ", " these ",
        " what about ", " how about ", " and ", " also ",
        " why ", " when ", " more ", " else",
    )
    lowered = f" {trimmed.lower()} "
    is_followup = (
        len(trimmed.split()) <= 8
        or any(m in lowered for m in followup_markers)
    )
    if not is_followup:
        return RewriteResult(query=trimmed, was_rewritten=False)

    # Build a compact history string
    recent = history[-max_history:]
    history_str = "".join(
        f"{'User' if h.role == 'user' else 'Assistant'}: {h.content[:200]}\n"
        for h in recent
    )
    user_prompt = (
        f"Conversation history:\n{history_str}\n\n"
        f"New user message:\n{trimmed}\n\n"
        f"Rewritten standalone question:"
    )

    try:
        result = await llm.complete(_REWRITE_PROMPT, user_prompt)
        rewritten = result.content.strip().strip('"').strip("'").strip()
        # Sanity: rewriter must return non-empty, reasonable length
        if not rewritten or len(rewritten) > 500:
            return RewriteResult(query=trimmed, was_rewritten=False)
        logger.info(
            "query_rewritten original_len=%d rewritten_len=%d",
            len(trimmed), len(rewritten),
        )
        return RewriteResult(query=rewritten, was_rewritten=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "query_rewrite_failed error_type=%s", type(exc).__name__
        )
        return RewriteResult(query=trimmed, was_rewritten=False)