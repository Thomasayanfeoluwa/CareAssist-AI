from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app.api.deps import SettingsDep, get_or_set_session_id, require_bearer_token
from app.prompts.system import build_system_prompt
from app.ratelimit import limiter
from app.services.llm import LLMService
from app.services.rag import RAGService, RetrievedChunk
from app.services.session_store import SessionStore

router = APIRouter()

# --- Stage 6.5 helpers -------------------------------------------------------

# Only exact-match on short greeting/casual phrases. Anything longer, or any
# real question, goes through normal retrieval.
_GREETING_PHRASES = {
    "hi", "hello", "hey", "yo", "sup", "hiya", "howdy",
    "thanks", "thank you", "ty", "thx",
    "good morning", "good afternoon", "good evening", "good night",
    "bye", "goodbye", "see you", "see ya",
    "how are you", "how's it going", "how are you doing",
    "what's up", "whats up",
    "ok", "okay", "cool", "got it",
}

# Chunks below this score are dropped before the LLM sees them AND before
# they appear as clickable sources. 0.35 is above the greeting-noise level
# (~0.17) and below the fallback trigger (0.4).
_MIN_SOURCE_SCORE = 0.35
_SNIPPET_LEN = 200


def _is_greeting(text: str) -> bool:
    normalized = " ".join(text.strip().lower().rstrip("!?.,").split())
    return normalized in _GREETING_PHRASES


def _extract_snippet(text: str, max_len: int = _SNIPPET_LEN) -> str:
    """Snippet with leading/trailing ellipsis so users know it is a continuation."""
    text = text.strip()
    if not text:
        return ""
    snippet = text[:max_len]
    suffix = "…" if len(text) > max_len else ""
    return f"…{snippet}{suffix}"


def _filter_chunks_by_score(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Drop low-confidence chunks so neither the LLM nor the UI is polluted."""
    return [
        c for c in chunks
        if c.score is None or c.score >= _MIN_SOURCE_SCORE
    ]


# --- Pydantic models --------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class Source(BaseModel):
    type: str
    title: str
    page: int | None = None
    link: str | None = None
    snippet: str = ""


class ChatResponse(BaseModel):
    reply: str
    model: str
    sources: list[Source] = Field(default_factory=list)
    request_id: str


# --- Routes -----------------------------------------------------------------

@router.get("/healthz")
async def healthz(settings: SettingsDep) -> dict[str, str]:
    return {"status": "ok", "version": settings.APP_VERSION}


@router.post(
    "/api/chat",
    response_model=ChatResponse,
    dependencies=[Depends(require_bearer_token)],
)
@limiter.limit("30/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    settings: SettingsDep,
    session_id: str = Depends(get_or_set_session_id),
) -> ChatResponse:
    request_id: str = getattr(request.state, "request_id", "unknown")
    store: SessionStore = request.app.state.session_store
    rag: RAGService = request.app.state.rag

    # 1. Load prior conversation for this session
    history = await store.get(session_id)
    history_str = "".join(
        f"{'User' if h.role == 'user' else 'Assistant'}: {h.content}\n"
        for h in history
    )

    chunks: list[RetrievedChunk] = []

    if settings.LLM_MOCK:
        # Test/mock path — no retrieval, no LLM
        reply = f"[mock] You said: {body.message}"
        model = "mock"
    elif _is_greeting(body.message):
        # Greeting path — skip retrieval entirely (saves Tavily + Pinecone)
        llm = LLMService(settings)
        greeting_system = build_system_prompt() + (
            "\n\nThe user has sent a greeting or casual message. "
            "Reply briefly and politely. Do not fabricate medical information. "
            "Do not cite sources — none were retrieved for this message."
        )
        result = await llm.complete(greeting_system, body.message)
        reply = result.content
        model = result.model
        # chunks stays [] — no sources for a greeting
    else:
        # Normal RAG path
        chunks = await rag.retrieve(body.message)
        chunks = _filter_chunks_by_score(chunks)
        context_str = rag.assemble_context(chunks)

        parts: list[str] = []
        if history_str:
            parts.append(f"### Previous conversation:\n{history_str}")
        if context_str:
            parts.append(f"### Retrieved context:\n{context_str}")
        else:
            parts.append("### Retrieved context:\n(no relevant context found)")
        parts.append(f"### Current question:\n{body.message}")
        user_prompt = "\n\n".join(parts)

        llm = LLMService(settings)
        result = await llm.complete(build_system_prompt(), user_prompt)
        reply = result.content
        model = result.model

    # 3. Persist the exchange for the next request
    await store.append(session_id, "user", body.message)
    await store.append(session_id, "assistant", reply)

    # 4. Structured sources for the frontend popovers
    sources = [
        Source(
            type=c.source,
            title=c.title,
            page=c.page,
            link=c.link,
            snippet=_extract_snippet(c.content),
        )
        for c in chunks
    ]

    return ChatResponse(
        reply=reply,
        model=model,
        sources=sources,
        request_id=request_id,
    )