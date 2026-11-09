from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app.api.deps import SettingsDep, get_or_set_session_id, require_bearer_token
from app.prompts.system import build_system_prompt
from app.ratelimit import limiter
from app.services.llm import LLMService
from app.services.rag import RAGService, RetrievedChunk
from app.services.session_store import SessionStore

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class Source(BaseModel):
    type: str
    title: str
    page: int | None = None
    link: str | None = None


class ChatResponse(BaseModel):
    reply: str
    model: str
    sources: list[Source] = Field(default_factory=list)
    request_id: str


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

    # 2. Retrieval + LLM (skipped in mock mode)
    chunks: list[RetrievedChunk] = []
    if settings.LLM_MOCK:
        reply = f"[mock] You said: {body.message}"
        model = "mock"
    else:
        # Real retrieval: PDF first, web fallback if PDF weak
        chunks = await rag.retrieve(body.message)
        context_str = rag.assemble_context(chunks)

        # Build user prompt: history + context + current question
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

    # 4. Structured sources for the frontend (hover popover)
    sources = [
        Source(type=c.source, title=c.title, page=c.page, link=c.link)
        for c in chunks
    ]

    return ChatResponse(
        reply=reply,
        model=model,
        sources=sources,
        request_id=request_id,
    )