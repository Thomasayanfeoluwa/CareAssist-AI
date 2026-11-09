from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app.api.deps import SettingsDep, get_or_set_session_id, require_bearer_token
from app.prompts.system import build_system_prompt
from app.ratelimit import limiter
from app.services.llm import LLMService
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

    # 1. Load prior history BEFORE appending the current message
    history = await store.get(session_id)
    history_str = "".join(
        f"{'User' if h.role == 'user' else 'Assistant'}: {h.content}\n"
        for h in history
    )

    # 2. Produce reply (mock short-circuits, no history-dependent reply)
    if settings.LLM_MOCK:
        reply = f"[mock] You said: {body.message}"
        model = "mock"
    else:
        llm = LLMService(settings)
        if history_str:
            user_prompt = (
                f"### Chat History (last {len(history)} messages):\n{history_str}\n"
                f"### Current User Question:\n{body.message}\n"
            )
        else:
            user_prompt = body.message
        result = await llm.complete(build_system_prompt(), user_prompt)
        reply = result.content
        model = result.model

    # 3. Persist the exchange so the next request sees it
    await store.append(session_id, "user", body.message)
    await store.append(session_id, "assistant", reply)

    return ChatResponse(reply=reply, model=model, request_id=request_id)