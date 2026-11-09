# from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app.api.deps import SettingsDep, require_bearer_token
from app.prompts.system import build_system_prompt
from app.ratelimit import limiter
from app.services.llm import LLMService

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class Source(BaseModel):
    type: str            # "pdf" or "web"
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
) -> ChatResponse:
    request_id: str = getattr(request.state, "request_id", "unknown")
    if settings.LLM_MOCK:
        return ChatResponse(
            reply=f"[mock] You said: {body.message}",
            model="mock",
            request_id=request_id,
        )
    llm = LLMService(settings)
    result = await llm.complete(build_system_prompt(), body.message)
    return ChatResponse(
        reply=result.content, model=result.model, request_id=request_id
    )