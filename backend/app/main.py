from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from app.api.routes import router
from app.config import Settings, get_settings
from app.logging_setup import configure_logging
from app.ratelimit import limiter

logger = logging.getLogger(__name__)


async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(status_code=429, content={"error": "rate_limited"})


async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id: str = getattr(request.state, "request_id", "unknown")
    logger.exception(
        "unhandled_error request_id=%s path=%s", request_id, request.url.path
    )
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "request_id": request_id},
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.LOG_LEVEL)

    app = FastAPI(title="CareAssist API", version=settings.APP_VERSION)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _unhandled_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.middleware("http")
    async def limit_body_size(request: Request, call_next):
        content_length = request.headers.get("Content-Length")
        if content_length and int(content_length) > settings.REQUEST_MAX_BYTES:
            return JSONResponse(status_code=413, content={"error": "payload_too_large"})
        return await call_next(request)

    app.include_router(router)
    return app


app = create_app()