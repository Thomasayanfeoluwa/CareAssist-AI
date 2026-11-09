import logging
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from app.api.routes import router
from app.config import Settings, get_settings
from app.logging_setup import configure_logging
from app.ratelimit import limiter

logger = logging.getLogger(__name__)

CallNext = Callable[[Request], Awaitable[Response]]


async def _rate_limit_handler(request: Request, exc: Exception) -> JSONResponse:
    # exc is a RateLimitExceeded at runtime; typed as Exception so this
    # signature matches Starlette's ExceptionHandler protocol (contravariance).
    del exc
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=429,
        content={"error": "rate_limited", "request_id": request_id},
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.LOG_LEVEL)

    app = FastAPI(title="CareAssist API", version=settings.APP_VERSION)
    app.state.limiter = limiter
    app.state.settings = settings
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

    @app.middleware("http")
    async def limit_body_size(request: Request, call_next: CallNext) -> Response:
        cl = request.headers.get("Content-Length")
        if cl:
            try:
                size = int(cl)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"error": "invalid_content_length"},
                )
            if size > settings.REQUEST_MAX_BYTES:
                rid = getattr(request.state, "request_id", "unknown")
                return JSONResponse(
                    status_code=413,
                    content={"error": "payload_too_large", "request_id": rid},
                )
        return await call_next(request)

    @app.middleware("http")
    async def catch_exceptions(request: Request, call_next: CallNext) -> Response:
        try:
            return await call_next(request)
        except Exception:
            request_id = getattr(request.state, "request_id", "unknown")
            logger.exception(
                "unhandled_error request_id=%s path=%s",
                request_id,
                request.url.path,
            )
            return JSONResponse(
                status_code=500,
                content={"error": "internal_error", "request_id": request_id},
            )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next: CallNext) -> Response:
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )

    app.include_router(router)
    return app


app = create_app()