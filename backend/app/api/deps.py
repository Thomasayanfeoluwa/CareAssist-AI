from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import Settings, get_settings

SettingsDep = Annotated[Settings, Depends(get_settings)]

_bearer = HTTPBearer(auto_error=False)

_SESSION_COOKIE = "careassist_session"


def require_bearer_token(
    settings: SettingsDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    """Reject requests without the pilot bearer token."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not hmac.compare_digest(credentials.credentials, settings.PILOT_AUTH_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _sign(session_id: str, secret: str) -> str:
    sig = hmac.new(secret.encode(), session_id.encode(), hashlib.sha256).hexdigest()
    return f"{session_id}.{sig}"


def _verify(signed: str, secret: str) -> str | None:
    if "." not in signed:
        return None
    sid, _, sig = signed.rpartition(".")
    if not sid or not sig:
        return None
    expected = hmac.new(secret.encode(), sid.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    return sid


def get_or_set_session_id(
    request: Request,
    response: Response,
    settings: SettingsDep,
) -> str:
    """Return the caller's session ID, issuing a signed cookie if absent.

    - Verifies HMAC signature; forged/tampered values are rejected.
    - Refreshes cookie Max-Age on every valid request (sliding window).
    - HttpOnly + SameSite=Lax: JS cannot read it, browser sends it with
      same-site fetch.
    """
    raw = request.cookies.get(_SESSION_COOKIE)
    if raw:
        sid = _verify(raw, settings.SESSION_SECRET)
        if sid:
            # Slide cookie expiry so an active conversation survives.
            response.set_cookie(
                key=_SESSION_COOKIE,
                value=raw,
                max_age=settings.SESSION_TTL_SECONDS,
                httponly=True,
                secure=settings.SESSION_COOKIE_SECURE,
                samesite="lax",
                path="/",
            )
            return sid

    sid = secrets.token_urlsafe(32)
    response.set_cookie(
        key=_SESSION_COOKIE,
        value=_sign(sid, settings.SESSION_SECRET),
        max_age=settings.SESSION_TTL_SECONDS,
        httponly=True,
        secure=settings.SESSION_COOKIE_SECURE,
        samesite="lax",
        path="/",
    )
    return sid