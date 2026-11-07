from __future__ import annotations

import hmac

from fastapi import Depends, HTTPException, Request, status
from typing import Annotated

from app.config import Settings, get_settings

SettingsDep = Annotated[Settings, Depends(get_settings)]


def require_bearer_token(request: Request, settings: SettingsDep) -> None:
    auth = request.headers.get("Authorization", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthorized"
        )
    if not hmac.compare_digest(token, settings.PILOT_AUTH_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthorized"
        )