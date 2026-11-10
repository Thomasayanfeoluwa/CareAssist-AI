from __future__ import annotations

import logging

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()
_storage_uri = _settings.REDIS_URL or "memory://"
logger.info("rate_limiter_backend=%s", "redis" if _settings.REDIS_URL else "memory")

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=_storage_uri,
)