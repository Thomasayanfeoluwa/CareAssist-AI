from slowapi import Limiter
from slowapi.util import get_remote_address

# In-memory limiter for Stage 1. Stage 5 swaps storage_uri=REDIS_URL
# once we run more than one Uvicorn worker and need cross-process limits.
limiter = Limiter(key_func=get_remote_address)



