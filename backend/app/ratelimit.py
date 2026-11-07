from slowapi import Limiter
from slowapi.util import get_remote_address

# In-memory limiter for Stage 1. Stage 5 swaps to storage_uri=REDIS_URL
# once we run more than one Uvicorn worker.
limiter = Limiter(key_func=get_remote_address)