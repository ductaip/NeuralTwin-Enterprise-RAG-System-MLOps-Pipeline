import time

import redis
from fastapi import HTTPException, Request, status

# Redis client should also use a singleton pattern or injection
# For now, connecting directly as per architecture demo style
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
except Exception:
    redis_client = None  # Fallback or error handling

RATE_LIMIT_DURATION = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 100


class RateLimiter:
    def __init__(self, key_prefix: str = "rate_limit"):
        self.key_prefix = key_prefix

    def is_allowed(self, user_id: str) -> bool:
        if not redis_client:
            return True  # Fail open if Redis is down for demo purposes
            
        key = f"{self.key_prefix}:{user_id}"
        
        # Simple fixed window counter
        current = redis_client.get(key)
        
        if current and int(current) >= RATE_LIMIT_MAX_REQUESTS:
            return False
            
        pipe = redis_client.pipeline()
        pipe.incr(key)
        pipe.expire(key, RATE_LIMIT_DURATION)
        pipe.execute()
        
        return True


async def rate_limit_dependency(request: Request):
    """
    FastAPI dependency to limit requests based on client IP.
    """
    client_ip = request.client.host
    limiter = RateLimiter()
    
    if not limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests",
        )
