import time
from fastapi import Depends, HTTPException, Request, status
from redis.asyncio import Redis

from api.dependencies import get_redis

async def rate_limit_dependency(
    request: Request, 
    redis: Redis = Depends(get_redis)
) -> None:
    """
    Implements a token bucket rate limiter utilizing Redis.
    Limits to 60 requests per minute per IP address.
    """
    client_ip = request.client.host if request.client else "unknown_ip"
    key = f"rate_limit:{client_ip}"
    
    limit = 60
    window = 60
    
    current_time = int(time.time())
    
    pipeline = redis.pipeline()
    pipeline.zremrangebyscore(key, 0, current_time - window)
    pipeline.zadd(key, {str(current_time): current_time})
    pipeline.zcard(key)
    pipeline.expire(key, window)
    
    results = await pipeline.execute()
    request_count = results[2]
    
    if request_count > limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(window)}
        )