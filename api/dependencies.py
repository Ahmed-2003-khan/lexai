from fastapi import Request
from redis.asyncio import Redis
from asyncpg import Pool

def get_redis(request: Request) -> Redis:
    """Provides the Redis client instance from application state."""
    return request.app.state.redis

def get_db_pool(request: Request) -> Pool:
    """Provides the asyncpg database pool from application state."""
    return request.app.state.db_pool