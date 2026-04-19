from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from redis.asyncio import Redis
from asyncpg import Pool

from api.dependencies import get_redis, get_db_pool

router = APIRouter(tags=["health"])

@router.get("/")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0"
    }

@router.get("/ready")
async def readiness_check(
    redis: Redis = Depends(get_redis),
    db: Pool = Depends(get_db_pool)
):
    """Deep health check verifying backing services."""
    services_status = {"redis": "down", "postgres": "down"}
    is_ready = True

    try:
        if await redis.ping():
            services_status["redis"] = "up"
    except Exception:
        is_ready = False

    try:
        async with db.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            if result == 1:
                services_status["postgres"] = "up"
    except Exception:
        is_ready = False

    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "services": services_status}
        )

    return {"status": "ready", "services": services_status}