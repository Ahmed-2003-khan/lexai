import logging
import uuid
import asyncpg
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis
from prometheus_fastapi_instrumentator import Instrumentator
from pythonjsonlogger import jsonlogger
import sentry_sdk

from api.config import get_settings
from api.routes import health, query

settings = get_settings()

def setup_logging() -> None:
    """Configures structured JSON logging."""
    logger = logging.getLogger()
    logger.setLevel(settings.LOG_LEVEL)
    
    logHandler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(levelname)s %(name)s %(message)s'
    )
    logHandler.setFormatter(formatter)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(logHandler)

if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        environment=settings.ENVIRONMENT,
        traces_sample_rate=1.0,
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and shutdown events."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Initializing database and cache connections.")
    
    app.state.redis = await Redis.from_url(settings.REDIS_URL, decode_responses=True)
    app.state.db_pool = await asyncpg.create_pool(settings.DATABASE_URL)
    
    yield
    
    logger.info("Closing connections.")
    await app.state.db_pool.close()
    await app.state.redis.aclose()


app = FastAPI(
    title="LexAI Legal Research API",
    version="1.0.0",
    lifespan=lifespan
)

if settings.ENVIRONMENT == "development":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Middleware to inject trace IDs and log request lifecycles."""
    trace_id = request.headers.get("X-Trace-Id", str(uuid.uuid4()))
    request.state.trace_id = trace_id
    
    logger = logging.getLogger(__name__)
    logger.info("Request started", extra={
        "trace_id": trace_id,
        "method": request.method,
        "url": str(request.url)
    })
    
    response = await call_next(request)
    
    logger.info("Request completed", extra={
        "trace_id": trace_id,
        "status_code": response.status_code
    })
    
    response.headers["X-Trace-Id"] = trace_id
    return response

Instrumentator().instrument(app).expose(app)

app.include_router(health.router, prefix="/health")
app.include_router(query.router, prefix="/api/v1")