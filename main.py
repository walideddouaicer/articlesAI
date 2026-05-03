import arq
from fastapi import FastAPI
from contextlib import asynccontextmanager
from core.config import settings
from api.routes import router as briefing_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create ARQ Redis pool
    app.state.redis_pool = await arq.create_pool(
        arq.connections.RedisSettings.from_dsn(settings.REDIS_URL)
    )
    yield
    # Shutdown: Close ARQ Redis pool
    await app.state.redis_pool.close()

app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan
)

app.include_router(briefing_router)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
