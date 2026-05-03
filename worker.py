import structlog
from arq.connections import RedisSettings
from qdrant_client import AsyncQdrantClient
from core.config import settings
from services.scraper import scrape_article
from services.vector_db import upsert_article_to_vector_db, init_qdrant, retrieve_recent_chunks
from services.slack_service import slack_service
from agents.synthesis_crew import generate_weekly_briefing
from core.schemas import CompanyProfile
from typing import Any, Dict

logger = structlog.get_logger()

async def startup(ctx: Dict[str, Any]):
    """
    Initialize Qdrant client and setup collection on worker startup.
    """
    logger.info("worker_startup")
    qdrant_client = AsyncQdrantClient(
        url=settings.QDRANT_URL, 
        api_key=settings.QDRANT_API_KEY
    )
    ctx['qdrant'] = qdrant_client
    
    # Ensure collection and indexes exist
    await init_qdrant(qdrant_client)

async def shutdown(ctx: Dict[str, Any]):
    """
    Close Qdrant client on worker shutdown.
    """
    logger.info("worker_shutdown")
    await ctx['qdrant'].close()

async def ingest_article_task(ctx: Dict[str, Any], url: str):
    """
    Background task to scrape and index an article.
    """
    log = logger.bind(url=url)
    log.info("ingest_task_started")
    
    # 1. Scrape
    article = await scrape_article(url)
    if not article:
        log.error("ingest_task_failed_scraping")
        return "failed_scraping"
    
    # 2. Vector DB Upsert
    try:
        qdrant_client: AsyncQdrantClient = ctx['qdrant']
        await upsert_article_to_vector_db(qdrant_client, article)
        log.info("ingest_task_success")
        return "success"
    except Exception as e:
        log.error("ingest_task_failed_upsert", error=str(e))
        raise # ARQ will retry based on max_tries

async def run_briefing_workflow(ctx: Dict[str, Any], profile_dict: dict):
    """
    Master task: Retrieve -> Synthesize -> Deliver.
    """
    profile = CompanyProfile(**profile_dict)
    log = logger.bind(company=profile.company_name)
    log.info("workflow_started")
    
    try:
        # 1. Retrieve
        qdrant_client: AsyncQdrantClient = ctx['qdrant']
        chunks = await retrieve_recent_chunks(qdrant_client, profile)
        
        # 2. Synthesize
        briefing = await generate_weekly_briefing(chunks, profile)
        
        # 3. Deliver
        await slack_service.send_briefing(briefing)
        
        log.info("workflow_completed")
        return "success"
    except Exception as e:
        log.error("workflow_failed", error=str(e))
        raise

class WorkerSettings:
    """
    ARQ Worker configuration.
    """
    functions = [ingest_article_task, run_briefing_workflow]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)
    max_tries = 3
    job_timeout = 600 # 10 minutes for CrewAI
