import secrets
from datetime import date
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from arq.connections import ArqRedis
from arq.jobs import Job
from core.config import settings
from core.schemas import CompanyProfile

router = APIRouter(prefix="/api/v1/briefings", tags=["Briefings"])

async def verify_api_key(x_api_key: str = Header(...)):
    if not secrets.compare_digest(x_api_key, settings.API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API Key")

@router.post("/trigger", status_code=202)
async def trigger_briefing(
    profile: CompanyProfile,
    request: Request,
    _ = Depends(verify_api_key)
):
    redis_pool: ArqRedis = request.app.state.redis_pool
    
    # Idempotency: One briefing per company per day
    job_id = f"brief_{profile.company_name}_{date.today()}"
    
    job = await redis_pool.enqueue_job(
        'run_briefing_workflow', 
        profile.model_dump(), 
        _job_id=job_id
    )
    
    if job is None:
        return {"message": "Briefing already in progress or completed for today", "job_id": job_id}
        
    return {"message": "Briefing job accepted", "job_id": job_id}

@router.get("/status/{job_id}")
async def get_briefing_status(
    job_id: str,
    request: Request,
    _ = Depends(verify_api_key)
):
    redis_pool: ArqRedis = request.app.state.redis_pool
    job = Job(job_id, redis_pool)
    status = await job.status()
    
    return {"job_id": job_id, "status": status.value}
