"""
GET /health — readiness check.
Returns 200 immediately. Used by evaluator for cold-start wake-up (up to 2 min).
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health")
async def health_check():
    return JSONResponse(status_code=200, content={"status": "ok"})
