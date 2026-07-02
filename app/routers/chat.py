"""
POST /chat — main conversational endpoint.
Stateless: full conversation history is passed on every call.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import logging

from app.models import ChatRequest, ChatResponse
from app.chat import handle_chat

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages list cannot be empty")
    try:
        return await handle_chat(request.messages)
    except Exception as exc:
        logger.error("Chat handler error: %s", exc, exc_info=True)
        # Include error detail for diagnostics — tighten before final submission
        raise HTTPException(status_code=500, detail=f"LLM error: {str(exc)[:200]}") from exc
