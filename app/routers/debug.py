"""
GET /debug — production diagnostics endpoint.
Returns system state without exposing secret values.
"""

import os
import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.vectorstore import get_catalog, get_index, _embedder

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/debug")
async def debug():
    catalog = get_catalog()
    index, _ = get_index()

    provider = os.getenv("LLM_PROVIDER", "groq")
    if provider == "groq":
        key_present = bool(os.getenv("GROQ_API_KEY", ""))
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    else:
        key_present = bool(os.getenv("GEMINI_API_KEY", ""))
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    return JSONResponse({
        "catalog_loaded": len(catalog) > 0,
        "catalog_items": len(catalog),
        "faiss_loaded": index is not None,
        "faiss_vectors": index.ntotal if index is not None else 0,
        "embedder_loaded": _embedder is not None,
        "llm_provider": provider,
        "llm_model": model,
        "api_key_present": key_present,
    })
