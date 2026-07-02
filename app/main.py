"""
FastAPI application entry point.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.vectorstore import build_index
from app.routers import health, chat
from app.routers import debug

logger = logging.getLogger(__name__)

# ── Startup validation ────────────────────────────────────────────────────────

def _validate_startup() -> None:
    """Fail fast with a clear message if critical files or env vars are missing."""
    import os

    catalog_path = os.path.join(os.path.dirname(__file__), "..", "data", "catalog.json")
    embeddings_path = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings.npy")

    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"catalog.json not found at {catalog_path}")
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"embeddings.npy not found at {embeddings_path}")

    provider = os.getenv("LLM_PROVIDER", "groq")
    if provider == "groq" and not os.getenv("GROQ_API_KEY"):
        logger.warning("GROQ_API_KEY is not set — LLM calls will fail")
    elif provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        logger.warning("GEMINI_API_KEY is not set — LLM calls will fail")

    logger.info("Startup validation passed. provider=%s", provider)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup validation then build FAISS index in a thread (non-blocking)."""
    _validate_startup()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, build_index)
    logger.info("Application startup complete")
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for SHL Individual Test Solutions.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "SHL Assessment Recommender API is running.",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "debug": "/debug",
    }

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(debug.router)
