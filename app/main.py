"""
FastAPI application entry point.
Registers routes and initializes the vector store on startup.
"""

from contextlib import asynccontextmanager
import asyncio

from dotenv import load_dotenv
load_dotenv()  # loads .env into os.environ before anything else runs

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.vectorstore import build_index
from app.routers import health, chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Build FAISS index at startup using run_in_executor so the async event loop
    is never blocked. This ensures /health responds immediately during cold start.
    """
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, build_index)
    yield


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

app.include_router(health.router)
app.include_router(chat.router)
