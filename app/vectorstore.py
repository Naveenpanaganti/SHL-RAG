"""
FAISS vector store — zero-model runtime design.

At startup, loads pre-computed embeddings from data/embeddings.npy.
No embedding model is loaded on the server — peak RAM ~30MB (numpy + faiss).

To embed a query at request time, uses fastembed with lazy init.
fastembed ONNX model is ~66MB — loaded once on first /chat request,
cached in _embedder global for subsequent requests.

Pre-computing:
    Run locally once:  python build_embeddings.py
    Commit the result: data/embeddings.npy  (~565KB for 377 items)

Catalog schema normalization:
    link  → url
    keys[] → test_type letter codes (spec: A B C D K P S only)
"""

import json
import logging
import os
from typing import Any, List, Optional, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)

_index: Optional[faiss.IndexFlatIP] = None
_embedder: Any = None          # fastembed.TextEmbedding — lazy init on first query
_catalog: List[dict] = []
_name_map: dict = {}

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "catalog.json")
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings.npy")
MODEL_NAME = "BAAI/bge-small-en-v1.5"

KEYS_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
    # "Assessment Exercises" omitted — no valid spec code
}


def build_index() -> None:
    """
    Load catalog + pre-computed embeddings, build FAISS index.
    Called once at startup via run_in_executor (non-blocking).
    No embedding model is loaded here.
    """
    global _index, _catalog, _name_map

    logger.info("Loading catalog from %s", CATALOG_PATH)
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    _catalog = [_normalize(item) for item in raw if item.get("name") and item.get("link")]
    _name_map = {item["name"].lower(): item for item in _catalog}
    logger.info("Loaded %d catalog items", len(_catalog))

    logger.info("Loading pre-computed embeddings from %s", EMBEDDINGS_PATH)
    vectors = np.load(EMBEDDINGS_PATH).astype(np.float32)

    if vectors.shape[0] != len(_catalog):
        raise RuntimeError(
            f"Embedding count mismatch: {vectors.shape[0]} vectors vs "
            f"{len(_catalog)} catalog items. Re-run build_embeddings.py."
        )

    dim = vectors.shape[1]
    _index = faiss.IndexFlatIP(dim)
    _index.add(vectors)
    logger.info("FAISS index built: %d vectors, dim=%d", _index.ntotal, dim)


def get_embedder() -> Any:
    """
    Lazy-load the fastembed model on first query.
    Only ~66MB ONNX model, loaded once and cached.
    """
    global _embedder
    if _embedder is None:
        logger.info("Lazy-loading fastembed model: %s", MODEL_NAME)
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(model_name=MODEL_NAME)
        logger.info("fastembed model loaded")
    return _embedder


def get_index() -> Tuple[Optional[faiss.IndexFlatIP], Any]:
    """Return (faiss_index, embedder_callable). embedder is lazy-loaded."""
    return _index, get_embedder


def get_catalog() -> List[dict]:
    return _catalog


def get_name_map() -> dict:
    return _name_map


def _normalize(item: dict) -> dict:
    keys = item.get("keys", [])
    test_type = ",".join(KEYS_TO_CODE[k] for k in keys if k in KEYS_TO_CODE)
    return {
        "name": item.get("name", "").strip(),
        "url": item.get("link", ""),
        "test_type": test_type,
        "description": item.get("description", ""),
        "keys": keys,
        "job_levels": item.get("job_levels", []),
        "languages": item.get("languages", []),
        "duration": item.get("duration", ""),
        "remote": item.get("remote", ""),
        "adaptive": item.get("adaptive", ""),
        "entity_id": item.get("entity_id", ""),
    }


def _item_to_text(item: dict) -> str:
    """Rich searchable text — used by build_embeddings.py, not at runtime."""
    name = item.get("name", "")
    description = item.get("description", "")

    key_synonyms = {
        "Ability & Aptitude": "cognitive ability aptitude reasoning numerical verbal deductive",
        "Biodata & Situational Judgment": "situational judgment scenarios sjt decision making biodata",
        "Competencies": "competencies skills assessment workplace competency",
        "Development & 360": "development feedback 360 coaching learning",
        "Knowledge & Skills": "knowledge skills technical programming domain expertise",
        "Personality & Behavior": "personality behaviour workplace style motivation character OPQ",
        "Simulations": "simulation work sample exercise realistic job preview",
        "Assessment Exercises": "assessment exercise in-tray e-tray role play",
    }
    expanded_keys = " ".join(
        key_synonyms.get(k, k) for k in item.get("keys", [])
    )

    level_synonyms = {
        "Director": "director senior leadership executive",
        "Executive": "executive C-suite CXO VP senior leader",
        "Graduate": "graduate entry-level fresh university new hire",
        "Entry-Level": "entry level junior trainee",
        "Manager": "manager management team lead",
        "Mid-Professional": "mid-level mid professional experienced",
        "Professional Individual Contributor": "senior IC individual contributor specialist",
        "Front Line Manager": "frontline supervisor team leader",
        "General Population": "all levels any role",
        "Supervisor": "supervisor frontline manager",
    }
    expanded_levels = " ".join(
        level_synonyms.get(jl, jl) for jl in item.get("job_levels", [])
    )

    parts = [name, name, name, description, expanded_keys, expanded_levels]
    return " | ".join(p for p in parts if p)
