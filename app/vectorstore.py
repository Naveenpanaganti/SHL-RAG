"""
FAISS vector store.

build_index()  — called once at startup via run_in_executor (non-blocking).
get_index()    — returns (faiss_index, embedder) for retrieval.
get_catalog()  — returns the normalized catalog list.
get_name_map() — returns precomputed lowercase-name → item map.

Memory design:
- Uses fastembed (ONNX runtime) instead of sentence-transformers + PyTorch.
  Peak RAM: ~120MB vs ~420MB — fits in Render free tier (512MB).
- Model: BAAI/bge-small-en-v1.5 (384-dim, same quality as MiniLM).
- FAISS IndexFlatIP on L2-normalized vectors = cosine similarity.
"""

import json
import logging
import os
from typing import Any, List, Optional, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)

_index: Optional[faiss.IndexFlatIP] = None
_embedder: Any = None          # fastembed.TextEmbedding instance
_catalog: List[dict] = []
_name_map: dict = {}           # lowercase name → catalog item, O(1) lookup

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "catalog.json")
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Maps catalog key strings → single-letter codes (spec: A B C D K P S only).
# "Assessment Exercises" intentionally omitted — no valid spec code.
KEYS_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


def build_index() -> None:
    """
    Load catalog.json, normalize, embed via fastembed, build FAISS index.
    Called once at startup via asyncio.run_in_executor (non-blocking).
    """
    global _index, _embedder, _catalog, _name_map

    logger.info("Loading catalog from %s", CATALOG_PATH)
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    _catalog = [_normalize(item) for item in raw if item.get("name") and item.get("link")]
    logger.info("Loaded and normalized %d catalog items", len(_catalog))

    _name_map = {item["name"].lower(): item for item in _catalog}

    logger.info("Loading fastembed model: %s", MODEL_NAME)
    from fastembed import TextEmbedding
    _embedder = TextEmbedding(model_name=MODEL_NAME)

    texts = [_item_to_text(item) for item in _catalog]
    logger.info("Encoding %d items...", len(texts))

    # fastembed.embed() returns a generator of 1-D numpy arrays
    vectors = np.array(list(_embedder.embed(texts)), dtype=np.float32)

    # L2-normalize so inner product == cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors /= norms

    dim = vectors.shape[1]
    _index = faiss.IndexFlatIP(dim)
    _index.add(vectors)
    logger.info("FAISS index built: %d vectors, dim=%d", _index.ntotal, dim)


def get_index() -> Tuple[Optional[faiss.IndexFlatIP], Any]:
    """Return (faiss_index, embedder). Both None if not yet built."""
    return _index, _embedder


def get_catalog() -> List[dict]:
    """Return the normalized catalog list."""
    return _catalog


def get_name_map() -> dict:
    """Return precomputed lowercase-name → catalog item map."""
    return _name_map


def _normalize(item: dict) -> dict:
    """Normalize raw catalog item: link→url, keys[]→test_type codes."""
    keys = item.get("keys", [])
    test_type = ",".join(
        KEYS_TO_CODE[k] for k in keys if k in KEYS_TO_CODE
    )
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
    """
    Rich searchable text for each catalog item.
    Name repeated 3x to dominate the embedding signal.
    Keys and job levels expanded to natural-language synonyms for recall.
    """
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
