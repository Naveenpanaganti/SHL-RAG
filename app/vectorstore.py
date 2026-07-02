"""
FAISS vector store.

build_index()  — called once at startup, loads catalog and builds the index.
get_index()    — returns (faiss_index, embedder) for retrieval.
get_catalog()  — returns the normalized catalog list for URL/name validation.

Design:
- all-MiniLM-L6-v2 is small (80MB), fast, and good enough for this domain.
- Inner-product index on L2-normalized vectors = cosine similarity.
- Index is held in module-level globals (process memory); no persistence needed.

Catalog schema normalization:
  Raw field   → Normalized field
  link        → url
  keys[]      → test_type (comma-separated letter codes, e.g. "K,S")
"""

import json
import logging
import os
from typing import Any, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_index: Optional[faiss.IndexFlatIP] = None
_embedder: Optional[SentenceTransformer] = None
_catalog: List[dict] = []
_name_map: dict = {}   # lowercase name → catalog item, for O(1) name lookup

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "catalog.json")
MODEL_NAME = "all-MiniLM-L6-v2"

# Maps full key strings → single letter codes used in the API response schema.
# Only codes defined in the SHL spec are included: A, B, C, D, K, P, S.
# "Assessment Exercises" has no valid single-letter code in the spec — omitted.
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
    Load catalog.json, normalize fields, embed each item, build the FAISS index.
    Called once via FastAPI lifespan hook.
    """
    global _index, _embedder, _catalog

    logger.info("Loading catalog from %s", CATALOG_PATH)
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Normalize — filter out items missing name or URL
    _catalog = [_normalize(item) for item in raw if item.get("name") and item.get("link")]
    logger.info("Loaded and normalized %d catalog items", len(_catalog))

    # Precompute name map for O(1) lookup during URL validation
    global _name_map
    _name_map = {item["name"].lower(): item for item in _catalog}

    logger.info("Loading embedding model: %s", MODEL_NAME)
    _embedder = SentenceTransformer(MODEL_NAME)

    # Build a rich text representation for each item to improve recall
    texts = [_item_to_text(item) for item in _catalog]

    logger.info("Encoding %d items...", len(texts))
    vectors = _embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize for cosine via inner product
        show_progress_bar=False,
        batch_size=64,
    )

    dim = vectors.shape[1]
    _index = faiss.IndexFlatIP(dim)
    _index.add(vectors.astype(np.float32))

    logger.info("FAISS index built: %d vectors, dim=%d", _index.ntotal, dim)


def get_index() -> Tuple[Optional[faiss.IndexFlatIP], Optional[SentenceTransformer]]:
    """Return the FAISS index and embedder. Both are None if not yet built."""
    return _index, _embedder


def get_catalog() -> List[dict]:
    """Return the normalized catalog list."""
    return _catalog


def get_name_map() -> dict:
    """Return the precomputed lowercase-name → catalog item map."""
    return _name_map


def _normalize(item: dict) -> dict:
    """
    Normalize a raw catalog item to the internal schema.
    - link → url
    - keys[] full strings → test_type letter codes (comma-separated)
    """
    keys = item.get("keys", [])
    codes = [KEYS_TO_CODE.get(k, "") for k in keys]
    test_type = ",".join(c for c in codes if c)

    return {
        "name": item.get("name", "").strip(),
        "url": item.get("link", ""),
        "test_type": test_type,
        "description": item.get("description", ""),
        "keys": keys,                            # keep full strings for prompt context
        "job_levels": item.get("job_levels", []),
        "languages": item.get("languages", []),
        "duration": item.get("duration", ""),
        "remote": item.get("remote", ""),
        "adaptive": item.get("adaptive", ""),
        "entity_id": item.get("entity_id", ""),
    }


def _item_to_text(item: dict) -> str:
    """
    Convert a normalized catalog item to a rich, weighted searchable string.

    Design decisions:
    - Name is repeated 3x so it dominates the embedding (boosts exact-name recall)
    - Description provides semantic context for role-based queries
    - Keys map to natural-language synonyms so "cognitive" matches "Ability & Aptitude"
    - Job levels help match seniority-based queries ("senior", "graduate", "executive")
    """
    name = item.get("name", "")
    description = item.get("description", "")
    keys = item.get("keys", [])
    job_levels = item.get("job_levels", [])

    # Expand keys to natural-language synonyms for better semantic matching
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
        key_synonyms.get(k, k) for k in keys
    )

    # Job level synonyms
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
        level_synonyms.get(jl, jl) for jl in job_levels
    )

    parts = [
        name, name, name,        # repeat name 3x for stronger embedding weight
        description,
        expanded_keys,
        expanded_levels,
    ]
    return " | ".join(p for p in parts if p)
