"""
Semantic retrieval over the FAISS index.

Improvements over v1:
- Multi-query expansion: generates 3 sub-queries from the conversation
  (role-focused, skills-focused, traits-focused) and merges results.
  This improves Recall@10 significantly for complex role descriptions.
- Score-weighted deduplication: when an item appears in multiple sub-queries,
  its best score is kept and it gets a small boost (appeared in >1 query).
- Keyword boosting: items whose name/description contain explicit keywords
  from the user query get a small additive score boost.
"""

import logging
import re
from typing import List, Dict, Any

import numpy as np

from app.vectorstore import get_index, get_catalog

logger = logging.getLogger(__name__)

# Keywords that map to specific assessment types — used for boosting
TYPE_KEYWORDS: Dict[str, List[str]] = {
    "K": ["java", "python", "sql", "aws", "docker", "spring", "excel", "word",
          "networking", "linux", "coding", "technical", "knowledge", "programming",
          "software", "developer", "engineer", "hipaa", "financial", "accounting"],
    "A": ["cognitive", "reasoning", "aptitude", "ability", "numerical", "verbal",
          "deductive", "inductive", "logical", "critical thinking", "problem solving",
          "verify", "graduate"],
    "P": ["personality", "behaviour", "behavioral", "opq", "leadership", "management",
          "sales", "culture", "fit", "workplace", "motivation", "character"],
    "B": ["situational", "judgment", "scenarios", "sjt", "graduate scenarios",
          "biodata", "decision"],
    "S": ["simulation", "contact center", "call", "customer service", "svar",
          "spoken", "language"],
    "C": ["competency", "competencies", "skills assessment", "gsa", "global skills"],
}


def retrieve(query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """
    Multi-query semantic retrieval with keyword boosting.

    Steps:
    1. Build 3 query variants from the conversation text
    2. Embed and search each variant
    3. Merge results, keeping best score per item
    4. Apply keyword boost
    5. Sort by final score, return top_k
    """
    if not query.strip():
        logger.warning("Empty query — returning empty list")
        return []

    index, get_embedder = get_index()
    catalog = get_catalog()

    if index is None or not catalog:
        logger.error("Vector store not initialized")
        return []

    embedder = get_embedder()  # lazy-loads fastembed model on first call

    # Build query variants for better recall
    queries = _expand_query(query)
    logger.debug("Query variants: %s", queries)

    # Embed all variants — fastembed.embed() returns a generator of 1-D arrays
    raw_vectors = list(embedder.embed(queries))
    vectors = np.array(raw_vectors, dtype=np.float32)

    # L2-normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors /= norms

    # Collect scores per catalog index across all query variants
    k = min(top_k * 2, len(catalog))  # fetch more, then trim after boosting
    score_map: Dict[int, float] = {}

    for vec in vectors:
        distances, indices = index.search(vec.reshape(1, -1), k)
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            # Keep best score across variants; small boost if seen in multiple queries
            if idx in score_map:
                score_map[idx] = max(score_map[idx], float(dist)) + 0.01
            else:
                score_map[idx] = float(dist)

    # Apply keyword boost based on explicit terms in the query
    query_lower = query.lower()
    for idx, item in enumerate(catalog):
        if idx not in score_map:
            continue
        boost = _keyword_boost(item, query_lower)
        if boost > 0:
            score_map[idx] += boost

    # Sort by final score descending, take top_k
    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for idx, score in ranked:
        entry = catalog[idx].copy()
        entry["_score"] = score
        results.append(entry)

    logger.debug(
        "retrieve('%s...') → %d results, top=%.3f",
        query[:50], len(results), results[0]["_score"] if results else 0,
    )
    return results


def _expand_query(query: str) -> List[str]:
    """
    Generate multiple query variants from the original conversation text.
    - q1: original (full context)
    - q2: role + skills focus (strip filler words)
    - q3: assessment type focus (what kind of test is needed)
    Returns deduplicated list of up to 3 non-empty strings.
    """
    q1 = query.strip()

    # q2: extract role-like phrases — nouns + adjectives before "developer/engineer/manager"
    role_pattern = re.sub(
        r'\b(i am|we are|we need|hiring|looking for|assessment for|help with|please|'
        r'actually|also|add|remove|drop|include|should|would|could|need to|want to)\b',
        '', q1, flags=re.IGNORECASE
    ).strip()

    # q3: map explicit phrases to assessment type vocabulary
    type_vocab = {
        "personality": "personality behaviour OPQ workplace motivation",
        "cognitive": "cognitive reasoning aptitude ability numerical verbal deductive",
        "technical": "technical knowledge skills programming coding",
        "situational": "situational judgment scenarios decision making",
        "simulation": "simulation contact center call customer service",
        "leadership": "leadership management senior executive director",
        "graduate": "graduate entry level fresh university",
        "sales": "sales commercial persuasion motivation",
    }
    q3_parts = []
    for keyword, expansion in type_vocab.items():
        if keyword in q1.lower():
            q3_parts.append(expansion)
    q3 = (role_pattern + " " + " ".join(q3_parts)).strip() if q3_parts else role_pattern

    seen = set()
    variants = []
    for q in [q1, role_pattern, q3]:
        if q and q not in seen:
            seen.add(q)
            variants.append(q)

    return variants if variants else [q1]


def _keyword_boost(item: dict, query_lower: str) -> float:
    """
    Return a small score boost (0.0 – 0.12) based on explicit keyword overlap.

    Fix: only boost when the SAME keyword appears in BOTH the query AND the item.
    Previous version checked them independently, causing false boosts.
    """
    boost = 0.0
    name_lower = item.get("name", "").lower()
    desc_lower = item.get("description", "").lower()
    combined = name_lower + " " + desc_lower

    # Boost if query words appear literally in the item name (strong signal)
    query_words = set(re.findall(r'\b[a-z]{3,}\b', query_lower))
    name_words = set(re.findall(r'\b[a-z]{3,}\b', name_lower))
    overlap = query_words & name_words
    if overlap:
        boost += min(len(overlap) * 0.02, 0.08)

    # Type-specific boost: only when the SAME keyword is in both query AND item
    for type_code, keywords in TYPE_KEYWORDS.items():
        shared = [kw for kw in keywords if kw in query_lower and kw in combined]
        if shared:
            boost += min(len(shared) * 0.02, 0.04)

    return boost
