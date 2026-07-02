"""
Core chat orchestration.

Flow:
  1. Extract previous shortlist from conversation history (for refine/compare)
  2. Build enriched search query — last user message double-weighted
  3. Retrieve top-k candidates via multi-query FAISS search
  4. For comparison turns: ensure both named assessments are in catalog context
  5. Build prompt with history + catalog context + previous shortlist
  6. Call LLM, parse structured JSON response
  7. Validate every URL against the FULL catalog — no hallucinations escape
  8. Deduplicate, filter empty objects, return valid ChatResponse

Schema contract (non-negotiable):
- recommendations: [] when clarifying/refusing, list[1-10] when committed
- end_of_conversation: true only on explicit user confirmation
"""

import json
import logging
import re
from typing import List, Dict, Optional

from app.models import Message, ChatResponse, Recommendation
from app.retriever import retrieve
from app.llm import call_llm
from app.prompts import build_system_prompt, build_user_prompt
from app.utils import count_turns, extract_json_block
from app.vectorstore import get_catalog, get_name_map

logger = logging.getLogger(__name__)

MAX_TURNS = 8


async def handle_chat(messages: List[Message]) -> ChatResponse:
    """Orchestrate a single conversation turn."""
    turn_count = count_turns(messages)

    # Guard: catalog must be ready
    catalog = get_catalog()
    if not catalog:
        return ChatResponse(
            reply="The service is still starting up. Please try again in a moment.",
            recommendations=[],
            end_of_conversation=False,
        )

    # Extract any committed shortlist so LLM can refine, not restart
    previous_shortlist = _extract_previous_shortlist(messages)

    # Build enriched search query
    search_query = _build_search_query(messages, previous_shortlist)

    # Retrieve candidates
    candidates = retrieve(search_query, top_k=25)

    # If retrieval returned nothing, return a clarification question
    if not candidates:
        return ChatResponse(
            reply="I couldn't find relevant assessments for that query. Could you describe the role or skills you're hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )

    # Build URL→item and name→item maps for validation
    retrieved_url_map: Dict[str, dict] = {c["url"]: c for c in candidates}
    full_catalog_url_map: Dict[str, dict] = {
        item["url"]: item for item in catalog if item.get("url")
    }
    # Use precomputed O(1) name map from vectorstore
    catalog_name_map: Dict[str, dict] = get_name_map()

    # For comparison turns: detect named assessments and ensure their full
    # catalog entries are included in the context passed to the LLM.
    # Without this, the LLM may not have enough grounding data to compare.
    comparison_items = _fetch_comparison_items(messages, catalog_name_map, full_catalog_url_map)

    # Ensure shortlist items are always in context (needed for compare/refine)
    shortlist_urls = {r.get("url") for r in previous_shortlist if r.get("url")}
    shortlist_items = [item for item in catalog if item["url"] in shortlist_urls]

    # Merge: shortlist + comparison items first, then retrieved (deduped)
    seen_in_context = {c["url"] for c in shortlist_items + comparison_items}
    deduplicated_candidates = (
        shortlist_items
        + comparison_items
        + [c for c in candidates if c["url"] not in seen_in_context]
    )

    # Serialize catalog context for prompt (trim description to control token budget)
    catalog_context = json.dumps(
        [
            {
                "name": c["name"],
                "url": c["url"],
                "test_type": c["test_type"],
                "description": c.get("description", "")[:300],
                "job_levels": c.get("job_levels", []),
                "duration": c.get("duration", ""),
                "keys": c.get("keys", []),
            }
            for c in deduplicated_candidates[:30]  # cap at 30 items for token budget
        ],
        indent=2,
    )

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(
        messages=messages,
        catalog_context=catalog_context,
        previous_shortlist=previous_shortlist,
        turn_count=turn_count,
        max_turns=MAX_TURNS,
    )

    try:
        raw_response = await call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
    except Exception as exc:
        logger.error("LLM call failed: %s", exc, exc_info=True)
        raise  # let the router return 500 with a clear error

    return _parse_llm_response(
        raw_response,
        retrieved_url_map,
        full_catalog_url_map,
        catalog_name_map,
        turn_count,
    )


def _build_search_query(messages: List[Message], previous_shortlist: List[dict]) -> str:
    """
    Build an enriched search query.
    - Last user message is double-weighted (carries the latest constraint)
    - Long messages (likely JD pastes) are trimmed to 300 chars
    - Previous shortlist names anchor retrieval to the right domain
    """
    user_texts = [m.content for m in messages if m.role == "user"]

    # Trim very long messages (JD pastes) to avoid diluting the embedding
    user_texts = [t[:300] if len(t) > 400 else t for t in user_texts]

    # Double-weight the last user message — it carries the freshest constraint
    if user_texts:
        weighted = user_texts[:-1] + [user_texts[-1], user_texts[-1]]
    else:
        weighted = user_texts

    shortlist_names = [r.get("name", "") for r in previous_shortlist]
    return " ".join(weighted + shortlist_names).strip()


def _extract_previous_shortlist(messages: List[Message]) -> List[dict]:
    """
    Walk backwards through assistant messages to find the last non-empty shortlist.
    Returns list of {name, url, test_type} dicts.
    """
    for msg in reversed(messages):
        if msg.role != "assistant":
            continue
        try:
            data = extract_json_block(msg.content)
            recs = data.get("recommendations", [])
            if isinstance(recs, list) and len(recs) > 0:
                return recs
        except Exception:
            pass
    return []


def _fetch_comparison_items(
    messages: List[Message],
    catalog_name_map: Dict[str, dict],
    full_catalog_url_map: Dict[str, dict],
) -> List[dict]:
    """
    Detect comparison queries in the latest user message and retrieve
    the full catalog entries for named assessments.

    This ensures the LLM always has grounding data when comparing two items,
    even if they're not in the top-25 retrieved results.

    Returns a list of catalog items (0-2 items).
    """
    last_user = next(
        (m.content for m in reversed(messages) if m.role == "user"), ""
    )

    # Only activate for comparison-like queries
    compare_patterns = [
        r"difference between",
        r"compare\b",
        r"vs\.?\s",
        r"versus",
        r"which is better",
        r"how does .+ differ",
    ]
    is_comparison = any(re.search(p, last_user, re.IGNORECASE) for p in compare_patterns)
    if not is_comparison:
        return []

    items = []
    # Look for assessment names in the query by checking against all catalog names
    # Use a simple containment check (case-insensitive)
    last_lower = last_user.lower()
    for name_lower, item in catalog_name_map.items():
        # Match on significant name fragments (skip very short names)
        if len(name_lower) >= 4 and name_lower in last_lower:
            if item not in items:
                items.append(item)
        if len(items) >= 2:
            break

    # If we found exact names, return them
    if items:
        return items

    # Fallback: check if items from the current shortlist are being compared
    # (user may refer to them by position: "compare the first two")
    return []


def _parse_llm_response(
    raw: str,
    retrieved_url_map: Dict[str, dict],
    full_catalog_url_map: Dict[str, dict],
    catalog_name_map: Dict[str, dict],
    turn_count: int,
) -> ChatResponse:
    """
    Parse the LLM's JSON response.
    - Validates all URLs against full catalog
    - Deduplicates by URL
    - Filters malformed objects (missing name or url)
    - Always returns valid ChatResponse, never raises
    """
    try:
        data = extract_json_block(raw)
    except Exception as exc:
        logger.error("JSON parse failed: %s | raw=%.300s", exc, raw)
        return ChatResponse(
            reply="I encountered an issue processing your request. Could you rephrase?",
            recommendations=[],
            end_of_conversation=(turn_count >= MAX_TURNS),
        )

    reply = (data.get("reply") or "").strip()
    if not reply:
        reply = "I'm here to help. What role are you hiring for?"

    end_of_conversation = bool(data.get("end_of_conversation", False))
    raw_recs = data.get("recommendations")

    # Force end_of_conversation at turn cap
    if turn_count >= MAX_TURNS:
        end_of_conversation = True

    # Empty list when clarifying or refusing
    if not raw_recs:
        return ChatResponse(
            reply=reply,
            recommendations=[],
            end_of_conversation=end_of_conversation,
        )

    validated: List[Recommendation] = []
    seen_urls: set = set()

    for r in raw_recs[:10]:
        # Filter malformed objects
        if not isinstance(r, dict):
            continue
        url = (r.get("url") or "").strip()
        name = (r.get("name") or "").strip()
        test_type = (r.get("test_type") or "").strip()

        if not name:  # skip objects with no name
            continue
        if url in seen_urls:  # skip duplicates
            continue

        catalog_item = None

        # 1. Exact URL match in retrieved set (fastest path)
        if url and url in retrieved_url_map:
            catalog_item = retrieved_url_map[url]

        # 2. Exact URL match in full catalog (valid but not in top-25)
        elif url and url in full_catalog_url_map:
            catalog_item = full_catalog_url_map[url]
            logger.info("URL '%s' valid in full catalog but outside top-25", url)

        # 3. Name match via precomputed O(1) map
        elif name:
            catalog_item = catalog_name_map.get(name.lower())
            if catalog_item:
                logger.warning("URL mismatch for '%s' — recovered via name map", name)
            else:
                # Last resort: partial name match
                for cat_name, cat_item in catalog_name_map.items():
                    if name.lower() in cat_name or cat_name in name.lower():
                        catalog_item = cat_item
                        logger.warning("Partial name match '%s' → '%s'", name, cat_item["name"])
                        break

        if catalog_item:
            final_test_type = test_type or catalog_item["test_type"]
            # Validate test_type — only allow known codes
            valid_codes = set("ABCDKPS")
            final_test_type = ",".join(
                c for c in final_test_type.replace(" ", "").split(",")
                if c in valid_codes
            )
            seen_urls.add(catalog_item["url"])
            validated.append(Recommendation(
                name=catalog_item["name"],
                url=catalog_item["url"],
                test_type=final_test_type,
            ))
        else:
            logger.warning("Dropping '%s' ('%s') — not found in catalog", name, url)

    return ChatResponse(
        reply=reply,
        recommendations=validated,
        end_of_conversation=end_of_conversation,
    )
