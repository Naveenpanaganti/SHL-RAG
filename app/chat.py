"""
Core chat orchestration with structured logging at every stage.

Flow:
  1. Request received → log
  2. Extract query from messages → log
  3. Retrieve candidates from FAISS → log count
  4. Build catalog context + prompt → log sizes
  5. Call LLM → log start/end
  6. Parse JSON response → log raw on failure
  7. Validate URLs → log any drops
  8. Return ChatResponse
"""

import json
import logging
import re
import traceback
from typing import List, Dict

from app.models import Message, ChatResponse, Recommendation
from app.retriever import retrieve
from app.llm import call_llm
from app.prompts import build_system_prompt, build_user_prompt
from app.utils import count_turns, extract_json_block
from app.vectorstore import get_catalog, get_name_map

logger = logging.getLogger(__name__)

MAX_TURNS = 8


async def handle_chat(messages: List[Message]) -> ChatResponse:
    """Orchestrate a single conversation turn with full structured logging."""

    # ── Stage 1: Request received ──────────────────────────────────────────
    turn_count = count_turns(messages)
    last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
    logger.info("[CHAT] turn=%d last_user_msg=%.80r", turn_count, last_user)

    # ── Stage 2: Catalog guard ─────────────────────────────────────────────
    catalog = get_catalog()
    if not catalog:
        logger.error("[CHAT] Catalog not loaded — build_index() may have failed at startup")
        return ChatResponse(
            reply="The service is still starting up. Please try again in a moment.",
            recommendations=[],
            end_of_conversation=False,
        )
    logger.info("[CHAT] catalog_size=%d", len(catalog))

    # ── Stage 3: Build search query ────────────────────────────────────────
    previous_shortlist = _extract_previous_shortlist(messages)
    search_query = _build_search_query(messages, previous_shortlist)
    logger.info("[CHAT] search_query=%.100r", search_query)

    # ── Stage 4: Retrieve candidates ───────────────────────────────────────
    try:
        candidates = retrieve(search_query, top_k=25)
        logger.info("[CHAT] retrieved=%d candidates", len(candidates))
    except Exception:
        logger.exception("[CHAT] Retrieval failed")
        return ChatResponse(
            reply="I couldn't search the catalog right now. Please try again in a moment.",
            recommendations=[],
            end_of_conversation=False,
        )

    if not candidates:
        logger.warning("[CHAT] Zero candidates returned for query=%.80r", search_query)
        return ChatResponse(
            reply="I couldn't find relevant assessments for that query. Could you describe the role or skills you're hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )

    # ── Stage 5: Build catalog context ────────────────────────────────────
    retrieved_url_map: Dict[str, dict] = {c["url"]: c for c in candidates}
    full_catalog_url_map: Dict[str, dict] = {
        item["url"]: item for item in catalog if item.get("url")
    }
    catalog_name_map: Dict[str, dict] = get_name_map()

    comparison_items = _fetch_comparison_items(messages, catalog_name_map, full_catalog_url_map)
    shortlist_urls = {r.get("url") for r in previous_shortlist if r.get("url")}
    shortlist_items = [item for item in catalog if item["url"] in shortlist_urls]

    seen_in_context = {c["url"] for c in shortlist_items + comparison_items}
    deduplicated_candidates = (
        shortlist_items
        + comparison_items
        + [c for c in candidates if c["url"] not in seen_in_context]
    )

    catalog_context = json.dumps(
        [
            {
                "name": c["name"],
                "url": c["url"],
                "test_type": c["test_type"],
                "description": c.get("description", "")[:150],
                "job_levels": c.get("job_levels", []),
                "duration": c.get("duration", ""),
                "keys": c.get("keys", []),
            }
            for c in deduplicated_candidates[:20]
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
    logger.info(
        "[CHAT] prompt_built system_len=%d user_len=%d context_items=%d",
        len(system_prompt), len(user_prompt), len(deduplicated_candidates[:20]),
    )

    # ── Stage 6: Call LLM ─────────────────────────────────────────────────
    logger.info("[CHAT] llm_request_start provider=%s", _get_provider())
    try:
        raw_response = await call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
        logger.info("[CHAT] llm_response_received len=%d preview=%.80r",
                    len(raw_response), raw_response)
    except Exception as exc:
        logger.exception("[CHAT] LLM call failed: %s", exc)
        raise  # propagate — router returns 500 with str(exc)

    # ── Stage 7: Parse + validate ─────────────────────────────────────────
    return _parse_llm_response(
        raw_response,
        retrieved_url_map,
        full_catalog_url_map,
        catalog_name_map,
        turn_count,
    )


def _get_provider() -> str:
    import os
    return os.getenv("LLM_PROVIDER", "groq")


def _build_search_query(messages: List[Message], previous_shortlist: List[dict]) -> str:
    user_texts = [m.content for m in messages if m.role == "user"]
    # Trim very long messages (JD pastes) to avoid token limit issues
    user_texts = [t[:250] if len(t) > 300 else t for t in user_texts]
    if user_texts:
        weighted = user_texts[:-1] + [user_texts[-1], user_texts[-1]]
    else:
        weighted = user_texts
    shortlist_names = [r.get("name", "") for r in previous_shortlist]
    return " ".join(weighted + shortlist_names).strip()


def _extract_previous_shortlist(messages: List[Message]) -> List[dict]:
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
    last_user = next(
        (m.content for m in reversed(messages) if m.role == "user"), ""
    )
    compare_patterns = [
        r"difference between", r"compare\b", r"vs\.?\s",
        r"versus", r"which is better", r"how does .+ differ",
    ]
    if not any(re.search(p, last_user, re.IGNORECASE) for p in compare_patterns):
        return []

    items = []
    last_lower = last_user.lower()
    for name_lower, item in catalog_name_map.items():
        if len(name_lower) >= 4 and name_lower in last_lower:
            if item not in items:
                items.append(item)
        if len(items) >= 2:
            break
    return items


def _parse_llm_response(
    raw: str,
    retrieved_url_map: Dict[str, dict],
    full_catalog_url_map: Dict[str, dict],
    catalog_name_map: Dict[str, dict],
    turn_count: int,
) -> ChatResponse:
    """
    Parse LLM JSON response with full logging on failure.
    Never silently swallows errors — logs full traceback before returning fallback.
    """
    # ── Parse JSON ────────────────────────────────────────────────────────
    try:
        data = extract_json_block(raw)
    except Exception:
        logger.exception(
            "[PARSE] JSON extraction failed. raw_response=%.500r", raw
        )
        # Return user-friendly message but LOG the real failure
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

    if turn_count >= MAX_TURNS:
        end_of_conversation = True

    if not raw_recs:
        logger.info("[PARSE] No recommendations in LLM response (clarifying/refusing)")
        return ChatResponse(
            reply=reply,
            recommendations=[],
            end_of_conversation=end_of_conversation,
        )

    # ── Validate recommendations ──────────────────────────────────────────
    validated: List[Recommendation] = []
    seen_urls: set = set()
    valid_codes = set("ABCDKPS")

    for r in raw_recs[:10]:
        if not isinstance(r, dict):
            logger.warning("[PARSE] Non-dict recommendation skipped: %r", r)
            continue
        url = (r.get("url") or "").strip()
        name = (r.get("name") or "").strip()
        test_type = (r.get("test_type") or "").strip()

        if not name:
            continue
        if url in seen_urls:
            continue

        catalog_item = None

        if url and url in retrieved_url_map:
            catalog_item = retrieved_url_map[url]
        elif url and url in full_catalog_url_map:
            catalog_item = full_catalog_url_map[url]
            logger.info("[PARSE] URL '%s' matched full catalog (outside top-25)", url)
        elif name:
            catalog_item = catalog_name_map.get(name.lower())
            if catalog_item:
                logger.warning("[PARSE] URL mismatch for '%s' — recovered via name map", name)
            else:
                for cat_name, cat_item in catalog_name_map.items():
                    if name.lower() in cat_name or cat_name in name.lower():
                        catalog_item = cat_item
                        logger.warning("[PARSE] Partial match '%s' → '%s'", name, cat_item["name"])
                        break

        if catalog_item:
            final_test_type = ",".join(
                c for c in (test_type or catalog_item["test_type"]).replace(" ", "").split(",")
                if c in valid_codes
            )
            seen_urls.add(catalog_item["url"])
            validated.append(Recommendation(
                name=catalog_item["name"],
                url=catalog_item["url"],
                test_type=final_test_type,
            ))
        else:
            logger.warning("[PARSE] Dropping '%s' ('%s') — not in catalog", name, url)

    logger.info("[PARSE] validated=%d recommendations, eoc=%s", len(validated), end_of_conversation)

    return ChatResponse(
        reply=reply,
        recommendations=validated,
        end_of_conversation=end_of_conversation,
    )
