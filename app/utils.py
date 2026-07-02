"""
Utility helpers.
"""

import json
import re
import logging
from typing import Any, Dict, List

from app.models import Message

logger = logging.getLogger(__name__)


def count_turns(messages: List[Message]) -> int:
    """
    Count total messages (user + assistant) to enforce the 8-turn cap.
    The spec counts each individual message as a turn.
    """
    return len(messages)


def extract_json_block(text: str) -> Dict[str, Any]:
    """
    Extract and parse a JSON object from LLM output.

    Handles four cases in priority order:
    1. Pure JSON (ideal — LLM followed instructions)
    2. JSON in ```json ... ``` fence (LLM sometimes adds fences despite instructions)
    3. JSON in ``` ... ``` fence (no language tag)
    4. First { ... } block found anywhere in the text (last resort)

    Raises ValueError if no valid JSON object is found.
    """
    text = text.strip()

    # Case 1: pure JSON
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # May have trailing text after the closing brace — try to extract
            pass

    # Case 2 & 3: fenced code block (with or without language tag)
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Case 4: find the outermost { ... } block by brace counting
    # This handles cases where prose precedes or follows the JSON
    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break  # malformed — stop trying

    raise ValueError(f"No valid JSON object found in LLM output: {text[:300]!r}")


def sanitize_reply(text: str) -> str:
    """Strip leaked JSON fences or braces from a reply string."""
    text = re.sub(r"```[a-z]*", "", text)
    text = text.replace("```", "")
    return text.strip()
