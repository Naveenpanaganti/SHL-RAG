"""
LLM client — Groq (primary) or Gemini (secondary).

Improvements:
- max_tokens increased to 2000 (was 1500) — needed for 6-item shortlists with explanations
- Retry once on transient 5xx errors before giving up
- asyncio.get_event_loop() replaced with asyncio.get_running_loop() (correct in async context)
- temperature=0.1 (was 0.2) for more deterministic JSON output
"""

import asyncio
import logging
import os

logger = logging.getLogger(__name__)

PROVIDER = os.getenv("LLM_PROVIDER", "gemini")   # "gemini" | "groq"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

_MAX_TOKENS = 2000
_TEMPERATURE = 0.1
_MAX_RETRIES = 2


async def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Call the configured LLM and return the raw text response.
    Retries once on transient failures.
    Raises RuntimeError if all retries fail.
    """
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            if PROVIDER == "gemini":
                return await _call_gemini(system_prompt, user_prompt)
            elif PROVIDER == "groq":
                return await _call_groq(system_prompt, user_prompt)
            else:
                raise ValueError(f"Unknown LLM_PROVIDER: {PROVIDER}")
        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                wait = 1.5 * (attempt + 1)
                logger.warning("LLM attempt %d failed (%s) — retrying in %.1fs", attempt + 1, exc, wait)
                await asyncio.sleep(wait)
            else:
                logger.error("LLM failed after %d attempts: %s", _MAX_RETRIES, exc)

    raise RuntimeError(f"LLM failed after {_MAX_RETRIES} attempts: {last_exc}") from last_exc


async def _call_gemini(system_prompt: str, user_prompt: str) -> str:
    """Call Google Gemini Flash via the official SDK."""
    try:
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=system_prompt,
            generation_config={
                "temperature": _TEMPERATURE,
                "max_output_tokens": _MAX_TOKENS,
            },
        )
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, lambda: model.generate_content(user_prompt)
        )
        return response.text

    except Exception as exc:
        logger.error("Gemini error: %s", exc)
        raise RuntimeError(f"Gemini error: {exc}") from exc


async def _call_groq(system_prompt: str, user_prompt: str) -> str:
    """Call Groq (Llama3) via the openai-compatible SDK."""
    try:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=_TEMPERATURE,
                max_tokens=_MAX_TOKENS,
            ),
        )
        return response.choices[0].message.content

    except Exception as exc:
        logger.error("Groq error: %s", exc)
        raise RuntimeError(f"Groq error: {exc}") from exc
