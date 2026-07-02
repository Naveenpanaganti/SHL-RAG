"""
LLM client — supports Groq and Gemini.
All configuration read at call time (never cached at import) so env var
changes take effect without restarting the process.
"""

import asyncio
import logging
import os

logger = logging.getLogger(__name__)

_MAX_TOKENS = 1500
_TEMPERATURE = 0.1
_MAX_RETRIES = 2


# ── All config read lazily — never module-level constants ─────────────────────

def _provider() -> str:
    return os.getenv("LLM_PROVIDER", "groq")

def _gemini_key() -> str:
    return os.getenv("GEMINI_API_KEY", "")

def _gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

def _groq_key() -> str:
    return os.getenv("GROQ_API_KEY", "")

def _groq_model() -> str:
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


# ── Public entry point ────────────────────────────────────────────────────────

async def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Call the configured LLM. Retries once ONLY on transient network failures.
    Does NOT retry on auth errors (401/403) or quota errors (429).
    """
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            provider = _provider()
            logger.info("[LLM] attempt=%d provider=%s", attempt + 1, provider)
            if provider == "gemini":
                return await _call_gemini(system_prompt, user_prompt)
            elif provider == "groq":
                return await _call_groq(system_prompt, user_prompt)
            else:
                raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Use 'gemini' or 'groq'.")
        except Exception as exc:
            last_exc = exc
            err_str = str(exc).lower()
            # Do NOT retry auth or quota failures — they won't resolve on retry
            if any(x in err_str for x in ["401", "403", "api key", "quota", "resource_exhausted", "rate_limit"]):
                logger.error("[LLM] Non-retryable failure: %s", exc)
                break
            if attempt < _MAX_RETRIES - 1:
                wait = 1.5 * (attempt + 1)
                logger.warning("[LLM] Transient failure attempt %d: %s — retrying in %.1fs",
                               attempt + 1, exc, wait)
                await asyncio.sleep(wait)
            else:
                logger.error("[LLM] All attempts failed: %s", exc)

    raise RuntimeError(f"LLM failed: {last_exc}") from last_exc


# ── Gemini ────────────────────────────────────────────────────────────────────

async def _call_gemini(system_prompt: str, user_prompt: str) -> str:
    """Call Google Gemini via the new google-genai SDK."""
    key = _gemini_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    model = _gemini_model()
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        loop = asyncio.get_running_loop()

        def _call():
            response = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=_TEMPERATURE,
                    max_output_tokens=_MAX_TOKENS,
                ),
            )
            return response.text

        return await loop.run_in_executor(None, _call)

    except Exception as exc:
        logger.error("Gemini error: %s", exc)
        raise RuntimeError(f"Gemini error: {exc}") from exc


# ── Groq ──────────────────────────────────────────────────────────────────────

async def _call_groq(system_prompt: str, user_prompt: str) -> str:
    """Call Groq via the openai-compatible SDK."""
    key = _groq_key()
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set")
    model = _groq_model()
    try:
        from groq import Groq

        client = Groq(api_key=key)
        loop = asyncio.get_running_loop()

        def _call():
            return client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=_TEMPERATURE,
                max_tokens=_MAX_TOKENS,
                timeout=25,  # explicit 25s timeout — stays within 30s evaluator deadline
            )

        response = await loop.run_in_executor(None, _call)
        logger.info("[LLM] groq_response_received model=%s", model)
        return response.choices[0].message.content

    except Exception as exc:
        logger.error("Groq error: %s", exc)
        raise RuntimeError(f"Groq error: {exc}") from exc
