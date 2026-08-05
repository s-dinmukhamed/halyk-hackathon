"""LLM access: Gemini (primary, vision + JSON) with Groq text fallback.

Design goals for the finals window (~3h, free-tier rate limits):
  * `complete_json()` — asks for strict JSON and parses it robustly.
  * `complete_vision()` — send page images to Gemini for scanned/table-heavy PDFs.
  * Exponential backoff on 429/5xx, and automatic Gemini->Groq failover so one
    provider hitting its RPM/RPD limit does not stall the whole run.

We deliberately keep this dependency-light (httpx only) — same approach as the
ai-support-bot llm.py — so there is nothing version-fragile to break on finals day.
"""
from __future__ import annotations

import base64
import json
import re
import time
from typing import Optional

import httpx

from .config import settings

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class LLMError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Low-level provider calls
# --------------------------------------------------------------------------- #
def _gemini(system: str, user: str, images: Optional[list[bytes]] = None,
            json_mode: bool = False) -> str:
    if not settings.gemini_api_key:
        raise LLMError("GEMINI_API_KEY is empty. Free key: https://aistudio.google.com/apikey")

    parts: list[dict] = [{"text": user}]
    for img in images or []:
        parts.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": base64.b64encode(img).decode(),
            }
        })

    body: dict = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": 0.1},
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"

    resp = httpx.post(
        GEMINI_URL.format(model=settings.gemini_model),
        params={"key": settings.gemini_api_key},
        json=body,
        timeout=120,
    )
    if resp.status_code != 200:
        raise LLMError(f"gemini {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as e:
        raise LLMError(f"gemini malformed response: {data}") from e


def _groq(system: str, user: str, json_mode: bool = False) -> str:
    if not settings.groq_api_key:
        raise LLMError("GROQ_API_KEY is empty. Free key: https://console.groq.com")
    body: dict = {
        "model": settings.groq_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.1,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    resp = httpx.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        json=body,
        timeout=120,
    )
    if resp.status_code != 200:
        raise LLMError(f"groq {resp.status_code}: {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"].strip()


# --------------------------------------------------------------------------- #
# Public API with retry + failover
# --------------------------------------------------------------------------- #
def _with_retry(fn, *args, **kwargs) -> str:
    last: Optional[Exception] = None
    for attempt in range(settings.llm_max_retries):
        try:
            return fn(*args, **kwargs)
        except LLMError as e:
            last = e
            msg = str(e)
            # Retry only on transient conditions (rate limit / server error).
            if any(code in msg for code in ("429", "500", "502", "503", "overloaded")):
                time.sleep(settings.llm_backoff_base ** attempt)
                continue
            raise
    raise LLMError(f"exhausted retries: {last}")


def complete(system: str, user: str, images: Optional[list[bytes]] = None,
             json_mode: bool = False) -> str:
    """Text (or vision) completion. Gemini first, Groq fallback on hard failure.

    Groq is text-only, so `images` force Gemini with no fallback.
    """
    provider = settings.llm_provider.lower()

    if provider == "gemini" or images:
        try:
            return _with_retry(_gemini, system, user, images, json_mode)
        except LLMError:
            if images:
                raise  # Groq can't see images
            return _with_retry(_groq, system, user, json_mode)

    # provider == "groq"
    try:
        return _with_retry(_groq, system, user, json_mode)
    except LLMError:
        return _with_retry(_gemini, system, user, None, json_mode)


def complete_json(system: str, user: str, images: Optional[list[bytes]] = None) -> dict | list:
    """Completion parsed as JSON. Tolerates code fences / stray prose."""
    raw = complete(system, user, images=images, json_mode=True)
    return _parse_json(raw)


def complete_vision(system: str, user: str, images: list[bytes],
                    json_mode: bool = True) -> dict | list | str:
    """Send page images to Gemini (scanned pages, complex tables)."""
    raw = _with_retry(_gemini, system, user, images, json_mode)
    return _parse_json(raw) if json_mode else raw


def _parse_json(raw: str) -> dict | list:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Strip ```json fences or extract the first {...}/[...] block.
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            raise LLMError(f"could not parse JSON from LLM output: {raw[:300]}")
