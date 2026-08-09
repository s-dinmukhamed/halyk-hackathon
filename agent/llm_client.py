"""LLM access: Gemini for text + vision, Groq as a text fallback.

Strict-JSON and vision helpers, exponential backoff on 429/5xx, and automatic
Gemini->Groq failover so one provider hitting its rate limit doesn't stall the
run. Kept to httpx only — nothing version-fragile to break on finals day.
"""
from __future__ import annotations

import base64
import json
import re
import sys
import threading
import time
from typing import Optional

import httpx

from .config import settings

# Global throttle: at most 2 concurrent LLM calls, with a minimum gap of 1.5s.
_llm_lock = threading.Semaphore(2)
_llm_last_call: float = 0.0
_llm_call_lock = threading.Lock()
_LLM_MIN_GAP = 1.5  # seconds between calls to stay under free-tier RPM

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class LLMError(RuntimeError):
    pass


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

    models_to_try = [settings.gemini_model]
    for fallback in ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-lite-latest"]:
        if fallback not in models_to_try:
            models_to_try.append(fallback)

    last_err = None
    for model in models_to_try:
        try:
            resp = httpx.post(
                GEMINI_URL.format(model=model),
                params={"key": settings.gemini_api_key},
                json=body,
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                try:
                    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
                except (KeyError, IndexError) as e:
                    raise LLMError(f"gemini malformed response: {data}") from e
            elif resp.status_code in (429, 404):
                print(f"[gemini] model {model} failed with status {resp.status_code}, trying next model in pool...", file=sys.stderr)
                last_err = LLMError(f"gemini {model} {resp.status_code}: {resp.text[:300]}")
                continue
            else:
                raise LLMError(f"gemini {model} {resp.status_code}: {resp.text[:300]}")
        except httpx.HTTPError as e:
            print(f"[gemini] model {model} connection error: {e}, trying next...", file=sys.stderr)
            last_err = LLMError(f"gemini {model} request failed: {e}")
            continue

    if last_err:
        raise last_err
    raise LLMError("No Gemini models succeeded")


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
    global _llm_last_call
    # Rate-limit: at most 2 concurrent callers, min 1.5s between any two calls.
    with _llm_lock:
        with _llm_call_lock:
            since = time.time() - _llm_last_call
            if since < _LLM_MIN_GAP:
                time.sleep(_LLM_MIN_GAP - since)
            _llm_last_call = time.time()

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
